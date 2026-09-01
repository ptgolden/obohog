"""Query helpers over a history artifact, backed by DuckDB.

Every interface (CLI, future API, hosted app) goes through :class:`HistoryDB` so
they all answer from the same Parquet files. DuckDB reads the Parquet lazily and
can point at local paths or HTTP URLs, so a hosted artifact needs no server.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

import duckdb

from . import model
from .obo import ParsedValue


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

_QUANTIFIERS = ("ever", "now")  # "during" (as-of) is reserved, unimplemented


class ClauseSyntaxError(ValueError):
    """Raised when a ``has`` clause doesn't parse."""


@dataclass(frozen=True)
class HasClause:
    """One parsed term-set predicate: ``[quantifier:]tag OP value``.

    ``quantifier`` is ``"ever"`` (the term has at some point carried a
    matching clause) or ``"now"`` (a matching clause is present at HEAD
    — or as of the ``until`` bound, when one is given). ``tag=None``
    means any tag; ``value=""`` (substring mode only) means any value.
    """

    quantifier: str  # "ever" | "now"
    tag: str | None
    match: str  # "substring" | "exact" | "regex"
    value: str


def parse_has_clause(raw: str) -> HasClause:
    """Parse ``[quantifier:]tag OP value`` into a :class:`HasClause`.

    The clause splits at its *first* ``~`` (contains) or ``=`` (exact
    body match), so values may freely contain ``~``, ``=``, ``:``, and
    ``/`` — OBO tags contain none of these. A ``~`` value wrapped in
    slashes (``~/pat/``) is a regex over the full value; a substring
    that itself starts and ends with ``/`` therefore needs ``=`` or a
    regex spelling. Examples::

        name~diabetes          ever: name contains "diabetes"
        ~diabetes              ever: any tag contains "diabetes"
        now:xref=XXX:1234567   present now: exact xref body
        synonym~               ever had any synonym clause
        ever:~/dia.*/          any tag matches the regex

    Regex *validity* is not checked here — a bad pattern surfaces as
    DuckDB's parse error on the first query that runs it.
    """
    clause = raw.strip()
    if not clause:
        raise ClauseSyntaxError("empty clause")
    positions = [i for i in (clause.find("~"), clause.find("=")) if i != -1]
    if not positions:
        # No square brackets in the message: the CLI prints it through
        # Rich, which would eat them as markup.
        raise ClauseSyntaxError(
            f"clause {clause!r} needs '~' (contains) or '=' (exact),"
            " e.g. name~diabetes"
        )
    at = min(positions)
    head, op, value = clause[:at].strip(), clause[at], clause[at + 1:]
    if ":" in head:
        quantifier, _, tag = head.partition(":")
        quantifier, tag = quantifier.strip(), tag.strip()
        if quantifier == "during":
            raise ClauseSyntaxError(
                "'during:' is reserved for as-of filtering; not supported yet"
            )
        if quantifier not in _QUANTIFIERS:
            raise ClauseSyntaxError(
                f"unknown quantifier {quantifier!r} (expected 'ever:' or 'now:')"
            )
    else:
        quantifier, tag = "ever", head
    if op == "=":
        if not value:
            raise ClauseSyntaxError(f"exact clause {clause!r} needs a value")
        match = "exact"
    elif len(value) >= 2 and value.startswith("/") and value.endswith("/"):
        match, value = "regex", value[1:-1]
        if not value:
            raise ClauseSyntaxError(f"empty regex in clause {clause!r}")
    else:
        match = "substring"
    return HasClause(
        quantifier=quantifier, tag=tag or None, match=match, value=value
    )


class EventCounts(NamedTuple):
    """Result-set scope for an events query: totals by three grains."""

    events: int
    terms: int
    commits: int


class ArtifactNotFound(Exception):
    """Raised when an artifact directory lacks the core history tables."""


class SchemaMismatch(Exception):
    """Raised when an artifact was built with a different schema version.

    Queries assume the current schema's columns exist; letting a stale
    artifact through surfaces as a confusing DuckDB binder error deep in
    some query instead of a clear "rebuild me".
    """


class RefNotFound(Exception):
    """Raised when a user-supplied ref matches no tag, sha, or commit_seq."""


@dataclass(frozen=True)
class RangeFilters:
    """Optional narrowings for range (diff) queries.

    ``term_id`` restricts to one term; ``namespace`` to term IDs with the
    given CURIE prefix (e.g. ``"MONDO"``).
    """

    term_id: str | None = None
    namespace: str | None = None


@dataclass(frozen=True)
class SearchFilters:
    """Match flags and optional narrowings for search queries.

    One frozen object feeds both the count and the iterator call for a
    search, so the two can't disagree. Field meanings as documented on
    :meth:`HistoryDB._search_where`.
    """

    term_id: str | None = None
    tag: str | None = None
    since_seq: int | None = None
    until_seq: int | None = None
    match: str = "substring"  # "substring" | "exact" | "regex"
    ignore_case: bool = False
    namespace: str | None = None
    # Term-set predicates: restrict to terms satisfying every clause
    # (see :func:`parse_has_clause`). A term-axis narrowing — it picks
    # *whose* events are eligible; the other fields (and the query) pick
    # which of those events show.
    has: tuple[HasClause, ...] = ()


def _wrap_parsed(body: str, qualifiers, comment: str | None) -> ParsedValue:
    """Convert the artifact's decomposition columns into a ParsedValue."""
    return ParsedValue(
        body=body, qualifiers=tuple(qualifiers or ()), comment=comment
    )


# The SELECT list that fully populates a :class:`Change`, in the exact order
# :func:`_change_from_row` unpacks. Queries that build Changes interpolate
# this (optionally after extra leading columns) so the column list and the
# row mapping can't drift apart.
_CHANGE_COLUMNS = """c.commit_seq, c.committed_date, c.sha, c.author_name,
       c.pr_number, c.message,
       e.operation, e.predicate, e.value,
       c.branch_commits, c.snapshot_url,
       e.body, e.qualifiers, e.comment"""


def _change_from_row(row) -> "Change":
    """Build a Change from a row SELECTed with ``_CHANGE_COLUMNS``."""
    return Change(
        *row[:9],
        branch_commits=_wrap_branch_commits(row[9]),
        snapshot_url=row[10],
        parsed=_wrap_parsed(row[11], row[12], row[13]),
    )


def _wrap_branch_commits(raw) -> tuple["BranchCommit", ...]:
    """Convert a duckdb list<struct> result into a tuple of BranchCommit."""
    if not raw:
        return ()
    return tuple(
        BranchCommit(
            sha=entry["sha"],
            author_name=entry["author_name"],
            committed_date=entry["committed_date"],
            message=entry["message"],
        )
        for entry in raw
    )


@dataclass(frozen=True)
class BranchCommit:
    """One commit on the merged branch of a merge commit (typically a PR)."""

    sha: str
    author_name: str
    committed_date: object
    message: str


@dataclass(frozen=True)
class Change:
    """One clause add/remove, joined to the commit that made it.

    ``parsed`` is the value's structural decomposition, read from the
    artifact's body/qualifiers/comment columns. It is always present on
    event rows; ``None`` only on synthetic commit-header rows (see
    :meth:`HistoryDB.commit_events`) whose value is a placeholder.
    """

    commit_seq: int
    committed_date: object
    sha: str
    author_name: str
    pr_number: int | None
    message: str
    operation: str
    tag: str
    value: str
    branch_commits: tuple[BranchCommit, ...] = ()
    snapshot_url: str | None = None
    parsed: ParsedValue | None = None


@dataclass(frozen=True)
class TermChange:
    """A ``Change`` tagged with its term's id and display name.

    Multi-term views (``commit``, ``diff``) enumerate events across many
    terms, so callers need the term id per row and a display name for the
    per-term section header. ``name`` is the term's name at the specific
    event's commit (from ``term_snapshots``), or ``None`` if that snapshot
    has no name (e.g. term-removal events).
    """

    term_id: str
    name: str | None
    change: Change


@dataclass(frozen=True)
class TermHeader:
    """Summary stats for a term, used to orient the timeline view."""

    term_id: str
    current_name: str | None
    event_count: int
    first_sha: str
    first_date: object
    last_sha: str
    last_date: object
    last_pr: int | None


class HistoryDB:
    _CORE = ("commits", "term_snapshots", "events")

    def __init__(self, artifact_dir: Path | str):
        self.dir = Path(artifact_dir)
        absent = [name for name in self._CORE if self._source(name) is None]
        if absent:
            raise ArtifactNotFound(
                f"No history artifact at '{self.dir}' (missing: {', '.join(absent)}). "
                "Run `obohog source sync <name>` first."
            )
        meta = model.read_build_meta(self.dir)
        built = meta.schema_version if meta else None
        if built != model.SCHEMA_VERSION:
            raise SchemaMismatch(
                f"Artifact at '{self.dir}' was built with schema "
                f"{built or 'unknown'}; this obohog reads schema "
                f"{model.SCHEMA_VERSION}. Rebuild it with `obohog source sync <name>`."
            )
        self.con = duckdb.connect(":memory:")
        for name in ("commits", "term_snapshots", "events", "releases", "skipped"):
            source = self._source(name)
            if source is None:
                continue  # table absent (single-file artifact, or older schema)
            self._create_view(name, source)

    def _create_view(self, name: str, source: str) -> None:
        """Create a DuckDB view over the parquet path ``source``.

        No per-column back-compat here: the schema check at open refuses
        artifacts that don't match ``model.SCHEMA_VERSION``, so every
        current column is guaranteed present.
        """
        # read_parquet needs a literal path (CREATE VIEW can't bind params);
        # escape single quotes in the path we control.
        literal = source.replace("'", "''")
        self.con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{literal}')"
        )

    def _source(self, name: str) -> str | None:
        """Resolve a table to a read_parquet path: part-file dir glob or single file."""
        directory = self.dir / name
        if directory.is_dir():
            return f"{directory}/*.parquet"
        single = self.dir / f"{name}.parquet"
        return str(single) if single.exists() else None

    def term_timeline(self, term_id: str, tag: str | None = None) -> list[Change]:
        """All changes to a term, oldest first, optionally one clause kind only."""
        where = "e.term_id = ?"
        params: list[object] = [term_id]
        if tag is not None:
            where += " AND e.predicate = ?"
            params.append(tag)
        rows = self.con.execute(
            f"""
            SELECT {_CHANGE_COLUMNS}
            FROM events e
            JOIN commits c USING (commit_seq)
            WHERE {where}
            ORDER BY c.commit_seq, e.operation, e.predicate, e.value
            """,
            params,
        ).fetchall()
        return [_change_from_row(row) for row in rows]

    def term_header(self, term_id: str) -> TermHeader | None:
        """Orientation stats for the term, or ``None`` if it has no events."""
        stats = self.con.execute(
            """
            SELECT min(commit_seq), max(commit_seq), count(*)
            FROM events WHERE term_id = ?
            """,
            [term_id],
        ).fetchone()
        if stats is None or stats[0] is None:
            return None
        first_seq, last_seq, event_count = stats
        first_sha, first_date = self.con.execute(
            "SELECT sha, committed_date FROM commits WHERE commit_seq = ?",
            [first_seq],
        ).fetchone()
        last_sha, last_date, last_pr = self.con.execute(
            "SELECT sha, committed_date, pr_number FROM commits WHERE commit_seq = ?",
            [last_seq],
        ).fetchone()
        name_row = self.con.execute(
            """
            SELECT name FROM term_snapshots
            WHERE term_id = ? AND name IS NOT NULL
            ORDER BY commit_seq DESC LIMIT 1
            """,
            [term_id],
        ).fetchone()
        return TermHeader(
            term_id=term_id,
            current_name=name_row[0] if name_row else None,
            event_count=event_count,
            first_sha=first_sha,
            first_date=first_date,
            last_sha=last_sha,
            last_date=last_date,
            last_pr=last_pr,
        )

    def term_at(self, term_id: str, commit_seq: int) -> list[tuple[str, str]]:
        """Reconstruct a term's clauses as of ``commit_seq`` (latest snapshot <=)."""
        row = self.con.execute(
            """
            SELECT clauses FROM term_snapshots
            WHERE term_id = ? AND commit_seq <= ?
            ORDER BY commit_seq DESC LIMIT 1
            """,
            [term_id, commit_seq],
        ).fetchone()
        if row is None:
            return []
        return [(c["predicate"], c["value"]) for c in row[0]]

    def commit_events(
        self,
        ref: str,
        namespace: str | None = None,
        after: str | None = None,
    ) -> tuple[Change | None, list[TermChange]]:
        """Full events for one commit, plus a Change-shaped commit header row.

        ``ref`` is anything :meth:`resolve_ref` accepts — a sha prefix, a
        release tag, a commit_seq, or HEAD; synthetic-history sources
        (whose shas reference nothing) are addressed by seq.

        Returns ``(head, events)``. ``head`` is a ``Change`` whose commit-level
        fields describe the matched commit (its operation/predicate/value are
        empty placeholders — the CLI uses it purely for the ``sha/date/PR/message``
        header). ``events`` is ordered by ``(term_id, operation, predicate, value)``
        so ``groupby(events, key=term_id)`` gives per-term event lists directly
        consumable by :func:`obohog.render.pair_events`. Optionally
        restricted to term IDs with a given CURIE prefix via ``namespace``;
        ``after`` is a term_id keyset cursor — only events for strictly
        later term IDs are returned (paged consumers resume with the last
        term they rendered).

        Returns ``(None, [])`` when the ref matches no commit.
        """
        try:
            seq = self.resolve_ref(ref)
        except RefNotFound:
            return None, []
        row = self.con.execute(
            """SELECT commit_seq, sha, author_name, committed_date, pr_number,
                      message, branch_commits, snapshot_url
               FROM commits WHERE commit_seq = ?""",
            [seq],
        ).fetchone()
        if row is None:
            return None, []
        commit_seq, sha, author, date, pr, message, raw_bc, snapshot_url = row
        branch_commits = _wrap_branch_commits(raw_bc)
        head = Change(
            commit_seq, date, sha, author, pr, message, "", "", "",
            branch_commits=branch_commits,
            snapshot_url=snapshot_url,
        )

        where = "e.commit_seq = ?"
        params: list[object] = [commit_seq]
        if namespace is not None:
            where += " AND starts_with(e.term_id, ? || ':')"
            params.append(namespace)
        if after is not None:
            where += " AND e.term_id > ?"
            params.append(after)
        rows = self.con.execute(
            f"""
            SELECT e.term_id, s.name, e.operation, e.predicate, e.value,
                   e.body, e.qualifiers, e.comment
            FROM events e
            LEFT JOIN term_snapshots s
              ON s.term_id = e.term_id AND s.commit_seq = e.commit_seq
            WHERE {where}
            ORDER BY e.term_id, e.operation, e.predicate, e.value
            """,
            params,
        ).fetchall()
        events = [
            TermChange(
                term_id=term_id,
                name=name,
                change=Change(
                    commit_seq, date, sha, author, pr, message, op, pred, val,
                    branch_commits=branch_commits,
                    snapshot_url=snapshot_url,
                    parsed=_wrap_parsed(body, qualifiers, comment),
                ),
            )
            for term_id, name, op, pred, val, body, qualifiers, comment in rows
        ]
        return head, events

    def pr_terms(self, pr_number: int) -> list[tuple[str, str | None]]:
        """Terms changed by any commit belonging to a pull request."""
        return self.con.execute(
            """
            SELECT DISTINCT e.term_id, s.name
            FROM events e
            JOIN commits c USING (commit_seq)
            LEFT JOIN term_snapshots s
              ON s.term_id = e.term_id AND s.commit_seq = e.commit_seq
            WHERE c.pr_number = ?
            ORDER BY e.term_id
            """,
            [pr_number],
        ).fetchall()

    def resolve_ref(self, ref: str) -> int:
        """Resolve HEAD, a release tag, a commit_seq, or a sha prefix to a seq."""
        if ref.upper() == "HEAD":
            return self.con.execute("SELECT max(commit_seq) FROM commits").fetchone()[0]
        row = self.con.execute(
            "SELECT commit_seq FROM releases WHERE tag = ?", [ref]
        ).fetchone() if self._has_releases() else None
        if row is not None:
            return row[0]
        if ref.isdigit():
            row = self.con.execute(
                "SELECT commit_seq FROM commits WHERE commit_seq = ?",
                [int(ref)],
            ).fetchone()
            if row is not None:
                return int(ref)
            # No such seq — an all-digit sha prefix falls through.
        row = self.con.execute(
            "SELECT commit_seq FROM commits WHERE sha LIKE ? || '%' ORDER BY commit_seq LIMIT 1",
            [ref],
        ).fetchone()
        if row is None:
            raise RefNotFound(
                f"could not resolve ref {ref!r} — expected a release tag, "
                "short sha, HEAD, or commit_seq"
            )
        return row[0]

    def resolve_bound(self, ref: str, *, end: bool = False) -> int:
        """Resolve a ref *or* a ``YYYY-MM-DD`` date to a commit_seq bound.

        Non-dates go through :meth:`resolve_ref` unchanged. A date is
        direction-aware: as a start bound it resolves to the first commit
        on or after that day, as an end bound (``end=True``) to the last
        commit on or before the end of it — so ``since=2025-03-01`` with
        ``until=2025-03-31`` means "during March 2025", both ends
        inclusive. A date beyond the history's edge resolves to a
        sentinel seq that matches nothing in that direction. (A release
        tag that happens to look like a date wins — same commit either
        way in practice.)
        """
        if not _DATE_RE.fullmatch(ref):
            return self.resolve_ref(ref)
        if self._has_releases():
            row = self.con.execute(
                "SELECT commit_seq FROM releases WHERE tag = ?", [ref]
            ).fetchone()
            if row is not None:
                return row[0]
        agg, cmp = ("max", "<=") if end else ("min", ">=")
        row = self.con.execute(
            f"SELECT {agg}(commit_seq) FROM commits"
            f" WHERE CAST(committed_date AS DATE) {cmp} CAST(? AS DATE)",
            [ref],
        ).fetchone()
        if row[0] is None:
            return -1 if end else self.resolve_ref("HEAD") + 1
        return row[0]

    def _range_where(
        self, ref_a: str, ref_b: str, f: RangeFilters
    ) -> tuple[str, list[object]]:
        """WHERE clause + params for events in the range ``(lo, hi]``."""
        lo, hi = sorted((self.resolve_ref(ref_a), self.resolve_ref(ref_b)))
        where = "e.commit_seq > ? AND e.commit_seq <= ?"
        params: list[object] = [lo, hi]
        if f.term_id is not None:
            where += " AND e.term_id = ?"
            params.append(f.term_id)
        if f.namespace is not None:
            where += " AND starts_with(e.term_id, ? || ':')"
            params.append(f.namespace)
        return where, params

    @staticmethod
    def _search_where(query: str | None, f: SearchFilters) -> tuple[str, list[object]]:
        """WHERE clause + params for events whose value matches ``query``.

        ``query=None`` means no text constraint at all — every event
        matches, and the optional narrowings below do the filtering
        (browse mode). Otherwise, three match modes (``f.match``):

        * ``"substring"`` (default): substring of the full ``value`` via
          DuckDB's ``contains()`` — no LIKE wildcard escape logic to write.
        * ``"exact"``: equality against ``body`` — the value minus its
          trailing ``{...}`` modifiers and ``!`` comment — so an exact
          clause body matches regardless of qualifiers.
        * ``"regex"``: full regex over ``value`` via DuckDB's
          ``regexp_matches()``. Invalid regex raises DuckDB's parse error
          up to the caller.

        ``ignore_case=True`` applies to every mode — ``LOWER()`` on both
        sides for substring/exact, the ``'i'`` option flag for regex.

        Optional narrowings (all AND'd together): ``term_id`` restricts to
        one term, ``tag`` restricts to one clause kind (``xref``,
        ``is_a``, ...), ``since_seq``/``until_seq`` cut off commits
        outside ``[since_seq, until_seq]`` (resolve external refs or
        dates via :meth:`resolve_bound` in the caller), ``namespace``
        restricts to
        term IDs whose CURIE prefix is the given value (e.g. ``"MONDO"``).

        ``has`` clauses narrow along the *term* axis: only events of
        terms satisfying every clause (see :meth:`_has_subquery`) are
        eligible. They compose freely with everything above — the
        canonical shape is "changes to xrefs containing NCIT (query +
        tag) among terms whose names have ever contained diabetes
        (has)". With no query at all, the eligible terms' histories
        stream whole (clipped only by the date window, if any).
        """
        if query is None:
            where = "TRUE"
        elif f.match == "regex":
            if f.ignore_case:
                where = "regexp_matches(e.value, ?, 'i')"
            else:
                where = "regexp_matches(e.value, ?)"
        elif f.match == "exact":
            if f.ignore_case:
                where = "LOWER(e.body) = LOWER(?)"
            else:
                where = "e.body = ?"
        else:
            if f.ignore_case:
                where = "contains(LOWER(e.value), LOWER(?))"
            else:
                where = "contains(e.value, ?)"
        params: list[object] = [] if query is None else [query]
        if f.term_id is not None:
            where += " AND e.term_id = ?"
            params.append(f.term_id)
        if f.tag is not None:
            where += " AND e.predicate = ?"
            params.append(f.tag)
        if f.since_seq is not None:
            where += " AND e.commit_seq >= ?"
            params.append(f.since_seq)
        if f.until_seq is not None:
            where += " AND e.commit_seq <= ?"
            params.append(f.until_seq)
        if f.namespace is not None:
            where += " AND starts_with(e.term_id, ? || ':')"
            params.append(f.namespace)
        if f.has:
            sub, sub_params = HistoryDB._has_subquery(f.has, f.ignore_case)
            where += f" AND e.term_id IN {sub}"
            params.extend(sub_params)
        return where, params

    @staticmethod
    def _has_subquery(
        clauses: Sequence[HasClause], ignore_case: bool
    ) -> tuple[str, list[object]]:
        """A ``SELECT term_id`` subquery for terms satisfying every clause.

        Per clause the WHERE comes from :meth:`_search_where` (a
        clause-only :class:`SearchFilters`, so no recursion); clauses
        combine by INTERSECT, so a term must satisfy all of them (AND).
        Membership is deliberately timeless — no date bounds in here;
        the outer WHERE's ``since_seq``/``until_seq`` clip which of the
        eligible terms' events *show*, never who qualifies.

        ``ever`` clauses match any event in history. ``now`` clauses
        keep a term when some matching ``(term, tag, value)`` group's
        last operation is an add — the clause is present at HEAD. The
        arg_max can't tie because extraction diffs snapshots and so
        never emits an add and a remove of the identical
        ``(term, predicate, value)`` in one commit.
        """
        parts: list[str] = []
        params: list[object] = []
        for clause in clauses:
            per = SearchFilters(
                tag=clause.tag, match=clause.match, ignore_case=ignore_case
            )
            where, p = HistoryDB._search_where(clause.value or None, per)
            select = f"SELECT e.term_id FROM events e WHERE {where}"
            if clause.quantifier == "now":
                select += (
                    " GROUP BY e.term_id, e.predicate, e.value"
                    " HAVING arg_max(e.operation, e.commit_seq) = 'add'"
                )
            parts.append(select)
            params.extend(p)
        return "(" + "\nINTERSECT\n".join(parts) + ")", params

    def _iter_term_changes(
        self,
        where: str,
        params: list[object],
        batch_size: int = 10_000,
        order: str = "term",
        reverse: bool = False,
        after: str | int | None = None,
    ) -> Iterator[TermChange]:
        """Stream ``TermChange`` rows for a WHERE over events, in render order.

        Rows are ordered ``(term_id, commit_seq, operation, predicate,
        value)`` so grouping-by-term-then-commit feeds the render pipeline
        directly — and so ``(term_id, commit_seq)`` works as a resume
        cursor for paged consumers. DuckDB produces sorted results
        incrementally, so the first batch is available almost immediately
        regardless of total result size; Change construction is amortized
        across consumption instead of paid up front.

        ``order`` picks the stream's grouping spine:

        * ``"term"`` — ``(term_id, commit_seq, ...)``: per-term sections,
          each term's history chronological. The default everywhere.
        * ``"date"`` — ``(commit_seq DESC, term_id, ...)``: newest commit
          first, terms grouped within each commit — the ``git log`` shape.
          A consumer that stops after N commit groups (``--limit``) makes
          DuckDB produce only a prefix of the sort.

        ``reverse`` flips the commit-time direction within the chosen
        spine (``git log --reverse`` analog): date order becomes oldest
        first; term order keeps its A→Z sections but lists each term's
        history newest first.

        ``after`` is the keyset cursor for paged consumers: the *section
        key* of the last section a previous page emitted — a ``term_id``
        for term order (sections are always A→Z regardless of
        ``reverse``), a ``commit_seq`` for date order. Resumption is
        strictly-after in the stream's direction, so pages concatenate to
        exactly the unpaged stream.
        """
        direction = {
            ("term", False): "ASC", ("term", True): "DESC",
            ("date", False): "DESC", ("date", True): "ASC",
        }[(order, reverse)]
        if after is not None:
            if order == "term":
                where = f"({where}) AND e.term_id > ?"
                params = [*params, str(after)]
            else:
                cmp = ">" if direction == "ASC" else "<"
                where = f"({where}) AND e.commit_seq {cmp} ?"
                params = [*params, int(after)]
        order_by = {
            "term": f"e.term_id, c.commit_seq {direction}, e.operation, e.predicate, e.value",
            "date": f"c.commit_seq {direction}, e.term_id, e.operation, e.predicate, e.value",
        }[order]
        cur = self.con.execute(
            f"""
            SELECT e.term_id, s.name,
                   {_CHANGE_COLUMNS}
            FROM events e
            JOIN commits c USING (commit_seq)
            LEFT JOIN term_snapshots s
              ON s.term_id = e.term_id AND s.commit_seq = e.commit_seq
            WHERE {where}
            ORDER BY {order_by}
            """,
            params,
        )
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                return
            for row in rows:
                yield TermChange(
                    term_id=row[0], name=row[1], change=_change_from_row(row[2:])
                )

    def _count_events(self, where: str, params: list[object]) -> EventCounts:
        row = self.con.execute(
            f"""
            SELECT count(*), count(DISTINCT e.term_id), count(DISTINCT e.commit_seq)
            FROM events e
            WHERE {where}
            """,
            params,
        ).fetchone()
        return EventCounts(*row)

    def iter_range_events(
        self,
        ref_a: str,
        ref_b: str,
        filters: RangeFilters = RangeFilters(),
        *,
        after: str | None = None,
    ) -> Iterator[TermChange]:
        """Stream events in ``(lo, hi]``, one row per clause change.

        ``lo``/``hi`` are the two refs (any of tag, short sha, HEAD, or
        commit_seq — via :meth:`resolve_ref`), sorted so order doesn't
        matter. Term-ordered; ``after`` is the term-id section cursor
        (see :meth:`_iter_term_changes`).
        """
        return self._iter_term_changes(
            *self._range_where(ref_a, ref_b, filters), after=after
        )

    def range_events(
        self, ref_a: str, ref_b: str, filters: RangeFilters = RangeFilters()
    ) -> list[TermChange]:
        """Materialized :meth:`iter_range_events`."""
        return list(self.iter_range_events(ref_a, ref_b, filters))

    def range_counts(
        self, ref_a: str, ref_b: str, filters: RangeFilters = RangeFilters()
    ) -> EventCounts:
        """Event/term/commit counts for a range — exact (no post-filter)."""
        return self._count_events(*self._range_where(ref_a, ref_b, filters))

    def iter_search_events(
        self,
        query: str | None,
        filters: SearchFilters = SearchFilters(),
        *,
        order: str = "term",
        reverse: bool = False,
        after: str | int | None = None,
    ) -> Iterator[TermChange]:
        """Stream events whose clause ``value`` matches ``query``.

        "Which commits added or removed a clause matching this?" —
        analogous to ``git log -S<string>`` (default substring mode) or
        ``git log -G<pattern>`` (``match="regex"``) at the file-line level,
        but on our clause-event granularity. Match semantics and filters
        as in :meth:`_search_where`; ``order``/``reverse``/``after`` as
        in :meth:`_iter_term_changes`.

        The stream carries **whole (term, commit) groups**: the text match
        picks candidate groups via a semi-join, then every event of those
        groups (under the same non-text narrowings) flows through. Pairing
        needs both sides of an edit even when only one side's text matches
        — a query hitting only the removed value must render as a ``~`` of
        that clause, not as a fake whole-clause deletion.

        These are *candidate* rows: the clause-aware op filter
        (``obohog.render.filter_ops_by_delta_match``) re-matches the query
        per op — the paired-partner and same-group events pulled in by the
        semi-join don't show unless their own delta involves the query.
        """
        where, params = self._search_where(query, filters)
        if query is not None:
            outer, outer_params = self._search_where(None, filters)
            where = (
                f"{outer} AND (e.term_id, e.commit_seq) IN "
                f"(SELECT e.term_id, e.commit_seq FROM events e WHERE {where})"
            )
            params = [*outer_params, *params]
        return self._iter_term_changes(
            where, params, order=order, reverse=reverse, after=after,
        )

    def search_events(
        self, query: str | None, filters: SearchFilters = SearchFilters()
    ) -> list[TermChange]:
        """Materialized :meth:`iter_search_events`."""
        return list(self.iter_search_events(query, filters))

    def search_counts(
        self, query: str | None, filters: SearchFilters = SearchFilters()
    ) -> EventCounts:
        """Candidate event/term/commit counts for a search.

        Counts SQL-level value matches — an upper bound on what survives
        the delta filter. Cheap (~fraction of a second on millions of
        events), so callers can show scope up front and paged consumers
        can size result sets.
        """
        return self._count_events(*self._search_where(query, filters))

    def commit_stats(
        self,
        filters: SearchFilters = SearchFilters(),
        *,
        reverse: bool = False,
        after: int | None = None,
    ) -> list[tuple[int, int, int]]:
        """Per-commit ``(commit_seq, events, terms)`` on the date spine.

        The cheap skeleton a browse page is planned from: an aggregate
        with no join and no big sort, so a caller can decide which
        commits a page shows — and knows each one's exact term count —
        before fetching a single event. Same filters as the event
        stream; commits with no eligible events don't appear. ``after``
        is the date-order keyset cursor, strictly-after in the stream's
        direction (newest first unless ``reverse``).
        """
        where, params = self._search_where(None, filters)
        direction = "ASC" if reverse else "DESC"
        if after is not None:
            cmp = ">" if reverse else "<"
            where = f"({where}) AND e.commit_seq {cmp} ?"
            params = [*params, int(after)]
        return self.con.execute(
            f"""
            SELECT e.commit_seq, count(*), count(DISTINCT e.term_id)
            FROM events e
            WHERE {where}
            GROUP BY e.commit_seq
            ORDER BY e.commit_seq {direction}
            """,
            params,
        ).fetchall()

    def capped_term_bounds(
        self,
        seqs: Sequence[int],
        cap: int,
        filters: SearchFilters = SearchFilters(),
    ) -> dict[int, str]:
        """Per commit, the ``cap``-th eligible term_id in term order.

        The inclusive upper bound a planned browse page fetches for a
        commit too wide to show whole: events of terms past it stay
        unread in DuckDB. Term order here (``ORDER BY term_id``) must
        match the event stream's within-commit order.
        """
        where, params = self._search_where(None, filters)
        marks = ", ".join("?" for _ in seqs)
        rows = self.con.execute(
            f"""
            SELECT commit_seq, max(term_id) FROM (
                SELECT e.commit_seq, e.term_id,
                       dense_rank() OVER (
                           PARTITION BY e.commit_seq ORDER BY e.term_id
                       ) AS rk
                FROM events e
                WHERE ({where}) AND e.commit_seq IN ({marks})
            )
            WHERE rk <= ?
            GROUP BY commit_seq
            """,
            [*params, *seqs, cap],
        ).fetchall()
        return {seq: bound for seq, bound in rows}

    def iter_browse_page_events(
        self,
        seqs: Sequence[int],
        term_bounds: dict[int, str],
        filters: SearchFilters = SearchFilters(),
        *,
        reverse: bool = False,
    ) -> Iterator[TermChange]:
        """Stream exactly one planned browse page's events, date-ordered.

        ``seqs`` are the page's commits (from :meth:`commit_stats`);
        those in ``term_bounds`` contribute only terms up to their bound
        (from :meth:`capped_term_bounds`). The sort input is one page's
        worth of rows — no wasted work on events past the page.
        """
        where, params = self._search_where(None, filters)
        parts: list[str] = []
        extra: list[object] = []
        whole = [s for s in seqs if s not in term_bounds]
        if whole:
            marks = ", ".join("?" for _ in whole)
            parts.append(f"e.commit_seq IN ({marks})")
            extra.extend(whole)
        for seq in seqs:
            if seq in term_bounds:
                parts.append("(e.commit_seq = ? AND e.term_id <= ?)")
                extra.extend([seq, term_bounds[seq]])
        where = f"({where}) AND ({' OR '.join(parts)})"
        return self._iter_term_changes(
            where, [*params, *extra], order="date", reverse=reverse
        )

    def facets(self) -> tuple[list[str], list[str]]:
        """Distinct ``(tags, namespaces)`` present in the events.

        Both are low-cardinality (a dozen-odd values even on millions of
        events) and ordered alphabetically, so filter UIs can offer them
        as controlled choices instead of free text. Namespace is the
        CURIE prefix of ``term_id``, matching the ``namespace`` filters'
        ``starts_with(term_id, ns || ':')`` semantics.
        """
        tags = [
            row[0]
            for row in self.con.execute(
                "SELECT predicate FROM events"
                " GROUP BY predicate ORDER BY predicate"
            ).fetchall()
        ]
        namespaces = [
            row[0]
            for row in self.con.execute(
                "SELECT split_part(term_id, ':', 1) AS ns FROM events"
                " GROUP BY ns ORDER BY ns"
            ).fetchall()
        ]
        return tags, namespaces

    def releases(self) -> list[tuple[str, int, object]]:
        if not self._has_releases():
            return []
        return self.con.execute(
            "SELECT tag, commit_seq, date FROM releases ORDER BY commit_seq"
        ).fetchall()

    def _has_releases(self) -> bool:
        return (self.dir / "releases.parquet").exists()

    def fork(self) -> "HistoryDB":
        """A second handle on this artifact, for concurrent use.

        A DuckDB connection must not run queries from multiple threads at
        once, but cursors on one in-memory connection share its catalog
        (the views built at open) while executing independently. A server
        opens one ``HistoryDB`` per artifact and hands each request a
        fork, closing it after the response; the artifact validation
        already happened when the parent opened. Closing the parent
        invalidates its forks.
        """
        clone = object.__new__(HistoryDB)
        clone.dir = self.dir
        clone.con = self.con.cursor()
        return clone

    def close(self) -> None:
        self.con.close()
