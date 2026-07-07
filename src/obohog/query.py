"""Query helpers over a history artifact, backed by DuckDB.

Every interface (CLI, future API, hosted app) goes through :class:`HistoryDB` so
they all answer from the same Parquet files. DuckDB reads the Parquet lazily and
can point at local paths or HTTP URLs, so a hosted artifact needs no server.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb


class ArtifactNotFound(Exception):
    """Raised when an artifact directory lacks the core history tables."""


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
    """One clause add/remove, joined to the commit that made it."""

    commit_seq: int
    committed_date: object
    sha: str
    author_name: str
    pr_number: int | None
    message: str
    operation: str
    predicate: str
    value: str
    branch_commits: tuple[BranchCommit, ...] = ()
    snapshot_url: str | None = None


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
        self.con = duckdb.connect(":memory:")
        for name in ("commits", "term_snapshots", "events", "releases", "skipped"):
            source = self._source(name)
            if source is None:
                continue  # table absent (single-file artifact, or older schema)
            self._create_view(name, source)

    def _create_view(self, name: str, source: str) -> None:
        """Create a DuckDB view over the parquet path ``source``.

        Older artifacts predate the ``snapshot_url`` column on ``commits``;
        wrap them so callers can always ``SELECT c.snapshot_url`` without
        branching. Add-on columns projected here should always be nullable.
        """
        # read_parquet needs a literal path (CREATE VIEW can't bind params);
        # escape single quotes in the path we control.
        literal = source.replace("'", "''")
        self.con.execute(
            f"CREATE VIEW {name}_raw AS SELECT * FROM read_parquet('{literal}')"
        )
        cols = {row[1] for row in self.con.execute(f"PRAGMA table_info('{name}_raw')").fetchall()}
        projections = [f"*"]
        if name == "commits" and "snapshot_url" not in cols:
            projections.append("CAST(NULL AS VARCHAR) AS snapshot_url")
        select_list = ", ".join(projections)
        self.con.execute(
            f"CREATE VIEW {name} AS SELECT {select_list} FROM {name}_raw"
        )

    def _source(self, name: str) -> str | None:
        """Resolve a table to a read_parquet path: part-file dir glob or single file."""
        directory = self.dir / name
        if directory.is_dir():
            return f"{directory}/*.parquet"
        single = self.dir / f"{name}.parquet"
        return str(single) if single.exists() else None

    def term_timeline(self, term_id: str, predicate: str | None = None) -> list[Change]:
        """All changes to a term, oldest first, optionally one clause kind only."""
        where = "e.term_id = ?"
        params: list[object] = [term_id]
        if predicate is not None:
            where += " AND e.predicate = ?"
            params.append(predicate)
        rows = self.con.execute(
            f"""
            SELECT c.commit_seq, c.committed_date, c.sha, c.author_name,
                   c.pr_number, c.message,
                   e.operation, e.predicate, e.value,
                   c.branch_commits, c.snapshot_url
            FROM events e
            JOIN commits c USING (commit_seq)
            WHERE {where}
            ORDER BY c.commit_seq, e.operation, e.predicate, e.value
            """,
            params,
        ).fetchall()
        return [
            Change(
                *row[:9],
                branch_commits=_wrap_branch_commits(row[9]),
                snapshot_url=row[10],
            )
            for row in rows
        ]

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
        self, sha_prefix: str, namespace: str | None = None
    ) -> tuple[Change | None, list[TermChange]]:
        """Full events for one commit, plus a Change-shaped commit header row.

        Returns ``(head, events)``. ``head`` is a ``Change`` whose commit-level
        fields describe the matched commit (its operation/predicate/value are
        empty placeholders — the CLI uses it purely for the ``sha/date/PR/message``
        header). ``events`` is ordered by ``(term_id, operation, predicate, value)``
        so ``groupby(events, key=term_id)`` gives per-term event lists directly
        consumable by :func:`obohog.render.pair_events`. Optionally
        restricted to term IDs with a given CURIE prefix via ``namespace``.

        Returns ``(None, [])`` when no commit matches the sha prefix.
        """
        row = self.con.execute(
            """SELECT commit_seq, sha, author_name, committed_date, pr_number,
                      message, branch_commits, snapshot_url
               FROM commits WHERE sha LIKE ? || '%' ORDER BY commit_seq LIMIT 1""",
            [sha_prefix],
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
        rows = self.con.execute(
            f"""
            SELECT e.term_id, s.name, e.operation, e.predicate, e.value
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
                ),
            )
            for term_id, name, op, pred, val in rows
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
            return int(ref)
        row = self.con.execute(
            "SELECT commit_seq FROM commits WHERE sha LIKE ? || '%' ORDER BY commit_seq LIMIT 1",
            [ref],
        ).fetchone()
        if row is None:
            raise KeyError(f"could not resolve ref {ref!r} to a commit")
        return row[0]

    def range_events(
        self,
        ref_a: str,
        ref_b: str,
        term_id: str | None = None,
        namespace: str | None = None,
    ) -> list[TermChange]:
        """Events in ``(lo, hi]``, one row per clause change.

        ``lo``/``hi`` are the two refs (any of tag, short sha, HEAD, or
        commit_seq — via :meth:`resolve_ref`), sorted so order doesn't matter.
        Optionally restricted to one term (``term_id``) or one CURIE
        prefix (``namespace``, e.g. ``"MONDO"``). Rows are ordered by
        ``(term_id, commit_seq, operation, predicate, value)`` so grouping
        by term (then by commit within term) feeds directly into the render
        pipeline.
        """
        lo, hi = sorted((self.resolve_ref(ref_a), self.resolve_ref(ref_b)))
        where = "e.commit_seq > ? AND e.commit_seq <= ?"
        params: list[object] = [lo, hi]
        if term_id is not None:
            where += " AND e.term_id = ?"
            params.append(term_id)
        if namespace is not None:
            where += " AND starts_with(e.term_id, ? || ':')"
            params.append(namespace)
        rows = self.con.execute(
            f"""
            SELECT e.term_id, s.name,
                   c.commit_seq, c.committed_date, c.sha, c.author_name,
                   c.pr_number, c.message,
                   e.operation, e.predicate, e.value,
                   c.branch_commits, c.snapshot_url
            FROM events e
            JOIN commits c USING (commit_seq)
            LEFT JOIN term_snapshots s
              ON s.term_id = e.term_id AND s.commit_seq = e.commit_seq
            WHERE {where}
            ORDER BY e.term_id, c.commit_seq, e.operation, e.predicate, e.value
            """,
            params,
        ).fetchall()
        return [
            TermChange(
                term_id=term_id,
                name=name,
                change=Change(
                    seq, date, sha, author, pr, message, op, pred, val,
                    branch_commits=_wrap_branch_commits(bc),
                    snapshot_url=snapshot_url,
                ),
            )
            for term_id, name, seq, date, sha, author, pr, message, op, pred, val, bc, snapshot_url in rows
        ]

    def search_events(
        self,
        query: str,
        term_id: str | None = None,
        predicate: str | None = None,
        since_seq: int | None = None,
        regex: bool = False,
        ignore_case: bool = False,
        namespace: str | None = None,
    ) -> list[TermChange]:
        """Events whose clause ``value`` matches ``query``.

        "Which commits added or removed a clause matching this?" —
        analogous to ``git log -S<string>`` (default substring mode) or
        ``git log -G<pattern>`` (``regex=True``) at the file-line level,
        but on our clause-event granularity.

        * ``regex=False`` (default): substring match via DuckDB's
          ``contains()`` — no LIKE wildcard escape logic to write.
        * ``regex=True``: full regex match via DuckDB's
          ``regexp_matches()``. Invalid regex raises DuckDB's parse error
          up to the caller.
        * ``ignore_case=True``: applies to both modes — via ``LOWER()`` on
          both sides for substring, via the ``'i'`` option flag for regex.

        Optional narrowings (all AND'd together): ``term_id`` restricts to
        one term, ``predicate`` restricts to one clause kind (``xref``,
        ``is_a``, ...), ``since_seq`` cuts off commits older than the
        supplied ``commit_seq`` (resolve external refs via
        :meth:`resolve_ref` in the caller), ``namespace`` restricts to
        term IDs whose CURIE prefix is the given value (e.g. ``"MONDO"``).

        Rows come back ordered ``(term_id, commit_seq, operation,
        predicate, value)`` so grouping-by-term-then-commit feeds the
        render pipeline directly.
        """
        if regex:
            if ignore_case:
                where = "regexp_matches(e.value, ?, 'i')"
            else:
                where = "regexp_matches(e.value, ?)"
        else:
            if ignore_case:
                where = "contains(LOWER(e.value), LOWER(?))"
            else:
                where = "contains(e.value, ?)"
        params: list[object] = [query]
        if term_id is not None:
            where += " AND e.term_id = ?"
            params.append(term_id)
        if predicate is not None:
            where += " AND e.predicate = ?"
            params.append(predicate)
        if since_seq is not None:
            where += " AND e.commit_seq >= ?"
            params.append(since_seq)
        if namespace is not None:
            where += " AND starts_with(e.term_id, ? || ':')"
            params.append(namespace)
        rows = self.con.execute(
            f"""
            SELECT e.term_id, s.name,
                   c.commit_seq, c.committed_date, c.sha, c.author_name,
                   c.pr_number, c.message,
                   e.operation, e.predicate, e.value,
                   c.branch_commits, c.snapshot_url
            FROM events e
            JOIN commits c USING (commit_seq)
            LEFT JOIN term_snapshots s
              ON s.term_id = e.term_id AND s.commit_seq = e.commit_seq
            WHERE {where}
            ORDER BY e.term_id, c.commit_seq, e.operation, e.predicate, e.value
            """,
            params,
        ).fetchall()
        return [
            TermChange(
                term_id=term_id,
                name=name,
                change=Change(
                    seq, date, sha, author, pr, message, op, pred, val,
                    branch_commits=_wrap_branch_commits(bc),
                    snapshot_url=snapshot_url,
                ),
            )
            for term_id, name, seq, date, sha, author, pr, message, op, pred, val, bc, snapshot_url in rows
        ]

    def releases(self) -> list[tuple[str, int, object]]:
        if not self._has_releases():
            return []
        return self.con.execute(
            "SELECT tag, commit_seq, date FROM releases ORDER BY commit_seq"
        ).fetchall()

    def _has_releases(self) -> bool:
        return (self.dir / "releases.parquet").exists()

    def close(self) -> None:
        self.con.close()
