"""The shared service layer: every server-facing query goes through here.

The JSON API, the HTML pages, and any future agent adapter (MCP) call
these functions and get back Pydantic models — bounded pages with keyset
cursors, ops pre-rendered to role-tagged spans (:func:`obohog.render.op_view`),
commit metadata resolved against the source's :class:`~obohog.views.SourceStyle`.
Adapters only translate shapes to a wire format; no query or presentation
decision happens outside this module.

Deliberately framework-free: nothing here may import FastAPI (or any web
machinery), so a future MCP server stays a page of thin wrappers.
"""

from collections import Counter
from datetime import datetime
from itertools import groupby
from typing import Generic, Iterator, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator

from . import model, render
from .config import AnySource, Config
from .query import Change, HistoryDB, RangeFilters, SearchFilters
from .views import SourceStyle, pr_title_from_merge

# Bounded-by-default paging: a page holds at most this many *sections*
# (a term for term-ordered streams, a commit for date-ordered ones).
DEFAULT_PAGE = 50
MAX_PAGE = 500


class InvalidCursor(ValueError):
    """Raised when an ``after`` cursor doesn't fit the requested order."""


# ---------------------------------------------------------------------------
# Response models. Dates arrive from DuckDB as datetimes and serialize to
# ISO-8601 via pydantic; ops carry both the role-tagged spans (presentation)
# and the raw clause values (data).


class SpanOut(BaseModel):
    role: str  # "same" | "del" | "ins" | "note"
    text: str


class QualLineOut(BaseModel):
    kind: str  # "context" | "del" | "ins" | "edit"
    spans: list[SpanOut]


class OpOut(BaseModel):
    kind: str  # "add" | "remove" | "edit"
    tag: str
    before: str | None  # raw removed value (remove/edit)
    after: str | None  # raw added value (add/edit)
    head: list[SpanOut]
    quals: list[QualLineOut]


class BranchCommitOut(BaseModel):
    sha: str
    author_name: str
    committed_date: datetime
    message: str


class CommitOut(BaseModel):
    """Commit metadata as shown in headers, style already applied.

    ``subject`` is the editorial line the CLI prints: the embedded PR
    title for classic GitHub merges, else the raw first message line,
    with the source's ``subject_prefix`` prepended. ``raw_subject`` keeps
    the unprocessed first line whenever ``subject`` replaced it.
    """

    commit_seq: int
    sha: str
    hide_sha: bool  # synthetic-git sources: sha references nothing public
    committed_date: datetime
    author_name: str
    subject: str
    raw_subject: str | None
    pr_number: int | None
    pr_url: str | None
    snapshot_url: str | None
    branch_commits: list[BranchCommitOut]


class CommitGroupOut(BaseModel):
    """One commit's paired ops (within one term's history)."""

    commit: CommitOut
    ops: list[OpOut]


class TermGroupOut(BaseModel):
    """One term's paired ops (within one commit's changes)."""

    term_id: str
    name: str | None
    ops: list[OpOut]


class TermSectionOut(BaseModel):
    """A term-major page section: one term, its matching commits in order."""

    term_id: str
    name: str | None
    commits: list[CommitGroupOut]


class CommitSectionOut(BaseModel):
    """A date-major page section: one commit, its affected terms."""

    commit: CommitOut
    terms: list[TermGroupOut]


class CountsOut(BaseModel):
    events: int
    terms: int
    commits: int
    # Search counts are SQL-level candidates — an upper bound on what
    # survives the clause-aware delta filter. Range counts are exact.
    approximate: bool = False


S = TypeVar("S", TermSectionOut, CommitSectionOut)


class PageOut(BaseModel, Generic[S]):
    sections: list[S]
    # Section key of the last section on this page, when more remain:
    # pass back as `after` to resume. None on the final page.
    next_cursor: str | None
    counts: CountsOut


class TermHeaderOut(BaseModel):
    term_id: str
    current_name: str | None
    event_count: int
    first_sha: str
    first_date: datetime
    last_sha: str
    last_date: datetime
    last_pr: int | None


class TimelineOut(BaseModel):
    term_id: str
    header: TermHeaderOut | None
    total_events: int
    shown_events: int
    by_tag: dict[str, int]
    commits: list[CommitGroupOut]  # oldest first


class StateOut(BaseModel):
    term_id: str
    ref: str
    commit_seq: int
    clauses: list[dict[str, str]]  # {"tag": ..., "value": ...}


class CommitViewOut(BaseModel):
    commit: CommitOut
    terms: list[TermGroupOut]


class PrTermOut(BaseModel):
    term_id: str
    name: str | None


class PrOut(BaseModel):
    number: int
    url: str | None
    terms: list[PrTermOut]


class ReleaseOut(BaseModel):
    tag: str
    commit_seq: int
    date: datetime


class SourceInfo(BaseModel):
    name: str
    type: str
    display: str
    tracked_path: str
    status: Literal["built", "stale", "not built"]
    schema_version: str | None
    n_commits: int | None


class FacetsOut(BaseModel):
    """Distinct filter values present in a source, alphabetical."""

    tags: list[str]
    namespaces: list[str]


class SearchParams(BaseModel):
    """Everything a search request can say, validated once at the edge.

    ``q`` is optional like every other filter: absent means no text
    constraint, so a request carrying only narrowings (tag, namespace,
    term, since/until) browses everything under them. ``since`` and
    ``until`` take a ref (sha, release tag, seq, HEAD) or a
    ``YYYY-MM-DD`` date, both ends inclusive.
    """

    q: str | None = None
    term: str | None = None
    tag: str | None = None
    namespace: str | None = None
    since: str | None = None
    until: str | None = None
    match: Literal["substring", "exact", "regex"] = "substring"
    ignore_case: bool = False
    order: Literal["term", "date"] = "term"
    reverse: bool = False
    limit: int = Field(default=DEFAULT_PAGE, ge=1, le=MAX_PAGE)
    after: str | None = None
    full: bool = False

    @field_validator(
        "q", "term", "tag", "namespace", "since", "until", "after", mode="before"
    )
    @classmethod
    def _blank_is_absent(cls, value):
        """HTML forms submit untouched fields as empty strings; an empty
        filter means "no filter", not "match the empty string"."""
        return None if value == "" else value


# ---------------------------------------------------------------------------
# Serialization helpers.


def _commit_out(head: Change, style: SourceStyle) -> CommitOut:
    subject = head.message.splitlines()[0] if head.message else ""
    pr_title = pr_title_from_merge(head.message)
    pr_url = (
        f"{style.pr_url_base}{head.pr_number}"
        if style.pr_url_base and head.pr_number is not None
        else None
    )
    return CommitOut(
        commit_seq=head.commit_seq,
        sha=head.sha,
        hide_sha=style.hide_sha,
        committed_date=head.committed_date,
        author_name=head.author_name,
        subject=style.subject_prefix + (pr_title or subject),
        raw_subject=subject if pr_title else None,
        pr_number=head.pr_number,
        pr_url=pr_url,
        snapshot_url=head.snapshot_url,
        branch_commits=[
            BranchCommitOut(
                sha=bc.sha,
                author_name=bc.author_name,
                committed_date=bc.committed_date,
                message=bc.message,
            )
            for bc in head.branch_commits
        ],
    )


def _op_out(op: render.Op, cap: int | None) -> OpOut:
    view = render.op_view(op, truncate=cap)
    if isinstance(op, render.Edit):
        before, after = op.before.value, op.after.value
    elif isinstance(op, render.Remove):
        before, after = op.change.value, None
    else:
        before, after = None, op.change.value
    return OpOut(
        kind=view.kind,
        tag=view.tag,
        before=before,
        after=after,
        head=[SpanOut(role=s.role, text=s.text) for s in view.head],
        quals=[
            QualLineOut(
                kind=ql.kind,
                spans=[SpanOut(role=s.role, text=s.text) for s in ql.spans],
            )
            for ql in view.quals
        ],
    )


def _ops_out(ops: list[render.Op], full: bool) -> list[OpOut]:
    cap = None if full else render.DEFAULT_TRUNCATE
    return [_op_out(op, cap) for op in ops]


def _term_sections(
    groups: list[render.PairedCommit], style: SourceStyle, full: bool
) -> list[TermSectionOut]:
    sections = []
    for term_id, run in groupby(groups, key=lambda g: g.term_id):
        entries = list(run)
        # Most recent non-None name in the term's range, as the section
        # header — same rule as views.render_paired_groups.
        name = next(
            (e.name for e in reversed(entries) if e.name is not None), None
        )
        sections.append(
            TermSectionOut(
                term_id=term_id,
                name=name,
                commits=[
                    CommitGroupOut(
                        commit=_commit_out(e.head, style),
                        ops=_ops_out(e.ops, full),
                    )
                    for e in entries
                ],
            )
        )
    return sections


def _commit_sections(
    groups: list[render.PairedCommit], style: SourceStyle, full: bool
) -> list[CommitSectionOut]:
    sections = []
    for _, run in groupby(groups, key=lambda g: g.head.commit_seq):
        entries = list(run)
        sections.append(
            CommitSectionOut(
                commit=_commit_out(entries[0].head, style),
                terms=[
                    TermGroupOut(
                        term_id=e.term_id,
                        name=e.name,
                        ops=_ops_out(e.ops, full),
                    )
                    for e in entries
                ],
            )
        )
    return sections


def _parse_after(after: str | None, order: str) -> str | int | None:
    """Wire cursor → query-layer cursor: ints for date order."""
    if after is None or order == "term":
        return after
    try:
        return int(after)
    except ValueError:
        raise InvalidCursor(
            f"cursor {after!r} is not valid for date-ordered results "
            "(expected a commit sequence number)"
        ) from None


def _take_page(
    groups: Iterator[render.PairedCommit], order: str, limit: int
) -> tuple[list[render.PairedCommit], str | None]:
    """Consume up to ``limit`` sections; return them + the resume cursor.

    Wraps :func:`obohog.render.take_sections`, converting its truncation
    out-param into the ``next_cursor`` contract: the section key of the
    last emitted section when the limit actually cut something off.
    """
    section_key = (
        (lambda g: g.term_id) if order == "term"
        else (lambda g: g.head.commit_seq)
    )
    truncated = [False]
    taken = list(render.take_sections(groups, limit, section_key, truncated))
    if truncated[0] and taken:
        return taken, str(section_key(taken[-1]))
    return taken, None


# ---------------------------------------------------------------------------
# The service functions — one per query surface.


def list_sources(cfg: Config) -> list[SourceInfo]:
    """Status of every configured source, from build_meta (no HistoryDB)."""
    return [source_info(name, src) for name, src in cfg.sources.items()]


def source_info(name: str, source: AnySource) -> SourceInfo:
    """One source's build status, markup-free (the CLI adds the color)."""
    meta = model.read_build_meta(source.db_dir) if source.db_dir.exists() else None
    if meta is None:
        # Distinguish "nothing there" from an artifact too old to even
        # carry build_meta (which the query layer refuses as schema-unknown).
        core_present = any(
            (source.db_dir / f"{n}.parquet").exists()
            or (source.db_dir / n).is_dir()
            for n in ("commits", "term_snapshots", "events")
        )
        status = "stale" if core_present else "not built"
        return SourceInfo(
            name=name,
            type=source.type,
            display=source.source_display,
            tracked_path=source.tracked_path,
            status=status,
            schema_version=None,
            n_commits=None,
        )
    stale = meta.schema_version != model.SCHEMA_VERSION
    return SourceInfo(
        name=name,
        type=source.type,
        display=source.source_display,
        tracked_path=source.tracked_path,
        status="stale" if stale else "built",
        schema_version=meta.schema_version,
        n_commits=meta.n_commits,
    )


def get_timeline(
    db: HistoryDB,
    style: SourceStyle,
    term_id: str,
    *,
    tag: str | None = None,
    since: str | None = None,
    limit: int | None = None,
    full: bool = False,
) -> TimelineOut | None:
    """A term's full history, commit-grouped and paired. None if unknown."""
    changes = db.term_timeline(term_id, tag=tag)
    if not changes:
        return None
    header = db.term_header(term_id)
    total = len(changes)
    if since is not None:
        since_seq = db.resolve_bound(since)
        changes = [c for c in changes if c.commit_seq >= since_seq]
    if limit is not None:
        seqs = sorted({c.commit_seq for c in changes})
        keep = set(seqs[-limit:])
        changes = [c for c in changes if c.commit_seq in keep]

    commits = [
        CommitGroupOut(
            commit=_commit_out(rows[0], style),
            ops=_ops_out(render.pair_events(rows), full),
        )
        for _, group in groupby(changes, key=lambda c: c.commit_seq)
        if (rows := list(group))
    ]
    return TimelineOut(
        term_id=term_id,
        header=_header_out(header) if header else None,
        total_events=total,
        shown_events=len(changes),
        by_tag=dict(Counter(c.tag for c in changes).most_common()),
        commits=commits,
    )


def _header_out(h) -> TermHeaderOut:
    return TermHeaderOut(
        term_id=h.term_id,
        current_name=h.current_name,
        event_count=h.event_count,
        first_sha=h.first_sha,
        first_date=h.first_date,
        last_sha=h.last_sha,
        last_date=h.last_date,
        last_pr=h.last_pr,
    )


def get_state(db: HistoryDB, term_id: str, at: str) -> StateOut | None:
    """A term's reconstructed clause set as of a ref. None if no snapshot."""
    seq = db.resolve_ref(at)
    clauses = db.term_at(term_id, seq)
    if not clauses:
        return None
    return StateOut(
        term_id=term_id,
        ref=at,
        commit_seq=seq,
        clauses=[{"tag": p, "value": v} for p, v in clauses],
    )


def search(
    db: HistoryDB, style: SourceStyle, params: SearchParams
) -> PageOut[TermSectionOut] | PageOut[CommitSectionOut]:
    """One page of search results — the CLI ``search`` pipeline, paged.

    Same stages as the CLI: SQL candidates → pair by (term, commit) →
    clause-aware delta filter on edits → section cutoff. ``counts`` are
    the SQL candidates — an upper bound (``approximate=True``) when a
    query ran the delta filter, exact when browsing without one.
    """
    filters = SearchFilters(
        term_id=params.term,
        tag=params.tag,
        since_seq=db.resolve_bound(params.since) if params.since else None,
        until_seq=(
            db.resolve_bound(params.until, end=True) if params.until else None
        ),
        match=params.match,
        ignore_case=params.ignore_case,
        namespace=params.namespace,
    )
    # Surfaces an invalid regex here, before the stream starts.
    counts = db.search_counts(params.q, filters)
    events = db.iter_search_events(
        params.q,
        filters,
        order=params.order,
        reverse=params.reverse,
        after=_parse_after(params.after, params.order),
    )
    groups = render.pair_by_term_and_commit(events, order=params.order)
    if params.q is None:
        filtered = groups  # nothing to delta-match; every paired op shows
    else:
        filtered = (
            g
            for g in (
                g._replace(
                    ops=render.filter_ops_by_delta_match(
                        g.ops, params.q, params.match, params.ignore_case
                    )
                )
                for g in groups
            )
            if g.ops
        )
    taken, next_cursor = _take_page(filtered, params.order, params.limit)
    counts_out = CountsOut(
        events=counts.events,
        terms=counts.terms,
        commits=counts.commits,
        approximate=params.q is not None,
    )
    if params.order == "term":
        return PageOut(
            sections=_term_sections(taken, style, params.full),
            next_cursor=next_cursor,
            counts=counts_out,
        )
    return PageOut(
        sections=_commit_sections(taken, style, params.full),
        next_cursor=next_cursor,
        counts=counts_out,
    )


def diff(
    db: HistoryDB,
    style: SourceStyle,
    ref_a: str,
    ref_b: str,
    *,
    term: str | None = None,
    namespace: str | None = None,
    limit: int = DEFAULT_PAGE,
    after: str | None = None,
    full: bool = False,
) -> PageOut[TermSectionOut]:
    """One page of changes between two refs, term-major. Counts are exact."""
    filters = RangeFilters(term_id=term or None, namespace=namespace or None)
    counts = db.range_counts(ref_a, ref_b, filters)
    events = db.iter_range_events(ref_a, ref_b, filters, after=after or None)
    groups = render.pair_by_term_and_commit(events)
    taken, next_cursor = _take_page(groups, "term", limit)
    return PageOut(
        sections=_term_sections(taken, style, full),
        next_cursor=next_cursor,
        counts=CountsOut(
            events=counts.events, terms=counts.terms, commits=counts.commits
        ),
    )


def get_commit(
    db: HistoryDB,
    style: SourceStyle,
    sha: str,
    *,
    namespace: str | None = None,
    full: bool = False,
) -> CommitViewOut | None:
    """One commit's changes, grouped by term. None if the sha matches nothing."""
    head, events = db.commit_events(sha, namespace=namespace)
    if head is None:
        return None
    terms = [
        TermGroupOut(
            term_id=term_id,
            name=rows[0].name,
            ops=_ops_out(
                render.pair_events([tc.change for tc in rows]), full
            ),
        )
        for term_id, group in groupby(events, key=lambda tc: tc.term_id)
        if (rows := list(group))
    ]
    return CommitViewOut(commit=_commit_out(head, style), terms=terms)


def get_pr(db: HistoryDB, style: SourceStyle, number: int) -> PrOut | None:
    """Terms a PR touched. None when the PR isn't in the history."""
    terms = db.pr_terms(number)
    if not terms:
        return None
    url = f"{style.pr_url_base}{number}" if style.pr_url_base else None
    return PrOut(
        number=number,
        url=url,
        terms=[PrTermOut(term_id=t, name=n) for t, n in terms],
    )


def list_releases(db: HistoryDB) -> list[ReleaseOut]:
    return [
        ReleaseOut(tag=tag, commit_seq=seq, date=date)
        for tag, seq, date in db.releases()
    ]


def get_facets(db: HistoryDB) -> FacetsOut:
    tags, namespaces = db.facets()
    return FacetsOut(tags=tags, namespaces=namespaces)
