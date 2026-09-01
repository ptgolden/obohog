"""Console views over query results: the terminal presentation layer.

Everything here is about *printing* — turning the query layer's typed rows
and the render pipeline's ops into styled terminal output. The module owns
the process console (with a fast plain-text path for pipes) and the
per-source :class:`SourceStyle` knobs. Nothing here touches typer or the
config-file plumbing; that stays in :mod:`obohog.cli`.
"""

import re
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from typing import Iterable

from rich.console import Console
from rich.text import Text

from . import render
from .config import BioPortalSource, GitFileSource, SourceConfig
from .query import Change, EventCounts, TermChange, TermHeader

# GitHub HTTPS URL, with or without a trailing ``.git``. Anything else
# (SSH URLs, local paths, non-GitHub hosts) → no PR link.
_GITHUB_HTTPS = re.compile(r"^https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$")


@dataclass(frozen=True)
class SourceStyle:
    """Per-source presentation knobs, resolved once from the source config.

    Threaded explicitly through the view functions (never module state) so
    concurrent consumers — one process serving several sources, like the
    future HTTP API — each render with their own source's knobs.
    """

    # GitHub PR link base (``https://github.com/{owner}/{repo}/pull/``), or
    # None when the repo isn't on GitHub and PR numbers render as bare text.
    pr_url_base: str | None = None
    # Synthetic-git source types (github-release, bioportal): commit shas
    # reference nothing outside the local materialized repo — noise, not
    # signal. Skip them in the commit-header line.
    hide_sha: bool = False
    # Prepended to the commit subject in the header line. Used to tag
    # BioPortal commits with ``"BioPortal: "`` since their subjects (e.g.
    # ``2025-08-29``, ``Submission #4``) don't otherwise carry the source's
    # identity, and can collide visually with the date column.
    subject_prefix: str = ""


def style_for(src: SourceConfig) -> SourceStyle:
    """Resolve the presentation knobs for a configured source."""
    # Only git-file / github-release sources have a `repo` URL that could
    # yield a GitHub PR link base. BioPortal sources render without one.
    return SourceStyle(
        pr_url_base=_pr_url_base(src.repo) if hasattr(src, "repo") else None,
        hide_sha=not isinstance(src, GitFileSource),
        subject_prefix="BioPortal: " if isinstance(src, BioPortalSource) else "",
    )


def _pr_url_base(repo: str) -> str | None:
    m = _GITHUB_HTTPS.match(repo)
    return f"https://github.com/{m.group(1)}/pull/" if m else None


class _PlainConsole:
    """Console stand-in when stdout isn't a terminal.

    Piped output has its styles stripped anyway, but rich's per-print
    machinery (measure, wrap, segment) still costs ~45µs/line — ~97% of
    render time on large result sets. Bare ``Text`` prints go straight to
    the stream as their plain form instead (~35x faster; long lines stay
    unwrapped, which suits pipes and greps). Everything else — markup
    strings, tables, empty separator prints — delegates to the wrapped
    Console, which already renders styleless when piped.
    """

    def __init__(self, rich_console: Console):
        self._rich = rich_console

    def print(self, *args, **kwargs) -> None:
        if len(args) == 1 and isinstance(args[0], Text) and not kwargs:
            self._rich.file.write(args[0].plain + "\n")
        else:
            self._rich.print(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._rich, name)


def _make_console() -> "Console | _PlainConsole":
    rich_console = Console()
    return rich_console if rich_console.is_terminal else _PlainConsole(rich_console)


console = _make_console()


def _print_pr_link(pr_number: int, style: SourceStyle) -> None:
    """Print an indented line for the PR — clickable URL if the source is on
    GitHub, otherwise the bare number so pre-2023 pattern still gets tagged
    visibly even when there's no place to link to."""
    if style.pr_url_base is None:
        console.print(Text(f"    → PR #{pr_number}", style="dim"))
        return
    url = f"{style.pr_url_base}{pr_number}"
    line = Text("    → ", style="dim")
    line.append(url, style=f"link {url} dim cyan")
    console.print(line)


def _print_snapshot_link(url: str) -> None:
    """Print the ``→ <url>`` line for a release-based commit. Same visual
    slot as the PR link (release commits don't have a PR to link to; this
    is the page where curators can dig into the release's PRs, notes, etc.).
    """
    line = Text("    → ", style="dim")
    line.append(url, style=f"link {url} dim cyan")
    console.print(line)


# Classic GitHub merge commit: line 1 is boilerplate, line 3+ is the PR title
# (whatever the PR was named on GitHub — usually the branch's last commit
# subject, or a manual title set on the merge screen). We use that title as
# the primary editorial line, demote the boilerplate to a dim sub-line, and
# hide branch commits by default (drill in via `obohog pr <N>` if needed).
_MERGE_BOILERPLATE = re.compile(r"^Merge pull request #(\d+) from ")


def pr_title_from_merge(message: str) -> str | None:
    """Return the PR title embedded in a classic GitHub merge commit body.

    GitHub-specific heuristic: when line 1 matches ``Merge pull request #N
    from …``, GitHub's default merge screen puts the PR title in the body
    (line 3+). Returns the first non-empty body line, or None if the
    message isn't a classic merge or has no body content. For non-GitHub
    sources (or GitHub merges with empty bodies) callers should fall back
    to rendering the raw subject line.
    """
    lines = message.splitlines()
    if not lines or not _MERGE_BOILERPLATE.match(lines[0]):
        return None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _commit_header_prefix(head, lead: str, style: SourceStyle) -> Text:
    """Build the ``<lead><sha> <date> <author>  `` prefix that leads a
    commit-header line. ``lead`` is the caller-specific leading text
    (e.g. ``"\\n● "``, ``"  ● "``) that differs by view.

    Skips the sha when ``style.hide_sha`` says it references only the
    local materialized repo — noise, not signal.
    """
    line = Text(lead)
    if not style.hide_sha:
        line.append(head.sha[:7], style="bold yellow")
        line.append(f"  {_date(head.committed_date)}  ")
    else:
        line.append(f"{_date(head.committed_date)}  ")
    if head.author_name:
        line.append(f"{head.author_name}  ", style="cyan")
    return line


def _render_commit_header(
    head, prefix: Text, style: SourceStyle, show_commits: bool = False
) -> None:
    """Given an already-built ``sha  date  author  `` prefix Text, append the
    editorial subject line and print, followed by any demoted boilerplate,
    PR link, and branch commits.

    Classic GitHub merge commits with a PR title in the body:
      * primary line: the PR title (editorial)
      * next line: the boilerplate ``Merge pull request #N from …`` demoted
        to dim italic
      * branch commits: hidden unless ``show_commits=True``

    Everything else (squash-and-merge, non-GitHub, or classic merge with
    empty body):
      * primary line: the raw subject
      * branch commits: shown when present (only editorial signal for
        old-style merges with empty bodies)

    ``style.subject_prefix`` is prepended to the editorial line — e.g.
    ``BioPortal: `` for BioPortal sources so their bare-date subjects
    don't visually collide with the date column.
    """
    pr_title = pr_title_from_merge(head.message)
    subject = head.message.splitlines()[0] if head.message else ""
    snapshot_url = getattr(head, "snapshot_url", None)
    if pr_title is not None:
        prefix.append(style.subject_prefix + pr_title, style="dim")
        console.print(prefix)
        console.print(Text("      " + subject, style="dim italic"))
        if head.pr_number is not None:
            _print_pr_link(head.pr_number, style)
        if snapshot_url:
            _print_snapshot_link(snapshot_url)
        if show_commits and head.branch_commits:
            _print_branch_commits(head.branch_commits)
    else:
        prefix.append(style.subject_prefix + subject, style="dim")
        console.print(prefix)
        if head.pr_number is not None:
            _print_pr_link(head.pr_number, style)
        if snapshot_url:
            _print_snapshot_link(snapshot_url)
        if head.branch_commits:
            _print_branch_commits(head.branch_commits)


def _print_branch_commits(branch_commits) -> None:
    """Print one line per PR-branch commit under a merge commit's header.

    Newest first (matching what a reader sees on GitHub); short sha + subject
    line only. For a typical PR-branch this is 1–3 lines that give the real
    editorial intent, since the mainline header just says
    "Merge pull request #N from ...".
    """
    for bc in branch_commits:
        line = Text("    ⤷ ", style="dim")
        line.append(bc.sha[:7], style="yellow")
        line.append("  ", style="dim")
        subject = bc.message.splitlines()[0] if bc.message else ""
        line.append(subject, style="dim")
        console.print(line)


def render_timeline(
    term_id: str,
    header: TermHeader | None,
    changes: list[Change],
    style: SourceStyle,
    limit: int | None = None,
    since_seq: int | None = None,
    full: bool = False,
    show_commits: bool = False,
) -> None:
    """A term's change history: orientation header + per-commit event groups."""
    if not changes:
        console.print(f"[yellow]No history for[/] {term_id}")
        return
    total_events = len(changes)
    if since_seq is not None:
        changes = [c for c in changes if c.commit_seq >= since_seq]
    if limit is not None:
        seqs: list[int] = []
        for c in changes:
            if not seqs or seqs[-1] != c.commit_seq:
                seqs.append(c.commit_seq)
        keep = set(seqs[-limit:])
        changes = [c for c in changes if c.commit_seq in keep]

    _render_header(term_id, header, changes, total_events, limit, since_seq, style)

    cap = None if full else render.DEFAULT_TRUNCATE
    for _, group in groupby(changes, key=lambda c: c.commit_seq):
        rows = list(group)
        head = rows[0]
        header_line = _commit_header_prefix(head, "\n● ", style)
        _render_commit_header(head, header_line, style, show_commits=show_commits)
        for op in render.pair_events(rows):
            console.print(render.render_op(op, truncate=cap))


def _render_header(
    term_id: str,
    header: TermHeader | None,
    changes: list[Change],
    total_events: int,
    limit: int | None,
    since_seq: int | None,
    style: SourceStyle,
) -> None:
    """Print the orientation header: name, span, and tag counts."""
    title = Text()
    title.append(term_id, style="bold cyan")
    if header is not None and header.current_name:
        title.append(f" — {header.current_name}", style="bold")
    console.print(title)

    if header is not None:
        n_commits = len({c.commit_seq for c in changes})
        shown_events = len(changes)
        summary = Text()
        if shown_events != total_events:
            summary.append(
                f"showing {shown_events} of {total_events} events "
                f"across {n_commits} commits",
                style="dim",
            )
        else:
            summary.append(
                f"{total_events} events across {n_commits} commits",
                style="dim",
            )
        console.print(summary)

        span = Text()
        if style.hide_sha:
            span.append(f"first {_date(header.first_date)}", style="dim")
            span.append("  ·  ", style="dim")
            span.append(f"last {_date(header.last_date)}", style="dim")
            if header.last_pr is not None:
                span.append(f" (PR #{header.last_pr})", style="dim")
        else:
            span.append(
                f"first {_date(header.first_date)} ({header.first_sha[:7]})",
                style="dim",
            )
            span.append("  ·  ", style="dim")
            span.append(
                f"last {_date(header.last_date)} ({header.last_sha[:7]}",
                style="dim",
            )
            if header.last_pr is not None:
                span.append(f", PR #{header.last_pr}", style="dim")
            span.append(")", style="dim")
        console.print(span)

    counts = Counter(c.tag for c in changes)
    if counts:
        by_pred = Text("by tag: ", style="dim")
        parts = [f"{p} {n}" for p, n in counts.most_common()]
        by_pred.append(", ".join(parts), style="dim")
        console.print(by_pred)


def render_commit_view(
    head: Change, events: list[TermChange], style: SourceStyle,
    full: bool = False, show_commits: bool = False,
) -> None:
    """Structural view of one commit: header + per-term event groups."""
    header_line = _commit_header_prefix(head, "● ", style)
    _render_commit_header(head, header_line, style, show_commits=show_commits)
    n_terms = len({tc.term_id for tc in events})
    console.print(Text(f"{n_terms} terms changed", style="dim"))

    cap = None if full else render.DEFAULT_TRUNCATE
    for term_id, group in groupby(events, key=lambda tc: tc.term_id):
        rows = list(group)
        title = Text("\n")
        title.append(term_id, style="bold cyan")
        if rows[0].name:
            title.append(f" — {rows[0].name}", style="bold")
        console.print(title)
        changes = [tc.change for tc in rows]
        for op in render.pair_events(changes):
            console.print(render.render_op(op, truncate=cap))


def render_paired_groups(
    groups: Iterable[render.PairedCommit], style: SourceStyle,
    full: bool = False, show_commits: bool = False,
) -> EventCounts:
    """Render (term, commit) op groups as per-term sections.

    Shared between ``diff`` and ``search``. Expects ``groups`` ordered by
    term so per-term runs are contiguous. Streams: only one term's groups
    are buffered at a time (the section title shows the most recent name
    in the term's range, which needs the term's full run — never more).

    Returns the exact rendered totals, for footers — with a post-render
    filter in the pipeline (search), these can be below any pre-count.
    """
    cap = None if full else render.DEFAULT_TRUNCATE
    n_events = n_terms = 0
    commit_seqs: set[int] = set()
    for term_id, term_group in groupby(groups, key=lambda g: g.term_id):
        term_entries = list(term_group)
        n_terms += 1
        title = Text("\n")
        title.append(term_id, style="bold cyan")
        # Take the most recent name we saw in the range as the section header.
        latest_name = next(
            (e.name for e in reversed(term_entries) if e.name is not None), None
        )
        if latest_name:
            title.append(f" — {latest_name}", style="bold")
        console.print(title)
        for entry in term_entries:
            commit_seqs.add(entry.head.commit_seq)
            commit_header = _commit_header_prefix(entry.head, "  ● ", style)
            _render_commit_header(entry.head, commit_header, style, show_commits=show_commits)
            for op in entry.ops:
                n_events += 2 if isinstance(op, render.Edit) else 1
                console.print(render.render_op(op, truncate=cap))
    return EventCounts(events=n_events, terms=n_terms, commits=len(commit_seqs))


def render_commit_ordered_groups(
    groups: Iterable[render.PairedCommit], style: SourceStyle,
    full: bool = False, show_commits: bool = False,
) -> EventCounts:
    """Render (term, commit) op groups as commit blocks, newest first.

    The date-ordered counterpart of :func:`render_paired_groups`:
    expects a commit-major stream (``order="date"``), renders one block
    per commit — header, then each affected term's ops. Streams with
    one commit's groups buffered at a time.
    """
    cap = None if full else render.DEFAULT_TRUNCATE
    n_events = n_commits = 0
    term_ids: set[str] = set()
    for _, commit_group in groupby(groups, key=lambda g: g.head.commit_seq):
        entries = list(commit_group)
        n_commits += 1
        head = entries[0].head
        _render_commit_header(
            head, _commit_header_prefix(head, "\n● ", style), style,
            show_commits=show_commits,
        )
        for entry in entries:
            term_ids.add(entry.term_id)
            title = Text("  ")
            title.append(entry.term_id, style="bold cyan")
            if entry.name:
                title.append(f" — {entry.name}", style="bold")
            console.print(title)
            for op in entry.ops:
                n_events += 2 if isinstance(op, render.Edit) else 1
                console.print(render.render_op(op, truncate=cap))
    return EventCounts(events=n_events, terms=len(term_ids), commits=n_commits)


def counts_phrase(
    events: int, terms: int, commits: int, *, noun: str = "events", tail: str = ""
) -> Text:
    """``<events> <noun> across <terms> terms and <commits> commits<tail>``,
    numbers bold, connective text dim — the scope/summary sentence shared by
    ``diff`` and ``search``."""
    t = Text()
    t.append(str(events), style="bold")
    t.append(f" {noun} across ", style="dim")
    t.append(str(terms), style="bold")
    t.append(" terms and ", style="dim")
    t.append(str(commits), style="bold")
    t.append(f" commits{tail}", style="dim")
    return t


def render_state(term_id: str, at: str, clauses: list[tuple[str, str]]) -> None:
    """A term's reconstructed clause set as of a resolved ref."""
    if not clauses:
        console.print(f"[yellow]{term_id} has no snapshot at or before {at}[/]")
        return
    console.print(f"[bold cyan]{term_id}[/] as of {at}:")
    # A synthetic id clause carries the file's own spelling when namespace
    # mapping canonicalized it — it IS the id line, not a clause.
    written_id = next((v for t, v in clauses if t == "id"), term_id)
    console.print(Text(f"  id: {written_id}"))
    for tag, value in clauses:
        if tag != "id":
            console.print(Text(f"  {tag}: {value}"))


def _date(value: object) -> str:
    return str(value)[:10]
