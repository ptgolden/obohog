"""Command-line interface for building and querying the history artifact."""

import re
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.text import Text

from . import render
from .config import (
    BioPortalSource,
    Config,
    ConfigError,
    GitFileSource,
    GitHubReleaseSource,
    SourceConfig,
    load_config,
)
from .providers import get_provider
from .extract import build_parallel
from .extract import extract as run_extract
from .gitsource import GitSource
from .query import ArtifactNotFound, Change, HistoryDB, TermChange, TermHeader

# Set at each query command's entry by ``_open_source`` from the resolved
# source config. Query commands are single-threaded and run one at a time
# per CLI invocation, so a module-level slot is safe here.
_PR_URL_BASE: str | None = None

# For synthetic-git source types (github-release, bioportal), commit shas
# reference nothing outside the local materialized repo — noise, not
# signal. Skip them in the commit-header line for those source types.
_HIDE_COMMIT_SHA: bool = False

# Prepended to the commit subject in the header line. Used to tag BioPortal
# commits with ``"BioPortal: "`` since their subjects (e.g. ``2025-08-29``,
# ``Submission #4``) don't otherwise carry the source's identity, and can
# collide visually with the date column.
_SUBJECT_PREFIX: str = ""

# GitHub HTTPS URL, with or without a trailing ``.git``. Anything else
# (SSH URLs, local paths, non-GitHub hosts) → no PR link.
_GITHUB_HTTPS = re.compile(r"^https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$")


def _pr_url_base(repo: str) -> str | None:
    m = _GITHUB_HTTPS.match(repo)
    return f"https://github.com/{m.group(1)}/pull/" if m else None


def _print_pr_link(pr_number: int) -> None:
    """Print an indented line for the PR — clickable URL if the source is on
    GitHub, otherwise the bare number so pre-2023 pattern still gets tagged
    visibly even when there's no place to link to."""
    if _PR_URL_BASE is None:
        console.print(Text(f"    → PR #{pr_number}", style="dim"))
        return
    url = f"{_PR_URL_BASE}{pr_number}"
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


def _pr_title_from_merge(message: str) -> str | None:
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


def _commit_header_prefix(head, lead: str) -> Text:
    """Build the ``<lead><sha> <date> <author>  `` prefix that leads a
    commit-header line. ``lead`` is the caller-specific leading text
    (e.g. ``"\\n● "``, ``"  ● "``) that differs by view.

    Skips the sha for synthetic-git source types where it references
    only the local materialized repo — noise, not signal.
    """
    line = Text(lead)
    if not _HIDE_COMMIT_SHA:
        line.append(head.sha[:7], style="bold yellow")
        line.append(f"  {_date(head.committed_date)}  ")
    else:
        line.append(f"{_date(head.committed_date)}  ")
    if head.author_name:
        line.append(f"{head.author_name}  ", style="cyan")
    return line


def _render_commit_header(head, prefix: Text, show_commits: bool = False) -> None:
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

    ``_SUBJECT_PREFIX`` (set per-source via :func:`_open_source`) is
    prepended to the editorial line — e.g. ``BioPortal: `` for
    BioPortal sources so their bare-date subjects don't visually collide
    with the date column.
    """
    pr_title = _pr_title_from_merge(head.message)
    subject = head.message.splitlines()[0] if head.message else ""
    snapshot_url = getattr(head, "snapshot_url", None)
    if pr_title is not None:
        prefix.append(_SUBJECT_PREFIX + pr_title, style="dim")
        console.print(prefix)
        console.print(Text("      " + subject, style="dim italic"))
        if head.pr_number is not None:
            _print_pr_link(head.pr_number)
        if snapshot_url:
            _print_snapshot_link(snapshot_url)
        if show_commits and head.branch_commits:
            _print_branch_commits(head.branch_commits)
    else:
        prefix.append(_SUBJECT_PREFIX + subject, style="dim")
        console.print(prefix)
        if head.pr_number is not None:
            _print_pr_link(head.pr_number)
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

app = typer.Typer(add_completion=False, help="Build and query an OBO ontology history index.")
console = Console()


def _open(artifact: Path) -> HistoryDB:
    """Open an artifact, exiting cleanly with guidance if it isn't there."""
    try:
        return HistoryDB(artifact)
    except ArtifactNotFound as err:
        console.print(f"[red]{err}[/]")
        raise typer.Exit(1)


def _resolve_source(source: str, config: Optional[Path]) -> SourceConfig:
    """Load the config file and look up the requested source; exit on error."""
    try:
        cfg = load_config(config)
        return cfg.get_source(source)
    except ConfigError as err:
        console.print(f"[red]{err}[/]")
        raise typer.Exit(1)


def _open_source(source: str, config: Optional[Path]) -> HistoryDB:
    """Combine config lookup and DB open into one call for query commands.

    Also configures per-source render knobs (PR URL base, sha visibility,
    subject prefix) as module-level state — query commands are
    single-threaded, one-source-per-invocation, so a global slot is
    the pragmatic choice.
    """
    src = _resolve_source(source, config)
    global _PR_URL_BASE, _HIDE_COMMIT_SHA, _SUBJECT_PREFIX
    # Only git-file / github-release sources have a `repo` URL that could
    # yield a GitHub PR link base. BioPortal sources render without one.
    _PR_URL_BASE = _pr_url_base(src.repo) if hasattr(src, "repo") else None
    # Commit shas are meaningful (they identify a real upstream commit)
    # only for git-file sources. The other providers materialize a
    # synthetic repo, and their shas are build-time artifacts.
    _HIDE_COMMIT_SHA = not isinstance(src, GitFileSource)
    # BioPortal commit subjects (a bare date, or "Submission #N") don't
    # otherwise identify their source, so tag them.
    _SUBJECT_PREFIX = "BioPortal: " if isinstance(src, BioPortalSource) else ""
    return _open(src.db_dir)


source_app = typer.Typer(add_completion=False, help="Manage configured ontology sources.")
app.add_typer(source_app, name="source")


@source_app.command("list")
def source_list(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml."),
):
    """Show configured sources with build status and disk usage."""
    try:
        cfg = load_config(config)
    except ConfigError as err:
        console.print(f"[red]{err}[/]")
        raise typer.Exit(1)
    console.print(f"[dim]Config:[/]  {cfg.path}")
    console.print(f"[dim]Storage:[/] {cfg.storage}")
    if not cfg.sources:
        console.print("\n[yellow]No sources configured.[/]")
        return
    from rich.table import Table

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("name", style="bold cyan", no_wrap=True)
    table.add_column("repo", style="dim", overflow="fold")
    table.add_column("file", style="dim", overflow="fold")
    table.add_column("status", style="dim", no_wrap=True)
    table.add_column("commits", style="dim", no_wrap=True, justify="right")
    table.add_column("clone", style="dim", no_wrap=True, justify="right")
    table.add_column("db", style="dim", no_wrap=True, justify="right")
    for name, source in cfg.sources.items():
        status, commits = _source_status(source)
        clone = _fmt_size(_dir_size(source.clone_dir))
        db = _fmt_size(_dir_size(source.db_dir))
        table.add_row(
            name, source.source_display, source.tracked_path,
            status, commits, clone, db,
        )
    console.print()
    console.print(table)


def _source_status(source: SourceConfig) -> tuple[str, str]:
    """Best-effort status label + commit count for a configured source."""
    if not source.db_dir.exists():
        return "not built", "—"
    try:
        db = HistoryDB(source.db_dir)
    except ArtifactNotFound:
        return "not built", "—"
    row = db.con.execute("SELECT COUNT(*) FROM commits").fetchone()
    db.close()
    return "built", f"{row[0]:,}" if row else "—"


def _dir_size(path: Path) -> int:
    """Total bytes on disk for everything under path, or 0 if it doesn't exist."""
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _fmt_size(nbytes: int) -> str:
    """Human-readable size, one decimal place."""
    if nbytes == 0:
        return "—"
    units = ["B", "K", "M", "G", "T"]
    size = float(nbytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}{units[-1]}"


@source_app.command("sync")
def source_sync(
    name: str = typer.Argument(..., help="Source name (as declared in obohog.toml)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml."),
    since: Optional[str] = typer.Option(
        None, help="Only index history at/after this git date, e.g. 2026-06-01."
    ),
    limit: Optional[int] = typer.Option(None, help="Index only the most recent N versions."),
    jobs: int = typer.Option(
        0, help="Parser processes: 0 = auto (cores-2), 1 = single-threaded, N = that many."
    ),
    chunk_size: int = typer.Option(
        0, help="Commits per chunk (0 = auto, ~4 chunks/worker)."
    ),
    progress: bool = typer.Option(True, help="Show a per-commit progress bar (parallel builds)."),
):
    """Clone (or update) a source's history and rebuild its database."""
    source = _resolve_source(name, config)
    clone_path = get_provider(source, console).ensure_synced(source, since=since)
    if jobs == 1:
        with GitSource(clone_path) as src:
            counts = run_extract(src, source.tracked_path, source.db_dir, limit=limit)
    else:
        counts = build_parallel(
            clone_path, source.tracked_path, source.db_dir, jobs=(jobs or None),
            chunk_size=(chunk_size or None), limit=limit, progress=progress,
        )
    msg = (
        f"[green]Built[/] {source.db_dir} — {counts['commits']} commits, "
        f"{counts['snapshots']} snapshots, {counts['events']} events"
    )
    if counts.get("skipped"):
        msg += f", [yellow]{counts['skipped']} skipped[/]"
    console.print(msg + ".")


@source_app.command("repack")
def source_repack(
    name: str = typer.Argument(..., help="Source name (as declared in obohog.toml)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml."),
    window_memory: str = typer.Option(
        "1g",
        "--window-memory",
        help=(
            "Per-thread cap on pack-objects' delta search window "
            "(git pack.windowMemory). Prevents SIGKILL on macOS for multi-GB "
            "packs. Set to '0' to remove the cap (Linux-safe, macOS-risky)."
        ),
    ),
):
    """Consolidate a source's git object storage to reclaim disk space.

    A fresh backfill of a large OBO file lands as a loosely-deltified pack —
    for Mondo, ~10 GB. A client-side repack redeltas across the whole history
    and typically shrinks it by ~5×. One-time cost; not required for
    correctness.
    """
    source = _resolve_source(name, config)
    if not source.clone_dir.exists():
        console.print(f"[red]No clone at {source.clone_dir}. Run `obohog source sync {name}` first.[/]")
        raise typer.Exit(1)
    before = _dir_size(source.clone_dir / ".git")
    console.print(f"Repacking [cyan]{source.clone_dir}[/] …")
    GitSource(source.clone_dir).repack(window_memory=window_memory)
    after = _dir_size(source.clone_dir / ".git")
    console.print(
        f"[green]Repacked[/] — {_fmt_size(before)} → {_fmt_size(after)} "
        f"([yellow]saved {_fmt_size(before - after)}[/])."
    )


@app.command()
def term(
    term_id: str = typer.Argument(..., help="e.g. MONDO:0007739"),
    source: str = typer.Option(..., "--source", help="Configured source name (see obohog source list)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml (default: ./obohog.toml)."),
    only: Optional[str] = typer.Option(None, help="Restrict to one clause kind, e.g. synonym."),
    at: Optional[str] = typer.Option(
        None, help="Reconstruct state as of this ref (short sha, tag, or commit_seq)."
    ),
    limit: Optional[int] = typer.Option(
        None, help="Show only the most recent N commits' events."
    ),
    since: Optional[str] = typer.Option(
        None, help="Show only events at/after this ref (short sha, tag, or commit_seq)."
    ),
    full: bool = typer.Option(False, help="Do not truncate long values."),
    commits: bool = typer.Option(
        False, "--commits",
        help="For classic-merge PRs with a PR title, also list the PR-branch commits.",
    ),
):
    """Show a term's change history, or its reconstructed state at a point."""
    db = _open_source(source, config)
    if at is not None:
        at_seq = db.resolve_ref(at)
        _render_state(term_id, at, db.term_at(term_id, at_seq))
    else:
        header = db.term_header(term_id)
        changes = db.term_timeline(term_id, predicate=only)
        since_seq = db.resolve_ref(since) if since is not None else None
        _render_timeline(
            term_id, header, changes,
            limit=limit, since_seq=since_seq, full=full, show_commits=commits,
        )
    db.close()


@app.command()
def commit(
    sha: str = typer.Argument(..., help="Commit sha or unique prefix."),
    source: str = typer.Option(..., "--source", help="Configured source name."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml."),
    namespace: Optional[str] = typer.Option(
        None, help="Restrict to terms whose CURIE prefix is PREFIX (e.g. MONDO)."
    ),
    full: bool = typer.Option(False, help="Do not truncate long values."),
    commits: bool = typer.Option(
        False, "--commits",
        help="For classic-merge PRs with a PR title, also list the PR-branch commits.",
    ),
):
    """Show what changed at one commit, structurally rendered per term."""
    db = _open_source(source, config)
    head, events = db.commit_events(sha, namespace=namespace)
    if head is None:
        console.print(f"[yellow]No indexed changes for commit[/] {sha}")
        db.close()
        return
    _render_commit_view(head, events, full=full, show_commits=commits)
    db.close()


@app.command()
def pr(
    number: int = typer.Argument(..., help="Pull request number, e.g. 10343."),
    source: str = typer.Option(..., "--source", help="Configured source name."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml."),
):
    """List the terms changed by a pull request."""
    db = _open_source(source, config)
    terms = db.pr_terms(number)
    if not terms:
        console.print(f"[yellow]No indexed changes for PR[/] #{number}")
    else:
        console.print(f"[bold]{len(terms)}[/] terms changed in PR #{number}:")
        for term_id, name in terms:
            line = Text("  ")
            line.append(term_id, style="cyan")
            if name:
                line.append(f"  {name}", style="dim")
            console.print(line)
    db.close()


@app.command()
def diff(
    ref_a: str = typer.Argument(..., help="Release tag, short sha, HEAD, or commit_seq."),
    ref_b: str = typer.Argument(..., help="Release tag, short sha, HEAD, or commit_seq."),
    source: str = typer.Option(..., "--source", help="Configured source name."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml."),
    term: Optional[str] = typer.Option(None, help="Restrict to one term."),
    namespace: Optional[str] = typer.Option(
        None, help="Restrict to terms whose CURIE prefix is PREFIX (e.g. MONDO)."
    ),
    full: bool = typer.Option(False, help="Do not truncate long values."),
    commits: bool = typer.Option(
        False, "--commits",
        help="For classic-merge PRs with a PR title, also list the PR-branch commits.",
    ),
):
    """Show clause changes between two points, grouped by term and commit."""
    db = _open_source(source, config)
    events = db.range_events(ref_a, ref_b, term_id=term, namespace=namespace)
    if not events:
        console.print(f"[yellow]No changes between[/] {ref_a} [yellow]and[/] {ref_b}")
        db.close()
        return
    _render_diff_view(ref_a, ref_b, events, full=full, show_commits=commits)
    db.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Substring (or regex, with --regex) to match in event values."),
    source: str = typer.Option(..., "--source", help="Configured source name."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml."),
    term: Optional[str] = typer.Option(None, help="Restrict to one term."),
    predicate: Optional[str] = typer.Option(
        None, help="Restrict to one clause kind, e.g. xref."
    ),
    namespace: Optional[str] = typer.Option(
        None, help="Restrict to terms whose CURIE prefix is PREFIX (e.g. MONDO)."
    ),
    since: Optional[str] = typer.Option(
        None, help="Show events at/after this ref (short sha, tag, or commit_seq)."
    ),
    regex: bool = typer.Option(False, "--regex", help="Treat QUERY as a regular expression."),
    ignore_case: bool = typer.Option(
        False, "--ignore-case", "-i", help="Case-insensitive match."
    ),
    full: bool = typer.Option(False, help="Do not truncate long values."),
    commits: bool = typer.Option(
        False, "--commits",
        help="For classic-merge PRs with a PR title, also list the PR-branch commits.",
    ),
):
    """Find commits that added or removed a clause matching QUERY."""
    db = _open_source(source, config)
    since_seq = db.resolve_ref(since) if since is not None else None
    events = db.search_events(
        query, term_id=term, predicate=predicate, since_seq=since_seq,
        regex=regex, ignore_case=ignore_case, namespace=namespace,
    )
    if not events:
        console.print(f'[yellow]No events matching[/] "{query}"')
        db.close()
        return
    _render_search_view(
        query, events, regex=regex, ignore_case=ignore_case, full=full,
        show_commits=commits,
    )
    db.close()


@app.command()
def releases(
    source: str = typer.Option(..., "--source", help="Configured source name."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml."),
):
    """List release tags indexed for a source."""
    db = _open_source(source, config)
    rows = db.releases()
    db.close()
    if not rows:
        console.print("[yellow]No releases indexed in this artifact.[/]")
        return
    for tag, commit_seq, date in rows:
        line = Text()
        line.append(tag, style="bold green")
        line.append(f"  commit {commit_seq}  {str(date)[:10]}", style="dim")
        console.print(line)


def _render_timeline(
    term_id: str,
    header: TermHeader | None,
    changes: list[Change],
    limit: int | None = None,
    since_seq: int | None = None,
    full: bool = False,
    show_commits: bool = False,
) -> None:
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

    _render_header(term_id, header, changes, total_events, limit, since_seq)

    cap = None if full else render.DEFAULT_TRUNCATE
    for _, group in groupby(changes, key=lambda c: c.commit_seq):
        rows = list(group)
        head = rows[0]
        header_line = _commit_header_prefix(head, "\n● ")
        _render_commit_header(head, header_line, show_commits=show_commits)
        for op in render.pair_events(rows):
            console.print(render.render_op(op, truncate=cap))


def _render_header(
    term_id: str,
    header: TermHeader | None,
    changes: list[Change],
    total_events: int,
    limit: int | None,
    since_seq: int | None,
) -> None:
    """Print the orientation header: name, span, and predicate counts."""
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
        if _HIDE_COMMIT_SHA:
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

    counts = Counter(c.predicate for c in changes)
    if counts:
        by_pred = Text("by predicate: ", style="dim")
        parts = [f"{p} {n}" for p, n in counts.most_common()]
        by_pred.append(", ".join(parts), style="dim")
        console.print(by_pred)


def _render_commit_view(
    head: Change, events: list[TermChange], full: bool = False,
    show_commits: bool = False,
) -> None:
    """Structural view of one commit: header + per-term event groups."""
    header_line = _commit_header_prefix(head, "● ")
    _render_commit_header(head, header_line, show_commits=show_commits)
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


def _render_diff_view(
    ref_a: str, ref_b: str, events: list[TermChange], full: bool = False,
    show_commits: bool = False,
) -> None:
    """Structural view of a range diff: per-term sections, per-commit sub-groups."""
    n_terms = len({tc.term_id for tc in events})
    n_commits = len({tc.change.commit_seq for tc in events})
    summary = Text()
    summary.append(f"{len(events)}", style="bold")
    summary.append(f" events across ", style="dim")
    summary.append(f"{n_terms}", style="bold")
    summary.append(f" terms and ", style="dim")
    summary.append(f"{n_commits}", style="bold")
    summary.append(f" commits between {ref_a} and {ref_b}", style="dim")
    console.print(summary)
    _render_events_by_term_and_commit(events, full=full, show_commits=show_commits)


def _render_search_view(
    query: str,
    events: list[TermChange],
    regex: bool = False,
    ignore_case: bool = False,
    full: bool = False,
    show_commits: bool = False,
) -> None:
    """Structural view of search hits: same layout as the diff view.

    Before rendering, apply a clause-aware post-filter: for any paired
    ``Edit`` whose delta (body-diff, comment-diff, or qualifier symmetric
    difference) doesn't actually contain the query, drop both constituent
    events. Unpaired adds/removes are always kept — their whole clause is
    "the change" by definition, so the SQL match already tells us the query
    is in the changed portion.

    See :func:`obohog.render.edit_delta_matches` for the exact rule.
    """
    events = _filter_events_by_delta_match(events, query, regex, ignore_case)
    if not events:
        console.print(f'[yellow]No events matching[/] "{query}"')
        return
    n_terms = len({tc.term_id for tc in events})
    n_commits = len({tc.change.commit_seq for tc in events})
    summary = Text()
    summary.append(f"Found {len(events)}", style="bold")
    summary.append(" events matching ", style="dim")
    summary.append(f'"{query}"', style="bold")
    summary.append(" across ", style="dim")
    summary.append(f"{n_terms}", style="bold")
    summary.append(" terms and ", style="dim")
    summary.append(f"{n_commits}", style="bold")
    summary.append(" commits", style="dim")
    console.print(summary)
    _render_events_by_term_and_commit(events, full=full, show_commits=show_commits)


def _filter_events_by_delta_match(
    events: list[TermChange], query: str, regex: bool, ignore_case: bool
) -> list[TermChange]:
    """Drop paired-edit event pairs whose delta doesn't contain the query.

    Pairs events per commit, drops both halves of any ``Edit`` whose
    :func:`~obohog.render.edit_delta_matches` returns False, and
    keeps every unpaired ``Add`` / ``Remove`` as-is. Preserves the input
    order at the (term_id, commit_seq) granularity so the downstream
    ``groupby`` in :func:`_render_events_by_term_and_commit` still sees
    contiguous groupings.
    """
    surviving: list[TermChange] = []
    for _, group in groupby(
        events, key=lambda tc: (tc.term_id, tc.change.commit_seq)
    ):
        group_events = list(group)
        by_change_id = {id(tc.change): tc for tc in group_events}
        changes = [tc.change for tc in group_events]
        for op in render.pair_events(changes):
            if isinstance(op, render.Edit):
                if render.edit_delta_matches(op, query, regex, ignore_case):
                    surviving.append(by_change_id[id(op.before)])
                    surviving.append(by_change_id[id(op.after)])
            elif isinstance(op, render.Add):
                surviving.append(by_change_id[id(op.change)])
            else:  # Remove
                surviving.append(by_change_id[id(op.change)])
    return surviving


def _render_events_by_term_and_commit(
    events: list[TermChange], full: bool = False, show_commits: bool = False
) -> None:
    """Group ``events`` by term, then by commit within each term, and render.

    Shared between ``diff`` and ``search``. Expects ``events`` already
    ordered by ``(term_id, commit_seq, ...)`` so the groupings are
    contiguous.
    """
    cap = None if full else render.DEFAULT_TRUNCATE
    for term_id, term_group in groupby(events, key=lambda tc: tc.term_id):
        term_rows = list(term_group)
        title = Text("\n")
        title.append(term_id, style="bold cyan")
        # Take the most recent name we saw in the range as the section header.
        latest_name = next(
            (tc.name for tc in reversed(term_rows) if tc.name is not None), None
        )
        if latest_name:
            title.append(f" — {latest_name}", style="bold")
        console.print(title)
        for _, commit_group in groupby(term_rows, key=lambda tc: tc.change.commit_seq):
            commit_rows = list(commit_group)
            head = commit_rows[0].change
            commit_header = _commit_header_prefix(head, "  ● ")
            _render_commit_header(head, commit_header, show_commits=show_commits)
            changes = [tc.change for tc in commit_rows]
            for op in render.pair_events(changes):
                console.print(render.render_op(op, truncate=cap))


def _render_state(term_id: str, at: str, clauses: list[tuple[str, str]]) -> None:
    if not clauses:
        console.print(f"[yellow]{term_id} has no snapshot at or before {at}[/]")
        return
    console.print(f"[bold cyan]{term_id}[/] as of {at}:")
    console.print(Text(f"  id: {term_id}"))
    for predicate, value in clauses:
        console.print(Text(f"  {predicate}: {value}"))


def _date(value: object) -> str:
    return str(value)[:10]


if __name__ == "__main__":
    app()
