"""Command-line interface for building and querying the history artifact."""

import enum
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, Optional

import duckdb
import typer
from rich.text import Text

from . import model, render, service
from .config import ConfigError, SourceConfig, load_config
from .providers import get_provider
from .extract import BuildMode, build_parallel
from .gitsource import GitSource
from .query import (
    ArtifactNotFound,
    HistoryDB,
    RangeFilters,
    RefNotFound,
    SchemaMismatch,
    SearchFilters,
)
from .views import (
    SourceStyle,
    console,
    counts_phrase,
    render_commit_ordered_groups,
    render_commit_view,
    render_paired_groups,
    render_state,
    render_timeline,
    style_for,
)

app = typer.Typer(add_completion=False, help="Build and query an OBO ontology history index.")


class SearchOrder(str, enum.Enum):
    """Output grouping for ``search``: term-major sections or commit-major
    blocks (newest first, the ``git log`` shape)."""

    term = "term"
    date = "date"


@app.callback()
def _app_setup() -> None:
    # Die silently on a closed pipe (`search ... | head`), like other unix
    # filters: restore SIGPIPE's default disposition instead of letting
    # Python turn it into a BrokenPipeError traceback mid-render.
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def _open(artifact: Path) -> HistoryDB:
    """Open an artifact, exiting cleanly with guidance if it isn't usable."""
    try:
        return HistoryDB(artifact)
    except (ArtifactNotFound, SchemaMismatch) as err:
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


def _open_source(source: str, config: Optional[Path]) -> tuple[HistoryDB, SourceStyle]:
    """Resolve a source, open its artifact, and return its presentation style."""
    src = _resolve_source(source, config)
    return _open(src.db_dir), style_for(src)


@contextmanager
def _query_errors():
    """Turn expected bad-input failures into clean CLI errors.

    Covers a ref that resolves to nothing (``--at``, ``--since``, diff
    refs) and a ``--regex`` pattern DuckDB rejects — user input, not
    bugs, so no traceback.
    """
    try:
        yield
    except (RefNotFound, duckdb.InvalidInputException) as err:
        console.print(f"[red]{err}[/]")
        raise typer.Exit(1)


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
    table.add_column("schema", style="dim", no_wrap=True, justify="right")
    table.add_column("commits", style="dim", no_wrap=True, justify="right")
    table.add_column("clone", style="dim", no_wrap=True, justify="right")
    table.add_column("db", style="dim", no_wrap=True, justify="right")
    any_stale = False
    for name, source in cfg.sources.items():
        st = _source_status(source)
        any_stale = any_stale or st.stale
        clone = _fmt_size(_dir_size(source.clone_dir))
        db = _fmt_size(_dir_size(source.db_dir))
        table.add_row(
            name, source.source_display, source.tracked_path,
            st.status, st.schema, st.commits, clone, db,
        )
    console.print()
    console.print(table)
    if any_stale:
        console.print(
            "\n[yellow]Stale sources were built with an older schema and can't "
            "be queried; run [cyan]obohog source sync <name>[/] to rebuild.[/]"
        )


class _SourceStatus(NamedTuple):
    status: str
    schema: str
    commits: str
    stale: bool


def _source_status(source: SourceConfig) -> _SourceStatus:
    """Rich-marked status columns for a source, from the service layer."""
    info = service.source_info(source.name, source)
    commits = f"{info.n_commits:,}" if info.n_commits is not None else "—"
    if info.status == "stale":
        schema = (
            f"[yellow]{info.schema_version} → {model.SCHEMA_VERSION}[/]"
            if info.schema_version is not None
            else "[yellow]?[/]"
        )
        return _SourceStatus("[yellow]stale[/]", schema, commits, True)
    if info.status == "not built":
        return _SourceStatus("not built", "—", "—", False)
    return _SourceStatus("built", info.schema_version or "?", commits, False)


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
        0, help="Parser worker processes: 0 = auto (cores-2), N = that many."
    ),
    chunk_size: int = typer.Option(
        0, help="Commits per chunk (0 = auto, ~4 chunks/worker)."
    ),
    progress: bool = typer.Option(True, help="Show a per-commit progress bar (parallel builds)."),
    rebuild: bool = typer.Option(
        False, "--rebuild",
        help="Rebuild the database from scratch instead of appending new commits.",
    ),
):
    """Update a source's clone and bring its database up to date.

    An existing database is extended in place with just the commits that are
    new since the last sync; a full rebuild happens automatically when
    appending isn't possible (first sync, schema change, rewritten upstream
    history) or on --rebuild.
    """
    source = _resolve_source(name, config)
    clone_path = get_provider(source, console).ensure_synced(source, since=since)
    report = build_parallel(
        clone_path, source.tracked_path, source.db_dir, jobs=(jobs or None),
        chunk_size=(chunk_size or None), limit=limit, progress=progress,
        update=not rebuild,
    )
    if report.mode is BuildMode.UP_TO_DATE:
        console.print(
            f"[green]Up to date[/] — {source.db_dir} already covers "
            f"{report.total_commits} commits."
        )
        return
    if report.mode is BuildMode.INCREMENTAL:
        msg = (
            f"[green]Appended[/] {report.commits} new commits to {source.db_dir} "
            f"(now {report.total_commits}) — {report.snapshots} snapshots, "
            f"{report.events} events"
        )
    else:
        msg = (
            f"[green]Built[/] {source.db_dir} — {report.commits} commits, "
            f"{report.snapshots} snapshots, {report.events} events"
        )
    if report.skipped:
        msg += f", [yellow]{report.skipped} skipped[/]"
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
    db, style = _open_source(source, config)
    if at is not None:
        with _query_errors():
            at_seq = db.resolve_ref(at)
        render_state(term_id, at, db.term_at(term_id, at_seq))
    else:
        header = db.term_header(term_id)
        changes = db.term_timeline(term_id, predicate=only)
        with _query_errors():
            since_seq = db.resolve_ref(since) if since is not None else None
        render_timeline(
            term_id, header, changes, style,
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
    db, style = _open_source(source, config)
    head, events = db.commit_events(sha, namespace=namespace)
    if head is None:
        console.print(f"[yellow]No indexed changes for commit[/] {sha}")
        db.close()
        return
    render_commit_view(head, events, style, full=full, show_commits=commits)
    db.close()


@app.command()
def pr(
    number: int = typer.Argument(..., help="Pull request number, e.g. 10343."),
    source: str = typer.Option(..., "--source", help="Configured source name."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml."),
):
    """List the terms changed by a pull request."""
    db, _ = _open_source(source, config)
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
    db, style = _open_source(source, config)
    filters = RangeFilters(term_id=term, namespace=namespace)
    with _query_errors():
        counts = db.range_counts(ref_a, ref_b, filters)
    if counts.events == 0:
        console.print(f"[yellow]No changes between[/] {ref_a} [yellow]and[/] {ref_b}")
        db.close()
        return
    console.print(counts_phrase(
        counts.events, counts.terms, counts.commits,
        tail=f" between {ref_a} and {ref_b}",
    ))
    groups = render.pair_by_term_and_commit(db.iter_range_events(ref_a, ref_b, filters))
    render_paired_groups(groups, style, full=full, show_commits=commits)
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
    order: SearchOrder = typer.Option(
        SearchOrder.term, "--order",
        help="term: per-term sections, each chronological. "
             "date: newest commit first, terms grouped within each commit.",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", min=1,
        help="Stop after this many term sections (--order term) or "
             "commit blocks (--order date).",
    ),
    reverse: bool = typer.Option(
        False, "--reverse",
        help="Flip the commit-time direction: oldest commits first with "
             "--order date; newest-first within each term section with "
             "--order term.",
    ),
):
    """Find commits that added or removed a clause matching QUERY."""
    db, style = _open_source(source, config)
    with _query_errors():
        since_seq = db.resolve_ref(since) if since is not None else None
        filters = SearchFilters(
            term_id=term, predicate=predicate, since_seq=since_seq,
            regex=regex, ignore_case=ignore_case, namespace=namespace,
        )
        # An invalid --regex pattern surfaces here, on the first query
        # that reaches regexp_matches; the later stream reuses the same
        # pattern, so success here means the stream won't hit it.
        counts = db.search_counts(query, filters)
    if counts.events == 0:
        console.print(f'[yellow]No events matching[/] "{query}"')
        db.close()
        return
    # Provisional scope, printed before results start streaming. These are
    # SQL-level candidate counts — an upper bound on what survives the
    # clause-aware delta filter; the exact totals land in the footer.
    scope = Text("Scanning ", style="dim")
    scope.append_text(counts_phrase(
        counts.events, counts.terms, counts.commits,
        noun="candidate events", tail=" …",
    ))
    console.print(scope)
    groups = render.pair_by_term_and_commit(
        db.iter_search_events(query, filters, order=order.value, reverse=reverse),
        order=order.value,
    )
    filtered = (
        g for g in
        (g._replace(ops=render.filter_ops_by_delta_match(g.ops, query, regex, ignore_case))
         for g in groups)
        if g.ops
    )
    truncated = [False]
    if limit is not None:
        section_key = (
            (lambda g: g.term_id) if order is SearchOrder.term
            else (lambda g: g.head.commit_seq)
        )
        filtered = render.take_sections(filtered, limit, section_key, truncated)
    if order is SearchOrder.term:
        stats = render_paired_groups(filtered, style, full=full, show_commits=commits)
    else:
        stats = render_commit_ordered_groups(filtered, style, full=full, show_commits=commits)
    if stats.events == 0:
        console.print(f'\n[yellow]No events matching[/] "{query}"')
    else:
        verb = "Showed" if truncated[0] else "Found"
        footer = Text("\n")
        footer.append(f"{verb} {stats.events}", style="bold")
        footer.append(" events matching ", style="dim")
        footer.append(f'"{query}"', style="bold")
        footer.append(" across ", style="dim")
        footer.append(f"{stats.terms}", style="bold")
        footer.append(" terms and ", style="dim")
        footer.append(f"{stats.commits}", style="bold")
        footer.append(" commits", style="dim")
        console.print(footer)
        if truncated[0]:
            if order is SearchOrder.term:
                unit = "terms"
            else:
                unit = "oldest commits" if reverse else "most recent commits"
            upper = counts.terms if order is SearchOrder.term else counts.commits
            console.print(Text(
                f"(limited to {limit} {unit}; up to {upper} total — "
                "drop --limit for everything)",
                style="dim",
            ))
    db.close()


@app.command()
def releases(
    source: str = typer.Option(..., "--source", help="Configured source name."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to obohog.toml."),
):
    """List release tags indexed for a source."""
    db, _ = _open_source(source, config)
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


if __name__ == "__main__":
    app()
