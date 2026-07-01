"""Command-line interface for building and querying the history artifact."""

from itertools import groupby
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.text import Text

from .extract import build_parallel
from .extract import extract as run_extract
from .gitsource import GitSource
from .query import ArtifactNotFound, Change, HistoryDB

DEFAULT_PATH = "src/ontology/mondo-edit.obo"
DEFAULT_ARTIFACT = Path("artifact")
DEFAULT_URL = "https://github.com/monarch-initiative/mondo.git"
DEFAULT_CLONE = Path("mondo-clone")

app = typer.Typer(add_completion=False, help="Build and query the Mondo history index.")
console = Console()


def _open(artifact: Path) -> HistoryDB:
    """Open an artifact, exiting cleanly with guidance if it isn't there."""
    try:
        return HistoryDB(artifact)
    except ArtifactNotFound as err:
        console.print(f"[red]{err}[/]")
        raise typer.Exit(1)


@app.command()
def build(
    out: Path = typer.Option(DEFAULT_ARTIFACT, help="Output artifact directory."),
    url: str = typer.Option(DEFAULT_URL, help="Mondo repo to clone (blob-filtered)."),
    since: Optional[str] = typer.Option(
        None, help="Only index history at/after this git date, e.g. 2026-06-01."
    ),
    clone_dir: Path = typer.Option(DEFAULT_CLONE, help="Where the blob-filtered clone lives."),
    repo: Optional[str] = typer.Option(
        None, help="Use an existing local clone instead of cloning --url."
    ),
    path: str = typer.Option(DEFAULT_PATH, help="File whose history to index."),
    limit: Optional[int] = typer.Option(None, help="Index only the most recent N versions."),
    jobs: int = typer.Option(
        0, help="Parser processes: 0 = auto (cores-2), 1 = single-threaded, N = that many."
    ),
    progress: bool = typer.Option(True, help="Show a per-commit progress bar (parallel builds)."),
):
    """Extract history into a Parquet artifact, cloning Mondo if needed."""
    clone_path = _ensure_clone(url, since, clone_dir, repo)
    if jobs == 1:
        with GitSource(clone_path) as src:
            counts = run_extract(src, path, out, limit=limit)
    else:
        counts = build_parallel(
            clone_path, path, out, jobs=(jobs or None), limit=limit, progress=progress
        )
    msg = (
        f"[green]Built[/] {out} — {counts['commits']} commits, "
        f"{counts['snapshots']} snapshots, {counts['events']} events"
    )
    if counts.get("skipped"):
        msg += f", [yellow]{counts['skipped']} skipped[/]"
    console.print(msg + ".")


def _ensure_clone(url: str, since: Optional[str], clone_dir: Path, repo: Optional[str]) -> str:
    """Return a path to a local clone, cloning if necessary."""
    if repo is not None:
        console.print(f"Reading history from existing clone [cyan]{repo}[/].")
        return repo
    if clone_dir.exists():
        console.print(
            f"Reusing clone at [cyan]{clone_dir}[/] "
            "(delete it to re-clone with different bounds)."
        )
        return str(clone_dir)
    bound = f", since {since}" if since else ""
    console.print(f"Cloning [cyan]{url}[/] (blob-filtered{bound}) → {clone_dir} …")
    GitSource.clone(url, clone_dir, since=since).close()
    return str(clone_dir)


@app.command()
def term(
    mondo_id: str = typer.Argument(..., help="e.g. MONDO:0007739"),
    artifact: Path = typer.Option(DEFAULT_ARTIFACT, help="Artifact directory."),
    only: Optional[str] = typer.Option(None, help="Restrict to one clause kind, e.g. synonym."),
    at: Optional[int] = typer.Option(None, help="Reconstruct state as of this commit_seq."),
):
    """Show a term's change history, or its reconstructed state at a point."""
    db = _open(artifact)
    if at is not None:
        _render_state(mondo_id, at, db.term_at(mondo_id, at))
    else:
        _render_timeline(mondo_id, db.term_timeline(mondo_id, predicate=only))
    db.close()


@app.command()
def commit(
    sha: str = typer.Argument(..., help="Commit sha or unique prefix."),
    artifact: Path = typer.Option(DEFAULT_ARTIFACT, help="Artifact directory."),
):
    """List the terms changed together in one commit."""
    db = _open(artifact)
    terms = db.commit_terms(sha)
    if not terms:
        console.print(f"[yellow]No indexed changes for commit[/] {sha}")
    else:
        console.print(f"[bold]{len(terms)}[/] terms changed in {sha}:")
        for mondo_id, name in terms:
            line = Text("  ")
            line.append(mondo_id, style="cyan")
            if name:
                line.append(f"  {name}", style="dim")
            console.print(line)
    db.close()


@app.command()
def pr(
    number: int = typer.Argument(..., help="Pull request number, e.g. 10343."),
    artifact: Path = typer.Option(DEFAULT_ARTIFACT, help="Artifact directory."),
):
    """List the terms changed by a pull request."""
    db = _open(artifact)
    terms = db.pr_terms(number)
    if not terms:
        console.print(f"[yellow]No indexed changes for PR[/] #{number}")
    else:
        console.print(f"[bold]{len(terms)}[/] terms changed in PR #{number}:")
        for mondo_id, name in terms:
            line = Text("  ")
            line.append(mondo_id, style="cyan")
            if name:
                line.append(f"  {name}", style="dim")
            console.print(line)
    db.close()


@app.command()
def diff(
    ref_a: str = typer.Argument(..., help="Release tag, commit_seq, or sha."),
    ref_b: str = typer.Argument(..., help="Release tag, commit_seq, or sha."),
    artifact: Path = typer.Option(DEFAULT_ARTIFACT, help="Artifact directory."),
    term: Optional[str] = typer.Option(None, help="Restrict to one term."),
):
    """Show clause changes between two points (e.g. two releases)."""
    db = _open(artifact)
    rows = db.changes_between(ref_a, ref_b, mondo_id=term)
    if not rows:
        console.print(f"[yellow]No changes between[/] {ref_a} [yellow]and[/] {ref_b}")
        db.close()
        return
    n_terms = len({r[0] for r in rows})
    console.print(
        f"[bold]{len(rows)}[/] changes across [bold]{n_terms}[/] terms "
        f"between {ref_a} and {ref_b}:"
    )
    for mondo_id, group in groupby(rows, key=lambda r: r[0]):
        console.print(Text(mondo_id, style="bold cyan"))
        for _, operation, predicate, value, _seq, _pr in group:
            line = Text("    ")
            line.append("+ " if operation == "add" else "- ",
                        style="bold green" if operation == "add" else "bold red")
            line.append(f"{predicate}: {value}")
            console.print(line)
    db.close()


@app.command()
def releases(artifact: Path = typer.Option(DEFAULT_ARTIFACT, help="Artifact directory.")):
    """List release tags indexed in this artifact."""
    db = _open(artifact)
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


def _render_timeline(mondo_id: str, changes: list[Change]) -> None:
    if not changes:
        console.print(f"[yellow]No history for[/] {mondo_id}")
        return
    console.print(f"[bold cyan]{mondo_id}[/] — {len(changes)} changes")
    for _, group in groupby(changes, key=lambda c: c.commit_seq):
        rows = list(group)
        head = rows[0]
        header = Text("\n● ")
        header.append(f"commit {head.commit_seq}", style="bold")
        header.append(f"  {_date(head.committed_date)}  ")
        if head.pr_number is not None:
            header.append(f"PR #{head.pr_number}  ", style="cyan")
        header.append(head.message.splitlines()[0], style="dim")
        console.print(header)
        for change in rows:
            line = Text("    ")
            if change.operation == "add":
                line.append("+ ", style="bold green")
            else:
                line.append("- ", style="bold red")
            line.append(f"{change.predicate}: {change.value}")
            console.print(line)


def _render_state(mondo_id: str, at: int, clauses: list[tuple[str, str]]) -> None:
    if not clauses:
        console.print(f"[yellow]{mondo_id} has no snapshot at or before commit {at}[/]")
        return
    console.print(f"[bold cyan]{mondo_id}[/] as of commit {at}:")
    console.print(Text(f"  id: {mondo_id}"))
    for predicate, value in clauses:
        console.print(Text(f"  {predicate}: {value}"))


def _date(value: object) -> str:
    return str(value)[:10]


if __name__ == "__main__":
    app()
