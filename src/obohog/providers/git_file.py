"""GitFileProvider — the source is a git repo and the tracked file's
commit history *is* the history we index.

Blob-filters the clone so we don't drag down every blob in the repo,
then backfills only the tracked file's historical blobs in one
delta-packed transfer. The result is a working clone the extract
pipeline can walk with ``git log --follow`` without triggering
per-commit lazy fetches.
"""

from rich.console import Console

from ..config import GitFileSource
from ..gitsource import GitSource


class GitFileProvider:
    """Clone + scoped-backfill the tracked file's history."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def ensure_synced(
        self, source: GitFileSource, *, since: str | None = None
    ) -> str:
        """Return the path to source's clone, cloning it (blob-filtered) if
        missing. Backfills the tracked file's blobs on every call — idempotent
        after the first successful run.

        The scoped backfill is what makes ``git log --follow`` on the tracked
        file fast: without it, git triggers a lazy-fetch per commit during
        rename detection, and a fresh Mondo build would spend minutes chatting
        with the promisor remote instead of parsing.
        """
        if not source.clone_dir.exists():
            bound = f", since {since}" if since else ""
            self.console.print(
                f"Cloning [cyan]{source.repo}[/] (blob-filtered{bound}) → "
                f"{source.clone_dir} …"
            )
            source.clone_dir.parent.mkdir(parents=True, exist_ok=True)
            GitSource.clone(source.repo, source.clone_dir, since=since).close()
        else:
            self.console.print(
                f"Reusing clone at [cyan]{source.clone_dir}[/] "
                "(delete it to re-clone with different bounds)."
            )
        self.console.print(
            f"Backfilling blobs for [cyan]{source.file}[/] "
            "(one delta-packed fetch of all historical versions) …"
        )
        GitSource(source.clone_dir).backfill_file(source.file)
        # Server-side backfill packs are transfer-optimized, not size-optimized —
        # a client-side repack often shrinks them ~5×. Nudge, don't force.
        self.console.print(
            f"[dim]Tip: [cyan]obohog source repack {source.name}[/] to reclaim "
            "disk space on the clone.[/]"
        )
        return str(source.clone_dir)
