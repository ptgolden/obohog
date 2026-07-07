"""History-acquisition providers per source type.

Each provider's job is to sync a source into a git repo at
``source.clone_dir``. The extract pipeline then walks that repo
identically regardless of provider — see :mod:`obohog.extract`. This
keeps a clean seam at *acquisition* while letting all downstream code
stay git-shaped and single-code-path.

Three providers today:

* :class:`~obohog.providers.git_file.GitFileProvider` clones the source
  repo (blob-filtered) and backfills the tracked file's history. Nothing
  in the resulting clone is synthetic — it *is* the source's git history.

* :class:`~obohog.providers.github_release.GitHubReleaseProvider`
  materializes GitHub Releases into a synthetic git repo — one commit
  per release, tagged with the release name. Author, date, and message
  come from release metadata; the release page URL is recorded so
  renderers can link back.

* :class:`~obohog.providers.bioportal.BioPortalProvider` materializes
  BioPortal submissions into a synthetic git repo — one commit per
  OBO-format submission, tagged with the submission's version.
"""

from typing import Protocol

from rich.console import Console

from ..config import AnySource, BioPortalSource, GitFileSource, GitHubReleaseSource


class Provider(Protocol):
    """The single contract every provider satisfies."""

    def ensure_synced(
        self,
        source: AnySource,
        *,
        since: str | None = None,
    ) -> str:
        """Bring the source's clone up to date; return the clone path.

        Idempotent: re-run should do only the incremental work needed.
        Should stream progress to the provider's console for long steps.
        """


def get_provider(source: AnySource, console: Console) -> Provider:
    """Return the provider that knows how to sync ``source``."""
    if isinstance(source, GitFileSource):
        from .git_file import GitFileProvider

        return GitFileProvider(console)
    if isinstance(source, GitHubReleaseSource):
        from .github_release import GitHubReleaseProvider

        return GitHubReleaseProvider(console)
    if isinstance(source, BioPortalSource):
        from .bioportal import BioPortalProvider

        return BioPortalProvider(console)
    raise TypeError(f"No provider registered for source type: {type(source).__name__}")
