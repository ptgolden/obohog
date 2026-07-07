"""GitHubReleaseProvider — materialize published releases as synthetic git commits.

The tool doesn't parse or diff snapshots directly — it converts each release
into a single commit in a synthetic git repo, then feeds that repo to the same
extract pipeline that git-file sources use. This preserves the diff-scoped
parsing optimization (git's blob-level diff tells us which stanzas changed;
we only fastobo-parse those) and every render/query feature at zero
architectural cost.

Implementation lands in a follow-up commit — see the roadmap for
``providers/github_release.py``. This stub keeps the discriminated union
functional (config parses, ``get_provider`` dispatches) but any actual
``source sync`` on a github-release source will raise until the
materializer arrives.
"""

from rich.console import Console

from ..config import GitHubReleaseSource


class GitHubReleaseProvider:
    """Placeholder for the release materializer — implementation to follow."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def ensure_synced(
        self, source: GitHubReleaseSource, *, since: str | None = None
    ) -> str:
        raise NotImplementedError(
            "GitHubReleaseProvider.ensure_synced is not yet implemented."
        )
