"""BioPortalProvider — materialize BioPortal submissions as synthetic git commits.

BioPortal (https://bioportal.bioontology.org) publishes many ontologies as
a series of *submissions*, each of which can be downloaded in OBO format
(when the submission was uploaded as OBO). This provider walks a source's
submission list, downloads each OBO-format submission, and commits it into
a synthetic git repo — one commit per submission, tagged with the
submission's version string (or ``sub-<id>`` if the version is empty).

Provenance mapping:

* ``author_name`` / ``author_email`` = first contact on the submission
  (fallback ``"BioPortal" / "bioportal@bioontology.org"``)
* ``committed_date`` = ``released`` if present, else ``creationDate``
* ``message`` = version (or ``sub-<id>``) — nothing else; ontology-level
  ``description`` is typically copy-pasted across every submission and
  adds noise, not signal.
* git tag = version if usable, else ``sub-<submissionId>``

BioPortal doesn't publish per-submission web pages worth linking to —
the ontology's submission table lives on a single ``/ontologies/<acronym>``
page — so commits from this provider carry no ``snapshot_url``.

**No OBO conversion.** Submissions with ``hasOntologyLanguage != "OBO"``
are skipped. If the ontology publishes zero OBO submissions, the sync
errors out rather than producing an empty artifact.
"""

import json
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from rich.console import Console

from ..config import BioPortalSource, ConfigError
from ..settings import get_settings
from ._synthetic_git import (
    commit_or_tag_head,
    git_init,
    list_local_tags,
    run_git,
)

_API_BASE = "https://data.bioontology.org"
_TAG_SAFE = re.compile(r"^[A-Za-z0-9._/-]+$")


class BioPortalProvider:
    """Fetch OBO submissions from BioPortal and materialize them into a git repo."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def ensure_synced(
        self, source: BioPortalSource, *, since: str | None = None
    ) -> str:
        """Materialize every unseen OBO-format submission into a synthetic
        commit, oldest first, tagged with the submission's version.

        ``since`` is accepted for interface compatibility with the other
        providers but ignored — BioPortal incremental sync compares local
        tags to remote submissions.
        """
        api_key = _require_api_key()
        clone_dir = source.clone_dir

        if not clone_dir.exists():
            clone_dir.parent.mkdir(parents=True, exist_ok=True)
            self.console.print(
                f"Initializing synthetic clone at [cyan]{clone_dir}[/] "
                f"for BioPortal source [cyan]{source.name}[/] …"
            )
            git_init(clone_dir)
            local_tags: set[str] = set()
        else:
            self.console.print(
                f"Reusing clone at [cyan]{clone_dir}[/]."
            )
            local_tags = list_local_tags(clone_dir)

        self.console.print(
            f"Fetching submission list for [cyan]{source.acronym}[/] from BioPortal …"
        )
        submissions = _list_submissions(source.acronym, api_key)

        # Keep OBO only. No conversion — we don't want to index something
        # the ontology didn't publish.
        obo_only = [s for s in submissions if s.get("hasOntologyLanguage") == "OBO"]
        if not obo_only:
            raise ConfigError(
                f"BioPortal ontology {source.acronym!r} has no OBO-format "
                f"submissions; can't index. Only OBO downloads are supported "
                f"(no OWL→OBO conversion)."
            )
        skipped = len(submissions) - len(obo_only)
        if skipped:
            self.console.print(
                f"[dim]Skipping {skipped} non-OBO submission(s).[/]"
            )

        # Attach a picked date to each submission and sort oldest-first
        # so the git history reads chronologically.
        for s in obo_only:
            s["_picked_date"] = _pick_date(s)
        obo_only.sort(key=lambda s: s["_picked_date"])

        # Skip anything already tagged locally.
        new_subs = [s for s in obo_only if _tag_for(s) not in local_tags]

        if not new_subs:
            self.console.print(
                f"[dim]No new submissions to materialize for {source.name}.[/]"
            )
            return str(clone_dir)

        self.console.print(
            f"Materializing [bold]{len(new_subs)}[/] submission(s) …"
        )
        for sub in new_subs:
            self._materialize_submission(source, sub, clone_dir, api_key)

        return str(clone_dir)

    def _materialize_submission(
        self,
        source: BioPortalSource,
        sub: dict,
        clone_dir: Path,
        api_key: str,
    ) -> None:
        sub_id = sub["submissionId"]
        tag = _tag_for(sub)
        self.console.print(
            f"  [dim]· fetching[/] [cyan]{tag}[/] "
            f"[dim](submission #{sub_id})[/]"
        )
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = _download_obo(source.acronym, sub_id, api_key, Path(tmp))
            dest = clone_dir / source.tracked_path
            shutil.copy(downloaded, dest)

        run_git(clone_dir, "add", source.tracked_path)
        author_name, author_email = _author(sub)
        commit_or_tag_head(
            clone_dir,
            tag,
            author_name=author_name,
            author_email=author_email,
            committed_date=sub["_picked_date"],
            message=tag,
            console=self.console,
        )


def _require_api_key() -> str:
    """Return the API key or raise a clear error pointing at ``.env``."""
    key = get_settings().bioportal_api_key
    if not key:
        raise ConfigError(
            "BIOPORTAL_API_KEY is not set. Put it in .env at the project "
            "root: `BIOPORTAL_API_KEY=<your key>` (get one at "
            "https://bioportal.bioontology.org/account)."
        )
    return key


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"apikey token={api_key}"}


def _list_submissions(acronym: str, api_key: str) -> list[dict]:
    """Return every submission for ``acronym`` as a list of dicts.

    Handles two response shapes BioPortal sometimes uses for list
    endpoints:

    * plain JSON array (small ontologies, no pagination) — return as-is
    * ``{"collection": [...], "nextPage": "url" | null}`` — walk pages
    """
    display = (
        "submissionId,version,released,creationDate,hasOntologyLanguage,"
        "description,contact"
    )
    url = (
        f"{_API_BASE}/ontologies/{acronym}/submissions"
        f"?display={display}&pagesize=200"
    )
    out: list[dict] = []
    while url:
        payload = _get_json(url, api_key)
        if isinstance(payload, list):
            out.extend(payload)
            url = None
        elif isinstance(payload, dict):
            out.extend(payload.get("collection", []))
            url = payload.get("nextPage")
        else:
            raise RuntimeError(
                f"BioPortal returned an unexpected shape for {url!r}: "
                f"{type(payload).__name__}"
            )
    return out


def _get_json(url: str, api_key: str) -> object:
    """GET a JSON URL with BioPortal auth. Translates 401/403 into a
    friendly ConfigError so users know exactly what went wrong."""
    req = urllib.request.Request(url, headers=_auth_headers(api_key))
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code in (401, 403):
            raise ConfigError(
                f"BioPortal rejected the API key (HTTP {err.code}). "
                "Check BIOPORTAL_API_KEY in .env."
            )
        raise


def _download_obo(
    acronym: str, submission_id: int, api_key: str, dest_dir: Path
) -> Path:
    """Download the OBO representation of submission ``submission_id`` into
    ``dest_dir``. Returns the local file path."""
    url = (
        f"{_API_BASE}/ontologies/{acronym}/submissions/{submission_id}/download"
        "?download_format=obo"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{acronym}.obo"
    req = urllib.request.Request(url, headers=_auth_headers(api_key))
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    return dest


def _pick_date(sub: dict) -> str:
    """Prefer ``released``; fall back to ``creationDate``. Both are ISO
    strings that git accepts as-is via GIT_AUTHOR_DATE."""
    return sub.get("released") or sub["creationDate"]


def _tag_for(sub: dict) -> str:
    """The tag we give this submission's commit.

    Prefer the submission's ``version`` string, but only if it's non-empty
    and syntactically valid as a git ref name; otherwise fall back to a
    stable synthetic ``sub-<submissionId>``.
    """
    version = sub.get("version")
    if version and _TAG_SAFE.match(version):
        return version
    return f"sub-{sub['submissionId']}"


def _author(sub: dict) -> tuple[str, str]:
    contacts = sub.get("contact") or []
    if contacts and isinstance(contacts[0], dict):
        name = contacts[0].get("name") or "BioPortal"
        email = contacts[0].get("email") or "bioportal@bioontology.org"
        return str(name), str(email)
    return "BioPortal", "bioportal@bioontology.org"
