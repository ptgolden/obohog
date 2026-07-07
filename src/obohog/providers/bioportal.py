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

import re
import shutil
import tempfile
from pathlib import Path

import requests
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

# Version strings BioPortal (or its uploaders) use as placeholders for
# "we don't have a real version to report." Treated the same as ``null`` /
# empty: fall back to the synthetic ``sub-<id>`` tag so each submission
# with such a placeholder gets its own commit rather than colliding with
# every other unversioned one.
_PLACEHOLDER_VERSIONS = frozenset({"unknown"})


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

        # Dedupe. Two ways a submission gets filtered here:
        #
        # 1. Its tag is already committed from a previous sync (idempotency).
        #    Silent skip — expected on every re-run.
        # 2. Its tag matches an earlier submission in this same batch. That
        #    means BioPortal re-processed the same released tarball without
        #    bumping ``version`` (ZFA does this a lot: runs of a dozen
        #    submissions all tagged ``releases/YYYY-MM-DD``). Skip and log
        #    so the user sees how many redundant API hits were avoided.
        seen = set(local_tags)
        new_subs: list[dict] = []
        dup_in_batch = 0
        for s in obo_only:
            tag = _tag_for(s)
            if tag in local_tags:
                continue
            if tag in seen:
                dup_in_batch += 1
                continue
            new_subs.append(s)
            seen.add(tag)

        if dup_in_batch:
            self.console.print(
                f"[dim]Skipping {dup_in_batch} duplicate-version "
                "submission(s) (BioPortal re-processed a release without "
                "bumping the version string).[/]"
            )

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
            try:
                downloaded = _download_obo(source.acronym, sub_id, api_key, Path(tmp))
            except _NotOBO:
                # BioPortal's per-submission metadata sometimes claims OBO
                # for content that was actually uploaded in another format
                # (ODT, docx, older OWL exports). Skip and move on.
                self.console.print(
                    f"  [yellow]· skipping[/] [cyan]{tag}[/] "
                    f"[dim](submission #{sub_id}: metadata says OBO but "
                    "download isn't)[/]"
                )
                return
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
            message=_commit_subject(sub, tag),
            console=self.console,
        )


def _commit_subject(sub: dict, tag: str) -> str:
    """Human-facing commit subject for a BioPortal submission.

    If the submission had a real version (any value other than a
    placeholder / null / empty), use it verbatim so the subject reads
    like the version-labeled tag. Otherwise say ``Submission #<id>``
    rather than exposing the synthetic ``sub-<id>`` git tag we picked
    for the ref.
    """
    if tag.startswith("sub-"):
        return f"Submission #{sub['submissionId']}"
    return tag


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


def _session(api_key: str) -> requests.Session:
    """Requests session pre-configured with BioPortal auth."""
    session = requests.Session()
    session.headers["Authorization"] = f"apikey token={api_key}"
    return session


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
    url: str | None = (
        f"{_API_BASE}/ontologies/{acronym}/submissions"
        f"?display={display}&pagesize=200"
    )
    session = _session(api_key)
    out: list[dict] = []
    while url:
        payload = _get_json(session, url)
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


def _get_json(session: requests.Session, url: str) -> object:
    """GET a JSON URL. Translates 401/403 into a friendly ConfigError so
    users know exactly what went wrong."""
    resp = session.get(url)
    if resp.status_code in (401, 403):
        raise ConfigError(
            f"BioPortal rejected the API key (HTTP {resp.status_code}). "
            "Check BIOPORTAL_API_KEY in .env."
        )
    resp.raise_for_status()
    return resp.json()


def _download_obo(
    acronym: str, submission_id: int, api_key: str, dest_dir: Path
) -> Path:
    """Download submission ``submission_id`` as OBO bytes into ``dest_dir``.

    BioPortal's ``download`` endpoint returns the file *as uploaded* — no
    server-side format conversion despite what the metadata claims (some
    submissions self-report as OBO but were uploaded as ODT / OWL / OFN).
    Verify the downloaded content looks like OBO before returning; on
    failure, raise :class:`_NotOBO` so the caller can log a skip.
    """
    url = f"{_API_BASE}/ontologies/{acronym}/submissions/{submission_id}/download"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{acronym}.obo"
    with _session(api_key).get(url, stream=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
    if not _looks_like_obo(dest):
        raise _NotOBO(submission_id, acronym)
    return dest


class _NotOBO(RuntimeError):
    """Downloaded submission bytes don't look like OBO — the ``hasOntologyLanguage``
    metadata claimed OBO but the actual upload wasn't."""

    def __init__(self, submission_id: int, acronym: str):
        self.submission_id = submission_id
        self.acronym = acronym
        super().__init__(
            f"BioPortal submission {acronym}/{submission_id} claims OBO in "
            "metadata but the download bytes aren't OBO."
        )


# Canonical OBO 1.x header. Any submission whose file starts with this line
# is (structurally) OBO; anything else is a mislabeled upload we shouldn't
# try to index.
_OBO_HEADER = re.compile(rb"^\s*format-version:\s*\d")


def _looks_like_obo(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(2048)
    return bool(_OBO_HEADER.search(head))


def _pick_date(sub: dict) -> str:
    """Prefer ``released``; fall back to ``creationDate``. Both are ISO
    strings that git accepts as-is via GIT_AUTHOR_DATE."""
    return sub.get("released") or sub["creationDate"]


def _tag_for(sub: dict) -> str:
    """The tag we give this submission's commit.

    Prefer the submission's ``version`` string, but only if it's non-empty,
    isn't a known placeholder (``"unknown"``), and is syntactically valid
    as a git ref name. Otherwise fall back to a stable synthetic
    ``sub-<submissionId>`` — this keeps each unversioned submission on
    its own commit rather than collapsing them by mistake.
    """
    version = sub.get("version")
    if (
        version
        and version.strip().lower() not in _PLACEHOLDER_VERSIONS
        and _TAG_SAFE.match(version)
    ):
        return version
    return f"sub-{sub['submissionId']}"


def _author(sub: dict) -> tuple[str, str]:
    contacts = sub.get("contact") or []
    if contacts and isinstance(contacts[0], dict):
        name = contacts[0].get("name") or "BioPortal"
        email = contacts[0].get("email") or "bioportal@bioontology.org"
        return str(name), str(email)
    return "BioPortal", "bioportal@bioontology.org"
