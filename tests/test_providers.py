"""Tests for the GitHubReleaseProvider materializer.

Mocks the two subprocess entry points that talk to GitHub (`gh api` for
listing releases, `gh release download` for fetching assets), then
inspects the resulting synthetic git repo to confirm the commit graph
matches the release sequence.
"""

import contextlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from obohog.config import GitHubReleaseSource
from obohog.providers.github_release import GitHubReleaseProvider


@contextlib.contextmanager
def _patch_subprocess(side_effect):
    """Patch subprocess.run in both modules that call out to gh/git.

    Kept as one wrapper so tests don't have to remember every module
    that shells out — the intent is "intercept subprocess in the
    provider machinery", not "patch a specific module".
    """
    with patch("obohog.providers.github_release.subprocess.run", side_effect=side_effect), \
         patch("obohog.providers._synthetic_git.subprocess.run", side_effect=side_effect):
        yield


def _source(clone_dir: Path) -> GitHubReleaseSource:
    return GitHubReleaseSource(
        name="fake",
        repo="https://github.com/example/fake",
        asset="fake.obo",
        clone_dir=clone_dir,
        db_dir=clone_dir.parent / "db",
    )


def _release(
    tag: str,
    published_at: str,
    body: str = "",
    author: str = "publisher",
    *,
    draft: bool = False,
    prerelease: bool = False,
    with_asset: bool = True,
) -> dict:
    return {
        "tag_name": tag,
        "published_at": published_at,
        "body": body,
        "html_url": f"https://github.com/example/fake/releases/tag/{tag}",
        "draft": draft,
        "prerelease": prerelease,
        "author": {"login": author},
        "assets": [{"name": "fake.obo"}] if with_asset else [],
    }


def _fake_gh(releases: list[dict], asset_bytes_by_tag: dict[str, bytes]):
    """Return a function that replaces subprocess.run for gh-related calls.

    Falls through to the real subprocess.run for git commands so we can
    genuinely materialize commits in a real repo.
    """
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        # gh api ... releases → return the release list as JSON on stdout
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[0] == "gh" and cmd[1] == "api":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps(releases), stderr=""
            )
        # gh release download <tag> --dir <tmp> → write the asset bytes
        if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] == "gh" and cmd[1] == "release" and cmd[2] == "download":
            tag = cmd[3]
            # Find --dir <path> and --pattern <name> in the flags.
            it = iter(cmd[4:])
            dest_dir: Path | None = None
            pattern: str | None = None
            for tok in it:
                if tok == "--dir":
                    dest_dir = Path(next(it))
                elif tok == "--pattern":
                    pattern = next(it)
            assert dest_dir is not None and pattern is not None, cmd
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / pattern).write_bytes(asset_bytes_by_tag.get(tag, b""))
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        # Everything else (git commands) — run for real.
        return real_run(cmd, *args, **kwargs)

    return fake_run


def _run_git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def test_materializes_releases_in_oldest_first_order(tmp_path: Path):
    clone_dir = tmp_path / "clone"
    src = _source(clone_dir)
    releases = [
        # Deliberately out of order to prove we sort by published_at.
        _release("v2.0", "2024-05-01T12:00:00Z", body="second"),
        _release("v1.0", "2024-01-01T12:00:00Z", body="first"),
        _release("v3.0", "2024-09-01T12:00:00Z", body="third"),
    ]
    asset_bytes = {
        "v1.0": b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: one\n",
        "v2.0": b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: uno\n",
        "v3.0": b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: uno\n[Term]\nid: FAKE:2\nname: two\n",
    }

    provider = GitHubReleaseProvider(Console(quiet=True))
    with _patch_subprocess(_fake_gh(releases, asset_bytes)):
        returned = provider.ensure_synced(src)

    assert Path(returned) == clone_dir
    # Commits are ordered oldest-first (v1.0 first, v3.0 last).
    log = _run_git("log", "--reverse", "--format=%s", cwd=clone_dir).splitlines()
    assert log == ["v1.0", "v2.0", "v3.0"]
    # Each commit has a matching git tag.
    tags = set(_run_git("tag", "--list", cwd=clone_dir).splitlines())
    assert tags == {"v1.0", "v2.0", "v3.0"}
    # The message body carries the Release URL trailer for extract to parse.
    body = _run_git("log", "-1", "--format=%B", "v2.0", cwd=clone_dir)
    assert "Release URL: https://github.com/example/fake/releases/tag/v2.0" in body


def test_skips_drafts_prereleases_and_missing_assets(tmp_path: Path):
    clone_dir = tmp_path / "clone"
    src = _source(clone_dir)
    releases = [
        _release("v1.0", "2024-01-01T12:00:00Z"),
        _release("v2.0-draft", "2024-01-15T12:00:00Z", draft=True),
        _release("v2.0-rc1", "2024-02-01T12:00:00Z", prerelease=True),
        _release("v2.0-no-asset", "2024-02-15T12:00:00Z", with_asset=False),
        _release("v2.0", "2024-03-01T12:00:00Z"),
    ]
    # Distinct contents so each accepted release is a genuine git commit.
    asset_bytes = {
        "v1.0": b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: one\n",
        "v2.0": b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: uno\n",
    }

    provider = GitHubReleaseProvider(Console(quiet=True))
    with _patch_subprocess(_fake_gh(releases, asset_bytes)):
        provider.ensure_synced(src)

    log = _run_git("log", "--reverse", "--format=%s", cwd=clone_dir).splitlines()
    # Only the two proper releases with matching assets should show up.
    assert log == ["v1.0", "v2.0"]


def test_incremental_sync_only_commits_new_releases(tmp_path: Path):
    clone_dir = tmp_path / "clone"
    src = _source(clone_dir)
    initial = [_release("v1.0", "2024-01-01T12:00:00Z")]
    later = initial + [_release("v2.0", "2024-06-01T12:00:00Z")]
    asset_bytes = {
        "v1.0": b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: one\n",
        "v2.0": b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: uno\n",
    }

    provider = GitHubReleaseProvider(Console(quiet=True))
    with _patch_subprocess(_fake_gh(initial, asset_bytes)):
        provider.ensure_synced(src)
    assert _run_git("log", "--format=%s", cwd=clone_dir).splitlines() == ["v1.0"]

    with _patch_subprocess(_fake_gh(later, asset_bytes)):
        provider.ensure_synced(src)
    # v1.0 is not re-committed; only v2.0 lands as a new commit.
    log = _run_git("log", "--reverse", "--format=%s", cwd=clone_dir).splitlines()
    assert log == ["v1.0", "v2.0"]


def test_rolling_tag_shares_a_commit(tmp_path: Path):
    """A release whose asset is byte-identical to the previous release (e.g.
    a rolling "current"/"latest" tag) shouldn't crash the materializer or
    produce an empty commit — it should tag the existing commit."""
    clone_dir = tmp_path / "clone"
    src = _source(clone_dir)
    releases = [
        _release("v1.0", "2024-01-01T12:00:00Z"),
        _release("current", "2024-01-02T12:00:00Z", body="rolling latest"),
    ]
    same_bytes = b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: one\n"
    asset_bytes = {"v1.0": same_bytes, "current": same_bytes}

    provider = GitHubReleaseProvider(Console(quiet=True))
    with _patch_subprocess(_fake_gh(releases, asset_bytes)):
        provider.ensure_synced(src)

    # Exactly one commit lands (for v1.0's actual content).
    log = _run_git("log", "--format=%s", cwd=clone_dir).splitlines()
    assert log == ["v1.0"]
    # Both tags exist and point to the same sha.
    v1_sha = _run_git("rev-parse", "v1.0", cwd=clone_dir)
    current_sha = _run_git("rev-parse", "current", cwd=clone_dir)
    assert v1_sha == current_sha


def test_non_github_repo_url_raises(tmp_path: Path):
    clone_dir = tmp_path / "clone"
    src = GitHubReleaseSource(
        name="bad",
        repo="https://gitlab.com/example/fake",
        asset="fake.obo",
        clone_dir=clone_dir,
        db_dir=clone_dir.parent / "db",
    )
    provider = GitHubReleaseProvider(Console(quiet=True))
    with pytest.raises(ValueError, match="github.com"):
        provider.ensure_synced(src)


# ---------------------------------------------------------------------------
# BioPortalProvider tests
# ---------------------------------------------------------------------------

import io  # noqa: E402
import re  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from obohog.config import BioPortalSource, ConfigError  # noqa: E402
from obohog.providers.bioportal import BioPortalProvider  # noqa: E402


def _bioportal_source(clone_dir: Path, acronym: str = "FAKE") -> BioPortalSource:
    return BioPortalSource(
        name=acronym.lower(),
        acronym=acronym,
        clone_dir=clone_dir,
        db_dir=clone_dir.parent / "db",
    )


def _submission(
    submission_id: int,
    version: str | None,
    released: str,
    creation_date: str | None = None,
    *,
    language: str = "OBO",
    contact_name: str = "A. Curator",
    contact_email: str = "a@example.org",
) -> dict:
    return {
        "submissionId": submission_id,
        "version": version,
        "released": released,
        "creationDate": creation_date or released,
        "hasOntologyLanguage": language,
        "description": "",
        "contact": [{"name": contact_name, "email": contact_email}],
    }


def _fake_response(*, status: int = 200, json_body=None, raw_bytes: bytes = b""):
    """A ``requests.Response``-shaped MagicMock for use in test fakes."""
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"status {status}")
    if json_body is not None:
        resp.json.return_value = json_body
    resp.raw = io.BytesIO(raw_bytes)
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda self, *a: False
    return resp


@contextmanager
def _fake_bioportal(
    submissions: list[dict],
    obo_bytes_by_id: dict[int, bytes],
    *,
    api_key: str | None = "test-key",
):
    """Patch the requests session BioPortalProvider uses.

    ``settings.get_settings`` is monkeypatched to return the given API
    key (or None to simulate a missing key).
    """
    def fake_session_get(session_self, url, *args, **kwargs):
        if "/submissions?" in url:
            return _fake_response(json_body=submissions)
        m = re.search(r"/submissions/(\d+)/download", url)
        if m:
            sub_id = int(m.group(1))
            return _fake_response(raw_bytes=obo_bytes_by_id.get(sub_id, b""))
        raise AssertionError(f"unexpected GET: {url}")

    fake_settings = MagicMock()
    fake_settings.bioportal_api_key = api_key

    import requests as _requests
    with patch.object(_requests.Session, "get", new=fake_session_get), \
         patch("obohog.providers.bioportal.get_settings", return_value=fake_settings), \
         patch("obohog.providers._synthetic_git.subprocess.run", side_effect=subprocess.run):
        yield


# Add missing import for `requests` module used in _fake_response.
import requests  # noqa: E402


def test_bioportal_materializes_oldest_first_and_tags(tmp_path: Path):
    clone_dir = tmp_path / "clone"
    src = _bioportal_source(clone_dir, "FAKE")
    submissions = [
        # Deliberately out of order to prove we sort by picked date.
        _submission(3, "v3.0", "2024-09-01T00:00:00Z"),
        _submission(1, "v1.0", "2024-01-01T00:00:00Z"),
        _submission(2, "v2.0", "2024-05-01T00:00:00Z"),
    ]
    obo_bytes = {
        1: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: one\n",
        2: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: uno\n",
        3: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: uno\n[Term]\nid: FAKE:2\nname: two\n",
    }
    provider = BioPortalProvider(Console(quiet=True))
    with _fake_bioportal(submissions, obo_bytes):
        provider.ensure_synced(src)

    log = _run_git("log", "--reverse", "--format=%s", cwd=clone_dir).splitlines()
    assert log == ["v1.0", "v2.0", "v3.0"]
    tags = set(_run_git("tag", "--list", cwd=clone_dir).splitlines())
    assert tags == {"v1.0", "v2.0", "v3.0"}


def test_bioportal_skips_non_obo_submissions(tmp_path: Path):
    clone_dir = tmp_path / "clone"
    src = _bioportal_source(clone_dir, "FAKE")
    submissions = [
        _submission(1, "v1.0", "2024-01-01T00:00:00Z"),
        _submission(2, "v2.0-owl", "2024-02-01T00:00:00Z", language="OWL"),
        _submission(3, "v2.0", "2024-03-01T00:00:00Z"),
    ]
    obo_bytes = {
        1: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: one\n",
        3: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: uno\n",
    }
    provider = BioPortalProvider(Console(quiet=True))
    with _fake_bioportal(submissions, obo_bytes):
        provider.ensure_synced(src)

    log = _run_git("log", "--reverse", "--format=%s", cwd=clone_dir).splitlines()
    # OWL submission dropped; only the two OBO ones landed.
    assert log == ["v1.0", "v2.0"]


def test_bioportal_fails_when_no_obo_submissions(tmp_path: Path):
    clone_dir = tmp_path / "clone"
    src = _bioportal_source(clone_dir, "OWLY")
    submissions = [
        _submission(1, "v1.0", "2024-01-01T00:00:00Z", language="OWL"),
        _submission(2, "v2.0", "2024-05-01T00:00:00Z", language="RDF/XML"),
    ]
    provider = BioPortalProvider(Console(quiet=True))
    with _fake_bioportal(submissions, obo_bytes_by_id={}):
        with pytest.raises(ConfigError, match="no OBO-format submissions"):
            provider.ensure_synced(src)


def test_bioportal_empty_version_falls_back_to_sub_id(tmp_path: Path):
    clone_dir = tmp_path / "clone"
    src = _bioportal_source(clone_dir, "FAKE")
    submissions = [
        _submission(7, None, "2024-01-01T00:00:00Z"),
        _submission(8, "", "2024-02-01T00:00:00Z"),
        _submission(9, "bad:tag", "2024-03-01T00:00:00Z"),  # colons aren't valid ref names
    ]
    obo_bytes = {
        7: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: a\n",
        8: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: b\n",
        9: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: c\n",
    }
    provider = BioPortalProvider(Console(quiet=True))
    with _fake_bioportal(submissions, obo_bytes):
        provider.ensure_synced(src)

    tags = set(_run_git("tag", "--list", cwd=clone_dir).splitlines())
    assert tags == {"sub-7", "sub-8", "sub-9"}


def test_bioportal_rolling_submission_shares_a_commit(tmp_path: Path):
    """Byte-identical OBO downloads share a commit with the previous
    submission (same rolling-tag path the github-release provider uses)."""
    clone_dir = tmp_path / "clone"
    src = _bioportal_source(clone_dir, "FAKE")
    submissions = [
        _submission(1, "v1.0", "2024-01-01T00:00:00Z"),
        _submission(2, "v1.0-rerun", "2024-02-01T00:00:00Z"),
    ]
    identical = b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: one\n"
    obo_bytes = {1: identical, 2: identical}
    provider = BioPortalProvider(Console(quiet=True))
    with _fake_bioportal(submissions, obo_bytes):
        provider.ensure_synced(src)

    log = _run_git("log", "--format=%s", cwd=clone_dir).splitlines()
    assert log == ["v1.0"]
    v1_sha = _run_git("rev-parse", "v1.0", cwd=clone_dir)
    rerun_sha = _run_git("rev-parse", "v1.0-rerun", cwd=clone_dir)
    assert v1_sha == rerun_sha


def test_bioportal_skips_mislabeled_non_obo_download(tmp_path: Path):
    """BioPortal's per-submission metadata sometimes claims OBO for content
    that was actually uploaded in a different format (ODT, docx). The
    provider should log a skip rather than commit garbage."""
    clone_dir = tmp_path / "clone"
    src = _bioportal_source(clone_dir, "FAKE")
    submissions = [
        _submission(1, "sub-1", "2015-01-01T00:00:00Z"),
        _submission(2, "v1.0", "2016-01-01T00:00:00Z"),
    ]
    obo_bytes = {
        # Submission 1: metadata claims OBO but bytes are actually an ODT
        # zip container. Should be skipped.
        1: b"PK\x03\x04garbage",
        # Submission 2: real OBO. Should commit.
        2: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: uno\n",
    }
    provider = BioPortalProvider(Console(quiet=True))
    with _fake_bioportal(submissions, obo_bytes):
        provider.ensure_synced(src)

    log = _run_git("log", "--reverse", "--format=%s", cwd=clone_dir).splitlines()
    assert log == ["v1.0"]


def test_bioportal_incremental_sync(tmp_path: Path):
    clone_dir = tmp_path / "clone"
    src = _bioportal_source(clone_dir, "FAKE")
    first = [_submission(1, "v1.0", "2024-01-01T00:00:00Z")]
    later = first + [_submission(2, "v2.0", "2024-06-01T00:00:00Z")]
    obo_bytes = {
        1: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: a\n",
        2: b"format-version: 1.2\n[Term]\nid: FAKE:1\nname: b\n",
    }
    provider = BioPortalProvider(Console(quiet=True))
    with _fake_bioportal(first, obo_bytes):
        provider.ensure_synced(src)
    with _fake_bioportal(later, obo_bytes):
        provider.ensure_synced(src)

    log = _run_git("log", "--reverse", "--format=%s", cwd=clone_dir).splitlines()
    assert log == ["v1.0", "v2.0"]


def test_bioportal_missing_api_key_raises_config_error(tmp_path: Path):
    clone_dir = tmp_path / "clone"
    src = _bioportal_source(clone_dir, "FAKE")
    provider = BioPortalProvider(Console(quiet=True))
    with _fake_bioportal([], obo_bytes_by_id={}, api_key=None):
        with pytest.raises(ConfigError, match="BIOPORTAL_API_KEY"):
            provider.ensure_synced(src)
