"""Shared fixtures: a tiny, deterministic multi-commit OBO git repository."""

import os
import subprocess
from pathlib import Path

import pytest

HEADER = "format-version: 1.2\n\n"


def _term(mondo_id: str, *clauses: str) -> str:
    body = "\n".join([f"id: {mondo_id}", *clauses])
    return f"[Term]\n{body}\n"


def _git(repo: Path, *args: str, date: str | None = None) -> None:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="Test Author",
        GIT_AUTHOR_EMAIL="author@example.org",
        GIT_COMMITTER_NAME="Test Author",
        GIT_COMMITTER_EMAIL="author@example.org",
    )
    if date is not None:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, env=env
    )


def _write(repo: Path, rel: str, content: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def obo_repo(tmp_path: Path) -> Path:
    """A git repo whose OBO file evolves over five commits, including a rename.

    History (oldest first), following the file to its final path ``src/onto.obo``:

    * c0  ``onto.obo``      MONDO:0000001 {name}
    * c1  ``onto.obo``      + synonym on MONDO:0000001
    * c2  ``src/onto.obo``  pure rename (git mv), content unchanged
    * c3  ``src/onto.obo``  + xref on MONDO:0000001
    * c4  ``src/onto.obo``  + new term MONDO:0000002
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    t1_v0 = _term("MONDO:0000001", "name: disease")
    _write(repo, "onto.obo", HEADER + t1_v0)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c0 create", date="2021-01-01T00:00:00+00:00")

    t1_v1 = _term("MONDO:0000001", "name: disease", 'synonym: "illness" EXACT []')
    _write(repo, "onto.obo", HEADER + t1_v1)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c1 add synonym", date="2021-01-02T00:00:00+00:00")

    (repo / "src").mkdir()
    _git(repo, "mv", "onto.obo", "src/onto.obo")
    _git(repo, "commit", "-qm", "c2 rename (#42)", date="2021-01-03T00:00:00+00:00")

    t1_v3 = _term(
        "MONDO:0000001",
        "name: disease",
        'synonym: "illness" EXACT []',
        "xref: DOID:4",
    )
    _write(repo, "src/onto.obo", HEADER + t1_v3)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c3 add xref", date="2021-01-04T00:00:00+00:00")
    _git(repo, "tag", "v1.0")  # release tag on c3

    t2 = _term("MONDO:0000002", "name: cancer")
    _write(repo, "src/onto.obo", HEADER + t1_v3 + "\n" + t2)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c4 add term", date="2021-01-05T00:00:00+00:00")

    return repo


@pytest.fixture
def bad_then_removed_repo(tmp_path: Path) -> Path:
    """A repo where an unparseable term appears, then is removed the next commit.

    Exercises the state/raw divergence: the bad term lands in ``raw`` (so it is
    not re-tried) but never in ``state`` (it never parsed), so removing it must
    not raise.
    """
    repo = tmp_path / "repo_bad"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    good = _term("MONDO:0000001", "name: good")
    bad = _term("MONDO:0000002", "name: bad", 'synonym: "x" WRONGSCOPE []')
    _write(repo, "onto.obo", HEADER + good + "\n" + bad)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c0 good + bad", date="2021-02-01T00:00:00+00:00")

    _write(repo, "onto.obo", HEADER + good)  # bad term removed
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c1 remove bad", date="2021-02-02T00:00:00+00:00")

    return repo
