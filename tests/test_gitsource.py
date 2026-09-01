"""Tests for the single-file git history walk."""

from pathlib import Path

from obohog.gitsource import GitSource

OBO = "src/onto.obo"


def test_walk_is_ordered_oldest_first(obo_repo: Path):
    with GitSource(obo_repo) as src:
        versions = list(src.iter_file_history(OBO))

    assert [v.commit.seq for v in versions] == [0, 1, 2, 3, 4]
    dates = [v.commit.committed_date for v in versions]
    assert dates == sorted(dates)
    assert versions[0].commit.message == "c0 create"
    assert versions[4].commit.message == "c4 add term"


def test_follow_tracks_rename(obo_repo: Path):
    with GitSource(obo_repo) as src:
        paths = [v.path for v in src.iter_file_history(OBO)]

    # File was created as onto.obo, renamed to src/onto.obo at c2.
    assert paths == ["onto.obo", "onto.obo", "src/onto.obo", "src/onto.obo", "src/onto.obo"]


def test_pure_rename_keeps_same_blob(obo_repo: Path):
    with GitSource(obo_repo) as src:
        versions = list(src.iter_file_history(OBO))
        # c1 -> c2 is a content-free rename, so the blob is byte-identical.
        assert versions[1].blob_oid == versions[2].blob_oid
        assert src.read_blob(versions[1].blob_oid) == src.read_blob(versions[2].blob_oid)


def test_read_blob_matches_committed_content(obo_repo: Path):
    with GitSource(obo_repo) as src:
        versions = list(src.iter_file_history(OBO))
        first = src.read_blob(versions[0].blob_oid).decode()
        last = src.read_blob(versions[-1].blob_oid).decode()

    assert "MONDO:0000001" in first
    assert "MONDO:0000002" not in first  # second term added only at c4
    assert "MONDO:0000002" in last
    assert "xref: DOID:4" in last


def test_first_commit_has_no_parent(obo_repo: Path):
    with GitSource(obo_repo) as src:
        versions = list(src.iter_file_history(OBO))
    assert versions[0].commit.parent_sha is None
    assert versions[1].commit.parent_sha == versions[0].commit.sha


def test_fetch_advances_local_branch(obo_repo: Path, tmp_path: Path) -> None:
    """fetch() must make new upstream commits visible to the history walk.

    Our clones are --no-checkout, so a bare ``git fetch`` only moves
    refs/remotes/origin/* — the walk starts at HEAD and would stay stale.
    """
    from conftest import HEADER, _git, _term, _write

    clone = tmp_path / "clone"
    GitSource.clone(str(obo_repo), clone).close()

    # A sixth commit lands upstream after the clone.
    two_terms = (
        _term("MONDO:0000001", "name: disease", 'synonym: "illness" EXACT []',
              "xref: DOID:4")
        + "\n"
        + _term("MONDO:0000002", "name: cancer", "xref: NCIT:C9305")
    )
    _write(obo_repo, "src/onto.obo", HEADER + two_terms)
    _git(obo_repo, "add", "-A")
    _git(obo_repo, "commit", "-qm", "c5 add xref to term 2",
         date="2021-01-06T00:00:00+00:00")

    with GitSource(clone) as src:
        before = [v.commit.message for v in src.iter_file_history(OBO)]
        src.fetch()
        after = [v.commit.message for v in src.iter_file_history(OBO)]

    assert "c5 add xref to term 2" not in before
    assert after == before + ["c5 add xref to term 2"]


def test_merge_populates_branch_commits(tmp_path: Path) -> None:
    """iter_file_history resolves branch commits via the in-memory DAG walk.

    Build a small repo with a merge and assert the merge commit's
    ``branch_commits`` are the PR-branch commits in newest-first order.
    """
    from conftest import HEADER, _git, _term, _write

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    _write(repo, "onto.obo", HEADER + _term("MONDO:0000001", "name: disease"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c0 create", date="2021-01-01T00:00:00+00:00")

    # branch off main, add two commits touching the file
    _git(repo, "checkout", "-qb", "feature")
    _write(repo, "onto.obo",
           HEADER + _term("MONDO:0000001", "name: disease", 'synonym: "illness" EXACT []'))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "b0 add synonym", date="2021-01-02T00:00:00+00:00")
    _write(repo, "onto.obo",
           HEADER + _term("MONDO:0000001", "name: disease",
                          'synonym: "illness" EXACT []', "xref: DOID:4"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "b1 add xref", date="2021-01-03T00:00:00+00:00")

    # back to main, create a merge commit (touches the file only via the merge)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "--no-ff", "-qm", "m0 Merge pull request #1 from feature",
         "feature", date="2021-01-04T00:00:00+00:00")

    with GitSource(repo) as src:
        versions = list(src.iter_file_history("onto.obo"))

    # --first-parent visits: c0, m0 (feature commits themselves not on mainline)
    assert [v.commit.message.splitlines()[0] for v in versions] == [
        "c0 create",
        "m0 Merge pull request #1 from feature",
    ]
    m0 = versions[-1]
    # PR-branch commits in newest-first order (tip first, matching GitHub).
    assert [bc.message for bc in m0.commit.branch_commits] == ["b1 add xref", "b0 add synonym"]
    # Correct dates carried through.
    assert m0.commit.branch_commits[0].committed_date.date().isoformat() == "2021-01-03"
    assert m0.commit.branch_commits[1].committed_date.date().isoformat() == "2021-01-02"
