"""End-to-end: build an artifact from the fixture repo and query it."""

from pathlib import Path

import duckdb
import pytest

from obohog.extract import build_parallel, extract
from obohog.gitsource import GitSource
from obohog.query import ArtifactNotFound, HistoryDB

OBO = "src/onto.obo"


@pytest.fixture
def artifact(obo_repo: Path, tmp_path: Path) -> Path:
    out = tmp_path / "artifact"
    with GitSource(obo_repo) as src:
        extract(src, OBO, out)
    return out


def _multiset(db: HistoryDB, table: str, cols: str):
    return sorted(
        db.con.execute(f"SELECT {cols} FROM {table}").fetchall()
    )


def test_parallel_build_matches_single(obo_repo: Path, tmp_path: Path):
    # Same history built single-threaded vs with multiple worker processes must
    # produce identical events and snapshots (chunk seeding + no stale files).
    single = tmp_path / "single"
    parallel = tmp_path / "parallel"
    with GitSource(obo_repo) as src:
        extract(src, OBO, single)
    build_parallel(str(obo_repo), OBO, parallel, jobs=3)

    ds, dp = HistoryDB(single), HistoryDB(parallel)
    ev_cols = "term_id, commit_seq, operation, predicate, value"
    sn_cols = "term_id, commit_seq, content_hash"
    assert _multiset(ds, "events", ev_cols) == _multiset(dp, "events", ev_cols)
    assert _multiset(ds, "term_snapshots", sn_cols) == _multiset(dp, "term_snapshots", sn_cols)
    ds.close()
    dp.close()


def test_chunk_size_does_not_change_output(obo_repo: Path, tmp_path: Path):
    # Many tiny chunks (seed at every boundary) must match the single-threaded build.
    single = tmp_path / "single"
    many = tmp_path / "many"
    with GitSource(obo_repo) as src:
        extract(src, OBO, single)
    build_parallel(str(obo_repo), OBO, many, jobs=2, chunk_size=1)

    ds, dm = HistoryDB(single), HistoryDB(many)
    cols = "term_id, commit_seq, operation, predicate, value"
    assert _multiset(ds, "events", cols) == _multiset(dm, "events", cols)
    ds.close()
    dm.close()


def test_parallel_rebuild_clears_stale_partfiles(obo_repo: Path, tmp_path: Path):
    # Re-running into the same dir must not accumulate/duplicate rows.
    out = tmp_path / "art"
    build_parallel(str(obo_repo), OBO, out, jobs=2)
    first = HistoryDB(out)
    n_events = first.con.execute("SELECT count(*) FROM events").fetchone()[0]
    first.close()

    build_parallel(str(obo_repo), OBO, out, jobs=3)  # different chunking
    again = HistoryDB(out)
    assert again.con.execute("SELECT count(*) FROM events").fetchone()[0] == n_events
    again.close()


def test_removing_an_unparseable_term_does_not_crash(bad_then_removed_repo: Path, tmp_path: Path):
    out = tmp_path / "art"
    build_parallel(str(bad_then_removed_repo), "onto.obo", out, jobs=1)  # must not raise

    db = HistoryDB(out)
    good = db.con.execute(
        "SELECT count(*) FROM term_snapshots WHERE term_id = 'MONDO:0000001'"
    ).fetchone()[0]
    skipped_ids = {r[0] for r in db.con.execute("SELECT DISTINCT term_id FROM skipped").fetchall()}
    db.close()

    assert good >= 1  # the good term is indexed
    assert "MONDO:0000002" in skipped_ids  # the bad term is recorded, not fatal


def test_missing_artifact_raises_clear_error(tmp_path: Path):
    with pytest.raises(ArtifactNotFound, match="Run `obohog source sync"):
        HistoryDB(tmp_path / "does-not-exist")


def test_term_events_are_clause_deltas(artifact: Path):
    db = HistoryDB(artifact)
    kinds = [(c.operation, c.predicate) for c in db.term_timeline("MONDO:0000001")]
    db.close()

    # name added at c0 (birth of the term — ∅ → full clause set is a valid
    # diff), synonym added at c1, xref added at c3.
    assert ("add", "name") in kinds
    assert ("add", "synonym") in kinds
    assert ("add", "xref") in kinds


def test_pure_rename_emits_no_events(artifact: Path):
    # c2 (commit_seq 2) is a content-free rename.
    n = duckdb.connect().execute(
        f"SELECT count(*) FROM read_parquet('{artifact}/events.parquet') WHERE commit_seq = 2"
    ).fetchone()[0]
    assert n == 0


def test_reconstruct_state_at_commit(artifact: Path):
    db = HistoryDB(artifact)
    clauses = dict(db.term_at("MONDO:0000001", 4))
    db.close()

    assert clauses["name"] == "disease"
    assert clauses["synonym"] == '"illness" EXACT []'
    assert clauses["xref"] == "DOID:4"


def test_new_term_appears_as_creation(artifact: Path):
    db = HistoryDB(artifact)
    # MONDO:0000002 is created at c4 with just a name.
    changes = [(c.operation, c.predicate) for c in db.term_timeline("MONDO:0000002")]
    before = db.term_at("MONDO:0000002", 0)
    after = dict(db.term_at("MONDO:0000002", 4))
    db.close()

    assert changes == [("add", "name")]
    assert before == []  # did not exist at the baseline
    assert after["name"] == "cancer"


def test_pr_number_parsed_from_message(artifact: Path):
    pr = duckdb.connect().execute(
        f"SELECT pr_number FROM read_parquet('{artifact}/commits.parquet') "
        "WHERE message LIKE 'c2%'"
    ).fetchone()[0]
    assert pr == 42


def test_pr_number_handles_both_github_conventions():
    from obohog.extract import _extract_pr_number

    # Squash-and-merge (post-2023 Mondo): title ends with "(#N)".
    assert _extract_pr_number("add venom terms (#10409)") == 10409
    # Classic merge commit (pre-2023 Mondo, PATO): "Merge pull request #N …"
    # The merge pattern must anchor to start-of-message so a PR body that
    # mentions "Merge pull request #x" in prose doesn't false-positive.
    assert _extract_pr_number(
        "Merge pull request #5013 from monarch-initiative/issue-4938"
    ) == 5013
    # No PR referenced.
    assert _extract_pr_number("misc fixes") is None
    # Body that quotes another PR in parens shouldn't win over the merge header.
    assert _extract_pr_number(
        "Merge pull request #123 from user/branch\n\nRelates to (#456)"
    ) == 123


def test_snapshot_url_extracted_from_release_trailer():
    from obohog.extract import _extract_snapshot_url

    # GitHubReleaseProvider stashes the release page URL on its own trailer line.
    msg = (
        "v2024.03.01\n\n"
        "Release notes body here.\n\n"
        "Release URL: https://github.com/obophenotype/zp/releases/tag/v2024.03.01"
    )
    assert (
        _extract_snapshot_url(msg)
        == "https://github.com/obophenotype/zp/releases/tag/v2024.03.01"
    )
    # A message without the trailer → None (git-file sources).
    assert _extract_snapshot_url("add venom terms (#10409)") is None
    # The trailer must be at the start of a line — a URL embedded inside prose
    # doesn't count.
    assert _extract_snapshot_url("See Release URL: https://x/y for context") is None


def test_snapshot_url_suppresses_pr_number():
    """Release-based commits shouldn't attribute PR numbers to release bodies.

    ``_commit_row`` in extract.py checks snapshot_url before pr_number, so
    a `(#123)` in release notes doesn't leak into `pr_number`. Verified via
    the row builder directly since the fixture obo_repo is git-file-only.
    """
    from datetime import datetime, timezone
    from obohog.extract import _commit_row
    from obohog.gitsource import CommitInfo

    commit = CommitInfo(
        seq=5,
        sha="a" * 40,
        author_name="bot",
        author_email="bot@example.com",
        committed_date=datetime(2024, 3, 1, tzinfo=timezone.utc),
        message=(
            "v2024.03.01\n\n"
            "Merged (#123) and (#456) into this release.\n\n"
            "Release URL: https://github.com/x/y/releases/tag/v2024.03.01"
        ),
        parent_sha="b" * 40,
    )
    row = _commit_row(commit)
    assert row["snapshot_url"] == "https://github.com/x/y/releases/tag/v2024.03.01"
    assert row["pr_number"] is None  # suppressed by snapshot_url presence


def test_releases_map_tag_to_commit(artifact: Path):
    db = HistoryDB(artifact)
    rels = db.releases()
    db.close()
    # v1.0 was tagged on c3 (commit_seq 3).
    assert [(tag, seq) for tag, seq, _date in rels] == [("v1.0", 3)]


def test_diff_between_release_and_head(artifact: Path):
    db = HistoryDB(artifact)
    # Between v1.0 (seq 3) and HEAD (seq 4): only the new term created at c4.
    rows = db.range_events("v1.0", "4")
    db.close()
    assert [
        (r.term_id, r.change.operation, r.change.predicate) for r in rows
    ] == [("MONDO:0000002", "add", "name")]


def test_diff_resolves_sha_and_seq_symmetrically(artifact: Path):
    db = HistoryDB(artifact)
    a = db.range_events("3", "4")
    b = db.range_events("4", "3")  # order shouldn't matter
    db.close()
    assert a == b


def test_diff_accepts_head(artifact: Path):
    db = HistoryDB(artifact)
    by_head = db.range_events("v1.0", "HEAD")
    by_seq = db.range_events("v1.0", "4")
    db.close()
    assert by_head == by_seq


def test_pr_terms_from_message(artifact: Path):
    db = HistoryDB(artifact)
    # c2 "c2 rename (#42)" is a pure rename → no term events → PR touches nothing.
    assert db.pr_terms(42) == []
    db.close()


def test_commit_events_lists_co_changed(artifact: Path):
    db = HistoryDB(artifact)
    # find c4's sha, then ask what changed in it.
    sha = duckdb.connect().execute(
        f"SELECT sha FROM read_parquet('{artifact}/commits.parquet') WHERE commit_seq = 4"
    ).fetchone()[0]
    head, events = db.commit_events(sha)
    db.close()

    assert head is not None and head.sha == sha
    terms = {tc.term_id for tc in events}
    assert "MONDO:0000002" in terms


def test_search_events_finds_substring(artifact: Path):
    # The "illness" synonym was added on c1 for MONDO:0000001.
    db = HistoryDB(artifact)
    events = db.search_events("illness")
    db.close()
    assert len(events) == 1
    assert events[0].term_id == "MONDO:0000001"
    assert events[0].change.operation == "add"
    assert events[0].change.predicate == "synonym"
    assert "illness" in events[0].change.value


def test_search_events_empty_when_no_match(artifact: Path):
    db = HistoryDB(artifact)
    assert db.search_events("nonexistent-string-that-cannot-occur") == []
    db.close()


def test_search_events_predicate_filter_narrows(artifact: Path):
    # DOID:4 was added as an xref on c3. Filtering by predicate=xref keeps it;
    # filtering by predicate=synonym drops it even though it's the same needle.
    db = HistoryDB(artifact)
    xrefs = db.search_events("DOID:4", predicate="xref")
    synonyms = db.search_events("DOID:4", predicate="synonym")
    db.close()
    assert len(xrefs) == 1 and xrefs[0].change.predicate == "xref"
    assert synonyms == []


def test_search_events_term_filter_narrows(artifact: Path):
    # MONDO:0000002 is created at c4 with `name: cancer` — that lands in the
    # events table because c4 is diffed against c3 (which had no such term).
    # Restricting to MONDO:0000002 keeps the hit; restricting to a term
    # without the string drops it.
    db = HistoryDB(artifact)
    hits = db.search_events("cancer", term_id="MONDO:0000002")
    off_term = db.search_events("cancer", term_id="MONDO:0000001")
    db.close()
    assert len(hits) == 1
    assert hits[0].term_id == "MONDO:0000002"
    assert hits[0].change.predicate == "name"
    assert off_term == []


def test_search_events_since_filter_cuts_off_early_commits(artifact: Path):
    # The "illness" synonym was added on c1 (commit_seq 1). A --since cutoff
    # of seq 2 must exclude it.
    db = HistoryDB(artifact)
    all_hits = db.search_events("illness")
    after_c1 = db.search_events("illness", since_seq=2)
    db.close()
    assert len(all_hits) == 1
    assert after_c1 == []


def test_search_events_ignore_case_substring(artifact: Path):
    # "ILLNESS" (all caps) matches the "illness" synonym under --ignore-case;
    # without the flag, it doesn't.
    db = HistoryDB(artifact)
    sensitive = db.search_events("ILLNESS")
    insensitive = db.search_events("ILLNESS", ignore_case=True)
    db.close()
    assert sensitive == []
    assert len(insensitive) == 1
    assert insensitive[0].change.predicate == "synonym"


def test_search_events_regex_matches(artifact: Path):
    # The DOID:4 xref matches ^DOID:\d+$; the OMIM-style patterns don't.
    db = HistoryDB(artifact)
    doid_hits = db.search_events(r"^DOID:\d+$", regex=True)
    omim_hits = db.search_events(r"^OMIM:\d+$", regex=True)
    db.close()
    assert len(doid_hits) == 1
    assert doid_hits[0].change.predicate == "xref"
    assert doid_hits[0].change.value == "DOID:4"
    assert omim_hits == []


def test_search_events_regex_ignore_case_combined(artifact: Path):
    # Regex + --ignore-case: DOID uppercase pattern still matches even if the
    # regex uses lowercase. Both flags combine via the 'i' option to
    # regexp_matches.
    db = HistoryDB(artifact)
    sensitive = db.search_events(r"^doid:\d+$", regex=True)
    insensitive = db.search_events(r"^doid:\d+$", regex=True, ignore_case=True)
    db.close()
    assert sensitive == []
    assert len(insensitive) == 1
    assert insensitive[0].change.value == "DOID:4"


def test_search_events_namespace_filter_keeps_matching_prefix(artifact: Path):
    # The fixture has only MONDO: term_ids, so namespace="MONDO" is a no-op
    # from a "which rows" perspective — but the SQL wire-up must be right.
    db = HistoryDB(artifact)
    unfiltered = db.search_events("cancer")
    with_ns = db.search_events("cancer", namespace="MONDO")
    db.close()
    assert with_ns == unfiltered
    assert len(with_ns) >= 1


def test_search_events_namespace_filter_excludes_other_prefixes(artifact: Path):
    db = HistoryDB(artifact)
    hits = db.search_events("cancer", namespace="FOO")
    db.close()
    assert hits == []


def test_range_events_namespace_filter(artifact: Path):
    db = HistoryDB(artifact)
    unfiltered = db.range_events("v1.0", "HEAD")
    with_ns = db.range_events("v1.0", "HEAD", namespace="MONDO")
    empty = db.range_events("v1.0", "HEAD", namespace="FOO")
    db.close()
    assert with_ns == unfiltered
    assert empty == []


def test_commit_events_namespace_filter(artifact: Path):
    db = HistoryDB(artifact)
    sha = duckdb.connect().execute(
        f"SELECT sha FROM read_parquet('{artifact}/commits.parquet') WHERE commit_seq = 4"
    ).fetchone()[0]
    head_a, events_a = db.commit_events(sha)
    head_b, events_b = db.commit_events(sha, namespace="MONDO")
    _, events_empty = db.commit_events(sha, namespace="FOO")
    db.close()
    assert head_a is not None and head_b is not None
    assert events_a == events_b
    assert events_empty == []
