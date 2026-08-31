"""End-to-end: build an artifact from the fixture repo and query it."""

from dataclasses import asdict, replace
from pathlib import Path

import duckdb
import pytest

from obohog import model
from obohog.extract import BuildMode, build_parallel, extract
from obohog.gitsource import GitSource
from obohog.query import (
    ArtifactNotFound,
    HistoryDB,
    RangeFilters,
    RefNotFound,
    SchemaMismatch,
    SearchFilters,
)

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


def test_build_matches_naive_full_parse_oracle(obo_repo: Path, tmp_path: Path):
    """The diff-scoped parse must match an obviously-correct reference.

    The shipped core skips stanzas whose raw bytes didn't change and parses
    changed stanzas in isolation against the header context. This oracle
    fully re-parses every version and diffs whole document states — if the
    stanza splitter mis-carved a boundary or the byte-hash shortcut ever
    diverged from a real parse, the two would disagree.
    """
    from obohog.obo import clause_delta, parse_terms

    out = tmp_path / "art"
    build_parallel(str(obo_repo), OBO, out, jobs=2)

    events: list[tuple] = []
    snapshots: list[tuple] = []
    prev: dict = {}
    with GitSource(obo_repo) as src:
        for v in src.iter_file_history(OBO):
            current = parse_terms(src.read_blob(v.blob_oid))
            for term_id, term in current.items():
                before = prev.get(term_id)
                if before is not None and before.content_hash == term.content_hash:
                    continue
                snapshots.append((term_id, v.commit.seq, term.content_hash))
                added, removed = clause_delta(
                    before.clauses if before else (), term.clauses
                )
                events += [(term_id, v.commit.seq, "add", c.tag, c.value)
                           for c in added]
                events += [(term_id, v.commit.seq, "remove", c.tag, c.value)
                           for c in removed]
            for term_id in prev.keys() - current.keys():
                events += [(term_id, v.commit.seq, "remove", c.tag, c.value)
                           for c in prev[term_id].clauses]
            prev = current

    db = HistoryDB(out)
    ev_cols = "term_id, commit_seq, operation, predicate, value"
    assert sorted(events) == _multiset(db, "events", ev_cols)
    assert sorted(snapshots) == _multiset(
        db, "term_snapshots", "term_id, commit_seq, content_hash"
    )
    db.close()


def _extend_repo(repo: Path) -> None:
    """Two more upstream commits + a tag, landing after an initial build.

    c5 removes MONDO:0000002 and adds MONDO:0000003 — a term removal across
    the incremental boundary is exactly what the seed state must get right.
    """
    from conftest import HEADER, _git, _term, _write

    t1 = _term(
        "MONDO:0000001", "name: disease", 'synonym: "illness" EXACT []', "xref: DOID:4"
    )
    t3 = _term("MONDO:0000003", "name: syndrome")
    _write(repo, "src/onto.obo", HEADER + t1 + "\n" + t3)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c5 swap terms", date="2021-01-06T00:00:00+00:00")

    t3b = _term("MONDO:0000003", "name: syndrome", "xref: NCIT:C123")
    _write(repo, "src/onto.obo", HEADER + t1 + "\n" + t3b)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c6 add xref", date="2021-01-07T00:00:00+00:00")
    _git(repo, "tag", "v2.0")


def _assert_same_artifact(a: Path, b: Path) -> None:
    da, db = HistoryDB(a), HistoryDB(b)
    checks = [
        ("events", "term_id, commit_seq, operation, predicate, value, body, comment"),
        ("term_snapshots", "term_id, commit_seq, content_hash"),
        ("commits", "commit_seq, sha, pr_number, message"),
        ("releases", "tag, sha, commit_seq"),
    ]
    for table, cols in checks:
        assert _multiset(da, table, cols) == _multiset(db, table, cols), table
    da.close()
    db.close()
    assert model.read_build_meta(a) == model.read_build_meta(b)


def test_incremental_update_matches_full_rebuild(obo_repo: Path, tmp_path: Path):
    inc = tmp_path / "inc"
    build_parallel(str(obo_repo), OBO, inc, jobs=2)
    _extend_repo(obo_repo)
    report = build_parallel(str(obo_repo), OBO, inc, jobs=2, update=True)
    assert report.mode is BuildMode.INCREMENTAL
    assert report.commits == 2
    assert report.total_commits == 7

    fresh = tmp_path / "fresh"
    build_parallel(str(obo_repo), OBO, fresh, jobs=2)
    _assert_same_artifact(inc, fresh)


def test_incremental_after_serial_build(obo_repo: Path, tmp_path: Path):
    # A serial artifact stores single files, not part-file dirs; appending
    # must migrate them so the union stays complete.
    inc = tmp_path / "inc"
    with GitSource(obo_repo) as src:
        extract(src, OBO, inc)
    _extend_repo(obo_repo)
    report = build_parallel(str(obo_repo), OBO, inc, jobs=2, update=True)
    assert report.mode is BuildMode.INCREMENTAL

    fresh = tmp_path / "fresh"
    build_parallel(str(obo_repo), OBO, fresh, jobs=2)
    _assert_same_artifact(inc, fresh)


def test_incremental_up_to_date_still_refreshes_releases(obo_repo: Path, tmp_path: Path):
    from conftest import _git

    out = tmp_path / "a"
    build_parallel(str(obo_repo), OBO, out, jobs=2)
    # A release tagged after the last file-touching commit: no new file
    # versions, but the releases table must gain the tag.
    _git(obo_repo, "tag", "v1.1")
    report = build_parallel(str(obo_repo), OBO, out, jobs=2, update=True)
    assert report.mode is BuildMode.UP_TO_DATE
    assert report.commits == 0
    db = HistoryDB(out)
    tags = {row[0] for row in db.con.execute("SELECT tag FROM releases").fetchall()}
    db.close()
    assert tags == {"v1.0", "v1.1"}


def test_incremental_falls_back_on_history_rewrite(obo_repo: Path, tmp_path: Path):
    from conftest import _git

    out = tmp_path / "a"
    build_parallel(str(obo_repo), OBO, out, jobs=2)
    _git(obo_repo, "commit", "--amend", "-qm", "c4 amended",
         date="2021-01-05T00:00:00+00:00")
    report = build_parallel(str(obo_repo), OBO, out, jobs=2, update=True)
    assert report.mode is BuildMode.FULL
    db = HistoryDB(out)
    last = db.con.execute(
        "SELECT message FROM commits ORDER BY commit_seq DESC LIMIT 1"
    ).fetchone()[0]
    db.close()
    assert last == "c4 amended"


def test_incremental_falls_back_on_schema_mismatch(obo_repo: Path, tmp_path: Path):
    out = tmp_path / "a"
    build_parallel(str(obo_repo), OBO, out, jobs=2)
    meta = replace(model.read_build_meta(out), schema_version="0")
    model.write_table([asdict(meta)], model.BUILD_META, out, "build_meta")
    report = build_parallel(str(obo_repo), OBO, out, jobs=2, update=True)
    assert report.mode is BuildMode.FULL
    assert model.read_build_meta(out).schema_version == model.SCHEMA_VERSION


def test_incremental_cleans_aborted_parts(obo_repo: Path, tmp_path: Path):
    out = tmp_path / "a"
    build_parallel(str(obo_repo), OBO, out, jobs=2)
    # A part-file from an increment that died before its build_meta write:
    # its starting seq (5) is beyond the recorded last_commit_seq (4).
    stray = {
        "term_id": "MONDO:9999999", "commit_seq": 99, "sha": "dead",
        "predicate": "name", "value": "ghost", "operation": "add",
        "body": "ghost", "qualifiers": [], "comment": None,
    }
    model.write_part([stray], model.EVENTS, out / "events" / "inc-0000005-000-0000.parquet")
    report = build_parallel(str(obo_repo), OBO, out, jobs=2, update=True)
    assert report.mode is BuildMode.UP_TO_DATE
    db = HistoryDB(out)
    n = db.con.execute(
        "SELECT count(*) FROM events WHERE term_id = 'MONDO:9999999'"
    ).fetchone()[0]
    db.close()
    assert n == 0


def test_missing_artifact_raises_clear_error(tmp_path: Path):
    with pytest.raises(ArtifactNotFound, match="Run `obohog source sync"):
        HistoryDB(tmp_path / "does-not-exist")


def test_stale_schema_raises_clear_error(artifact: Path):
    meta = replace(model.read_build_meta(artifact), schema_version="0")
    model.write_table([asdict(meta)], model.BUILD_META, artifact, "build_meta")
    with pytest.raises(SchemaMismatch, match="built with schema 0"):
        HistoryDB(artifact)


def test_absent_build_meta_raises_schema_mismatch(artifact: Path):
    (artifact / "build_meta.parquet").unlink()
    with pytest.raises(SchemaMismatch, match="built with schema unknown"):
        HistoryDB(artifact)


def test_term_events_are_clause_deltas(artifact: Path):
    db = HistoryDB(artifact)
    kinds = [(c.operation, c.tag) for c in db.term_timeline("MONDO:0000001")]
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
    changes = [(c.operation, c.tag) for c in db.term_timeline("MONDO:0000002")]
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
        (r.term_id, r.change.operation, r.change.tag) for r in rows
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


def test_unresolvable_ref_raises_typed_error(artifact: Path):
    db = HistoryDB(artifact)
    with pytest.raises(RefNotFound, match="no-such-ref"):
        db.resolve_ref("no-such-ref")
    db.close()


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
    assert events[0].change.tag == "synonym"
    assert "illness" in events[0].change.value


def test_search_events_empty_when_no_match(artifact: Path):
    db = HistoryDB(artifact)
    assert db.search_events("nonexistent-string-that-cannot-occur") == []
    db.close()


def test_search_events_tag_filter_narrows(artifact: Path):
    # DOID:4 was added as an xref on c3. Filtering by tag=xref keeps it;
    # filtering by tag=synonym drops it even though it's the same needle.
    db = HistoryDB(artifact)
    xrefs = db.search_events("DOID:4", SearchFilters(tag="xref"))
    synonyms = db.search_events("DOID:4", SearchFilters(tag="synonym"))
    db.close()
    assert len(xrefs) == 1 and xrefs[0].change.tag == "xref"
    assert synonyms == []


def test_search_events_term_filter_narrows(artifact: Path):
    # MONDO:0000002 is created at c4 with `name: cancer` — that lands in the
    # events table because c4 is diffed against c3 (which had no such term).
    # Restricting to MONDO:0000002 keeps the hit; restricting to a term
    # without the string drops it.
    db = HistoryDB(artifact)
    hits = db.search_events("cancer", SearchFilters(term_id="MONDO:0000002"))
    off_term = db.search_events("cancer", SearchFilters(term_id="MONDO:0000001"))
    db.close()
    assert len(hits) == 1
    assert hits[0].term_id == "MONDO:0000002"
    assert hits[0].change.tag == "name"
    assert off_term == []


def test_search_events_since_filter_cuts_off_early_commits(artifact: Path):
    # The "illness" synonym was added on c1 (commit_seq 1). A --since cutoff
    # of seq 2 must exclude it.
    db = HistoryDB(artifact)
    all_hits = db.search_events("illness")
    after_c1 = db.search_events("illness", SearchFilters(since_seq=2))
    db.close()
    assert len(all_hits) == 1
    assert after_c1 == []


def test_search_events_ignore_case_substring(artifact: Path):
    # "ILLNESS" (all caps) matches the "illness" synonym under --ignore-case;
    # without the flag, it doesn't.
    db = HistoryDB(artifact)
    sensitive = db.search_events("ILLNESS")
    insensitive = db.search_events("ILLNESS", SearchFilters(ignore_case=True))
    db.close()
    assert sensitive == []
    assert len(insensitive) == 1
    assert insensitive[0].change.tag == "synonym"


def test_search_events_regex_matches(artifact: Path):
    # The DOID:4 xref matches ^DOID:\d+$; the OMIM-style patterns don't.
    db = HistoryDB(artifact)
    doid_hits = db.search_events(r"^DOID:\d+$", SearchFilters(regex=True))
    omim_hits = db.search_events(r"^OMIM:\d+$", SearchFilters(regex=True))
    db.close()
    assert len(doid_hits) == 1
    assert doid_hits[0].change.tag == "xref"
    assert doid_hits[0].change.value == "DOID:4"
    assert omim_hits == []


def test_search_events_regex_ignore_case_combined(artifact: Path):
    # Regex + --ignore-case: DOID uppercase pattern still matches even if the
    # regex uses lowercase. Both flags combine via the 'i' option to
    # regexp_matches.
    db = HistoryDB(artifact)
    sensitive = db.search_events(r"^doid:\d+$", SearchFilters(regex=True))
    insensitive = db.search_events(
        r"^doid:\d+$", SearchFilters(regex=True, ignore_case=True)
    )
    db.close()
    assert sensitive == []
    assert len(insensitive) == 1
    assert insensitive[0].change.value == "DOID:4"


def test_search_events_namespace_filter_keeps_matching_prefix(artifact: Path):
    # The fixture has only MONDO: term_ids, so namespace="MONDO" is a no-op
    # from a "which rows" perspective — but the SQL wire-up must be right.
    db = HistoryDB(artifact)
    unfiltered = db.search_events("cancer")
    with_ns = db.search_events("cancer", SearchFilters(namespace="MONDO"))
    db.close()
    assert with_ns == unfiltered
    assert len(with_ns) >= 1


def test_search_events_namespace_filter_excludes_other_prefixes(artifact: Path):
    db = HistoryDB(artifact)
    hits = db.search_events("cancer", SearchFilters(namespace="FOO"))
    db.close()
    assert hits == []


def test_range_events_namespace_filter(artifact: Path):
    db = HistoryDB(artifact)
    unfiltered = db.range_events("v1.0", "HEAD")
    with_ns = db.range_events("v1.0", "HEAD", RangeFilters(namespace="MONDO"))
    empty = db.range_events("v1.0", "HEAD", RangeFilters(namespace="FOO"))
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


def test_events_carry_recomposable_decomposition(artifact: Path):
    # Every events row stores body/qualifiers/comment alongside value, and
    # the decomposition recomposes to value exactly (the invariant that
    # makes the parsed columns trustworthy without re-parsing).
    rows = duckdb.connect().execute(
        f"SELECT value, body, qualifiers, comment "
        f"FROM read_parquet('{artifact}/events.parquet')"
    ).fetchall()
    assert rows
    for value, body, qualifiers, comment in rows:
        assert body is not None
        recomposed = body
        if qualifiers:
            recomposed += " {" + ", ".join(qualifiers) + "}"
        if comment is not None:
            recomposed += " ! " + comment
        assert recomposed == value


def test_iter_search_events_matches_materialized(artifact: Path):
    db = HistoryDB(artifact)
    assert list(db.iter_search_events("illness")) == db.search_events("illness")


def test_search_counts_match_candidate_rows(artifact: Path):
    db = HistoryDB(artifact)
    hits = db.search_events("illness")
    counts = db.search_counts("illness")
    assert counts.events == len(hits)
    assert counts.terms == len({tc.term_id for tc in hits})
    assert counts.commits == len({tc.change.commit_seq for tc in hits})


def test_range_counts_match_rows(artifact: Path):
    db = HistoryDB(artifact)
    hits = db.range_events("v1.0", "HEAD")
    counts = db.range_counts("v1.0", "HEAD")
    assert counts.events == len(hits)
    assert counts.terms == len({tc.term_id for tc in hits})
    assert counts.commits == len({tc.change.commit_seq for tc in hits})


def test_iter_search_events_date_order(artifact: Path):
    db = HistoryDB(artifact)
    hits = list(db.iter_search_events("illness", order="date"))
    seqs = [tc.change.commit_seq for tc in hits]
    assert seqs == sorted(seqs, reverse=True)
    # Same rows as term order, differently arranged.
    assert sorted(map(repr, hits)) == sorted(map(repr, db.search_events("illness")))


def test_iter_search_events_reverse_orders(artifact: Path):
    db = HistoryDB(artifact)
    date_rev = [tc.change.commit_seq for tc in
                db.iter_search_events("illness", order="date", reverse=True)]
    assert date_rev == sorted(date_rev)
    term_rev = list(db.iter_search_events("illness", order="term", reverse=True))
    # A->Z term sections, newest-first within each.
    for _, grp in __import__("itertools").groupby(term_rev, key=lambda tc: tc.term_id):
        seqs = [tc.change.commit_seq for tc in grp]
        assert seqs == sorted(seqs, reverse=True)
