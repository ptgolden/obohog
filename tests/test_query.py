"""Query-layer paging and concurrency: the ``after`` cursor and ``fork()``.

Uses a repo wide enough to page over — four terms sharing a searchable
xref prefix across four commits — so both section spines (term for
``order="term"``, commit for ``order="date"``) have several sections.
"""

from pathlib import Path

import pytest

from obohog.query import (
    ClauseSyntaxError,
    HasClause,
    HistoryDB,
    RangeFilters,
    SearchFilters,
    parse_has_clause,
)


@pytest.fixture(scope="module")
def db(paged_artifact: Path):
    db = HistoryDB(paged_artifact)
    yield db
    db.close()


def _section_keys(rows, order: str) -> list:
    """Distinct section keys, in stream order."""
    key = (lambda tc: tc.term_id) if order == "term" else (
        lambda tc: tc.change.commit_seq
    )
    keys: list = []
    for tc in rows:
        if not keys or keys[-1] != key(tc):
            keys.append(key(tc))
    return keys


@pytest.mark.parametrize("order", ["term", "date"])
@pytest.mark.parametrize("reverse", [False, True])
def test_search_after_resumes_exactly_after_each_section(db, order, reverse):
    full = list(db.iter_search_events("SHARED", order=order, reverse=reverse))
    keys = _section_keys(full, order)
    assert len(keys) >= 3  # the fixture must actually give us pages

    section = (lambda tc: tc.term_id) if order == "term" else (
        lambda tc: tc.change.commit_seq
    )
    for i, key in enumerate(keys):
        later = set(keys[i + 1 :])
        expected = [tc for tc in full if section(tc) in later]
        rest = list(
            db.iter_search_events(
                "SHARED", order=order, reverse=reverse, after=key
            )
        )
        assert rest == expected


def test_search_after_pages_concatenate_to_unpaged(db):
    for order, reverse in [
        ("term", False), ("term", True), ("date", False), ("date", True),
    ]:
        full = list(db.iter_search_events("SHARED", order=order, reverse=reverse))
        keys = _section_keys(full, order)
        # Page one section at a time via the cursor and re-concatenate.
        paged = []
        cursor = None
        while True:
            rows = list(
                db.iter_search_events(
                    "SHARED", order=order, reverse=reverse, after=cursor
                )
            )
            if not rows:
                break
            first_key = _section_keys(rows, order)[0]
            section = [
                tc for tc in rows if _section_keys([tc], order)[0] == first_key
            ]
            paged.extend(section)
            cursor = first_key
        assert paged == full
        assert _section_keys(paged, order) == keys


def test_range_after_skips_earlier_terms(db):
    full = list(db.iter_range_events("0", "HEAD"))
    keys = _section_keys(full, "term")
    assert len(keys) >= 2
    rest = list(db.iter_range_events("0", "HEAD", after=keys[0]))
    assert rest == [tc for tc in full if tc.term_id > keys[0]]


def test_search_filters_narrow_with_after(db):
    # Cursor composes with filters: same WHERE, just resumed.
    filters = SearchFilters(tag="xref")
    full = list(db.iter_search_events("SHARED", filters))
    keys = _section_keys(full, "term")
    rest = list(db.iter_search_events("SHARED", filters, after=keys[0]))
    assert rest == [tc for tc in full if tc.term_id > keys[0]]


def test_range_filters_type_errors_on_unknown_field():
    with pytest.raises(TypeError):
        RangeFilters(nonsense="x")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        SearchFilters(predicat="xref")  # type: ignore[call-arg]


def test_fork_shares_views_and_data(db):
    fork = db.fork()
    try:
        assert fork.search_events("SHARED") == db.search_events("SHARED")
        assert fork.resolve_ref("HEAD") == db.resolve_ref("HEAD")
    finally:
        fork.close()


def test_fork_close_leaves_parent_usable(db):
    fork = db.fork()
    fork.close()
    assert db.search_counts("SHARED").events > 0


def test_forks_query_concurrently(db):
    import threading

    results: list = [None] * 4
    errors: list = []

    def work(i: int) -> None:
        fork = db.fork()
        try:
            results[i] = fork.search_events("SHARED")
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)
        finally:
            fork.close()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(r == results[0] for r in results)


def test_facets_lists_distinct_values_alphabetically(db):
    tags, namespaces = db.facets()
    assert tags == ["name", "xref"]
    assert namespaces == ["MONDO"]


def test_resolve_bound_dates(db):
    # paged fixture: four commits, one per day 2021-01-01..04, seqs 0..3
    assert db.resolve_bound("2021-01-02") == 1
    assert db.resolve_bound("2021-01-02", end=True) == 1
    assert db.resolve_bound("2020-06-01") == 0  # before history: first commit
    assert db.resolve_bound("2020-06-01", end=True) == -1  # matches nothing
    assert db.resolve_bound("2022-01-01") == db.resolve_ref("HEAD") + 1
    assert db.resolve_bound("2022-01-01", end=True) == 3
    assert db.resolve_bound("HEAD") == 3  # non-dates still resolve as refs


def test_search_since_until_seq_window(db):
    rows = list(
        db.iter_search_events(
            "SHARED", SearchFilters(since_seq=1, until_seq=2)
        )
    )
    assert {tc.change.commit_seq for tc in rows} == {1, 2}


# ---------------------------------------------------------------------------
# Term-set predicates: ``has`` clause parsing and membership.
# Lifecycle fixture membership cheat sheet (see conftest):
#   ~diabetes   ever {T1, T3, T4, EX}   now {T1, T4, EX}
#   xref~DOID   ever {T2, T3, T4}       now {T2, T3}

T1, T2, T3, T4 = (f"MONDO:000000{i}" for i in (1, 2, 3, 4))
EX = "EX:0000001"


@pytest.fixture(scope="module")
def ldb(lifecycle_artifact: Path):
    db = HistoryDB(lifecycle_artifact)
    yield db
    db.close()


def _filters(*raw_clauses, **fkw) -> SearchFilters:
    return SearchFilters(
        has=tuple(parse_has_clause(r) for r in raw_clauses), **fkw
    )


def _members(db, *raw_clauses, **fkw) -> set[str]:
    f = _filters(*raw_clauses, **fkw)
    return {tc.term_id for tc in db.iter_search_events(None, f)}


@pytest.mark.parametrize("raw, expected", [
    ("name~diabetes", HasClause("ever", "name", "substring", "diabetes")),
    ("~diabetes", HasClause("ever", None, "substring", "diabetes")),
    ("=diabetes mellitus", HasClause("ever", None, "exact", "diabetes mellitus")),
    ("now:xref=XXX:1234567", HasClause("now", "xref", "exact", "XXX:1234567")),
    ("ever:~/dia.*/", HasClause("ever", None, "regex", "dia.*")),
    ("synonym~a=b", HasClause("ever", "synonym", "substring", "a=b")),
    ("def=x~y", HasClause("ever", "def", "exact", "x~y")),
    ("~/", HasClause("ever", None, "substring", "/")),
    ("synonym~", HasClause("ever", "synonym", "substring", "")),
    ("now:name~x", HasClause("now", "name", "substring", "x")),
])
def test_parse_has_clause_forms(raw, expected):
    assert parse_has_clause(raw) == expected


@pytest.mark.parametrize("raw, hint", [
    ("~//", "empty regex"),
    ("=", "needs a value"),
    ("diabetes", "needs '~'"),
    ("during:name~x", "reserved"),
    ("foo:name~x", "unknown quantifier"),
    ("", "empty clause"),
])
def test_parse_has_clause_errors(raw, hint):
    with pytest.raises(ClauseSyntaxError, match=hint):
        parse_has_clause(raw)


def test_ever_vs_now_membership(ldb):
    # T3's matching synonym was removed and never re-added: ever only.
    # T1's was removed then re-added: present now.
    assert _members(ldb, "~diabetes") == {T1, T3, T4, EX}
    assert _members(ldb, "now:~diabetes") == {T1, T4, EX}


def test_dates_clip_events_not_membership(ldb):
    # Membership is timeless: ever {T1, T3, T4, EX}. The date window only
    # clips which of their events show — at c3 that's T1's synonym
    # re-add, T4's xref removal, and EX's creation; T3 sat c3 out.
    rows = list(
        ldb.iter_search_events(None, _filters("~diabetes", since_seq=3))
    )
    assert all(tc.change.commit_seq == 3 for tc in rows)
    assert {tc.term_id for tc in rows} == {T1, T4, EX}
    # T4's c3 event is its xref removal — no 'diabetes' in it: eligible
    # terms' events show whether or not they match any clause.
    assert {tc.change.tag for tc in rows if tc.term_id == T4} == {"xref"}


def test_now_means_head_even_with_dates(ldb):
    # 'now' is a fact about HEAD: membership stays {T1, T4, EX} under an
    # until bound; the bound clips display (only T1 has events <= c1).
    rows = list(
        ldb.iter_search_events(None, _filters("now:~diabetes", until_seq=1))
    )
    assert {tc.term_id for tc in rows} == {T1}
    assert all(tc.change.commit_seq <= 1 for tc in rows)


def test_clauses_intersect(ldb):
    assert _members(ldb, "~diabetes", "xref~DOID") == {T3, T4}
    assert _members(ldb, "~diabetes", "now:xref~DOID") == {T3}


def test_membership_narrowings(ldb):
    assert _members(ldb, "~diabetes", namespace="MONDO") == {T1, T3, T4}
    assert _members(ldb, "~diabetes", term_id=T3) == {T3}


def test_exact_regex_and_ignore_case_clauses(ldb):
    # Exact matches the body: only EX's name is exactly "diabetes".
    assert _members(ldb, "name=diabetes") == {EX}
    assert _members(ldb, "~/dia.*mell.*/") == {T1}
    assert _members(ldb, "~DIABETES") == set()
    assert _members(ldb, "~DIABETES", ignore_case=True) == {T1, T3, T4, EX}


def test_full_timelines_stream_nonmatching_events(ldb):
    rows = list(ldb.iter_search_events(None, _filters("~diabetes")))
    # T3 qualifies via its synonym; its xref events ride along anyway.
    assert any(tc.term_id == T3 and tc.change.tag == "xref" for tc in rows)
    # T1's name never matched the clause but its add event is present.
    assert any(tc.term_id == T1 and tc.change.tag == "name" for tc in rows)


def test_query_composes_with_has(ldb):
    # "changes to xrefs containing DOID among terms that have ever had
    # 'diabetes'": T2's DOID:9 is excluded (not a member); EX has no
    # xrefs; T3's add and T4's add + remove show.
    f = _filters("~diabetes", tag="xref")
    rows = list(ldb.iter_search_events("DOID", f))
    assert {tc.term_id for tc in rows} == {T3, T4}
    assert all(tc.change.tag == "xref" for tc in rows)
    assert {
        (tc.term_id, tc.change.operation) for tc in rows
    } == {(T3, "add"), (T4, "add"), (T4, "remove")}


@pytest.mark.parametrize("reverse", [False, True])
def test_has_after_resumes_and_concatenates(ldb, reverse):
    f = _filters("~diabetes")
    full = list(ldb.iter_search_events(None, f, reverse=reverse))
    keys = _section_keys(full, "term")
    assert len(keys) >= 3
    for i, key in enumerate(keys):
        later = set(keys[i + 1 :])
        rest = list(
            ldb.iter_search_events(None, f, reverse=reverse, after=key)
        )
        assert rest == [tc for tc in full if tc.term_id in later]


def test_has_counts_are_exact(ldb):
    f = _filters("~diabetes")
    counts = ldb.search_counts(None, f)
    rows = list(ldb.iter_search_events(None, f))
    assert counts.terms == 4
    assert counts.events == len(rows)
    assert counts.commits == len({tc.change.commit_seq for tc in rows})


# ---------------------------------------------------------------------------
# Whole-group streaming: a text match pulls in its (term, commit) group.


def test_search_streams_whole_groups_for_matching_commits(requalified_artifact):
    db = HistoryDB(requalified_artifact)
    try:
        rows = list(db.iter_search_events("NCIT"))
        # "NCIT" textually matches two events (c0's xref add, c1's xref
        # remove) — but both groups arrive whole: c0's name+xref adds,
        # c1's remove plus both adds (the requalified partner and the
        # unrelated MEDGEN xref). The op filter, not the SQL, decides
        # what shows.
        assert len(rows) == 5
        last_seq = max(tc.change.commit_seq for tc in rows)
        c1 = [tc.change for tc in rows if tc.change.commit_seq == last_seq]
        assert sorted(c.operation for c in c1) == ["add", "add", "remove"]
    finally:
        db.close()
