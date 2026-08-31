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


def _members(db, *raw_clauses, f=SearchFilters(), **kw) -> set[str]:
    clauses = [parse_has_clause(r) for r in raw_clauses]
    return {tc.term_id for tc in db.iter_term_set_events(clauses, f, **kw)}


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


def test_now_until_shifts_the_asof_point(ldb):
    # As of c1: T1's synonym is removed (though present at HEAD), T3's is
    # live (though gone at HEAD); T4/EX don't exist yet.
    f = SearchFilters(until_seq=1)
    assert _members(ldb, "now:~diabetes", f=f) == {T3}


def test_since_is_inert_for_now_clauses(ldb):
    f = SearchFilters(since_seq=3)
    assert _members(ldb, "now:~diabetes", f=f) == {T1, T4, EX}


def test_windowed_ever(ldb):
    # Only c3's matching events count: T1's re-add and EX's name.
    f = SearchFilters(since_seq=3)
    assert _members(ldb, "~diabetes", f=f) == {T1, EX}


def test_clauses_intersect(ldb):
    assert _members(ldb, "~diabetes", "xref~DOID") == {T3, T4}
    assert _members(ldb, "~diabetes", "now:xref~DOID") == {T3}


def test_membership_narrowings(ldb):
    assert _members(ldb, "~diabetes", f=SearchFilters(namespace="MONDO")) == {
        T1, T3, T4,
    }
    assert _members(ldb, "~diabetes", f=SearchFilters(term_id=T3)) == {T3}


def test_exact_regex_and_ignore_case_clauses(ldb):
    # Exact matches the body: only EX's name is exactly "diabetes".
    assert _members(ldb, "name=diabetes") == {EX}
    assert _members(ldb, "~/dia.*mell.*/") == {T1}
    assert _members(ldb, "~DIABETES") == set()
    f = SearchFilters(ignore_case=True)
    assert _members(ldb, "~DIABETES", f=f) == {T1, T3, T4, EX}


def test_full_timelines_stream_nonmatching_events(ldb):
    rows = list(ldb.iter_term_set_events([parse_has_clause("~diabetes")]))
    # T3 qualifies via its synonym; its xref events ride along anyway.
    assert any(tc.term_id == T3 and tc.change.tag == "xref" for tc in rows)
    # T1's name never matched the clause but its add event is present.
    assert any(tc.term_id == T1 and tc.change.tag == "name" for tc in rows)


@pytest.mark.parametrize("reverse", [False, True])
def test_term_set_after_resumes_and_concatenates(ldb, reverse):
    clauses = [parse_has_clause("~diabetes")]
    full = list(ldb.iter_term_set_events(clauses, reverse=reverse))
    keys = _section_keys(full, "term")
    assert len(keys) >= 3
    for i, key in enumerate(keys):
        later = set(keys[i + 1 :])
        rest = list(
            ldb.iter_term_set_events(clauses, reverse=reverse, after=key)
        )
        assert rest == [tc for tc in full if tc.term_id in later]


def test_term_set_counts_are_exact(ldb):
    clauses = [parse_has_clause("~diabetes")]
    counts = ldb.term_set_counts(clauses)
    rows = list(ldb.iter_term_set_events(clauses))
    assert counts.terms == 4
    assert counts.events == len(rows)
    assert counts.commits == len({tc.change.commit_seq for tc in rows})
