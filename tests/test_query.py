"""Query-layer paging and concurrency: the ``after`` cursor and ``fork()``.

Uses a repo wide enough to page over — four terms sharing a searchable
xref prefix across four commits — so both section spines (term for
``order="term"``, commit for ``order="date"``) have several sections.
"""

from pathlib import Path

import pytest

from obohog.query import HistoryDB, RangeFilters, SearchFilters


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


def test_facets_lists_distinct_values_most_frequent_first(db):
    tags, namespaces = db.facets()
    assert tags == ["name", "xref"]  # tied counts break alphabetically
    assert namespaces == ["MONDO"]
