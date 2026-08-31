"""Service layer: paged search/diff, timelines, and source status."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from obohog import service
from obohog.config import Config, GitFileSource
from obohog.query import HistoryDB
from obohog.service import InvalidCursor, SearchParams
from obohog.views import SourceStyle

STYLE = SourceStyle(pr_url_base="https://github.com/x/y/pull/")


@pytest.fixture(scope="module")
def db(paged_artifact: Path):
    db = HistoryDB(paged_artifact)
    yield db
    db.close()


def _sha_of(db: HistoryDB, seq: int) -> str:
    return db.con.execute(
        "SELECT sha FROM commits WHERE commit_seq = ?", [seq]
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Search paging.


def test_search_unpaged_sections_by_term(db):
    page = service.search(db, STYLE, SearchParams(q="SHARED"))
    assert [s.term_id for s in page.sections] == [
        "MONDO:0000001", "MONDO:0000002", "MONDO:0000003", "MONDO:0000004",
    ]
    assert page.next_cursor is None
    assert page.counts.approximate is True
    assert page.counts.events == 4
    # Each section: one commit group, one add op with spans + raw value.
    section = page.sections[0]
    assert section.name == "alpha"
    (group,) = section.commits
    (op,) = group.ops
    assert op.kind == "add"
    assert op.tag == "xref"
    assert op.after == "SHARED:1"
    assert op.before is None
    assert [s.role for s in op.head] == ["same"]


def test_search_pages_concatenate_to_unpaged(db):
    for order in ("term", "date"):
        unpaged = service.search(
            db, STYLE, SearchParams(q="SHARED", order=order, limit=500)
        )
        collected = []
        after = None
        pages = 0
        while True:
            page = service.search(
                db, STYLE,
                SearchParams(q="SHARED", order=order, limit=1, after=after),
            )
            collected.extend(page.sections)
            pages += 1
            if page.next_cursor is None:
                break
            after = page.next_cursor
        assert pages == len(unpaged.sections) == 4
        assert [s.model_dump() for s in collected] == [
            s.model_dump() for s in unpaged.sections
        ]


def test_search_date_order_sections_are_commits(db):
    page = service.search(db, STYLE, SearchParams(q="SHARED", order="date"))
    seqs = [s.commit.commit_seq for s in page.sections]
    assert seqs == sorted(seqs, reverse=True)  # newest first
    assert page.sections[0].terms[0].term_id == "MONDO:0000004"


def test_search_next_cursor_set_when_truncated(db):
    page = service.search(db, STYLE, SearchParams(q="SHARED", limit=2))
    assert len(page.sections) == 2
    assert page.next_cursor == page.sections[-1].term_id


def test_search_bad_date_cursor_raises_invalid_cursor(db):
    with pytest.raises(InvalidCursor, match="not valid"):
        service.search(
            db, STYLE,
            SearchParams(q="SHARED", order="date", after="MONDO:0000001"),
        )


def test_search_params_reject_out_of_range_limit():
    with pytest.raises(ValidationError):
        SearchParams(q="x", limit=0)
    with pytest.raises(ValidationError):
        SearchParams(q="x", limit=service.MAX_PAGE + 1)


# ---------------------------------------------------------------------------
# Diff.


def test_diff_pages_with_exact_counts(db):
    page = service.diff(db, STYLE, "0", "HEAD")
    assert [s.term_id for s in page.sections] == [
        "MONDO:0000002", "MONDO:0000003", "MONDO:0000004",
    ]
    assert page.counts.approximate is False
    # Creations: name + xref per term.
    assert page.counts.events == 6
    assert page.next_cursor is None

    first = service.diff(db, STYLE, "0", "HEAD", limit=1)
    assert first.next_cursor == "MONDO:0000002"
    rest = service.diff(db, STYLE, "0", "HEAD", after=first.next_cursor)
    assert [s.term_id for s in rest.sections] == [
        "MONDO:0000003", "MONDO:0000004",
    ]


def test_diff_term_filter(db):
    page = service.diff(db, STYLE, "0", "HEAD", term="MONDO:0000003")
    assert [s.term_id for s in page.sections] == ["MONDO:0000003"]


# ---------------------------------------------------------------------------
# Timeline / state / commit / pr / releases.


def test_timeline_groups_by_commit_oldest_first(db):
    out = service.get_timeline(db, STYLE, "MONDO:0000001")
    assert out is not None
    assert out.header is not None and out.header.current_name == "alpha"
    assert out.total_events == out.shown_events == 2  # name + xref creation
    (group,) = out.commits
    assert group.commit.commit_seq == 0
    assert {op.tag for op in group.ops} == {"name", "xref"}
    assert out.by_tag == {"name": 1, "xref": 1}


def test_timeline_unknown_term_is_none(db):
    assert service.get_timeline(db, STYLE, "MONDO:9999999") is None


def test_state_at_release(db):
    out = service.get_state(db, "MONDO:0000001", "v1.0")
    assert out is not None
    assert out.commit_seq == 3
    assert {"tag": "name", "value": "alpha"} in out.clauses


def test_state_unknown_term_is_none(db):
    assert service.get_state(db, "MONDO:9999999", "HEAD") is None


def test_commit_view_groups_terms(db):
    out = service.get_commit(db, STYLE, _sha_of(db, 1)[:7])
    assert out is not None
    assert out.commit.commit_seq == 1
    assert [t.term_id for t in out.terms] == ["MONDO:0000002"]
    assert out.commit.subject == "c1 beta"


def test_commit_view_unknown_sha_is_none(db):
    assert service.get_commit(db, STYLE, "abcdef9") is None


def test_pr_without_history_is_none(db):
    assert service.get_pr(db, STYLE, 12345) is None


def test_releases(db):
    (rel,) = service.list_releases(db)
    assert rel.tag == "v1.0"
    assert rel.commit_seq == 3


# ---------------------------------------------------------------------------
# Source status.


def _source(name: str, base: Path, db_dir: Path) -> GitFileSource:
    return GitFileSource(
        type="git-file",
        name=name,
        repo="https://github.com/example/repo",
        file="onto.obo",
        clone_dir=base / name / "clone",
        db_dir=db_dir,
    )


def test_list_sources_status(paged_artifact: Path, tmp_path: Path):
    cfg = Config(
        path=tmp_path / "obohog.toml",
        storage=tmp_path,
        sources={
            "built": _source("built", tmp_path, paged_artifact),
            "missing": _source("missing", tmp_path, tmp_path / "nope" / "db"),
        },
    )
    built, missing = service.list_sources(cfg)
    assert built.status == "built"
    assert built.schema_version is not None
    assert built.n_commits == 4
    assert missing.status == "not built"
    assert missing.n_commits is None


@pytest.fixture(scope="module")
def wide_db(wide_commit_artifact: Path):
    db = HistoryDB(wide_commit_artifact)
    yield db
    db.close()


def test_commit_view_pages_by_term(wide_db):
    page1 = service.get_commit(wide_db, STYLE, "0", limit=2)
    assert [t.term_id for t in page1.terms] == ["MONDO:0000001", "MONDO:0000002"]
    assert page1.counts.terms == 3  # whole-commit scope, not the window
    assert page1.next_cursor == "MONDO:0000002"
    page2 = service.get_commit(
        wide_db, STYLE, "0", limit=2, after=page1.next_cursor
    )
    assert [t.term_id for t in page2.terms] == ["MONDO:0000003"]
    assert page2.next_cursor is None


def test_date_order_sections_cap_terms(wide_db, monkeypatch):
    monkeypatch.setattr(service, "SECTION_TERM_CAP", 2)
    page = service.search(wide_db, STYLE, SearchParams(order="date"))
    (section,) = page.sections
    assert [t.term_id for t in section.terms] == [
        "MONDO:0000001", "MONDO:0000002",
    ]
    assert section.more_terms == 1
