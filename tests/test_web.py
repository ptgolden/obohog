"""The JSON API over a built artifact: routes, error mapping, paging."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from obohog.config import Config, GitFileSource
from obohog.web.app import create_app
from obohog.web.deps import SourceRegistry


def _source(name: str, base: Path, db_dir: Path) -> GitFileSource:
    return GitFileSource(
        type="git-file",
        name=name,
        repo="https://github.com/example/repo",
        file="onto.obo",
        clone_dir=base / name / "clone",
        db_dir=db_dir,
    )


def _config(base: Path, paged_artifact: Path) -> Config:
    return Config(
        path=base / "obohog.toml",
        storage=base,
        sources={
            "onto": _source("onto", base, paged_artifact),
            "empty": _source("empty", base, base / "empty" / "db"),
        },
    )


@pytest.fixture(scope="module")
def client(paged_artifact: Path, tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("web")
    app = create_app(_config(base, paged_artifact))
    with TestClient(app) as client:
        yield client


def test_sources_lists_status(client):
    body = client.get("/api/v1/sources").json()
    by_name = {s["name"]: s for s in body}
    assert by_name["onto"]["status"] == "built"
    assert by_name["onto"]["n_commits"] == 4
    assert by_name["empty"]["status"] == "not built"


def test_unknown_source_is_404(client):
    r = client.get("/api/v1/sources/nope/releases")
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]


def test_artifactless_source_is_503(client):
    r = client.get("/api/v1/sources/empty/releases")
    assert r.status_code == 503
    assert "source sync" in r.json()["detail"]


def test_term_timeline_with_colon_in_path(client):
    r = client.get("/api/v1/sources/onto/terms/MONDO:0000001")
    assert r.status_code == 200
    body = r.json()
    assert body["term_id"] == "MONDO:0000001"
    assert body["header"]["current_name"] == "alpha"
    assert body["total_events"] == 2
    (group,) = body["commits"]
    assert group["commit"]["commit_seq"] == 0
    assert {op["tag"] for op in group["ops"]} == {"name", "xref"}


def test_unknown_term_is_404(client):
    assert (
        client.get("/api/v1/sources/onto/terms/MONDO:9999999").status_code
        == 404
    )


def test_term_state_at_ref(client):
    r = client.get(
        "/api/v1/sources/onto/terms/MONDO:0000001/state", params={"at": "v1.0"}
    )
    assert r.status_code == 200
    assert {"tag": "name", "value": "alpha"} in r.json()["clauses"]


def test_unresolvable_ref_is_404(client):
    r = client.get(
        "/api/v1/sources/onto/terms/MONDO:0000001/state",
        params={"at": "not-a-ref"},
    )
    assert r.status_code == 404
    assert "not-a-ref" in r.json()["detail"]


def test_search_pages_roundtrip(client):
    unpaged = client.get(
        "/api/v1/sources/onto/search", params={"q": "SHARED"}
    ).json()
    assert unpaged["counts"]["approximate"] is True
    assert unpaged["next_cursor"] is None
    assert len(unpaged["sections"]) == 4

    sections = []
    after = None
    while True:
        params = {"q": "SHARED", "limit": 1}
        if after:
            params["after"] = after
        page = client.get("/api/v1/sources/onto/search", params=params).json()
        sections.extend(page["sections"])
        after = page["next_cursor"]
        if after is None:
            break
    assert sections == unpaged["sections"]


def test_search_bad_regex_is_400(client):
    r = client.get(
        "/api/v1/sources/onto/search", params={"q": "[", "match": "regex"}
    )
    assert r.status_code == 400


def test_search_exact_matches_body(client):
    # The fixture's xref bodies are SHARED:1..4 — an exact body hits one
    # term, a substring of it hits none.
    hit = client.get(
        "/api/v1/sources/onto/search", params={"q": "SHARED:1", "match": "exact"}
    ).json()
    assert [s["term_id"] for s in hit["sections"]] == ["MONDO:0000001"]
    miss = client.get(
        "/api/v1/sources/onto/search", params={"q": "SHARED", "match": "exact"}
    ).json()
    assert miss["sections"] == []


def test_search_bad_match_mode_is_422(client):
    r = client.get(
        "/api/v1/sources/onto/search", params={"q": "x", "match": "fuzzy"}
    )
    assert r.status_code == 422


def test_search_bad_date_cursor_is_400(client):
    r = client.get(
        "/api/v1/sources/onto/search",
        params={"q": "SHARED", "order": "date", "after": "MONDO:0000001"},
    )
    assert r.status_code == 400


def test_search_limit_out_of_range_is_422(client):
    r = client.get(
        "/api/v1/sources/onto/search", params={"q": "SHARED", "limit": 0}
    )
    assert r.status_code == 422


def test_diff_between_refs(client):
    r = client.get(
        "/api/v1/sources/onto/diff", params={"a": "0", "b": "HEAD"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["counts"] == {
        "events": 6, "terms": 3, "commits": 3, "approximate": False,
    }
    assert [s["term_id"] for s in body["sections"]] == [
        "MONDO:0000002", "MONDO:0000003", "MONDO:0000004",
    ]


def test_commit_view_by_sha_prefix(client):
    releases = client.get("/api/v1/sources/onto/releases").json()
    assert releases[0]["tag"] == "v1.0"
    # No sha endpoint yet in the fixture data — take one from a timeline.
    timeline = client.get("/api/v1/sources/onto/terms/MONDO:0000002").json()
    sha = timeline["commits"][0]["commit"]["sha"]
    r = client.get(f"/api/v1/sources/onto/commits/{sha[:7]}")
    assert r.status_code == 200
    assert r.json()["commit"]["sha"] == sha
    assert [t["term_id"] for t in r.json()["terms"]] == ["MONDO:0000002"]


def test_unknown_commit_is_404(client):
    assert (
        client.get("/api/v1/sources/onto/commits/0000000").status_code == 404
    )


def test_unknown_pr_is_404(client):
    assert client.get("/api/v1/sources/onto/prs/12345").status_code == 404


def test_facets_lists_distinct_filter_values(client):
    body = client.get("/api/v1/sources/onto/facets").json()
    assert body == {"tags": ["name", "xref"], "namespaces": ["MONDO"]}


def test_openapi_document_serves(client):
    r = client.get("/api/openapi.json")
    assert r.status_code == 200
    assert "/api/v1/sources/{src}/search" in r.json()["paths"]


# ---------------------------------------------------------------------------
# HTML pages.


def test_home_lists_sources_as_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'href="/onto/search"' in r.text
    assert "not built" in r.text  # the empty source, unlinked


def test_term_page_renders_ops(client):
    r = client.get("/onto/terms/MONDO:0000001")
    assert r.status_code == 200
    assert "alpha" in r.text
    assert 'class="op op-add"' in r.text
    assert "xref" in r.text


def test_term_page_unknown_term_is_html_404(client):
    r = client.get("/onto/terms/MONDO:9999999")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "no history" in r.text


def test_search_page_without_query_shows_form_only(client):
    r = client.get("/onto/search")
    assert r.status_code == 200
    assert "<form" in r.text
    assert "up to" not in r.text


def test_search_page_renders_first_page_with_load_more(client):
    r = client.get("/onto/search", params={"q": "SHARED", "limit": 1})
    assert r.status_code == 200
    assert "up to" in r.text
    assert "MONDO:0000001" in r.text
    assert 'hx-trigger="revealed"' in r.text
    assert "after=MONDO%3A0000001" in r.text


def test_search_form_blank_filters_do_not_filter(client):
    # The HTML form submits untouched fields as empty strings
    # (?q=...&tag=&namespace=&term=) — they must mean "no filter",
    # not "match the empty string".
    r = client.get(
        "/onto/search?q=SHARED&tag=&namespace=&term=&order=term"
    )
    assert r.status_code == 200
    assert "MONDO:0000001" in r.text
    api = client.get(
        "/api/v1/sources/onto/search",
        params={"q": "SHARED", "tag": "", "namespace": "", "term": ""},
    ).json()
    assert len(api["sections"]) == 4


def test_search_form_offers_derived_filter_choices(client):
    r = client.get("/onto/search")
    assert '<select name="tag">' in r.text
    assert '<option value="xref">xref</option>' in r.text
    assert '<select name="namespace">' in r.text
    assert '<option value="MONDO">MONDO</option>' in r.text


def test_search_form_keeps_selected_filter_value(client):
    r = client.get("/onto/search", params={"q": "SHARED", "tag": "xref"})
    assert '<option value="xref" selected>xref</option>' in r.text
    # A hand-edited URL value outside the derived list must still show as
    # selected (the filter is applied), not silently display "any".
    r = client.get("/onto/search", params={"q": "SHARED", "tag": "bogus"})
    assert '<option value="bogus" selected>bogus</option>' in r.text


def test_search_form_order_newest_and_oldest(client):
    # The order select bakes commit-time direction into date order:
    # newest → (date, reverse=False), oldest → (date, reverse=True).
    newest = client.get("/onto/search", params={"q": "SHARED", "order": "newest"})
    assert '<option value="newest" selected>' in newest.text
    assert newest.text.find("MONDO:0000004") < newest.text.find("MONDO:0000001")
    oldest = client.get("/onto/search", params={"q": "SHARED", "order": "oldest"})
    assert '<option value="oldest" selected>' in oldest.text
    assert oldest.text.find("MONDO:0000001") < oldest.text.find("MONDO:0000004")
    # Pre-existing links with the raw API params still work and map back.
    raw = client.get(
        "/onto/search", params={"q": "SHARED", "order": "date", "reverse": "true"}
    )
    assert '<option value="oldest" selected>' in raw.text


def test_search_results_fragment_pages(client):
    r = client.get(
        "/onto/search/results",
        params={"q": "SHARED", "limit": 1, "after": "MONDO:0000001"},
    )
    assert r.status_code == 200
    assert "<html" not in r.text  # bare fragment
    assert "MONDO:0000002" in r.text
    assert "MONDO:0000001" not in r.text.replace("after=MONDO%3A0000001", "")
    # Last page: no sentinel.
    last = client.get(
        "/onto/search/results",
        params={"q": "SHARED", "limit": 5, "after": "MONDO:0000003"},
    )
    assert "hx-trigger" not in last.text


def test_htmx_error_is_bare_fragment(client):
    r = client.get(
        "/nope/search",
        params={"q": "x"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 404
    assert "<html" not in r.text
    assert "nope" in r.text


def test_unknown_source_page_is_html_404(client):
    r = client.get("/nope/search")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")


def test_static_serves_htmx(client):
    r = client.get("/static/htmx.min.js")
    assert r.status_code == 200
    assert "htmx" in r.text[:200]


def test_registry_reopens_when_build_meta_changes(
    paged_artifact: Path, tmp_path: Path
):
    registry = SourceRegistry(_config(tmp_path, paged_artifact))
    db1, _ = registry.acquire("onto")
    entry1 = registry._entries["onto"].db
    db1.close()

    db2, _ = registry.acquire("onto")
    assert registry._entries["onto"].db is entry1  # unchanged artifact → cached
    db2.close()

    meta = paged_artifact / "build_meta.parquet"
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    db3, _ = registry.acquire("onto")
    assert registry._entries["onto"].db is not entry1  # stat change → reopened
    db3.close()
    registry.close()


def test_registry_facets_cached_until_artifact_changes(
    paged_artifact: Path, tmp_path: Path
):
    registry = SourceRegistry(_config(tmp_path, paged_artifact))
    f1 = registry.facets("onto")
    assert f1.tags == ["name", "xref"]
    assert f1.namespaces == ["MONDO"]
    assert registry.facets("onto") is f1  # unchanged artifact → cached

    meta = paged_artifact / "build_meta.parquet"
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    assert registry.facets("onto") is not f1  # stat change → recomputed
    registry.close()
