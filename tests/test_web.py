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
    assert {op["predicate"] for op in group["ops"]} == {"name", "xref"}


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
    assert {"predicate": "name", "value": "alpha"} in r.json()["clauses"]


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
        "/api/v1/sources/onto/search", params={"q": "[", "regex": "true"}
    )
    assert r.status_code == 400


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


def test_openapi_document_serves(client):
    r = client.get("/api/openapi.json")
    assert r.status_code == 200
    assert "/api/v1/sources/{src}/search" in r.json()["paths"]


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
