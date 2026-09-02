"""Tests for OWL→OBO conversion: import stripping, the OID-keyed cache,
ROBOT discovery, and (when ROBOT is available) a real conversion."""

from pathlib import Path

import pytest

from obohog.convert import (
    ConversionError,
    IdentityConverter,
    RobotConverter,
    converter_for,
    find_robot,
    strip_imports,
)
from obohog.settings import ObohogSettings


OFN_DOC = b"""Prefix(:=<http://example.org/test#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)
Prefix(obo:=<http://purl.obolibrary.org/obo/>)
Ontology(<http://example.org/test.owl>
Import(<http://example.org/missing.owl>)
Declaration(Class(obo:TST_0000001))
Declaration(Class(obo:TST_0000002))
AnnotationAssertion(rdfs:label obo:TST_0000001 "test term")
SubClassOf(obo:TST_0000002 obo:TST_0000001)
)
"""


def _robot_available() -> bool:
    try:
        find_robot()
        return True
    except ConversionError:
        return False


# --- strip_imports ---------------------------------------------------------


def test_strip_imports_ofn():
    stripped = strip_imports(OFN_DOC)
    assert b"Import(" not in stripped
    # Everything else survives untouched.
    assert b"Declaration(Class(obo:TST_0000001))" in stripped
    assert b'AnnotationAssertion(rdfs:label obo:TST_0000001 "test term")' in stripped


def test_strip_imports_rdfxml_self_closing():
    doc = (
        b'<?xml version="1.0"?>\n'
        b'<rdf:RDF xmlns:owl="http://www.w3.org/2002/07/owl#">\n'
        b'  <owl:Ontology rdf:about="http://example.org/test.owl">\n'
        b'    <owl:imports rdf:resource="http://example.org/missing.owl"/>\n'
        b'  </owl:Ontology>\n'
        b'  <owl:Class rdf:about="http://example.org/A"/>\n'
        b"</rdf:RDF>\n"
    )
    stripped = strip_imports(doc)
    assert b"owl:imports" not in stripped
    assert b'<owl:Class rdf:about="http://example.org/A"/>' in stripped


def test_strip_imports_rdfxml_multiline_element():
    doc = (
        b'<?xml version="1.0"?>\n'
        b"<rdf:RDF>\n"
        b"  <owl:imports\n"
        b'      rdf:resource="http://example.org/missing.owl"/>\n'
        b"  <owl:Class/>\n"
        b"</rdf:RDF>\n"
    )
    stripped = strip_imports(doc)
    assert b"owl:imports" not in stripped
    assert b"missing.owl" not in stripped
    assert b"<owl:Class/>" in stripped


def test_strip_imports_unknown_syntax_passes_through():
    doc = b"@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    assert strip_imports(doc) == doc


# --- cache behavior (no ROBOT needed) --------------------------------------


def _broken_converter(cache_dir: Path, version: str = "test") -> RobotConverter:
    """A RobotConverter whose subprocess could never succeed — any convert
    that escapes the cache fails loudly, proving cache hits never fork."""
    return RobotConverter(cache_dir, ("robot-does-not-exist",), version)


def test_cache_hit_skips_conversion_and_blob_read(tmp_path: Path):
    converter = _broken_converter(tmp_path / "converted")
    (tmp_path / "converted" / "abc123.obo").write_bytes(b"[Term]\nid: TST:1\n")

    def read() -> bytes:
        raise AssertionError("cache hit must not read the blob")

    assert converter.convert("abc123", read) == b"[Term]\nid: TST:1\n"


def test_cache_miss_with_broken_robot_raises(tmp_path: Path):
    converter = _broken_converter(tmp_path / "converted")
    with pytest.raises((ConversionError, OSError)):
        converter.convert("def456", lambda: OFN_DOC)


def test_conversion_failure_captures_robots_error_line(tmp_path: Path):
    # A failing conversion keeps ROBOT's first *meaningful* stderr line as
    # `reason` — the JVM's "Picked up JAVA_TOOL_OPTIONS" notice (always
    # present, since conversions cap the JVM to one core) doesn't count.
    fake_robot = (
        "/bin/sh", "-c",
        "echo 'Picked up JAVA_TOOL_OPTIONS: -XX:ActiveProcessorCount=1' >&2; "
        "echo 'FAKE PARSE ERROR: bad axiom' >&2; false",
    )
    converter = RobotConverter(tmp_path / "converted", fake_robot, "test")
    with pytest.raises(ConversionError) as excinfo:
        converter.convert("def456", lambda: OFN_DOC)
    assert excinfo.value.reason == "FAKE PARSE ERROR: bad axiom"


def test_prune_empties_the_cache(tmp_path: Path):
    cache = tmp_path / "converted"
    converter = _broken_converter(cache)
    (cache / "aaa.obo").write_bytes(b"a conversion")
    (cache / "bbb.obo").write_bytes(b"another")
    (cache / "ccc.12345.out.obo").write_bytes(b"orphaned temp")
    (cache / "ddd.12345.in.owl").write_bytes(b"orphaned temp")
    converter.prune()
    assert sorted(p.name for p in cache.iterdir()) == ["meta.json"]


def test_converter_change_clears_cache(tmp_path: Path):
    cache = tmp_path / "converted"
    old = _broken_converter(cache, version="1.0")
    (cache / "abc123.obo").write_bytes(b"old conversion")
    # Same identity → cache survives a new converter instance.
    _broken_converter(cache, version="1.0")
    assert (cache / "abc123.obo").exists()
    # Different ROBOT version → stale conversions are cleared.
    new = _broken_converter(cache, version="2.0")
    assert not (cache / "abc123.obo").exists()
    assert new.converter_id != old.converter_id


# --- discovery -------------------------------------------------------------


def test_find_robot_missing_everything(monkeypatch):
    monkeypatch.setattr("obohog.convert.shutil.which", lambda name: None)
    with pytest.raises(ConversionError, match="ROBOT"):
        find_robot(ObohogSettings(robot_jar=None, _env_file=None))


def test_find_robot_bad_jar_path(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("obohog.convert.shutil.which", lambda name: None)
    settings = ObohogSettings(robot_jar=str(tmp_path / "nope.jar"), _env_file=None)
    with pytest.raises(ConversionError, match="does not exist"):
        find_robot(settings)


def test_converter_for_obo_source():
    class FakeSource:
        format = "obo"

    assert isinstance(converter_for(FakeSource()), IdentityConverter)


def test_identity_converter_passes_bytes_through():
    assert IdentityConverter().convert("any", lambda: b"[Term]\nid: X:1\n") == (
        b"[Term]\nid: X:1\n"
    )


# --- wiring into builds ----------------------------------------------------


class _FakeConverter:
    """Identity conversion under a fake identity; optionally fails on
    chosen blob OIDs to exercise the skip path."""

    def __init__(self, bad_oids: set[str] = frozenset(), converter_id: str = "fake=1"):
        self.bad_oids = set(bad_oids)
        self.converter_id = converter_id

    def convert(self, oid, read):
        if oid in self.bad_oids:
            raise ConversionError(f"cannot convert {oid}")
        return read()

    def prune(self):
        pass


OBO = "src/onto.obo"


def test_conversion_failure_skips_commit_and_folds_forward(obo_repo, tmp_path: Path):
    from obohog.extract import extract
    from obohog.gitsource import GitSource

    out = tmp_path / "artifact"
    with GitSource(obo_repo) as src:
        versions = list(src.iter_file_history(OBO))
        # c1 adds the synonym; c2 is a pure rename with the same blob, so
        # failing this OID skips both commits.
        bad = versions[1].blob_oid
        extract(src, OBO, out, converter=_FakeConverter({bad}))

    import pyarrow.parquet as pq

    skipped = pq.read_table(out / "skipped.parquet").to_pylist()
    assert [(r["commit_seq"], r["error"]) for r in skipped] == [
        (1, "ConversionFailed"),
        (2, "ConversionFailed"),
    ]
    # The skipped commits' change (the synonym) folds into the next
    # convertible commit rather than being lost.
    events = pq.read_table(out / "events.parquet").to_pylist()
    synonym_adds = [
        e for e in events if e["tag"] == "synonym" and e["operation"] == "add"
    ]
    assert [e["commit_seq"] for e in synonym_adds] == [3]


def test_build_meta_records_converter_id(obo_repo, tmp_path: Path):
    from obohog import model
    from obohog.extract import extract
    from obohog.gitsource import GitSource

    fake_out, plain_out = tmp_path / "fake", tmp_path / "plain"
    with GitSource(obo_repo) as src:
        extract(src, OBO, fake_out, converter=_FakeConverter())
    with GitSource(obo_repo) as src:
        extract(src, OBO, plain_out)
    assert model.read_build_meta(fake_out).converter_id == "fake=1"
    assert model.read_build_meta(plain_out).converter_id == "obo"


def test_converter_change_forces_full_rebuild(obo_repo, tmp_path: Path):
    from obohog.extract import BuildMode, build_parallel, plan_build
    from obohog.gitsource import GitSource

    out = tmp_path / "artifact"
    build_parallel(str(obo_repo), OBO, out, jobs=1)
    with GitSource(obo_repo) as src:
        full = list(src.iter_file_history(OBO))
    same = plan_build(out, full, limit=None, update=True, converter_id="obo")
    assert same.mode is BuildMode.UP_TO_DATE
    changed = plan_build(out, full, limit=None, update=True, converter_id="robot=9.9")
    assert changed.mode is BuildMode.FULL


# --- end-to-end with a real ROBOT ------------------------------------------


@pytest.mark.skipif(not _robot_available(), reason="ROBOT not installed")
def test_owl_history_end_to_end(ofn_repo, tmp_path: Path):
    from obohog import model
    from obohog.extract import extract, build_parallel
    from obohog.gitsource import GitSource
    from obohog.query import HistoryDB

    command, version = find_robot()
    converter = RobotConverter(tmp_path / "cache", command, version)
    serial_out, parallel_out = tmp_path / "serial", tmp_path / "parallel"
    with GitSource(ofn_repo) as src:
        extract(src, "onto.owl", serial_out, converter=converter)

    meta = model.read_build_meta(serial_out)
    assert meta.converter_id.startswith("robot=")

    import pyarrow.parquet as pq

    assert pq.read_table(serial_out / "skipped.parquet").to_pylist() == []
    events = pq.read_table(serial_out / "events.parquet").to_pylist()
    names = [
        (e["commit_seq"], e["operation"], e["value"])
        for e in events
        if e["term_id"] == "TST:0000001" and e["tag"] == "name"
    ]
    # Created at c0; label renamed (remove + add) at c2.
    assert sorted(names) == [
        (0, "add", "test term"),
        (2, "add", "renamed term"),
        (2, "remove", "test term"),
    ]
    is_a = [
        (e["commit_seq"], e["operation"], e["body"])
        for e in events
        if e["term_id"] == "TST:0000002" and e["tag"] == "is_a"
    ]
    # Created at c1. c2's rename of the *parent's* label also touches this
    # clause: ROBOT writes the target label as a `!` comment
    # ("is_a: TST:0000001 ! test term"), so the referencing stanza changes
    # too — the render layer classifies exactly this as a target-label edit.
    assert sorted(is_a) == [
        (1, "add", "TST:0000001"),
        (2, "add", "TST:0000001"),
        (2, "remove", "TST:0000001"),
    ]

    # c3's object property lands as a [Typedef] stanza, tracked like a term.
    typedef = [
        (e["commit_seq"], e["tag"], e["value"])
        for e in events
        if e["term_id"] == "TST:9000001" and e["operation"] == "add"
    ]
    assert sorted(typedef) == [
        (3, "is_transitive", "true"),
        (3, "name", "part of thing"),
    ]

    # The parallel build (workers hit the now-warm conversion cache) must
    # match the serial one exactly.
    build_parallel(str(ofn_repo), "onto.owl", parallel_out, jobs=2, converter=converter)
    # A successful sync empties the cache — it's build-transient; the
    # next sync reconverts its one seed blob from the clone.
    assert list((tmp_path / "cache").glob("*.obo")) == []
    ds, dp = HistoryDB(serial_out), HistoryDB(parallel_out)
    cols = "term_id, commit_seq, operation, tag, value"
    q = f"SELECT {cols} FROM events"
    assert sorted(ds.con.execute(q).fetchall()) == sorted(dp.con.execute(q).fetchall())
    ds.close()
    dp.close()


@pytest.mark.skipif(not _robot_available(), reason="ROBOT not installed")
def test_robot_conversion_end_to_end(tmp_path: Path):
    import fastobo

    command, version = find_robot()
    converter = RobotConverter(tmp_path / "converted", command, version)
    reads = 0

    def read() -> bytes:
        nonlocal reads
        reads += 1
        return OFN_DOC

    obo_bytes = converter.convert("blob1", read)
    doc = fastobo.loads(obo_bytes.decode())
    frames = {str(f.id): f for f in doc}
    assert "TST:0000001" in frames
    labeled = frames["TST:0000001"]
    assert any("test term" in str(clause) for clause in labeled)
    # The unresolvable import was stripped, not fetched.
    assert b"missing.owl" not in obo_bytes
    # Second convert of the same OID is a pure cache hit.
    assert converter.convert("blob1", read) == obo_bytes
    assert reads == 1
