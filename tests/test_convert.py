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


# --- end-to-end with a real ROBOT ------------------------------------------


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
