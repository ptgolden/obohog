"""OWL→OBO conversion for sources whose tracked file isn't OBO format.

A source declaring ``format = "owl"`` in ``obohog.toml`` has its blobs
converted to OBO before they reach the extract pipeline; everything from
:class:`~obohog.obo.DocumentState` down never learns the source wasn't
OBO. The converter is ROBOT (http://robot.obolibrary.org) — the only
maintained implementation of the official OBO Foundry OWL↔OBO mapping —
invoked as a subprocess per version, either as ``robot`` on PATH or via
``ROBOT_JAR`` in ``.env``.

Conversion is the expensive step (~0.5–1 s per version, JVM startup
included), so results are cached on disk keyed by blob OID: a cache hit
skips both the conversion *and* the ``git cat-file`` read. The cache
directory carries a ``meta.json`` recording the converter identity
(ROBOT version + conversion settings); a mismatch — say, a ROBOT
upgrade that reorders output — clears the cache so stale conversions
can't manufacture phantom diffs against fresh ones.

Two conversion-time transformations, both deliberate:

* **Imports are stripped before conversion.** ODK edit files
  ``Import(...)`` sibling component files resolved through a local
  ``catalog-v001.xml`` — files the single-file history model never
  fetches, so ROBOT could not resolve them anyway. Semantically this
  matches what obohog tracks: the edit file's own axioms, not its
  import closure.
* **``--check false``.** Historical versions are routinely legal OWL
  but illegal OBO (e.g. two ``comment`` annotations on one class,
  where OBO allows one per frame). ROBOT's OBO writer hard-fails on
  such structure rules by default; with the check relaxed it emits the
  frame anyway, and fastobo parses the result without complaint — the
  "illegal" clauses are simply tracked as-is.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Protocol

from .settings import ObohogSettings, get_settings


class ConversionError(RuntimeError):
    """Raised when ROBOT is unavailable or a conversion fails. Preserves
    captured stderr in the message so failures aren't silent."""


class Converter(Protocol):
    """Turns one tracked-file blob into OBO bytes.

    ``convert`` takes the blob's OID plus a *lazy* reader so a cache hit
    can skip the underlying ``git cat-file`` entirely.
    """

    @property
    def converter_id(self) -> str: ...

    def convert(self, oid: str, read: Callable[[], bytes]) -> bytes: ...


class IdentityConverter:
    """The no-op converter for ``format = "obo"`` sources."""

    converter_id = "obo"

    def convert(self, oid: str, read: Callable[[], bytes]) -> bytes:
        return read()


# Bumped whenever strip_imports or the ROBOT invocation changes in a way
# that could alter conversion output; part of converter_id so such a
# change invalidates caches (and, once recorded in build_meta, artifacts).
_STRIP_RULES_VERSION = 1


class RobotConverter:
    """OWL→OBO via a ROBOT subprocess, with an OID-keyed on-disk cache."""

    def __init__(self, cache_dir: Path, command: tuple[str, ...], version: str):
        self._cache_dir = cache_dir
        self._command = command
        self.converter_id = f"robot={version};out=obo;strip={_STRIP_RULES_VERSION}"
        self._init_cache()

    def _init_cache(self) -> None:
        """Ensure the cache dir exists and belongs to this converter.

        A stale ``meta.json`` (different ROBOT version / strip rules)
        clears every cached conversion. Parallel workers race here only
        when the converter identity just changed; the worst case is a
        freshly written conversion getting deleted, which is simply a
        cache miss on its next request.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._cache_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            meta = None
        if meta and meta.get("converter_id") == self.converter_id:
            return
        for stale in self._cache_dir.glob("*.obo"):
            stale.unlink(missing_ok=True)
        meta_path.write_text(json.dumps({"converter_id": self.converter_id}))

    def convert(self, oid: str, read: Callable[[], bytes]) -> bytes:
        cached = self._cache_dir / f"{oid}.obo"
        try:
            return cached.read_bytes()
        except FileNotFoundError:
            pass
        stripped = strip_imports(read())
        # Per-pid temp names so parallel workers converting the same OID
        # can't collide; os.replace makes the cache write atomic. The
        # ``.owl`` input suffix lets OWLAPI sniff the concrete syntax
        # (OFN, RDF/XML, ...) itself.
        in_path = self._cache_dir / f"{oid}.{os.getpid()}.in.owl"
        out_path = self._cache_dir / f"{oid}.{os.getpid()}.out.obo"
        in_path.write_bytes(stripped)
        try:
            result = subprocess.run(
                [
                    *self._command,
                    "convert", "--check", "false",
                    "--input", str(in_path),
                    "--format", "obo",
                    "--output", str(out_path),
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0 or not out_path.exists():
                raise ConversionError(
                    f"ROBOT conversion of blob {oid} failed "
                    f"(exit {result.returncode})\n"
                    f"stderr: {result.stderr.strip()[-2000:]}"
                )
            data = out_path.read_bytes()
            os.replace(out_path, cached)
            return data
        finally:
            in_path.unlink(missing_ok=True)
            if out_path.exists():
                out_path.unlink(missing_ok=True)


# A self-closing or one-line RDF/XML owl:imports element.
_XML_IMPORT_ONE_LINE = re.compile(rb"<owl:imports\b[^>]*(?:/>|</owl:imports>)")
_XML_IMPORT_OPEN = re.compile(rb"<owl:imports\b")
_XML_IMPORT_CLOSE = re.compile(rb"(?:/>|</owl:imports>)")


def strip_imports(data: bytes) -> bytes:
    """Remove ``owl:imports`` declarations from an OWL document.

    Handles the two syntaxes obohog supports — OFN (``Import(...)``
    lines) and RDF/XML (``<owl:imports .../>`` elements). Anything else
    (Turtle, Manchester) passes through unchanged; if such a file has
    imports, ROBOT's unresolvable-import error will surface at
    conversion time and name the missing IRI.
    """
    head = data.lstrip()[:200]
    if head.startswith((b"<?xml", b"<rdf:RDF", b"<!DOCTYPE")):
        return _strip_xml_imports(data)
    if b"Prefix(" in head or b"Ontology(" in head:
        lines = data.split(b"\n")
        return b"\n".join(ln for ln in lines if not ln.lstrip().startswith(b"Import("))
    return data


def _strip_xml_imports(data: bytes) -> bytes:
    kept: list[bytes] = []
    in_element = False
    for line in data.split(b"\n"):
        if in_element:
            if _XML_IMPORT_CLOSE.search(line):
                in_element = False
            continue
        if _XML_IMPORT_ONE_LINE.search(line):
            kept.append(_XML_IMPORT_ONE_LINE.sub(b"", line))
            continue
        if _XML_IMPORT_OPEN.search(line):
            # Element opens here and closes on a later line.
            kept.append(_XML_IMPORT_OPEN.split(line, maxsplit=1)[0])
            in_element = True
            continue
        kept.append(line)
    return b"\n".join(kept)


def find_robot(settings: ObohogSettings | None = None) -> tuple[tuple[str, ...], str]:
    """Locate ROBOT and return ``(command, version)``.

    Prefers ``robot`` on PATH; falls back to ``ROBOT_JAR`` from ``.env``
    run through ``java -jar``. Raises :class:`ConversionError` with
    setup guidance when neither is available.
    """
    settings = settings if settings is not None else get_settings()
    if exe := shutil.which("robot"):
        command: tuple[str, ...] = (exe,)
    elif settings.robot_jar:
        jar = Path(settings.robot_jar).expanduser()
        if not jar.is_file():
            raise ConversionError(f"ROBOT_JAR points at {jar}, which does not exist.")
        java = shutil.which("java")
        if java is None:
            raise ConversionError(f"ROBOT_JAR is set ({jar}) but no `java` found on PATH.")
        command = (java, "-jar", str(jar))
    else:
        raise ConversionError(
            "Sources with format = \"owl\" need ROBOT (http://robot.obolibrary.org): "
            "install `robot` on PATH, or set ROBOT_JAR=/path/to/robot.jar in .env."
        )
    result = subprocess.run([*command, "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise ConversionError(
            f"`{' '.join(command)} --version` failed (exit {result.returncode})\n"
            f"stderr: {result.stderr.strip()}"
        )
    # "ROBOT version 1.9.8" → "1.9.8"; an unexpected shape keeps the whole
    # line, which still works as an opaque cache-identity token.
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"
    version = line.rsplit(None, 1)[-1] if line else "unknown"
    return command, version


def converter_for(source) -> Converter:
    """The converter matching a source's declared ``format``."""
    if source.format == "obo":
        return IdentityConverter()
    return RobotConverter(source.convert_dir, *find_robot())
