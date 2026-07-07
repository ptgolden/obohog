"""Configuration for obohog: which OBO ontologies to track, and where.

An ``obohog.toml`` file at the project root declares one or more ontology
*sources*. Each source declares a ``type`` — the history-acquisition
strategy — plus type-specific fields. The `--config <path>` CLI flag
overrides the default lookup; otherwise we read ``./obohog.toml`` from
the current working directory. There is no global / XDG search yet.

Two source types today:

``type = "git-file"`` — the ontology's edit file lives in a git repo and
its commit history *is* the history we index. Requires ``repo`` and
``file`` (path within the repo).

``type = "github-release"`` — the ontology is published as GitHub
Releases; each release is one snapshot. The provider materializes each
release into a synthetic git commit so the extract pipeline runs
unchanged. Requires ``repo`` and ``asset`` (release asset filename to
fetch, e.g. ``zp-base.obo``).

Example (also shipped as ``obohog.toml.example``)::

    storage = "./data"

    [source.mondo]
    type = "git-file"
    repo = "https://github.com/monarch-initiative/mondo"
    file = "src/ontology/mondo-edit.obo"

    [source.zp]
    type = "github-release"
    repo = "https://github.com/obophenotype/zebrafish-phenotype-ontology"
    asset = "zp-base.obo"

Per-source paths (``clone_dir``, ``db_dir``) default to
``{storage}/{name}/clone`` and ``{storage}/{name}/db`` respectively, and
can be overridden by explicit fields in the source table.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError


DEFAULT_CONFIG_NAME = "obohog.toml"
DEFAULT_STORAGE_DIR = Path("./data")


class ConfigError(Exception):
    """Raised for missing / malformed config files or unknown source lookups."""


class _BaseSource(BaseModel):
    """Fields shared by every source type. Not instantiated directly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    repo: str
    clone_dir: Path
    db_dir: Path


class GitFileSource(_BaseSource):
    """A single file's history within a git repo — the original OBOHOG shape."""

    type: Literal["git-file"] = "git-file"
    file: str  # path to the OBO file within the repo

    @property
    def tracked_path(self) -> str:
        """The OBO file's location inside the source clone."""
        return self.file


class GitHubReleaseSource(_BaseSource):
    """Published GitHub Releases materialized as a synthetic git history."""

    type: Literal["github-release"] = "github-release"
    asset: str  # asset filename fetched from each release, e.g. "zp-base.obo"

    @property
    def tracked_path(self) -> str:
        """The OBO file's location inside the source clone — the materializer
        writes each release's asset to ``<clone>/<asset>``, so the tracked
        path is just the asset filename."""
        return self.asset


# Discriminated union: pydantic dispatches on the `type` tag to the right
# subclass. Consumer code that needs `source.file` (git-file only) or
# `source.asset` (github-release only) must narrow with `isinstance` first.
# Callers wanting the OBO file's location in the clone regardless of source
# type should use `source.tracked_path`.
SourceConfig = Annotated[
    Union[GitFileSource, GitHubReleaseSource],
    Field(discriminator="type"),
]


@dataclass(frozen=True)
class Config:
    """The parsed obohog.toml — top-level storage root plus a source table."""

    path: Path  # the config file this was loaded from
    storage: Path
    sources: dict[str, GitFileSource | GitHubReleaseSource] = field(default_factory=dict)

    def get_source(self, name: str) -> GitFileSource | GitHubReleaseSource:
        """Look up a source by name; raise a helpful error if it doesn't exist."""
        source = self.sources.get(name)
        if source is not None:
            return source
        available = ", ".join(sorted(self.sources)) or "(none)"
        raise ConfigError(
            f"No source named {name!r} configured in {self.path}. "
            f"Available sources: {available}."
        )


def load_config(path: Path | None = None) -> Config:
    """Load and validate an obohog config file.

    ``path`` may be an explicit path (from ``--config``), or ``None`` — in
    which case we look for ``obohog.toml`` in the current working directory.
    Raises :class:`ConfigError` with a helpful message on any failure.
    """
    resolved = _resolve_config_path(path)
    try:
        data = tomllib.loads(resolved.read_text())
    except FileNotFoundError:
        raise ConfigError(
            f"No obohog config found at {resolved}. "
            f"Create one (see obohog.toml.example) or pass --config <path>."
        )
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(f"Malformed TOML at {resolved}: {err}")
    return _parse_config(resolved, data)


def _resolve_config_path(path: Path | None) -> Path:
    if path is not None:
        return path
    return Path.cwd() / DEFAULT_CONFIG_NAME


def _parse_config(path: Path, data: dict) -> Config:
    storage = Path(data.get("storage", DEFAULT_STORAGE_DIR)).expanduser()
    raw_sources = data.get("source", {})
    if not isinstance(raw_sources, dict):
        raise ConfigError(f"'source' must be a table in {path}, got {type(raw_sources).__name__}")
    sources: dict[str, GitFileSource | GitHubReleaseSource] = {}
    for name, section in raw_sources.items():
        if not isinstance(section, dict):
            raise ConfigError(
                f"'source.{name}' must be a table in {path}, got "
                f"{type(section).__name__}"
            )
        sources[name] = _parse_source(name, section, storage, path)
    return Config(path=path, storage=storage, sources=sources)


def _parse_source(
    name: str, section: dict, storage: Path, path: Path
) -> GitFileSource | GitHubReleaseSource:
    # Give a clearer error than pydantic's raw "Unable to extract tag using
    # discriminator 'type'" when the field is simply missing.
    if "type" not in section:
        raise ConfigError(
            f"source.{name} in {path} is missing required field 'type' "
            f"(expected one of: git-file, github-release)"
        )
    default_root = storage / name
    payload = {
        "name": name,
        "clone_dir": section.get("clone_dir", default_root / "clone"),
        "db_dir": section.get("db_dir", default_root / "db"),
        **section,
    }
    try:
        # Round-trip through the discriminated union so pydantic dispatches
        # on `type` and validates the type-specific fields.
        adapter = _source_adapter()
        return adapter.validate_python(payload)
    except ValidationError as err:
        raise ConfigError(f"source.{name} in {path} is invalid: {_format(err)}")


def _format(err: ValidationError) -> str:
    """Turn a pydantic ValidationError into a compact one-line summary.

    Pydantic surfaces the discriminator value (``"git-file"``,
    ``"github-release"``) as a prefix in ``loc``; strip that so the caller
    just sees the offending field name plus the error message.
    """
    parts = []
    for issue in err.errors():
        loc = [x for x in issue["loc"] if x not in ("git-file", "github-release")]
        loc_str = ".".join(str(x) for x in loc)
        msg = issue["msg"]
        parts.append(f"{loc_str}: {msg}" if loc_str else msg)
    return "; ".join(parts)


def _source_adapter():
    """Cache pydantic's TypeAdapter for the discriminated union."""
    global _ADAPTER
    if _ADAPTER is None:
        from pydantic import TypeAdapter

        _ADAPTER = TypeAdapter(SourceConfig)
    return _ADAPTER


_ADAPTER = None
