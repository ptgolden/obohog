"""Tests for obohog.toml loading and source resolution."""

from pathlib import Path

import pytest

from obohog.config import (
    BioPortalSource,
    Config,
    ConfigError,
    GitFileSource,
    GitHubReleaseSource,
    load_config,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_load_git_file_source(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        [source.mondo]
        type = "git-file"
        repo = "https://github.com/monarch-initiative/mondo"
        file = "src/ontology/mondo-edit.obo"
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.path == cfg_path
    assert cfg.storage == Path("data")
    assert set(cfg.sources) == {"mondo"}
    source = cfg.sources["mondo"]
    assert isinstance(source, GitFileSource)
    assert source.name == "mondo"
    assert source.repo == "https://github.com/monarch-initiative/mondo"
    assert source.file == "src/ontology/mondo-edit.obo"
    # tracked_path abstracts over source type — for git-file it's `file`.
    assert source.tracked_path == "src/ontology/mondo-edit.obo"
    # Default per-source path convention: {storage}/{name}/{clone,db}.
    assert source.clone_dir == Path("data/mondo/clone")
    assert source.db_dir == Path("data/mondo/db")


def test_load_bioportal_source(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        [source.exo]
        type = "bioportal"
        acronym = "EXO"
        """,
    )
    cfg = load_config(cfg_path)
    source = cfg.sources["exo"]
    assert isinstance(source, BioPortalSource)
    assert source.acronym == "EXO"
    # Materializer writes to <acronym>.obo in the synthetic clone.
    assert source.tracked_path == "EXO.obo"
    # No `repo` field on bioportal sources; display uses the acronym.
    assert source.source_display == "bioportal:EXO"
    assert not hasattr(source, "repo")


def test_bioportal_source_requires_acronym(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        [source.exo]
        type = "bioportal"
        """,
    )
    with pytest.raises(ConfigError, match=r"source\.exo .* acronym"):
        load_config(cfg_path)


def test_load_github_release_source(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        [source.zp]
        type = "github-release"
        repo = "https://github.com/obophenotype/zebrafish-phenotype-ontology"
        asset = "zp-base.obo"
        """,
    )
    cfg = load_config(cfg_path)
    source = cfg.sources["zp"]
    assert isinstance(source, GitHubReleaseSource)
    assert source.asset == "zp-base.obo"
    # tracked_path abstracts over source type — for github-release it's `asset`.
    assert source.tracked_path == "zp-base.obo"


def test_explicit_storage_and_paths(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        storage = "/tmp/obohog-data"

        [source.pato]
        type = "git-file"
        repo = "https://github.com/pato-ontology/pato"
        file = "src/ontology/pato-edit.obo"
        clone_dir = "/big/disk/pato/clone"
        db_dir = "/big/disk/pato/db"
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.storage == Path("/tmp/obohog-data")
    pato = cfg.sources["pato"]
    assert pato.clone_dir == Path("/big/disk/pato/clone")
    assert pato.db_dir == Path("/big/disk/pato/db")


def test_multiple_mixed_sources(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        [source.mondo]
        type = "git-file"
        repo = "https://github.com/monarch-initiative/mondo"
        file = "src/ontology/mondo-edit.obo"

        [source.zp]
        type = "github-release"
        repo = "https://github.com/obophenotype/zebrafish-phenotype-ontology"
        asset = "zp-base.obo"
        """,
    )
    cfg = load_config(cfg_path)
    assert set(cfg.sources) == {"mondo", "zp"}
    assert isinstance(cfg.sources["mondo"], GitFileSource)
    assert isinstance(cfg.sources["zp"], GitHubReleaseSource)


def test_missing_config_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="No obohog config found"):
        load_config(tmp_path / "does-not-exist.toml")


def test_malformed_toml(tmp_path: Path):
    cfg_path = _write(tmp_path / "obohog.toml", "[source.mondo\nunterminated")
    with pytest.raises(ConfigError, match="Malformed TOML"):
        load_config(cfg_path)


def test_git_file_source_requires_file(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        [source.mondo]
        type = "git-file"
        repo = "https://example/mondo"
        """,
    )
    with pytest.raises(ConfigError, match=r"source\.mondo .* file"):
        load_config(cfg_path)


def test_github_release_source_requires_asset(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        [source.zp]
        type = "github-release"
        repo = "https://example/zp"
        """,
    )
    with pytest.raises(ConfigError, match=r"source\.zp .* asset"):
        load_config(cfg_path)


def test_unknown_source_type(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        [source.zenodo_sample]
        type = "zenodo-record"
        repo = "https://zenodo.org/record/12345"
        """,
    )
    with pytest.raises(ConfigError, match="source.zenodo_sample"):
        load_config(cfg_path)


def test_missing_type_field(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        [source.mondo]
        repo = "https://example/mondo"
        file = "a.obo"
        """,
    )
    with pytest.raises(ConfigError, match=r"source\.mondo.*'type'"):
        load_config(cfg_path)


def test_get_source_unknown_lists_available(tmp_path: Path):
    cfg_path = _write(
        tmp_path / "obohog.toml",
        """
        [source.mondo]
        type = "git-file"
        repo = "https://example/mondo"
        file = "a.obo"

        [source.pato]
        type = "git-file"
        repo = "https://example/pato"
        file = "b.obo"
        """,
    )
    cfg = load_config(cfg_path)
    with pytest.raises(ConfigError, match=r"No source named 'go' .* Available sources: mondo, pato"):
        cfg.get_source("go")


def test_get_source_empty_config_error_message(tmp_path: Path):
    cfg_path = _write(tmp_path / "obohog.toml", "")
    cfg = load_config(cfg_path)
    with pytest.raises(ConfigError, match=r"Available sources: \(none\)"):
        cfg.get_source("mondo")
