"""Typed vocabulary and Parquet schemas for the history artifact.

The artifact is a small set of Parquet files sharing ``commit_seq`` as their time
axis: ``commits`` (git metadata), ``term_snapshots`` (a term's full normalized
state, written only where it changed), and ``events`` (clause-level add/remove
deltas derived by diffing adjacent snapshots). ``build_meta`` records provenance.
"""

import enum
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pyarrow as pa
import pyarrow.parquet as pq

# Bump whenever the artifact schema changes shape, so build_meta records
# which schema an artifact was built with.
#   1 — initial schema
#   2 — events gained body/qualifiers/comment decomposition columns
SCHEMA_VERSION = "2"


class Operation(enum.StrEnum):
    ADD = "add"
    REMOVE = "remove"


# Row shapes as TypedDicts, one per Parquet schema below. Rows stay plain
# dicts at runtime (pyarrow's ``Table.from_pylist`` consumes dicts, and the
# events path builds millions of them), but producers annotate with these so
# a type checker can verify construction — and
# ``test_row_typeddicts_match_schemas`` locks each declaration's keys to its
# ``pa.schema``, so the two can't drift.


class ClauseRow(TypedDict):
    predicate: str
    value: str


class BranchCommitRow(TypedDict):
    sha: str
    author_name: str
    committed_date: datetime
    message: str


class CommitRow(TypedDict):
    commit_seq: int
    sha: str
    author_name: str
    author_email: str
    committed_date: datetime
    message: str
    pr_number: int | None
    parent_sha: str | None
    branch_commits: list[BranchCommitRow]
    snapshot_url: str | None


class SnapshotRow(TypedDict):
    term_id: str
    commit_seq: int
    sha: str
    name: str | None
    is_obsolete: bool
    content_hash: str
    clauses: list[ClauseRow]


class EventRow(TypedDict):
    term_id: str
    commit_seq: int
    sha: str
    predicate: str
    value: str
    operation: str
    body: str
    qualifiers: list[str]
    comment: str | None


class ReleaseRow(TypedDict):
    tag: str
    sha: str
    date: datetime
    commit_seq: int


class SkipRow(TypedDict):
    commit_seq: int
    sha: str
    term_id: str | None
    error: str


_CLAUSE = pa.struct([("predicate", pa.string()), ("value", pa.string())])

# For merge commits (typically GitHub PR merges), the sequence of commits on
# the merged branch that landed as part of this merge. Each entry is
# (sha, author_name, committed_date, message) for one branch commit. Empty
# for non-merge commits. Newest-first, matching what a user sees on GitHub.
_BRANCH_COMMIT = pa.struct([
    ("sha", pa.string()),
    ("author_name", pa.string()),
    ("committed_date", pa.timestamp("us")),
    ("message", pa.string()),
])

COMMITS = pa.schema(
    [
        ("commit_seq", pa.int32()),
        ("sha", pa.string()),
        ("author_name", pa.string()),
        ("author_email", pa.string()),
        ("committed_date", pa.timestamp("us")),  # naive UTC
        ("message", pa.string()),
        ("pr_number", pa.int32()),  # nullable
        ("parent_sha", pa.string()),  # nullable
        ("branch_commits", pa.list_(_BRANCH_COMMIT)),
        # External URL for the snapshot this commit represents. Populated
        # by release-based providers (e.g. GitHubReleaseProvider stashes the
        # release page URL). NULL for git-file sources, where "the URL" is
        # nominally the commit page but we don't invent one.
        ("snapshot_url", pa.string()),  # nullable
    ]
)

TERM_SNAPSHOTS = pa.schema(
    [
        ("term_id", pa.string()),
        ("commit_seq", pa.int32()),
        ("sha", pa.string()),
        ("name", pa.string()),  # nullable convenience column
        ("is_obsolete", pa.bool_()),
        ("content_hash", pa.string()),
        ("clauses", pa.list_(_CLAUSE)),
    ]
)

EVENTS = pa.schema(
    [
        ("term_id", pa.string()),
        ("commit_seq", pa.int32()),
        ("sha", pa.string()),
        ("predicate", pa.string()),
        ("value", pa.string()),
        ("operation", pa.string()),  # Operation value
        # Structural decomposition of `value`, captured at parse time from
        # the fastobo clause object (see obo.decompose_clause). Invariant:
        # body + " {qualifiers}" + " ! comment" == value, byte for byte.
        # `value` stays stored as the primitive record; these columns let
        # query/render skip fastobo entirely.
        ("body", pa.string()),
        ("qualifiers", pa.list_(pa.string())),
        ("comment", pa.string()),  # nullable: absent trailing `!` comment
    ]
)

RELEASES = pa.schema(
    [
        ("tag", pa.string()),
        ("sha", pa.string()),
        ("date", pa.timestamp("us")),  # naive UTC
        ("commit_seq", pa.int32()),  # file-history commit at/before the tag
    ]
)

SKIPPED = pa.schema(
    [
        ("commit_seq", pa.int32()),
        ("sha", pa.string()),
        ("term_id", pa.string()),  # the single term whose stanza failed to parse
        ("error", pa.string()),
    ]
)

BUILD_META = pa.schema(
    [
        ("schema_version", pa.string()),
        ("generator_version", pa.string()),
        ("source_path", pa.string()),
        ("first_commit_seq", pa.int32()),
        ("last_commit_seq", pa.int32()),
        ("n_commits", pa.int32()),
        # JSON object, sorted keys ({"TBD": "MONDO"}); None when no mapping.
        # Part of build identity: a changed map forces a full rebuild.
        ("namespace_map", pa.string()),
        # Identity of the OWL→OBO conversion the blobs went through ("obo"
        # for untranslated sources; see obohog.convert). Part of build
        # identity: a changed converter (e.g. a ROBOT upgrade that reorders
        # output) forces a full rebuild rather than phantom-diffing appends.
        ("converter_id", pa.string()),
    ]
)

# Filenames within an artifact directory.
FILES = {
    "commits": COMMITS,
    "term_snapshots": TERM_SNAPSHOTS,
    "events": EVENTS,
    "releases": RELEASES,
    "skipped": SKIPPED,
    "build_meta": BUILD_META,
}


@dataclass(frozen=True)
class BuildMeta:
    """The artifact's ``build_meta`` row, typed: what's on disk and how it
    was built. The compatibility gate (``schema_version``) and the
    incremental-sync resume point (``last_commit_seq``) both live here."""

    schema_version: str
    generator_version: str
    source_path: str
    first_commit_seq: int | None
    last_commit_seq: int | None
    n_commits: int
    namespace_map: str | None = None  # JSON, sorted keys; None = no mapping
    converter_id: str = "obo"  # see obohog.convert; "obo" = no conversion


def read_build_meta(artifact_dir: Path | str) -> BuildMeta | None:
    """Return the artifact's single ``build_meta`` row, or None if absent.

    One tiny row — read it with pyarrow directly rather than spinning up a
    DuckDB connection.
    """
    path = Path(artifact_dir) / "build_meta.parquet"
    if not path.exists():
        return None
    rows = pq.read_table(path).to_pylist()
    if not rows:
        return None
    row = rows[0]
    return BuildMeta(
        schema_version=row["schema_version"],
        generator_version=row["generator_version"],
        source_path=row["source_path"],
        first_commit_seq=row["first_commit_seq"],
        last_commit_seq=row["last_commit_seq"],
        n_commits=row["n_commits"],
        # Absent in artifacts built before namespace mapping existed.
        namespace_map=row.get("namespace_map"),
        # Absent in artifacts built before OWL conversion existed — all
        # of which were unconverted OBO.
        converter_id=row.get("converter_id") or "obo",
    )


def write_table(rows: list[dict], schema: pa.Schema, out_dir: Path, name: str) -> Path:
    """Write ``rows`` as ``<out_dir>/<name>.parquet`` using ``schema``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    write_part(rows, schema, path)
    return path


def write_part(rows: list[dict], schema: pa.Schema, path: Path) -> Path:
    """Write ``rows`` to a single Parquet file at ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")
    return path
