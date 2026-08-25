"""Turn a stream of file versions into the history artifact.

Walks one file's versions oldest-first, applies each to a
:class:`~obohog.obo.DocumentState` (which parses only the stanzas whose bytes
changed), and:

* writes a ``term_snapshots`` row for every term that changed at that commit;
* writes ``events`` rows for the clause-level adds/removes that changed it.

The first version is diffed against nothing, so every term's creation appears
as clause additions — a change from ∅ to its full clause set — and a removed
term emits removal events for its last known clauses; creation and removal
are therefore recoverable from the events alone. (For a window-bounded build
this dates pre-window content to the window's first commit.)
"""

import enum
import multiprocessing
import os
import re
import shutil
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq

from . import model
from .gitsource import CommitInfo, FileVersion, GitError, GitSource, TagRef
from .obo import Clause, CommitDelta, DocumentState, TermState

# Flush a worker's accumulated rows to a part-file every this many processed
# commits, so peak memory stays bounded regardless of history length.
_FLUSH_EVERY = 200

# GitHub PR # in a commit message. Two common shapes:
#   * squash-and-merge (post-2023 Mondo, most repos): "Title text (#1234)".
#   * classic merge commit (pre-2023 Mondo, PATO):
#     "Merge pull request #1234 from user/branch".
# The classic pattern is very specific (near-zero false positive) so we check
# it first; the parenthesized form can incidentally appear inside PR bodies
# that quote other PRs.
_PR_MERGE = re.compile(r"^Merge pull request #(\d+) from ")
_PR_SQUASH = re.compile(r"\(#(\d+)\)")
# Release-based providers stash the snapshot's canonical URL as a trailer
# in the commit message body ("Release URL: <url>"). Parsed out here so the
# render layer can link back without needing per-source-type wiring.
_SNAPSHOT_URL = re.compile(r"^Release URL: (\S+)", re.MULTILINE)


class BuildMode(enum.StrEnum):
    """How a build run related to what was already on disk."""

    FULL = "full"
    INCREMENTAL = "incremental"
    UP_TO_DATE = "up-to-date"


@dataclass(frozen=True)
class BuildReport:
    """What a build run did, for callers to report."""

    mode: BuildMode
    commits: int  # commits processed by this run
    total_commits: int  # commits the artifact covers afterwards
    snapshots: int
    events: int
    skipped: int = 0


def _extract_pr_number(message: str) -> int | None:
    m = _PR_MERGE.match(message)
    if m:
        return int(m.group(1))
    m = _PR_SQUASH.search(message)
    if m:
        return int(m.group(1))
    return None


def _extract_snapshot_url(message: str) -> str | None:
    m = _SNAPSHOT_URL.search(message)
    return m.group(1) if m else None


def extract(
    src: GitSource, path: str, out_dir: Path, *, limit: int | None = None
) -> BuildReport:
    """Build an artifact under ``out_dir`` from ``path``'s history in ``src``.

    ``limit`` keeps only the most recent ``limit`` versions — useful for
    iterating on a recent slice.
    """
    versions = list(src.iter_file_history(path))
    if limit is not None:
        versions = versions[-limit:]
    return build(versions, src.read_blob, out_dir, source_path=path, tags=src.read_tags())


def build(
    versions: Iterable[FileVersion],
    read_blob,
    out_dir: Path,
    *,
    source_path: str,
    tags: Iterable[TagRef] = (),
) -> BuildReport:
    """Serial in-process build: the parallel path minus chunking and workers."""
    commits: list[dict] = []
    snapshots: list[dict] = []
    events: list[dict] = []
    skipped: list[dict] = []

    state = DocumentState()
    seqs: list[int] = []
    seq_dates: list[tuple[int, object]] = []  # (seq, naive-UTC date) for tag mapping
    for version in versions:
        row = _commit_row(version.commit)
        commits.append(row)
        seqs.append(version.commit.seq)
        seq_dates.append((version.commit.seq, row["committed_date"]))
        delta = state.apply(read_blob(version.blob_oid))
        snap_rows, event_rows, skip_rows = _delta_rows(version, delta)
        snapshots.extend(snap_rows)
        events.extend(event_rows)
        skipped.extend(skip_rows)

    releases = _release_rows(tags, seq_dates)

    model.write_table(commits, model.COMMITS, out_dir, "commits")
    model.write_table(snapshots, model.TERM_SNAPSHOTS, out_dir, "term_snapshots")
    model.write_table(events, model.EVENTS, out_dir, "events")
    model.write_table(releases, model.RELEASES, out_dir, "releases")
    model.write_table(skipped, model.SKIPPED, out_dir, "skipped")
    _write_build_meta(
        Path(out_dir), source_path=source_path,
        first=seqs[0] if seqs else None, last=seqs[-1] if seqs else None,
        n=len(seqs),
    )

    return BuildReport(
        mode=BuildMode.FULL,
        commits=len(commits),
        total_commits=len(commits),
        snapshots=len(snapshots),
        events=len(events),
        skipped=len(skipped),
    )


def _delta_rows(
    version: FileVersion, delta: CommitDelta
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shape one commit's delta into (snapshot, event, skipped) row dicts."""
    snapshots = [_snapshot_row(version, d.term) for d in delta.changed]
    events: list[dict] = []
    for d in delta.changed:
        events.extend(_event_rows(version, d.term.term_id, d.added, model.Operation.ADD))
        events.extend(
            _event_rows(version, d.term.term_id, d.removed, model.Operation.REMOVE)
        )
    for term in delta.removed:
        events.extend(
            _event_rows(version, term.term_id, term.clauses, model.Operation.REMOVE)
        )
    skipped = [
        {
            "commit_seq": version.commit.seq,
            "sha": version.commit.sha,
            "term_id": term_id,
            "error": error,
        }
        for term_id, error in delta.failed
    ]
    return snapshots, events, skipped


def _release_rows(
    tags: Iterable[TagRef], seq_dates: list[tuple[int, object]]
) -> list[dict]:
    """Map each tag to the latest file-history commit at or before its date.

    A release's file state is whatever the last commit touching the file left it
    as of the tag; tags predating the window map to no commit and are dropped.
    """
    rows: list[dict] = []
    for tag in tags:
        tag_date = tag.date.astimezone(timezone.utc).replace(tzinfo=None)
        seq = None
        for candidate_seq, date in seq_dates:
            if date <= tag_date:
                seq = candidate_seq
            else:
                break
        if seq is None:
            continue
        rows.append(
            {"tag": tag.name, "sha": tag.sha, "date": tag_date, "commit_seq": seq}
        )
    return rows


def _commit_row(commit: CommitInfo) -> dict:
    snapshot_url = _extract_snapshot_url(commit.message)
    # Release-based commits carry the release page URL as their identity;
    # any `(#N)` or `Merge pull request #N` in the release body is just an
    # incidental reference, not a claim about which PR produced this
    # snapshot. Suppress pr_number on those commits so `obohog pr <N>`
    # doesn't attribute release terms to unrelated PRs.
    pr_number = None if snapshot_url else _extract_pr_number(commit.message)
    return {
        "commit_seq": commit.seq,
        "sha": commit.sha,
        "author_name": commit.author_name,
        "author_email": commit.author_email,
        "committed_date": commit.committed_date.astimezone(timezone.utc).replace(tzinfo=None),
        "message": commit.message,
        "pr_number": pr_number,
        "parent_sha": commit.parent_sha,
        "branch_commits": [
            {
                "sha": bc.sha,
                "author_name": bc.author_name,
                "committed_date": bc.committed_date.astimezone(timezone.utc).replace(tzinfo=None),
                "message": bc.message,
            }
            for bc in commit.branch_commits
        ],
        "snapshot_url": snapshot_url,
    }


def _snapshot_row(version: FileVersion, term: TermState) -> dict:
    name = next((c.value for c in term.clauses if c.predicate == "name"), None)
    is_obsolete = any(
        c.predicate == "is_obsolete" and c.value == "true" for c in term.clauses
    )
    return {
        "term_id": term.term_id,
        "commit_seq": version.commit.seq,
        "sha": version.commit.sha,
        "name": name,
        "is_obsolete": is_obsolete,
        "content_hash": term.content_hash,
        "clauses": [{"predicate": c.predicate, "value": c.value} for c in term.clauses],
    }


def _event_rows(
    version: FileVersion,
    term_id: str,
    clauses: Iterable[Clause],
    operation: model.Operation,
) -> list[dict]:
    return [
        {
            "term_id": term_id,
            "commit_seq": version.commit.seq,
            "sha": version.commit.sha,
            "predicate": clause.predicate,
            "value": clause.value,
            "operation": str(operation),
            "body": clause.parsed.body,
            "qualifiers": list(clause.parsed.qualifiers),
            "comment": clause.parsed.comment,
        }
        for clause in clauses
    ]


def _version() -> str:
    from . import __version__

    return __version__


def _write_build_meta(
    out: Path, *, source_path: str, first: int | None, last: int | None, n: int
) -> None:
    """Record what the artifact now covers. ALWAYS the final write of a build.

    ``plan_build`` trusts this row to describe what's on disk, and for an
    in-place update it is the commit point: an increment that dies before
    this write leaves a consistent, merely stale artifact (its orphaned
    ``inc-*`` parts are cleaned up by prefix on the next run, and its extra
    ``commits``/``skipped`` rows are truncated by ``_validate_resume``).
    """
    row = {
        "schema_version": model.SCHEMA_VERSION,
        "generator_version": _version(),
        "source_path": source_path,
        "first_commit_seq": first,
        "last_commit_seq": last,
        "n_commits": n,
    }
    model.write_table([row], model.BUILD_META, out, "build_meta")


# --- parallel, streaming build over a local clone -----------------------

def build_parallel(
    clone_path: str,
    obo_path: str,
    out_dir: Path,
    *,
    jobs: int | None = None,
    chunk_size: int | None = None,
    limit: int | None = None,
    progress: bool = False,
    update: bool = False,
) -> BuildReport:
    """Build the artifact from a local clone using a pool of parsing workers.

    The commit range is split into contiguous chunks (one per worker). Each
    worker parses its own commits — plus one seed commit from the previous chunk
    so boundary diffs are correct — and streams ``term_snapshots`` and ``events``
    to per-chunk Parquet part-files. The parent writes ``commits``, ``releases``,
    ``skipped_commits`` and ``build_meta`` directly (no parsing needed).

    With ``update=True``, an existing artifact is extended in place: only the
    commits after its recorded ``last_commit_seq`` are parsed and appended (see
    :func:`_build_update`). Falls back to a full rebuild whenever appending
    isn't safe — no artifact, different schema, a ``limit`` bound, or a walk
    that no longer matches what was built (history rewrite). The report's
    ``mode`` says which of the three actually ran.

    Runs strictly offline: blobs must already be present in ``clone_path``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # The parent's `iter_file_history` uses `git log --follow`, which fires
    # rename detection and can need blobs for *former* paths of the tracked
    # file — outside the sparse cone the backfill was scoped to. Let the
    # parent lazy-fetch those (small: a handful of blobs at most), then
    # disable lazy fetching before spawning workers so they can't race on
    # parallel fetches.
    src = GitSource(clone_path)
    full = list(src.iter_file_history(obo_path))
    tags = src.read_tags()
    src.close()

    os.environ["GIT_NO_LAZY_FETCH"] = "1"

    plan = plan_build(out, full, limit=limit, update=update)
    if plan.resume is not None:
        _clear_aborted_parts(out, plan.resume.last)
    if plan.mode is BuildMode.UP_TO_DATE:
        return _refresh_releases(out, tags, plan.resume)
    if plan.mode is BuildMode.INCREMENTAL:
        return _build_update(
            clone_path, obo_path, out, full, tags, plan.resume,
            jobs=jobs, chunk_size=chunk_size, progress=progress,
        )
    return _build_full(
        clone_path, obo_path, out, full[plan.offset:], tags,
        jobs=jobs, chunk_size=chunk_size, progress=progress,
    )


class _ResumePlan(NamedTuple):
    """A validated go-ahead for appending to an existing artifact."""

    last: int  # index in the walk (== commit_seq) of the last built commit
    commit_rows: list[dict]  # the artifact's commits table, up to ``last``
    meta: dict  # the artifact's current build_meta row


@dataclass(frozen=True)
class BuildPlan:
    """A declarative decision about how a build run treats the artifact."""

    mode: BuildMode
    offset: int = 0  # FULL: window start in the walk (from ``limit``)
    resume: _ResumePlan | None = None  # INCREMENTAL / UP_TO_DATE only


def plan_build(
    out: Path, full: list, *, limit: int | None, update: bool
) -> BuildPlan:
    """Decide how a run over the walk ``full`` should treat the artifact at ``out``.

    Appending requires ``update`` with no ``limit`` window, and an artifact
    whose recorded state still matches the walk (see :func:`_validate_resume`);
    anything else falls back to a full rebuild. Only reads artifact metadata —
    the decision itself is pure.
    """
    if update and limit is None:
        meta = model.read_build_meta(out)
        commit_rows = (
            pq.read_table(out / "commits.parquet").to_pylist()
            if (out / "commits.parquet").exists()
            else []
        )
        resume = _validate_resume(meta, commit_rows, full)
        if resume is not None:
            mode = (
                BuildMode.UP_TO_DATE
                if resume.last == len(full) - 1
                else BuildMode.INCREMENTAL
            )
            return BuildPlan(mode=mode, resume=resume)
    offset = 0 if limit is None else max(0, len(full) - limit)
    return BuildPlan(mode=BuildMode.FULL, offset=offset)


def _validate_resume(
    meta: dict | None, commit_rows: list[dict], full: list
) -> _ResumePlan | None:
    """Check that an artifact with this metadata can be extended by this walk.

    None means "can't append" — no artifact, a different schema, or a walk
    that no longer matches what was built (history rewrite, changed clone
    bounds, tracked-file rename changing the ``--follow`` resolution). The
    check is positional: the artifact's last built commit must sit at index
    ``last_commit_seq`` of the new walk with the same sha, which holds
    exactly when the previously built prefix is unchanged.
    """
    if meta is None or meta["schema_version"] != model.SCHEMA_VERSION:
        return None
    last = meta["last_commit_seq"]
    if last is None or last >= len(full):
        return None
    last_sha = next((r["sha"] for r in commit_rows if r["commit_seq"] == last), None)
    if last_sha is None or full[last].commit.sha != last_sha:
        return None
    # Truncate to what build_meta actually recorded, in case a prior increment
    # died between its table writes and its build_meta write.
    kept = [r for r in commit_rows if r["commit_seq"] <= last]
    return _ResumePlan(last=last, commit_rows=kept, meta=meta)


def _refresh_releases(
    out: Path, tags: Iterable[TagRef], resume: _ResumePlan
) -> BuildReport:
    """The up-to-date case: no new file commits, but new release *tags* may
    still have appeared (they rarely touch the tracked file), so remap the
    releases table before reporting."""
    seq_dates = [(r["commit_seq"], r["committed_date"]) for r in resume.commit_rows]
    model.write_table(_release_rows(tags, seq_dates), model.RELEASES, out, "releases")
    return BuildReport(
        mode=BuildMode.UP_TO_DATE, commits=0,
        total_commits=len(resume.commit_rows), snapshots=0, events=0,
    )


def _build_full(
    clone_path: str,
    obo_path: str,
    out: Path,
    windowed: list[FileVersion],
    tags: Iterable[TagRef],
    *,
    jobs: int | None,
    chunk_size: int | None,
    progress: bool,
) -> BuildReport:
    """Build the artifact from scratch over ``windowed``."""
    _reset_part_tables(out)
    n = len(windowed)
    jobs, chunks = _plan_chunks(n, jobs, chunk_size)

    # Parent-written tables (derived from commit metadata alone).
    commit_rows = [_commit_row(v.commit) for v in windowed]
    model.write_table(commit_rows, model.COMMITS, out, "commits")
    seq_dates = [(r["commit_seq"], r["committed_date"]) for r in commit_rows]
    model.write_table(_release_rows(tags, seq_dates), model.RELEASES, out, "releases")

    results = _run_chunks(
        clone_path, windowed, out, chunks, jobs=jobs, progress=progress, total=n,
    )

    # Guarantee the core tables exist even if this (degenerate) build produced no
    # part-files, so queries never hit a missing table.
    for name, schema in (("term_snapshots", model.TERM_SNAPSHOTS), ("events", model.EVENTS)):
        if not (out / name).is_dir():
            model.write_table([], schema, out, name)

    skipped = [row for r in results for row in r["skipped"]]
    model.write_table(skipped, model.SKIPPED, out, "skipped")
    _write_build_meta(
        out, source_path=obo_path,
        first=windowed[0].commit.seq if windowed else None,
        last=windowed[-1].commit.seq if windowed else None,
        n=n,
    )

    return BuildReport(
        mode=BuildMode.FULL,
        commits=n,
        total_commits=n,
        snapshots=sum(r["snapshots"] for r in results),
        events=sum(r["events"] for r in results),
        skipped=len(skipped),
    )


def _build_update(
    clone_path: str,
    obo_path: str,
    out: Path,
    full: list,
    tags: Iterable[TagRef],
    resume: _ResumePlan,
    *,
    jobs: int | None,
    chunk_size: int | None,
    progress: bool,
) -> BuildReport:
    """Append the walk's commits after ``resume.last`` to an existing artifact.

    ``build_meta`` is the commit point and is written last: everything else is
    either derivable-and-truncatable (``commits``/``skipped`` rows beyond the
    recorded ``last_commit_seq`` are dropped before appending) or cleaned up by
    prefix (``inc-*`` part-files from an increment that never reached its
    ``build_meta`` write). An aborted increment therefore leaves a consistent,
    merely stale artifact behind.
    """
    last = resume.last
    old_commits = resume.commit_rows
    new_versions = full[last + 1:]
    n = len(new_versions)
    _adopt_single_file_tables(out)

    # Workers get the walk from the last built commit on: index 0 is the seed
    # (already-built state to diff the first new commit against), so chunk
    # bounds over the n new commits shift up by one.
    tail = full[last:]
    jobs, chunks = _plan_chunks(n, jobs, chunk_size)
    chunks = [Chunk(c.id, c.start + 1, c.end + 1) for c in chunks]
    results = _run_chunks(
        clone_path, tail, out, chunks,
        jobs=jobs, progress=progress, total=n, prefix=f"inc-{last + 1:07d}-",
    )

    new_commit_rows = [_commit_row(v.commit) for v in new_versions]
    all_commits = old_commits + new_commit_rows
    model.write_table(all_commits, model.COMMITS, out, "commits")
    seq_dates = [(r["commit_seq"], r["committed_date"]) for r in all_commits]
    model.write_table(_release_rows(tags, seq_dates), model.RELEASES, out, "releases")

    old_skipped: list[dict] = []
    if (out / "skipped.parquet").exists():
        old_skipped = [
            r for r in pq.read_table(out / "skipped.parquet").to_pylist()
            if r["commit_seq"] <= last
        ]
    new_skipped = [row for r in results for row in r["skipped"]]
    model.write_table(old_skipped + new_skipped, model.SKIPPED, out, "skipped")

    _write_build_meta(
        out, source_path=obo_path,
        first=resume.meta["first_commit_seq"],
        last=full[-1].commit.seq,
        n=len(all_commits),
    )

    return BuildReport(
        mode=BuildMode.INCREMENTAL,
        commits=n,
        total_commits=len(all_commits),
        snapshots=sum(r["snapshots"] for r in results),
        events=sum(r["events"] for r in results),
        skipped=len(new_skipped),
    )


def _reset_part_tables(out: Path) -> None:
    """Full-rebuild reset: drop the snapshot/event tables in both layouts.

    Workers append numbered part-files that a glob would union, so stale
    files from an aborted or earlier run must not survive.
    """
    for name in ("term_snapshots", "events"):
        shutil.rmtree(out / name, ignore_errors=True)
        (out / f"{name}.parquet").unlink(missing_ok=True)


def _adopt_single_file_tables(out: Path) -> None:
    """Migrate a serial artifact's single-file tables into part-file dirs.

    Appending writes part-files; without this, a leftover single file would
    shadow (or be shadowed by) the directory when queries glob the table.
    """
    for name in ("term_snapshots", "events"):
        single = out / f"{name}.parquet"
        if single.exists():
            (out / name).mkdir(exist_ok=True)
            single.rename(out / name / "base.parquet")


def _clear_aborted_parts(out: Path, last_recorded: int) -> None:
    """Delete ``inc-*`` part-files from increments never recorded in build_meta.

    A recorded increment always has its starting seq covered by
    ``last_commit_seq``; a part-file starting beyond it can only come from an
    increment that died before its ``build_meta`` write.
    """
    for name in ("term_snapshots", "events"):
        directory = out / name
        if not directory.is_dir():
            continue
        for part in directory.glob("inc-*.parquet"):
            try:
                first_seq = int(part.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            if first_seq > last_recorded:
                part.unlink()


class Chunk(NamedTuple):
    """A contiguous span of walk indices for one worker task."""

    id: int
    start: int
    end: int


def _plan_chunks(
    n: int, jobs: int | None, chunk_size: int | None
) -> tuple[int, list[Chunk]]:
    """Resolve the worker count and chunk spans for ``n`` commits."""
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    # More chunks than workers so the pool can load-balance dynamically (a worker
    # that finishes grabs the next queued chunk). Each chunk pays a one-parse seed
    # cost, so default to a handful per worker rather than one-per-commit.
    if chunk_size and chunk_size > 0:
        n_chunks = -(-n // chunk_size)  # ceil
    else:
        n_chunks = jobs * 4
    bounds = _chunk_bounds(n, max(1, min(n_chunks, n or 1)))
    return jobs, [Chunk(i, s, e) for i, (s, e) in enumerate(bounds)]


def _run_chunks(
    clone_path: str,
    versions: list[FileVersion],
    out: Path,
    chunks: list[Chunk],
    *,
    jobs: int,
    progress: bool,
    total: int,
    prefix: str = "",
) -> list[dict]:
    """Run ``_build_chunk`` over ``chunks`` in a spawn-based process pool."""
    # "spawn" (not fork): workers parse with fastobo's threaded runtime, and
    # fork() in a multi-threaded process risks deadlock.
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager() if progress else None
    ticks = manager.Queue() if manager else None  # workers report per-commit
    try:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            # Send the already-computed versions to each worker so they don't
            # each re-walk `git log --follow`. Pickle cost is small
            # (dataclasses of str/int/datetime) and pays for itself many times
            # over vs. per-worker subprocess overhead.
            futures = [
                pool.submit(
                    _build_chunk, clone_path, versions, str(out),
                    c.id, c.start, c.end, ticks, prefix,
                )
                for c in chunks
            ]
            if progress:
                _consume_ticks(futures, ticks, total)
            return [f.result() for f in futures]
    finally:
        if manager is not None:
            manager.shutdown()


def _consume_ticks(futures, ticks, total: int) -> None:
    """Drain per-commit ticks from workers into a single tqdm bar."""
    import queue as _queue

    from tqdm import tqdm

    seen = 0
    with tqdm(total=total, unit="commit", desc="building", smoothing=0.05) as bar:
        while seen < total:
            try:
                ticks.get(timeout=0.5)
                seen += 1
                bar.update(1)
            except _queue.Empty:
                if all(f.done() for f in futures):
                    break  # a worker finished/failed without emitting all ticks


def _chunk_bounds(n: int, k: int) -> list[tuple[int, int]]:
    """Split ``range(n)`` into ``k`` contiguous, balanced (start, end) spans."""
    k = max(1, min(k, n)) if n else 1
    base, rem = divmod(n, k)
    bounds, start = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def _build_chunk(
    clone_path: str,
    windowed: list[FileVersion],
    out_dir: str,
    chunk_id: int,
    start: int,
    end: int,
    ticks=None,
    prefix: str = "",
) -> dict:
    """Worker: apply ``windowed[start:end]`` to the document state, stream part-files.

    ``windowed`` is the pre-computed versions list from the parent process,
    so this worker doesn't re-walk `git log --follow` (which would repeat
    the expensive branch-commit resolution for every worker).
    """
    # fastobo prints Rust panics to stderr even though we catch them; a worker
    # has no other use for stderr (results and errors reach the parent via the
    # future), so silence it to keep the parent's progress bar clean.
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

    src = GitSource(clone_path)
    state = _seed_state(src, windowed, start)
    writer = _PartWriter(Path(out_dir), chunk_id, prefix)
    skipped: list[dict] = []

    for i in range(start, end):
        if ticks is not None:
            ticks.put(1)  # one tick per commit
        version = windowed[i]
        try:
            blob = src.read_blob(version.blob_oid)
        except GitError:
            # A blob absent from the (offline) clone can't be processed; skip the
            # commit and carry state forward rather than aborting the whole build.
            skipped.append(
                {"commit_seq": version.commit.seq, "sha": version.commit.sha,
                 "term_id": None, "error": "BlobMissing"}
            )
            continue
        snaps, events, skips = _delta_rows(version, state.apply(blob))
        writer.add(snaps, events)
        skipped.extend(skips)

    writer.close()
    src.close()
    return {
        "chunk": chunk_id, "snapshots": writer.n_snapshots,
        "events": writer.n_events, "skipped": skipped,
    }


class _PartWriter:
    """Streams one worker's snapshot/event rows to numbered part-files.

    Buffers rows and flushes every ``_FLUSH_EVERY`` commits so peak memory
    stays bounded regardless of history length. Part-files are named
    ``{prefix}{chunk:03d}-{batch:04d}.parquet`` — the batch counter keeps
    names unique across flushes, and the prefix is how incremental appends
    stay distinguishable (see ``_clear_aborted_parts``).
    """

    def __init__(self, out: Path, chunk_id: int, prefix: str) -> None:
        self._out = out
        self._chunk_id = chunk_id
        self._prefix = prefix
        self._snapshots: list[dict] = []
        self._events: list[dict] = []
        self._batch = 0
        self._commits_buffered = 0
        self.n_snapshots = 0
        self.n_events = 0

    def add(self, snapshots: list[dict], events: list[dict]) -> None:
        """Buffer one commit's rows, flushing if the batch is due."""
        self._snapshots.extend(snapshots)
        self._events.extend(events)
        self.n_snapshots += len(snapshots)
        self.n_events += len(events)
        self._commits_buffered += 1
        if self._commits_buffered >= _FLUSH_EVERY:
            self._flush()

    def close(self) -> None:
        self._flush()

    def _flush(self) -> None:
        name = f"{self._prefix}{self._chunk_id:03d}-{self._batch:04d}.parquet"
        if self._snapshots:
            model.write_part(
                self._snapshots, model.TERM_SNAPSHOTS, self._out / "term_snapshots" / name
            )
        if self._events:
            model.write_part(self._events, model.EVENTS, self._out / "events" / name)
        self._snapshots, self._events = [], []
        self._batch += 1
        self._commits_buffered = 0


def _seed_state(
    src: GitSource, windowed: list[FileVersion], start: int
) -> DocumentState:
    """Document state as of the version before ``start``.

    Empty for the first chunk (``start == 0``): its first version diffs
    against nothing, so every term appears as created.
    """
    if start == 0:
        return DocumentState()
    try:
        blob = src.read_blob(windowed[start - 1].blob_oid)
    except GitError:
        return DocumentState()  # missing seed blob → first diff treats all as new
    return DocumentState.from_blob(blob)
