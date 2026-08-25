"""Turn a stream of file versions into the history artifact.

Walks one file's versions oldest-first, parses each into per-term state, and:

* writes a ``term_snapshots`` row for every term that changed at that commit;
* writes ``events`` rows for the clause-level adds/removes that changed it.

The first version in the stream is a **baseline**: every term is snapshotted but
no events are emitted, because a term's clauses were added *before* the window and
dating those additions to the window's start would be a lie. Terms that first
appear *after* the baseline are diffed against nothing, so their creation shows up
as clause additions. Term creation/removal times are recoverable from snapshot
presence, so they need no dedicated event kind.
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
from .obo import (
    Clause,
    TermState,
    clause_delta,
    parse_stanzas,
    parse_terms,
    split_document,
    stanza_hash,
)

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
_EMPTY: tuple[Clause, ...] = ()


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

    ``limit`` keeps only the most recent ``limit`` versions (the oldest kept one
    becomes the baseline) — useful for iterating on a recent slice.
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
    commits: list[dict] = []
    snapshots: list[dict] = []
    events: list[dict] = []

    prev: dict[str, TermState] = {}
    seqs: list[int] = []
    seq_dates: list[tuple[int, object]] = []  # (seq, naive-UTC date) for tag mapping
    for i, version in enumerate(versions):
        current = parse_terms(read_blob(version.blob_oid))
        row = _commit_row(version.commit)
        commits.append(row)
        seqs.append(version.commit.seq)
        seq_dates.append((version.commit.seq, row["committed_date"]))

        # At the first commit `prev` is empty, so every term's `before` is
        # None and the delta falls out as "all clauses added". That's the
        # right story — a term's creation is a change from ∅ to its full
        # clause set, and we want that visible in the events table so the
        # timeline is complete.
        for term_id, term in current.items():
            before = prev.get(term_id)
            if before is not None and before.content_hash == term.content_hash:
                continue
            snapshots.append(_snapshot_row(version, term))
            added, removed = clause_delta(
                before.clauses if before else _EMPTY, term.clauses
            )
            events.extend(_event_rows(version, term_id, added, model.Operation.ADD))
            events.extend(_event_rows(version, term_id, removed, model.Operation.REMOVE))
        for term_id in prev.keys() - current.keys():
            events.extend(
                _event_rows(version, term_id, prev[term_id].clauses, model.Operation.REMOVE)
            )
        prev = current

    meta = [
        {
            "schema_version": model.SCHEMA_VERSION,
            "generator_version": _version(),
            "source_path": source_path,
            "first_commit_seq": seqs[0] if seqs else None,
            "last_commit_seq": seqs[-1] if seqs else None,
            "n_commits": len(seqs),
        }
    ]

    releases = _release_rows(tags, seq_dates)

    model.write_table(commits, model.COMMITS, out_dir, "commits")
    model.write_table(snapshots, model.TERM_SNAPSHOTS, out_dir, "term_snapshots")
    model.write_table(events, model.EVENTS, out_dir, "events")
    model.write_table(releases, model.RELEASES, out_dir, "releases")
    model.write_table(meta, model.BUILD_META, out_dir, "build_meta")

    return BuildReport(
        mode=BuildMode.FULL,
        commits=len(commits),
        total_commits=len(commits),
        snapshots=len(snapshots),
        events=len(events),
    )


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
    # Clear any prior part-files: workers append numbered files that a glob would
    # union, so stale files from an aborted or earlier run must not survive.
    for name in ("term_snapshots", "events"):
        shutil.rmtree(out / name, ignore_errors=True)
        (out / f"{name}.parquet").unlink(missing_ok=True)

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
    meta = [
        {
            "schema_version": model.SCHEMA_VERSION,
            "generator_version": _version(),
            "source_path": obo_path,
            "first_commit_seq": windowed[0].commit.seq if windowed else None,
            "last_commit_seq": windowed[-1].commit.seq if windowed else None,
            "n_commits": n,
        }
    ]
    model.write_table(meta, model.BUILD_META, out, "build_meta")

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

    # An artifact from the serial builder stores single files; move them into
    # the part-file directories so the appended parts union with them.
    for name in ("term_snapshots", "events"):
        single = out / f"{name}.parquet"
        if single.exists():
            (out / name).mkdir(exist_ok=True)
            single.rename(out / name / "base.parquet")

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

    meta = [
        {
            "schema_version": model.SCHEMA_VERSION,
            "generator_version": _version(),
            "source_path": obo_path,
            "first_commit_seq": resume.meta["first_commit_seq"],
            "last_commit_seq": full[-1].commit.seq,
            "n_commits": len(all_commits),
        }
    ]
    model.write_table(meta, model.BUILD_META, out, "build_meta")

    return BuildReport(
        mode=BuildMode.INCREMENTAL,
        commits=n,
        total_commits=len(all_commits),
        snapshots=sum(r["snapshots"] for r in results),
        events=sum(r["events"] for r in results),
        skipped=len(new_skipped),
    )


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
    """Worker: parse+diff ``windowed[start:end]`` and stream part-files.

    ``windowed`` is the pre-computed versions list from the parent process,
    so this worker doesn't re-walk `git log --follow` (which would repeat
    the expensive branch-commit resolution for every worker).
    """
    # fastobo prints Rust panics to stderr even though we catch them; a worker
    # has no other use for stderr (results and errors reach the parent via the
    # future), so silence it to keep the parent's progress bar clean.
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

    out = Path(out_dir)
    src = GitSource(clone_path)

    state, raw = _seed_state(src, windowed, start)
    snap_rows: list[dict] = []
    event_rows: list[dict] = []
    skipped: list[dict] = []
    n_snap = n_evt = batch = since_flush = 0

    def flush() -> None:
        nonlocal snap_rows, event_rows, batch
        if snap_rows:
            model.write_part(
                snap_rows, model.TERM_SNAPSHOTS,
                out / "term_snapshots" / f"{prefix}{chunk_id:03d}-{batch:04d}.parquet",
            )
        if event_rows:
            model.write_part(
                event_rows, model.EVENTS,
                out / "events" / f"{prefix}{chunk_id:03d}-{batch:04d}.parquet",
            )
        snap_rows, event_rows, batch = [], [], batch + 1

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
        # Split the file into stanzas by text (cheap) and hash each; only the
        # stanzas whose bytes changed are handed to fastobo.
        context, stanzas = split_document(blob)
        cur_hash = {mid: stanza_hash(s) for mid, s in stanzas.items()}

        # At the chunk-0 first commit `state` and `raw` are empty (the seed
        # returns empty when start == 0), so `changed` covers every term and
        # `before` falls out as None → every clause becomes an add event.
        # That's the right story: a term's creation is a change from ∅ to
        # its full clause set, and we want it visible in the events table.
        changed = [mid for mid in stanzas if cur_hash[mid] != raw.get(mid)]
        removed = raw.keys() - stanzas.keys()
        parsed, failed = parse_stanzas(context, {mid: stanzas[mid] for mid in changed})
        failed_set = set(failed)
        _record_skips(skipped, version, failed)
        for term_id in failed:
            # Mark the failing bytes as seen: keep the last good state and only
            # re-attempt if this stanza's content changes again (avoids
            # re-bisecting the same unparseable term at every later commit).
            raw[term_id] = cur_hash[term_id]
        for term_id in changed:
            if term_id in failed_set:
                continue
            term = parsed.get(term_id)
            if term is None:
                # Stanza parsed, but fastobo keyed it under a different id than
                # our text-level scan did; record and skip rather than crash.
                skipped.append(
                    {"commit_seq": version.commit.seq, "sha": version.commit.sha,
                     "term_id": term_id, "error": "IdMismatch"}
                )
                raw[term_id] = cur_hash[term_id]
                continue
            before = state.get(term_id)
            raw[term_id] = cur_hash[term_id]
            if before is not None and before.content_hash == term.content_hash:
                continue  # bytes changed but canonical content did not
            snap_rows.append(_snapshot_row(version, term))
            n_snap += 1
            added, gone = clause_delta(before.clauses if before else _EMPTY, term.clauses)
            event_rows.extend(_event_rows(version, term_id, added, model.Operation.ADD))
            event_rows.extend(_event_rows(version, term_id, gone, model.Operation.REMOVE))
            n_evt += len(added) + len(gone)
            state[term_id] = term
        for term_id in removed:
            del raw[term_id]
            term = state.pop(term_id, None)
            if term is None:
                continue  # only ever failed to parse; nothing was emitted to remove
            event_rows.extend(
                _event_rows(version, term_id, term.clauses, model.Operation.REMOVE)
            )
            n_evt += len(term.clauses)

        since_flush += 1
        if since_flush >= _FLUSH_EVERY:
            flush()
            since_flush = 0

    flush()
    src.close()
    return {"chunk": chunk_id, "snapshots": n_snap, "events": n_evt, "skipped": skipped}


def _seed_state(
    src: GitSource, windowed: list[FileVersion], start: int
) -> tuple[dict[str, TermState], dict[str, bytes]]:
    """Full state at the version before ``start``: parsed clauses + stanza hashes.

    Empty for the first chunk (``start == 0``), whose first version is the
    baseline. Per-stanza parse failures are isolated and simply omitted from the
    seed (they surface as skips when that term next changes).
    """
    state: dict[str, TermState] = {}
    raw: dict[str, bytes] = {}
    if start == 0:
        return state, raw
    try:
        blob = src.read_blob(windowed[start - 1].blob_oid)
    except GitError:
        return state, raw  # missing seed blob → empty seed (first diff treats new)
    context, stanzas = split_document(blob)
    parsed, _failed = parse_stanzas(context, stanzas)
    for term_id, term in parsed.items():
        state[term_id] = term
        raw[term_id] = stanza_hash(stanzas[term_id])
    return state, raw


def _record_skips(skipped: list[dict], version: FileVersion, failed: list[str]) -> None:
    for term_id in failed:
        skipped.append(
            {
                "commit_seq": version.commit.seq,
                "sha": version.commit.sha,
                "term_id": term_id,
                "error": "ParseError",
            }
        )
