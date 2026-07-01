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

import multiprocessing
import os
import re
import shutil
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import timezone
from pathlib import Path

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

_PR = re.compile(r"\(#(\d+)\)")
_EMPTY: tuple[Clause, ...] = ()


def extract(
    src: GitSource, path: str, out_dir: Path, *, limit: int | None = None
) -> dict[str, int]:
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
) -> dict[str, int]:
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

        if i == 0:
            for term in current.values():
                snapshots.append(_snapshot_row(version, term))
        else:
            for mondo_id, term in current.items():
                before = prev.get(mondo_id)
                if before is not None and before.content_hash == term.content_hash:
                    continue
                snapshots.append(_snapshot_row(version, term))
                added, removed = clause_delta(
                    before.clauses if before else _EMPTY, term.clauses
                )
                events.extend(_event_rows(version, mondo_id, added, model.Operation.ADD))
                events.extend(_event_rows(version, mondo_id, removed, model.Operation.REMOVE))
            for mondo_id in prev.keys() - current.keys():
                events.extend(
                    _event_rows(version, mondo_id, prev[mondo_id].clauses, model.Operation.REMOVE)
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

    return {
        "commits": len(commits),
        "snapshots": len(snapshots),
        "events": len(events),
        "releases": len(releases),
    }


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
    match = _PR.search(commit.message)
    return {
        "commit_seq": commit.seq,
        "sha": commit.sha,
        "author_name": commit.author_name,
        "author_email": commit.author_email,
        "committed_date": commit.committed_date.astimezone(timezone.utc).replace(tzinfo=None),
        "message": commit.message,
        "pr_number": int(match.group(1)) if match else None,
        "parent_sha": commit.parent_sha,
    }


def _snapshot_row(version: FileVersion, term: TermState) -> dict:
    name = next((c.value for c in term.clauses if c.predicate == "name"), None)
    is_obsolete = any(
        c.predicate == "is_obsolete" and c.value == "true" for c in term.clauses
    )
    return {
        "mondo_id": term.mondo_id,
        "commit_seq": version.commit.seq,
        "sha": version.commit.sha,
        "name": name,
        "is_obsolete": is_obsolete,
        "content_hash": term.content_hash,
        "clauses": [{"predicate": c.predicate, "value": c.value} for c in term.clauses],
    }


def _event_rows(
    version: FileVersion,
    mondo_id: str,
    clauses: Iterable[Clause],
    operation: model.Operation,
) -> list[dict]:
    return [
        {
            "mondo_id": mondo_id,
            "commit_seq": version.commit.seq,
            "sha": version.commit.sha,
            "predicate": clause.predicate,
            "value": clause.value,
            "operation": str(operation),
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
    limit: int | None = None,
    progress: bool = False,
) -> dict:
    """Build the artifact from a local clone using a pool of parsing workers.

    The commit range is split into contiguous chunks (one per worker). Each
    worker parses its own commits — plus one seed commit from the previous chunk
    so boundary diffs are correct — and streams ``term_snapshots`` and ``events``
    to per-chunk Parquet part-files. The parent writes ``commits``, ``releases``,
    ``skipped_commits`` and ``build_meta`` directly (no parsing needed).

    Runs strictly offline: blobs must already be present in ``clone_path``.
    """
    os.environ.setdefault("GIT_NO_LAZY_FETCH", "1")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Clear any prior part-files: workers append numbered files that a glob would
    # union, so stale files from an aborted or earlier run must not survive.
    for name in ("term_snapshots", "events"):
        shutil.rmtree(out / name, ignore_errors=True)
        (out / f"{name}.parquet").unlink(missing_ok=True)

    src = GitSource(clone_path)
    full = list(src.iter_file_history(obo_path))
    tags = src.read_tags()
    src.close()

    offset = 0 if limit is None else max(0, len(full) - limit)
    windowed = full[offset:]
    n = len(windowed)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    bounds = _chunk_bounds(n, jobs)

    # Parent-written tables (derived from commit metadata alone).
    commit_rows = [_commit_row(v.commit) for v in windowed]
    model.write_table(commit_rows, model.COMMITS, out, "commits")
    seq_dates = [(r["commit_seq"], r["committed_date"]) for r in commit_rows]
    model.write_table(_release_rows(tags, seq_dates), model.RELEASES, out, "releases")

    # "spawn" (not fork): workers parse with fastobo's threaded runtime, and
    # fork() in a multi-threaded process risks deadlock.
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager() if progress else None
    ticks = manager.Queue() if manager else None  # workers report per-commit
    try:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            futures = [
                pool.submit(_build_chunk, clone_path, obo_path, str(out), offset, i, s, e, ticks)
                for i, (s, e) in enumerate(bounds)
            ]
            if progress:
                _consume_ticks(futures, ticks, n)
            results = [f.result() for f in futures]
    finally:
        if manager is not None:
            manager.shutdown()

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

    return {
        "commits": n,
        "snapshots": sum(r["snapshots"] for r in results),
        "events": sum(r["events"] for r in results),
        "skipped": len(skipped),
    }


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
    obo_path: str,
    out_dir: str,
    offset: int,
    chunk_id: int,
    start: int,
    end: int,
    ticks=None,
) -> dict:
    """Worker: parse+diff ``windowed[start:end]`` and stream part-files."""
    # fastobo prints Rust panics to stderr even though we catch them; a worker
    # has no other use for stderr (results and errors reach the parent via the
    # future), so silence it to keep the parent's progress bar clean.
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

    out = Path(out_dir)
    src = GitSource(clone_path)
    windowed = list(src.iter_file_history(obo_path))[offset:]

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
                out / "term_snapshots" / f"{chunk_id:03d}-{batch:04d}.parquet",
            )
        if event_rows:
            model.write_part(
                event_rows, model.EVENTS,
                out / "events" / f"{chunk_id:03d}-{batch:04d}.parquet",
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
                 "mondo_id": None, "error": "BlobMissing"}
            )
            continue
        # Split the file into stanzas by text (cheap) and hash each; only the
        # stanzas whose bytes changed are handed to fastobo.
        context, stanzas = split_document(blob)
        cur_hash = {mid: stanza_hash(s) for mid, s in stanzas.items()}

        if i == 0:  # global baseline: snapshot every term, emit no events
            parsed, failed = parse_stanzas(context, stanzas)
            _record_skips(skipped, version, failed)
            for mondo_id in failed:
                raw[mondo_id] = cur_hash[mondo_id]  # don't retry identical bad bytes
            for mondo_id, term in parsed.items():
                snap_rows.append(_snapshot_row(version, term))
                n_snap += 1
                state[mondo_id] = term
                raw[mondo_id] = cur_hash[mondo_id]
        else:
            changed = [mid for mid in stanzas if cur_hash[mid] != raw.get(mid)]
            removed = raw.keys() - stanzas.keys()
            parsed, failed = parse_stanzas(context, {mid: stanzas[mid] for mid in changed})
            failed_set = set(failed)
            _record_skips(skipped, version, failed)
            for mondo_id in failed:
                # Mark the failing bytes as seen: keep the last good state and only
                # re-attempt if this stanza's content changes again (avoids
                # re-bisecting the same unparseable term at every later commit).
                raw[mondo_id] = cur_hash[mondo_id]
            for mondo_id in changed:
                if mondo_id in failed_set:
                    continue
                term = parsed[mondo_id]
                before = state.get(mondo_id)
                raw[mondo_id] = cur_hash[mondo_id]
                if before is not None and before.content_hash == term.content_hash:
                    continue  # bytes changed but canonical content did not
                snap_rows.append(_snapshot_row(version, term))
                n_snap += 1
                added, gone = clause_delta(before.clauses if before else _EMPTY, term.clauses)
                event_rows.extend(_event_rows(version, mondo_id, added, model.Operation.ADD))
                event_rows.extend(_event_rows(version, mondo_id, gone, model.Operation.REMOVE))
                n_evt += len(added) + len(gone)
                state[mondo_id] = term
            for mondo_id in removed:
                event_rows.extend(
                    _event_rows(version, mondo_id, state[mondo_id].clauses, model.Operation.REMOVE)
                )
                n_evt += len(state[mondo_id].clauses)
                del state[mondo_id]
                del raw[mondo_id]

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
    for mondo_id, term in parsed.items():
        state[mondo_id] = term
        raw[mondo_id] = stanza_hash(stanzas[mondo_id])
    return state, raw


def _record_skips(skipped: list[dict], version: FileVersion, failed: list[str]) -> None:
    for mondo_id in failed:
        skipped.append(
            {
                "commit_seq": version.commit.seq,
                "sha": version.commit.sha,
                "mondo_id": mondo_id,
                "error": "ParseError",
            }
        )
