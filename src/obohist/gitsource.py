"""Acquire and walk the git history of a single file.

``GitSource`` builds (or opens) a git repository and yields every version of one
tracked file, oldest first, following the file across historical renames. It is
the only part of the system that talks to git; everything downstream consumes the
:class:`FileVersion` stream it produces.

The intended acquisition path is a *blob-filtered* clone
(``--filter=blob:none``): git downloads the full commit graph and trees but no
file contents, and the contents of our one file are fetched lazily as we read
them. That is how the build only ever downloads the history of a single file
rather than the whole repository.
"""

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ASCII control characters used as field / record separators in ``git log
# --format`` output. They cannot appear in commit metadata, so parsing is
# unambiguous without escaping.
_FIELD = "\x1f"  # unit separator, between format fields
_RECORD = "\x1e"  # record separator, between commits

_NULL_OID = "0" * 40


@dataclass(frozen=True)
class BranchCommit:
    """One commit on the merged branch of a merge commit (typically a PR)."""

    sha: str
    author_name: str
    committed_date: datetime  # timezone-aware
    message: str


@dataclass(frozen=True)
class CommitInfo:
    """Git metadata for one commit that touched the followed file."""

    seq: int  # 0-based position in oldest-first history; the linear time axis
    sha: str
    author_name: str
    author_email: str
    committed_date: datetime  # timezone-aware
    parent_sha: str | None  # first parent, or None at a root commit
    message: str
    branch_commits: tuple["BranchCommit", ...] = ()  # newest-first, empty for non-merges


@dataclass(frozen=True)
class TagRef:
    """A git tag resolved to the commit it points at."""

    name: str
    sha: str  # commit the tag dereferences to
    date: datetime  # that commit's committer date, timezone-aware


@dataclass(frozen=True)
class FileVersion:
    """One version of the followed file, as it existed at a single commit."""

    commit: CommitInfo
    path: str  # the file's path *at this commit* (renames change it)
    blob_oid: str  # git object id of the file content at this commit


class GitError(RuntimeError):
    """A git subprocess exited non-zero."""


class GitSource:
    """A git repository we can walk one file's history through.

    Use :meth:`clone` to create a fresh blob-filtered clone, or construct
    directly around an existing repository directory (handy for tests and for
    reusing a clone). Use as a context manager so the backing ``git cat-file``
    reader is cleaned up::

        with GitSource.clone(url, dest) as src:
            for version in src.iter_file_history("src/ontology/mondo-edit.obo"):
                data = src.read_blob(version.blob_oid)
    """

    def __init__(self, repo_dir: Path | str):
        self.repo_dir = Path(repo_dir)
        if not (self.repo_dir / ".git").exists() and not (self.repo_dir / "HEAD").exists():
            raise GitError(f"{self.repo_dir} is not a git repository")
        self._reader: _BlobReader | None = None

    @classmethod
    def clone(
        cls,
        url: str,
        dest: Path | str,
        *,
        ref: str | None = None,
        since: str | None = None,
        depth: int | None = None,
    ) -> "GitSource":
        """Create a blob-filtered, checkout-free clone of ``url`` at ``dest``.

        ``--filter=blob:none`` fetches the commit graph and trees but no file
        contents; ``--no-checkout`` skips materializing a working tree we never
        use. File contents are fetched lazily from the promisor remote when
        :meth:`read_blob` first touches them — so a build only ever downloads the
        blobs of the one file it walks.

        ``since`` (a git date such as ``2026-06-01``) or ``depth`` bound the
        clone to a recent slice of history via a shallow clone; they are mutually
        exclusive. ``ref`` restricts to a single branch.
        """
        dest = Path(dest)
        cmd = ["git", "clone", "--filter=blob:none", "--no-checkout"]
        if ref is not None:
            cmd += ["--branch", ref]
        if since is not None:
            cmd += [f"--shallow-since={since}"]
        elif depth is not None:
            cmd += ["--depth", str(depth)]
        cmd += [url, str(dest)]
        _run(cmd, cwd=None)
        return cls(dest)

    def backfill_file(self, path: str) -> None:
        """Pre-fetch every historical blob of ``path`` in one delta-packed pass.

        A ``--filter=blob:none`` clone starts with zero blob content. Even a
        plain ``git log --follow -- <path>`` needs blob content for rename
        detection, so an on-demand lazy-fetch happens sooner than one might
        expect. Rather than let git make thousands of tiny fetches during the
        build, we set the sparse-checkout to just ``path`` and call
        ``git backfill --sparse``, which asks the promisor remote for every
        blob that is currently missing under the sparse pattern in one batched
        delta-packed transfer. Idempotent — subsequent runs are near-free.

        The backfill can take minutes on a large repo; stderr is left connected
        to the caller so git's own transfer progress ("Receiving objects: 42%
        (1234/2938)") streams to the terminal live. Sparse-checkout setup is
        instantaneous and stays captured.
        """
        _run(["git", "sparse-checkout", "init"], cwd=self.repo_dir)
        _run(["git", "sparse-checkout", "set", path], cwd=self.repo_dir)
        _run_streaming(["git", "backfill", "--sparse"], cwd=self.repo_dir)

    def iter_file_history(self, path: str) -> Iterator[FileVersion]:
        """Yield every version of ``path``, oldest first, following renames.

        ``--follow`` tracks the file across renames, so ``path`` need only be its
        *current* location; each yielded :class:`FileVersion` reports the path as
        it was at that commit. Commits that delete the file are skipped.

        Branch commits for each merge are resolved via per-merge
        ``git log first_parent..second_parent`` (the semantics we want),
        cached by (first_parent, second_parent) so repeated identical pairs
        pay only once. The heavier cost of that subprocess is contained to
        the parent process — workers receive the pre-computed versions list.
        """
        commits = self._read_commits(path)
        blobs = self._read_blob_refs(path)  # sha -> (path, blob_oid) at that commit
        branch_cache: dict[tuple[str, str], tuple[BranchCommit, ...]] = {}

        seq = 0
        for commit in commits:
            ref = blobs.get(commit.sha)
            if ref is None:
                # No content-bearing change for the file at this commit (e.g. a
                # pure delete, or a merge git listed but attributed no diff to).
                continue
            at_path, blob_oid = ref
            if blob_oid == _NULL_OID:
                continue  # file removed at this commit; nothing to snapshot
            branch_commits: tuple[BranchCommit, ...] = ()
            if commit.parent_sha and commit.second_parent_sha:
                key = (commit.parent_sha, commit.second_parent_sha)
                cached = branch_cache.get(key)
                if cached is None:
                    cached = self._read_branch_commits(*key)
                    branch_cache[key] = cached
                branch_commits = cached
            yield FileVersion(
                commit=CommitInfo(
                    seq=seq,
                    sha=commit.sha,
                    author_name=commit.author_name,
                    author_email=commit.author_email,
                    committed_date=commit.committed_date,
                    parent_sha=commit.parent_sha,
                    message=commit.message,
                    branch_commits=branch_commits,
                ),
                path=at_path,
                blob_oid=blob_oid,
            )
            seq += 1

    def read_tags(self) -> list[TagRef]:
        """All tags, each dereferenced to its target commit and that commit's date.

        Works for both lightweight and annotated tags (``^{commit}`` derefs the
        annotated case). Tags outside a shallow clone's range simply aren't here.
        """
        names = _run(
            ["git", "for-each-ref", "--format=%(refname:short)", "refs/tags"],
            cwd=self.repo_dir,
        ).split()
        tags: list[TagRef] = []
        for name in names:
            info = _run(
                ["git", "log", "-1", f"--format=%H{_FIELD}%aI", f"{name}^{{commit}}"],
                cwd=self.repo_dir,
            ).strip()
            sha, _, date = info.partition(_FIELD)
            tags.append(TagRef(name=name, sha=sha, date=datetime.fromisoformat(date)))
        return tags

    def read_blob(self, blob_oid: str) -> bytes:
        """Return the raw bytes of a blob, fetching it from the promisor if needed."""
        if self._reader is None:
            self._reader = _BlobReader(self.repo_dir)
        return self._reader.read(blob_oid)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self) -> "GitSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internals -------------------------------------------------------

    def _read_commits(self, path: str) -> list["_RawCommit"]:
        """Ordered commit metadata for every mainline commit touching ``path``.

        Uses ``--first-parent`` so the sequence is genuinely linear: each
        commit's predecessor in the returned list is its git first-parent, and
        the diff between adjacent commits is what a curator saw land on main.

        Without ``--first-parent``, ``git log --follow`` returns every commit
        touching the file across every branch, interleaved chronologically.
        Diffing adjacent commits then compares state on unrelated branches
        (their common ancestor may be many commits back), producing phantom
        "changes" that don't reflect any real edit. That was the source of the
        weird edit cycles on paired events: the walk zig-zagged between
        branch state and mainline state, and each hop looked like a change.

        The body (``%B``) is the last field and no path/diff output follows it,
        so multi-line messages parse unambiguously against the separators.
        """
        # NB: --follow is incompatible with --reverse (git rewrites history while
        # walking backward, which --reverse then mangles). Walk newest-first and
        # reverse in Python.
        fmt = _FIELD.join(["%H", "%an", "%ae", "%aI", "%P", "%B"]) + _RECORD
        out = _run(
            [
                "git",
                "log",
                "--follow",
                "--first-parent",
                f"--format={fmt}",
                "--",
                path,
            ],
            cwd=self.repo_dir,
        )
        commits: list[_RawCommit] = []
        for record in out.split(_RECORD):
            record = record.strip("\n")
            if not record:
                continue
            sha, an, ae, adate, parents, message = record.split(_FIELD)
            parts = parents.split(" ") if parents else []
            commits.append(
                _RawCommit(
                    sha=sha,
                    author_name=an,
                    author_email=ae,
                    committed_date=datetime.fromisoformat(adate),
                    parent_sha=parts[0] if parts else None,
                    second_parent_sha=parts[1] if len(parts) > 1 else None,
                    message=message.strip(),
                )
            )
        commits.reverse()  # oldest first
        return commits

    def _read_branch_commits(
        self, first_parent: str, second_parent: str
    ) -> tuple["BranchCommit", ...]:
        """Commits reachable from ``second_parent`` but not from ``first_parent``.

        These are the individual commits that landed as part of a merge — for
        a GitHub-merged PR, the PR-branch commits in the order they'll be
        listed on the PR page. Returned newest-first so the tip (typically the
        one with the most descriptive final message) is prominent.

        Delegates the reachability computation to git (``a..b`` semantics) —
        computing it in memory got tangled up on cross-branch merges (a PR
        whose branch itself merged main into it), which have PR-branch
        commits reachable through the merged-in tip that aren't on the
        first-parent chain from the tip.
        """
        fmt = _FIELD.join(["%H", "%an", "%aI", "%s"]) + _RECORD
        try:
            out = _run(
                [
                    "git", "log",
                    "--no-merges",
                    f"--format={fmt}",
                    f"{first_parent}..{second_parent}",
                ],
                cwd=self.repo_dir,
            )
        except GitError:
            return ()
        result: list[BranchCommit] = []
        for record in out.split(_RECORD):
            record = record.strip("\n")
            if not record:
                continue
            sha, an, adate, subject = record.split(_FIELD)
            result.append(
                BranchCommit(
                    sha=sha,
                    author_name=an,
                    committed_date=datetime.fromisoformat(adate),
                    message=subject,
                )
            )
        return tuple(result)

    def _read_blob_refs(self, path: str) -> dict[str, tuple[str, str]]:
        """Map each commit sha to (path_at_commit, blob_oid) via ``--raw``.

        ``--raw`` prints the post-image blob oid for the file at each commit, so
        content can be read by oid without re-deriving the (possibly renamed)
        path. The path is still captured for reporting.
        """
        # Newest-first (see _read_commits); order is irrelevant here since the
        # result is keyed by sha. `--first-parent` matches _read_commits so we
        # don't accidentally return blobs for commits that _read_commits
        # (correctly) omitted.
        out = _run(
            [
                "git",
                "log",
                "--follow",
                "--first-parent",
                "--raw",
                "--no-abbrev",
                f"--format={_RECORD}%H",
                "--",
                path,
            ],
            cwd=self.repo_dir,
        )
        refs: dict[str, tuple[str, str]] = {}
        for chunk in out.split(_RECORD):
            lines = [ln for ln in chunk.splitlines() if ln]
            if not lines:
                continue
            sha = lines[0]
            for raw in lines[1:]:
                if not raw.startswith(":"):
                    continue
                meta, _, names = raw.partition("\t")
                # meta: ":<omode> <nmode> <ooid> <noid> <status>"
                fields = meta[1:].split(" ")
                new_oid = fields[3]
                at_path = names.split("\t")[-1]  # new path for renames
                refs[sha] = (at_path, new_oid)
                break  # one file per commit under --follow
        return refs


@dataclass(frozen=True)
class _RawCommit:
    sha: str
    author_name: str
    author_email: str
    committed_date: datetime
    parent_sha: str | None
    second_parent_sha: str | None  # set when this is a merge (2+ parents)
    message: str


class _BlobReader:
    """A persistent ``git cat-file --batch`` process for fast repeated reads."""

    def __init__(self, repo_dir: Path):
        self._proc = subprocess.Popen(
            ["git", "cat-file", "--batch"],
            cwd=repo_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def read(self, blob_oid: str) -> bytes:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(f"{blob_oid}\n".encode())
        self._proc.stdin.flush()
        header = self._proc.stdout.readline().decode().rstrip("\n")
        parts = header.split(" ")
        if len(parts) < 3 or parts[1] != "blob":
            raise GitError(f"unexpected cat-file response: {header!r}")
        size = int(parts[2])
        data = self._proc.stdout.read(size)
        self._proc.stdout.read(1)  # trailing newline
        return data

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()


def _run(cmd: list[str], cwd: Path | None) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitError(f"{' '.join(cmd)} failed:\n{result.stderr}")
    return result.stdout


def _run_streaming(cmd: list[str], cwd: Path | None) -> None:
    """Run a git command with stderr inherited so the user sees its progress.

    Used for long-running operations like ``git backfill --sparse`` where
    git's own transfer-progress output (`Receiving objects: 42% (…)`) is
    much more informative than a "please wait" spinner from us. Stdout
    stays captured so it doesn't interleave with the rest of the CLI's
    rich rendering (backfill's stdout is empty on success anyway).
    """
    result = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=None, text=True,
    )
    if result.returncode != 0:
        raise GitError(f"{' '.join(cmd)} failed with exit code {result.returncode}")
