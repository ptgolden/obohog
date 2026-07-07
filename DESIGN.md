# obohog — Design

*A concrete design for the vision in [`PLAN.md`](./PLAN.md).*

## Context

`PLAN.md` describes the goal: turn an OBO ontology's git history into a
compact, queryable database so ontology developers can ask *ontology-centric*
historical questions ("when did this synonym appear?", "how did this term's
classification evolve?") without cloning the source repository or doing
GitHub archaeology. The vision doc leaves the implementation unspecified.
This document commits to concrete choices.

`obohog` builds its *own* clones of each configured ontology's repository —
scoped to a single OBO file's history per source (see §3) — as a build-time
input. Each ontology's database is entirely *derived from* the source's git
history (a single OBO file, e.g. Mondo's `src/ontology/mondo-edit.obo` or
PATO's `src/ontology/pato-edit.obo`), but once built it is self-contained:
**querying** it — via the CLI, API, or a future hosted app — needs no access
to the source repo at all. The build depends on that history; consumers do
not. Each database versions and releases on its own cadence.

Chosen stack:
- **TOML** (via stdlib `tomllib`) for the per-project `obohog.toml` config
  declaring one or more sources.
- **fastobo** (Python bindings) for OBO parsing.
- **Parquet + DuckDB** for storage and query.
- A **self-contained** extraction tool that fetches just the history it needs.

### What a target looks like
- OBO 1.2 format `src/ontology/<name>-edit.obo`. Sizes vary from a few MB
  (PATO, small terminologies) to ~45 MB (Mondo). Terms carry the multivalued
  fields queries care about: `synonym`, `xref`, `is_a`, `relationship`,
  `subset`, `def`, `is_obsolete`, `replaced_by`.
- Modern OBO Foundry commit messages tend to embed PR numbers like `(#10400)`
  → PR-linking is nearly free.
- Files sometimes move paths over years (Mondo's `tbd-edit.obo` →
  `mondo-edit.obo`) → extraction uses `git log --follow`.

---

## Core design decisions

### 1. Historical unit: snapshots are primitive, events are derived
- **Term-version snapshots**, stored *only on commits where a given term changed*
  (change detected by content-hashing each normalized term frame). Reconstructing
  "state of `MONDO:x` at commit `c`" = the latest snapshot with `commit_seq <= seq(c)`.
- **Change events**, materialized by diffing adjacent snapshots of the same term:
  `(term_id, commit_seq, predicate, value, operation)` where operation ∈ {add, remove}.
  A synonym text edit is naturally a remove+add of that clause. This table is the
  queryable spine for "when did X change".
- Snapshots are the source of truth; events are a convenience view over them.

### 2. Artifact: Parquet canonical, DuckDB query layer
- Canonical artifact = a few Parquet files. Highly compressible for repetitive
  ontology strings; engine-neutral; archival.
- CLI/API query them with **DuckDB**. DuckDB reads Parquet locally *and* range-queries
  it over HTTP, so the "hosted service" can be static file hosting (GitHub Releases /
  S3) with **no server** — clients fetch only the byte ranges they need.
- Optionally emit a convenience single-file `.duckdb` later; Parquet stays primitive.

### 3. History acquisition: providers materialize a git repo, extract walks it
- Every configured source declares a `type` in `obohog.toml`. Three types
  today: `git-file`, `github-release`, and `bioportal`. All produce a
  git repo at `{storage}/{name}/clone`; the extract pipeline walks that
  repo identically regardless of type. Adding a source type means
  writing one provider module, not touching extract/query/render.
- **`git-file`** — do a `git clone --filter=blob:none --no-checkout` of the
  source repo: full commit graph + trees, **no blob contents** until
  requested. Then walk `git log --follow --reverse -- <config-declared-file>`
  and lazily `git cat-file` **only that one file's blobs** across history.
  We only ever download the history of the declared OBO file, not the rest
  of the repo. Reproducible from a URL.
- **`github-release`** — enumerate releases via `gh api`, download the
  configured `asset` from each with `gh release download`, and commit the
  asset into a synthetic git repo (`git init` at first sync). Each release
  becomes one commit: `author` = release publisher, `date` = `published_at`,
  message body carries a `Release URL: <html_url>` trailer that the
  extractor parses back out into `snapshot_url`. The commit is git-tagged
  with the release tag. Renderers link to the release page via
  `snapshot_url`. Non-goal: reconstructing which PR touched which term —
  release notes often link PRs but there's no reliable machinery to
  attribute term changes to specific PRs.
- **`bioportal`** — enumerate submissions from BioPortal's REST API
  (`data.bioontology.org`), filter to `hasOntologyLanguage == "OBO"`,
  and download each via `?download_format=obo`. Non-OBO submissions
  are skipped — **no OWL→OBO conversion** — and if the ontology
  publishes zero OBO submissions the sync errors out cleanly. Requires
  `BIOPORTAL_API_KEY` in a project-root `.env` (loaded via
  `pydantic-settings`). Provenance: `author` = first contact name/email
  on the submission (fallback `BioPortal`), `date` = `released` or
  `creationDate`, tag = the submission's `version` string when it's a
  valid git ref name (fallback `sub-<submissionId>`). BioPortal doesn't
  publish per-submission web pages, so these commits carry no
  `snapshot_url`.
- The source URL and configured fields come from `obohog.toml` and are
  recorded in the built `build_meta`.

---

## Data model (Parquet schema)

- **`commits`** — one row per commit touching the OBO file:
  `commit_seq` (monotonic linear index, oldest=0), `sha`, `author_name`,
  `author_email`, `committed_date`, `message`, `pr_number` (nullable, parsed from
  message), `parent_sha`, `branch_commits` (nullable list of PR-branch commits
  for merges), `snapshot_url` (nullable — populated for release-based sources
  with the release page URL, parsed from a `Release URL:` trailer in the
  commit message body).
- **`releases`** — release tags mapped to commits: `tag`, `sha`, `date`, `commit_seq`.
  Enables "what changed between two releases".
- **`term_snapshots`** — one row per (term, commit-where-it-changed):
  `term_id`, `commit_seq`, `sha`, `name`, `is_obsolete`, `replaced_by`,
  `content_hash`, `clauses` (list<struct{predicate, value}> — canonical normalized
  frame), `frame_text` (canonical OBO serialization for exact reconstruction).
- **`events`** — derived: `term_id`, `commit_seq`, `sha`, `predicate`, `value`,
  `operation` (add|remove). Semantic events (term_created / term_obsoleted /
  term_merged) are just filtered views over this table.
- **`build_meta`** — schema version, generator version, source repo URL, source
  sha range (first/last `commit_seq`), obo path. Makes results deterministic and
  reproducible; supports incremental rebuilds.

Ordering: `commit_seq` linearizes history (first-parent walk of the OBO file's
commits) so point-in-time queries are simple range comparisons.

---

## Extraction algorithm (build step)

1. Acquire history (blobless clone as in §3); resolve the OBO path with `--follow`
   (handles historical renames).
2. `git log --follow --reverse --format=... -- <path>` → ordered commit list;
   parse PR numbers from messages; collect tags → `releases`.
3. Stream commits oldest→newest, holding only the *previous* version's
   `{term_id: (content_hash, clauses)}` in memory:
   - `git cat-file blob <sha>:<path>` → bytes.
   - Parse with **fastobo**. Trust it — no defensive fallback parser, no
     per-commit "unparseable" flagging. If fastobo raises, the build fails loudly.
   - Normalize each `[Term]` frame to a canonical clause set; hash it.
   - For each term whose hash changed (or is new): write a `term_snapshots` row and,
     by diffing clause sets vs the previous version, write `events` rows.
   - Removed terms (present before, absent now) → a removal marker event.
4. Write Parquet via pyarrow/DuckDB; write `build_meta`.
5. **Incremental mode** (`--since ARTIFACT`): read last `commit_seq` from prior
   `build_meta`, seed the "previous version" from the last snapshot state, process
   only newer commits, append. Keeps ongoing per-release rebuilds cheap.

Cost note: parsing is Rust-backed (fastobo) and one-time; term-level hashing avoids
storing/diffing unchanged terms; per-commit parse is independent and parallelizable
if needed.

**Robustness stance:** correctness comes from *types and libraries*, not defensive
code. Model snapshots/events/operations as typed dataclasses (or Pydantic/attrs) and
a small enum for `operation`/`predicate`; let fastobo, pyarrow, and DuckDB enforce
their own invariants and raise on violation. Avoid speculative edge-case handling.

---

## Interfaces (all thin DuckDB SQL over the same Parquet)

CLI (`obohog`):
- `term MONDO:x` — event timeline for a term.
- `term MONDO:x --at <sha|date|release>` — reconstructed snapshot at that point.
- `synonyms|xrefs|parents MONDO:x` — filtered event history for one field kind.
- `commit <sha>` — all terms changed together in that commit.
- `pr <n>` — terms affected by a PR.
- `diff <releaseA> <releaseB> [--term MONDO:x]` — changes between two releases.

### Commit-header rendering: GitHub-specific heuristics with a graceful fallback

`obohog term`, `commit`, `diff`, and `search` all render a per-commit header
block: `sha  date  author  <subject line>`, followed optionally by a PR link
and/or PR-branch commits. Two GitHub conventions inform how that subject line
is chosen; both are heuristics rather than schema, and both degrade cleanly
when the source isn't hosted on GitHub or the commit doesn't fit the shape.

**PR number extraction** (see `extract._extract_pr_number`):

- **Squash-and-merge** (post-2023 Mondo, most modern repos): the title ends
  with `(#N)` — e.g., `add venom terms (#10409)`.
- **Classic merge commit** (pre-2023 Mondo, PATO, older workflows): the first
  line is `Merge pull request #N from user/branch`.

The classic form is matched anchored to the start of the message so a PR body
that quotes `(#123)` in prose doesn't false-positive over a real merge header.
If neither matches, `pr_number` is `NULL` — no PR link, no `pr <N>` lookup.

**PR-title-from-body extraction** (see `cli._pr_title_from_merge`):

Classic GitHub merge commits carry the PR title in the message body:

```
Merge pull request #N from user/branch

<PR title — often the branch's last commit subject, or set on the merge screen>

<optional PR body>
```

When we find this shape, we render the PR title as the primary editorial line
and demote the boilerplate `Merge pull request #N from …` to a dim italic
sub-line. This makes classic-merge commits visually consistent with the
one-line squash-and-merge style. Because the PR title is usually more
informative than the aggregate of 30 tiny `revise xrefs` branch commits, we
also **hide branch commits by default** in this case; `--commits` on any
render command opts back in.

**Non-GitHub sources**: `_pr_url_base` returns `None` for anything that
isn't `https://github.com/owner/name`. In that case:

- PR links become plain dim text `→ PR #N` (no URL, but the tag is still
  visible).
- Classic-merge PR-title extraction still works — it operates on the
  message body, not on GitHub metadata — so any repo that uses GitHub's
  merge-commit format gets the nicer rendering even if it's mirrored to a
  different host. Repos with a different merge convention gracefully fall
  through to raw-subject rendering.

Programmatic API: a small Python module exposing the same queries (returns
DataFrames/dicts). Hosted web app: reads the identical Parquet (optionally over
HTTP via DuckDB httpfs), no independent representation.

---

## Repository layout

```
DESIGN.md                     # this document
PLAN.md                       # original vision
README.md                     # quick start
obohog.toml.example          # example config; user copies to obohog.toml
pyproject.toml                # deps: fastobo, duckdb, pyarrow, typer, rich, tqdm, pydantic
src/obohog/
  config.py                   # obohog.toml loading + Source resolution (pydantic discriminated union)
  extract.py                  # walks the clone, parses OBO, writes Parquet
  gitsource.py                # blobless clone / rename-aware log walk / cat-file blob reader
  obo.py                      # frame normalization, canonical clause set, hashing
  model.py                    # Parquet schemas / table writers
  query.py                    # DuckDB query helpers (shared by CLI + API)
  render.py                   # structural word-diff, pairing, delta search
  cli.py                      # command-line entry point (source subcommand + query commands)
  providers/
    __init__.py               # get_provider(source, console) → Provider dispatcher
    _synthetic_git.py         # shared helpers: git init/tag/commit-or-tag-head for materializer providers
    git_file.py               # GitFileProvider: blob-filtered clone + backfill
    github_release.py         # GitHubReleaseProvider: materialize releases via `gh` into a synthetic git repo
    bioportal.py              # BioPortalProvider: materialize BioPortal submissions (OBO only) via REST API
  settings.py                 # pydantic-settings loader for .env (BIOPORTAL_API_KEY, ...)
tests/
  fixtures/                   # tiny multi-commit OBO git repo for deterministic tests

data/                         # gitignored per-source working state
  <source>/clone/
  <source>/db/
```

---

## Verification

- **Unit**: build a tiny fixture git repo with a handful of commits mutating a
  small OBO (add synonym, remove xref, reparent, obsolete, merge). Assert exact
  `events` rows and that `--at` reconstruction equals the committed file per commit.
- **Round-trip**: for random (term, commit) pairs, assert the reconstructed
  `frame_text` matches the `git cat-file`-extracted stanza of that term at that commit.
- **Integration (real data)**: run extraction against a real Mondo history
  end-to-end; sanity-check known changes (e.g. a recent PR's terms appear under
  `pr <n>`); confirm artifact size is small and queries are sub-second.
- **Determinism**: rebuild twice → identical Parquet content hashes; incremental
  build from artifact N to N+k equals a full build at N+k.

---

## Open questions deferred to implementation
- Exact canonical clause normalization (whitespace, qualifier ordering, xref-in-def).
- Whether to store `frame_text` in `term_snapshots` or reconstruct purely from `clauses`.
- Release-tag discovery: git tags vs `data-version` header vs GitHub releases API.
- Whether the CLI ships a bundled recent artifact or always downloads one.

---

## Implementation status (2026-07-03)

**Working:**
- `config` — TOML-based `obohog.toml` declaring one or more ontology sources.
  Each source pins a git repo, the OBO file within it, and optionally clone/db
  paths. Default layout is `{storage}/{name}/{clone,db}`.
- CLI is source-aware. All query commands (`term`, `commit`, `pr`, `diff`,
  `search`, `releases`) take a required `--source <name>`. The `source`
  subcommand group manages sources: `source list` shows configured sources
  with disk usage, `source sync <name>` clones + builds a source's database.
- `gitsource` — blob-filtered clone; rename-following single-file walk; scoped,
  delta-packed history fetch via `git backfill --sparse` (sparse-checkout
  scoped to the source's OBO file); blob reads via a persistent
  `git cat-file --batch`.
- `obo` — fastobo normalization (single-threaded parse, `threads=1`), canonical
  clause sets, content hashing, clause diffing.
- `extract` — single-threaded `build()` (full-parse reference) and a **parallel,
  streaming `build_parallel()`**: the commit range is split into **more chunks
  than workers** (default ~4/worker, tunable via `--chunk-size`) and dispatched
  dynamically by the process pool, so a worker that finishes a light chunk grabs
  the next queued one instead of idling (the earlier tail-latency issue). Each
  chunk is seeded by the previous chunk's last commit — a one-parse cost that
  bounds how small chunks can usefully get. Parquet **part-files** are flushed
  periodically to bound memory; output dirs are cleared first so re-runs don't
  accumulate stale files.
  - **Diff-scoped parsing:** rather than fastobo-parsing all ~45 MB each commit,
    a worker splits the file into stanzas by text (cheap), hashes each, and hands
    fastobo *only the stanzas whose bytes changed*, carrying unchanged term state
    forward. This is ~10× faster (verified byte-identical to full parsing) and
    more resilient — an unparseable stanza only matters at the commit that
    touches it.
  - **Per-term skip-and-isolate:** a failing batch is bisected until the single
    offending stanza is found; that one term is recorded in `skipped` and skipped,
    never the whole commit.
- `model` — Parquet schemas incl. `releases` and `skipped_commits`.
- `query`/`cli` — DuckDB over part-file globs or single files; `source sync`
  (with `--jobs`), `term` (with `--limit`, `--since`, `--full`, `--only`,
  `--at` accepting sha/tag/seq), `commit`, `pr`, `diff`, `search` (with
  `--regex`, `--ignore-case`, `--namespace`, `--predicate`), `releases`;
  rich rendering. All query commands are scoped by `--source`.
- `render` — **structure-aware term timeline**: paired remove/add events on the
  same predicate render as `~` word-diff edits rather than two adjacent lines.
  Pairing is two-pass — parsed-body identity first (fastobo-parsed), then greedy
  lexical similarity — so a same-target clause whose qualifiers were reordered
  can't cross-pair with a different-target clause whose qualifier text happens
  to align (the classic commit-1476 pathology). Rendering is layered:
  - **Target-label rename** (body + qualifier set identical, only `!` comment
    differs) → shared form plain, comment change bracketed, tagged `(target label)`.
  - **Qualifier reorder** (body + comment identical, qualifier multiset identical,
    order differs) → current form displayed, tagged `(qualifier order rewritten)`.
  - **Qualifier-block edit** (qualifier multiset changed) → body + comment on the
    top `~` line, per-qualifier `+`/`-`/`~` sub-lines indented underneath with
    inline word-diff on `~` sub-lines. Reads like an axiom-annotation diff.
  - **Fallback** (body changed, no qualifiers, or fastobo can't parse) → single-
    line token word-diff using a compound-identifier-aware tokenizer that keeps
    CURIEs, URLs, and snake_case names whole while splitting on structural
    punctuation. Git `--word-diff=plain` markers stay readable when piped.
- **50 tests**, incl. parallel-build == single-threaded equivalence, stale
  part-file clearing, structure-aware rendering, and the commit-1476 pairing
  regression.

**Local state:**
- `./data/mondo/clone/` — full history of `mondo-edit.obo`, 2017-09→2026-06
  (7,487 versions), ~925 MB single pack, gitignored. The full build runs
  **offline** against it (`GIT_NO_LAZY_FETCH=1`).
- `./data/mondo/db/` — the built history database from that clone. The
  `obohog term --source mondo MONDO:0012350` examples in `README.md` read
  from it. ~656 MB of parquet.

**Validated:** parallel build produces byte-identical events/snapshots to the
single-threaded build (checksum match on a 12-commit slice).

**Next steps:**
1. **Incremental updates** — a `source sync --update` path: self-seed from
   the latest snapshot per term, `git fetch` + `git backfill --sparse` the
   new commits, append new part-files, extend `commit_seq`; ancestry check
   as a rewrite guard. This is the top of the queue.
2. **Prefix migration** — per-source `replaced_prefix` config to
   transparently include `TBD:0000450` events when querying
   `MONDO:0000450`; specified in
   `2026-07-03-note.term-identity-across-renames.md` (deferred note).
3. **Distribution** — publish part-files to GitHub Releases; document HTTP
   range-query use.
4. **N-to-M pairing** — detect commits like `1ac4db2^` (two same-target xrefs
   collapsed into one with a merged qualifier list). Now tractable given the
   fastobo-parsed body + qualifier sets; the missing piece is grouping
   events by body within a predicate bucket before pairing.
5. **Non-OBO serializations** — OFN, RDF/XML, Turtle. Would require
   abstracting the per-commit stanza scan and per-term parse behind a
   format strategy interface; today's diff-scoped parse depends on OBO's
   line-oriented `[Term]` stanzas.
6. **If size matters** — evaluate the keyframe + event-replay variant to
   shrink `term_snapshots`.
