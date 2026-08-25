# obohog

*OBO History over Git*

`obohog` builds a compact, queryable database describing how every term in
an OBO ontology has changed over time. It answers ontology-level history
questions — "when did this synonym appear?", "which commits touched
NCIT:C317?", "how did this classification evolve?" — directly, without
cloning the ontology repo's full git history.

Configure one or more ontologies in `obohog.toml`, run `obohog source
sync <name>` to build each database once, then query with `obohog term`,
`search`, `commit`, `diff`, `pr`, `releases`.

See [`DESIGN.md`](./DESIGN.md) for the architecture and
[`PLAN.md`](./PLAN.md) for the original vision.

## Status

Working end-to-end on the full Mondo history (7,487 commits, ~5 min build):

- `gitsource` — file-scoped, blob-filtered clone + single-file history walk,
  following renames, reading blob content by OID.
- `obo` — normalize term frames to canonical clause sets (via fastobo) and
  diff adjacent versions clause-by-clause.
- `extract` / `model` — stream the diff into a Parquet database
  (`commits`, `term_snapshots`, `events`, `releases`, `skipped`,
  `build_meta`).
- `config` — declarative `obohog.toml` supporting multiple ontology sources.
- `query` / `cli` — DuckDB-backed queries rendered with `rich`. The `term`,
  `commit`, `diff`, and `search` commands render paired add/remove events as
  inline word-diffs with fastobo-aware structural detection (target-label-
  only edits, qualifier reorderings, and qualifier-block adds/removes/edits
  each rendered distinctly — see `src/obohog/render.py`). `search`
  additionally filters at the delta level so hits reflect what actually
  changed.

Only the OBO file format is supported today. Future serializations
(OFN, RDF/XML, Turtle) would require abstracting the per-commit parse step;
see the Non-goals section of `PLAN.md`.

## Try it

### 1. Configure

Copy `obohog.toml.example` to `obohog.toml` and declare the ontologies
you want to track:

```toml
storage = "./data"

[source.mondo]
repo = "https://github.com/monarch-initiative/mondo"
file = "src/ontology/mondo-edit.obo"

[source.pato]
repo = "https://github.com/pato-ontology/pato"
file = "src/ontology/pato-edit.obo"
```

Each source clones into `{storage}/{name}/clone` and builds its database at
`{storage}/{name}/db`. Both paths can be overridden per source.

### 2. Sync

```sh
uv sync --extra dev

# Clone the git history (blob-filtered) + build the DB. Only the history
# of the declared OBO file is downloaded, lazily.
uv run obohog source sync mondo

# Re-running sync is incremental: it fetches what's new upstream and
# appends just those commits (seconds, not minutes). --rebuild starts
# over, and a rebuild also happens automatically when appending isn't
# safe (schema change, rewritten upstream history).
uv run obohog source sync mondo

# What's configured, built with which artifact schema (stale artifacts
# need a resync), and how much disk it's using:
uv run obohog source list
```

### 3. Query

All query commands take `--source <name>`:

```sh
# A term's change history (optionally one clause kind), a point-in-time
# state, and everything that changed together in a commit.
uv run obohog term    --source mondo MONDO:0012350
uv run obohog term    --source mondo MONDO:0012350 --only synonym
uv run obohog term    --source mondo MONDO:0012350 --limit 5
uv run obohog term    --source mondo MONDO:0012350 --since v2026-05-05
uv run obohog term    --source mondo MONDO:0012350 --full
uv run obohog term    --source mondo MONDO:0012350 --at 1ac4db2
uv run obohog commit  --source mondo 1ac4db2

# Release-oriented views: list releases, a PR's terms, or a range diff.
uv run obohog releases --source mondo
uv run obohog pr       --source mondo 10400
uv run obohog diff     --source mondo v2026-06-02 HEAD --term MONDO:0001213

# Search event values (git log -S style): commits that added or removed a
# clause where the query appears in the *changed* portion. Kept-unchanged
# qualifiers and unchanged body tokens don't count as hits.
uv run obohog search   --source mondo "OMIM:609814"
uv run obohog search   --source mondo "GARD:18551" --predicate xref
uv run obohog search   --source mondo "\bCML\b" --regex
uv run obohog search   --source mondo "CML" --namespace MONDO
uv run obohog search   --source mondo "MONDO:MalaCards" --term MONDO:0012350

# git-log style: the 10 most recent commits that touched an NCIT xref,
# newest first (--reverse for oldest first).
uv run obohog search   --source mondo "NCIT" --order date --limit 10
```

## Development

```sh
uv sync --extra dev
uv run pytest
```
