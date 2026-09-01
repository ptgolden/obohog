# Performance notes

Context for these numbers, and the constraint they serve: the web UI +
JSON API will be deployed on hardware weaker than the dev laptop every
figure below was measured on (2026-09-01, Apple Silicon; treat numbers
as relative, not absolute). Expected traffic is a *wide variety of
unique queries* — including LLM agents exploring the API, which can
issue many requests quickly — so response caching (HTTP or otherwise)
barely helps: the budget that matters is **CPU per novel query**.

The architecture's standing advantage: an artifact is immutable between
syncs, so anything derivable from the events table can be precomputed
at build time or memoized for a handle's lifetime without invalidation
logic. Prefer that lever whenever a query-time computation's answer
can't change between syncs.

## Anatomy of a page (mondo: 4.1M events, 5,249 commits)

Typical delta-filtered query (`q=NCIT tag=xref has=subset~nord_rare`,
newest-first): **0.39s** first page, **0.28s** per load-more.

- ~0.2–0.3s — DuckDB stream startup: three scans of the events parquet
  (outer WHERE, the semi-join's `contains()` scan, the has-subquery),
  then the sort. The `contains()` scans read the whole value column;
  nothing prunes them.
- 0.11s — `search_counts` (first page only; continuations skip it).
- ~0.05s — Python pairing, delta filter, output models. Not worth
  optimizing.

Blank date-ordered browse (the planned path): **~0.12–0.4s** per
50-commit page.

## Levers already pulled

- Page cost bounded by ops, not just section count (191c041).
- Browse consumption of a giant commit capped at the render cap
  (fc950d0), then superseded by planned pages: the browse page's commit
  skeleton comes from a cheap per-commit aggregate, and one event query
  fetches exactly the page — capped commits contribute only their first
  50 terms (6666b3a).
- Load-more fetches skip the counts scan they never display (806f972).
- The semi-join subquery carries the text narrowing only; the has
  clause is evaluated once, in the outer WHERE (2ae90b9). Same rows,
  one fewer full scan.
- `facets()` cached per artifact open (predates this work);
  `commit_stats` memoized per (filters, reverse) for the handle's
  lifetime, cursors sliced in Python (018ed66).

## Levers available, measured but not yet pulled

**`term_clause_spans` table (build-time, schema bump).** One row per
presence-span of a (term_id, predicate, value) triple: `from_seq`,
`until_seq` (NULL = present at HEAD). Derived purely from events — one
SQL window pass, no git re-walk, so existing artifacts can be migrated
in place. On mondo: 2.37M rows, 0.3s to build, ~80 MB (+18% artifact).
Measured (in-memory table; persisted parquet lands somewhat above):

- `has ever:` subquery 48ms → 8ms
- `has now:` 49ms → 6ms (the query-time `arg_max` GROUP BY disappears —
  its answer is precomputed; verified identical membership on real data)
- `during:` (reserved in the parser, unimplemented) → 3ms as a pure
  range predicate: `from_seq <= S AND (until_seq IS NULL OR S < until_seq)`

Worth pulling when `has`/`during:` traffic is real. Until then,
`during:` is also implementable with no schema change at ~50ms: filter
events to `commit_seq <= S`, then the same
`GROUP BY triple HAVING arg_max(operation, commit_seq) = 'add'`.

**Per-commit stats columns in the artifact.** Would replace the
memoized aggregate (one-time 0.03s per handle) and serve analytics.
Only worth bundling with another schema bump.

**HTTP caching (ETag = build id + query string).** Cheap to add and
harmless, but not the main lever here: it only helps repeated
identical queries, and the expected traffic is mostly novel queries.

## Round 2: layout + plan-then-fetch (2026-09-01, deeper pass)

**Artifact layout is the biggest untwisted knob, and it needs no schema
bump** — readers glob the table dirs, so file count/order/codec are
free to change. Measured on mondo, current code, only the files
changed:

| operation            | current | best layout |
|----------------------|--------:|------------:|
| typical search p1    |  392 ms |      244 ms |
| typical load-more    |  271 ms |      197 ms |
| browse newest (warm) |  110 ms |       68 ms |
| browse oldest (warm) |  327 ms |      126 ms |
| timeline (busy term) |  116 ms |       58 ms |
| state at HEAD        |   65 ms |       29 ms |
| artifact size        |  471 MB |      160 MB |

"Best layout" is two changes, ~2 s to produce from the existing
artifact (pure DuckDB COPY, no git):

1. **Compact events to one file, keeping commit order.** 56 small
   files cost ~40 ms of per-file overhead on *every* `contains()`
   scan (56→16 ms measured in isolation). Commit order preserves
   row-group zone maps on commit_seq, so browse-window pruning
   survives. Do NOT sort events by term: it bloats the file
   (92→129 MB — diff rows have no cross-row redundancy) and kills
   commit pruning.
2. **Sort term_snapshots by (term_id, commit_seq) and compact.**
   Successive snapshots of one term are near-identical, so zstd
   crushes the redundancy: 379→74 MB (5×). term_at (a term-major
   point lookup) halves. No consumer needs snapshots in commit order
   (term_at and the name joins are key lookups).

Caveats measured: ROW_GROUP_SIZE 500k regresses point lookups (state
274 ms) — keep the default; incremental syncs that append files will
decay performance again, so compaction belongs at the end of every
sync, not as a one-off. Compaction must respect extract's incremental
file-naming scheme (`{chunk}-{batch}.parquet` resumption) — study that
before wiring in.

**Plan-then-fetch generalizes to delta searches** (prototype, on best
layout): one match-scan returns `(term_id, commit_seq, n)` candidate
pairs (53 ms) — counts derive from it for free, the page plan is
Python over the pairs, and a commit-bounded fetch joined against the
page's pairs (an Arrow-registered relation) pulls only the page's wide
rows (49 ms; term-order variant 56 ms). Estimated end-to-end: cold
page ~140 ms (vs 244), and with the pairs memoized per (q, filters) a
load-more page ~90 ms (vs 197) — no scans at all, just the bounded
fetch. Stacked with the layout change: the original 0.45 s query lands
around 0.09–0.14 s. Behavior delta to accept: pages plan by candidate
commits, so a page can render fewer than `limit` sections after the
delta filter drops groups (cursor semantics unchanged).

Trap for the implementation: never `executemany` the pairs into a temp
table — 2.5k inserts cost 600 ms. `con.register()` an Arrow table
(zero-copy, instant). Restricting the name join to page pairs is not
worth it (49→45 ms).

## Measured dead ends (don't revisit without new evidence)

- Dropping the `term_snapshots` LEFT JOIN: no gain (DuckDB handles it).
- The ORDER BY: 0.06s over 133k rows — not the bottleneck.
- Re-issuing the stream past capped commits (tried, discarded before
  merge): each re-issue re-paid the full sort, ~0.4s × ~12 capped
  commits per page. Planning the page first is strictly better.
- Sorting *events* by term or (predicate, term): larger files, broken
  commit pruning, and the contains() win turned out to be file-count
  overhead, not sort order.
- `executemany` for passing a pair list to DuckDB (600ms for 2.5k
  rows); register an Arrow table instead.

## Deploy tuning (untested, check when the box exists)

- Each DuckDB query parallelizes across all cores; N concurrent
  requests contend multiplicatively on a small box. Consider capping
  DuckDB threads per connection (`SET threads`) and/or limiting server
  concurrency so one exploring agent can't saturate the machine.
- LLM agents should be pointed at the JSON API with explicit `limit`s;
  the same service layer serves it, so all of the above applies.
