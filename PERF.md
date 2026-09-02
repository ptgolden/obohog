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

**Plan-then-fetch generalizes to delta searches** — SHIPPED (1131606,
after the layout change shipped in d655a71 with `obohog source
compact` for pre-existing artifacts): one memoized candidate scan
serves counts, the page plan, and (via a commit-bounded, pair-joined
fetch) every page of the query. Measured end-to-end on compacted
mondo: cold first page 165 ms, warm 100 ms, load-more 64 ms (vs
392/271 ms before this round); walking all 11 pages of the typical
query costs 2.0 s total. Behavior delta shipped with it: pages plan by
candidate sections, so the delta filter can leave fewer than `limit`
on a page (cursor semantics unchanged; the empty-state renders only on
final pages).

Trap for the implementation: never `executemany` the pairs into a temp
table — 2.5k inserts cost 600 ms. `con.register()` an Arrow table
(zero-copy, instant). Restricting the name join to page pairs is not
worth it (49→45 ms).

## Round 3: pairing is the tail cost (2026-09-02)

A broad text query on vbo (`q=dog`, matches most values in a breed
ontology) cost 2.46 s/page with the planned path — profiled to ~1.99 s
of `pair_events`, which was ~185k `SequenceMatcher.ratio()` calls, 99%
of them inside per-term `property_value` buckets (release commits
rewrite dozens per term; the values are long URLs). Two facts found on
the way: render.py already uses **cydifflib** (stdlib difflib would be
~5× slower still — don't "simplify" that import away), and quick_ratio
prefilters barely help at threshold 0.5 (ontology values share
character distributions, the multiset bound passes almost everything).

Fix (8cd46d9): block property_value pairing by the property's *local
name* (survives prefix/IRI migrations like http://purl.org/dc/terms/
source → terms:source → dcterms:source), pair pure respellings —
value identical past the property token — by dict lookup, keep a
leftover similarity round across blocks for renamed properties.
Decision-identical on all 24,348 groups of two full real result sets;
pairing 1.98→0.44 s, the dog page 2.46→0.98 s (~0.7 s once vbo is
compacted). If pairing ever dominates again, the next candidates are a
per-(term, commit) ops memo (groups are immutable) or extending the
rest-of-value exact pass to other storm-prone tags.

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
- Partitioning events by namespace (e.g. a MONDO file + an "other"
  file, queries routed by the namespace filter): measured dead flat —
  scoped match scan 37 ms on both layouts, unscoped 40 ms on both, and
  the `starts_with` namespace predicate itself costs ~1 ms on a broad
  scan. MONDO is already 87.8% of mondo's events (96.3% of snapshots),
  and DuckDB evaluates the cheap dictionary-encoded term_id predicate
  before touching the value column, so physical partitioning has
  nothing left to skip. Defaulting the UI to the primary namespace is
  therefore purely a product choice — no performance stakes either way.

## Deploy tuning (untested, check when the box exists)

- Each DuckDB query parallelizes across all cores; N concurrent
  requests contend multiplicatively on a small box. Consider capping
  DuckDB threads per connection (`SET threads`) and/or limiting server
  concurrency so one exploring agent can't saturate the machine.
- LLM agents should be pointed at the JSON API with explicit `limit`s;
  the same service layer serves it, so all of the above applies.
