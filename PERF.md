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

## Measured dead ends (don't revisit without new evidence)

- Dropping the `term_snapshots` LEFT JOIN: no gain (DuckDB handles it).
- The ORDER BY: 0.06s over 133k rows — not the bottleneck.
- Re-issuing the stream past capped commits (tried, discarded before
  merge): each re-issue re-paid the full sort, ~0.4s × ~12 capped
  commits per page. Planning the page first is strictly better.

## Deploy tuning (untested, check when the box exists)

- Each DuckDB query parallelizes across all cores; N concurrent
  requests contend multiplicatively on a small box. Consider capping
  DuckDB threads per connection (`SET threads`) and/or limiting server
  concurrency so one exploring agent can't saturate the machine.
- LLM agents should be pointed at the JSON API with explicit `limit`s;
  the same service layer serves it, so all of the above applies.
