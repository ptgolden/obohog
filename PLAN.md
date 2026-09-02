# obohog — Original Vision

*A queryable history of OBO ontology evolution.*

> **Status:** partially implemented. This document has been updated to reflect
> decisions made during prototyping. See `DESIGN.md` for the concrete Parquet
> schema and `README.md` to run the CLI.
>
> This document was originally written for Mondo specifically; the tool has
> since been generalized to work with any OBO Foundry-style ontology whose
> source is in a git repository. Where "Mondo" appears below, treat it as an
> illustrative example — the same reasoning holds for GO, HP, ChEBI, PATO,
> and any other similarly maintained OBO ontology.

## Overview

Git is the authoritative source of an OBO ontology's history, but in practice that history is not available during normal development.

Large OBO repositories grow to the point where maintaining a complete clone with full history is impractical. Development is therefore performed using shallow clones (for example, `--depth 1`), which intentionally omit historical commits.

As a result, historical investigation becomes awkward. Questions such as

* When did this mapping change?
* Why was this synonym removed?
* When did this classification change?
* Has this issue occurred before?

cannot be answered directly from a local checkout.

Instead, developers must resort to the GitHub web interface, manually search commit history, inspect pull requests, or invent one-off techniques to reconstruct the evolution of the ontology. These approaches are often slow, incomplete, and disruptive to normal development workflows.

This is particularly unfortunate because many ontology debugging and maintenance tasks are fundamentally historical. Understanding the current state of the ontology frequently requires understanding how that state was reached.

`obohog` addresses this by treating complete Git history as a build-time dependency rather than a user dependency.

A history extraction process runs once per source, against a complete repository clone, and produces a compact, queryable database describing the evolution of one OBO file (e.g. `mondo-edit.obo`, `go-edit.obo`, `pato-edit.obo`). Developers query these databases directly rather than traversing repository history.

The result is that historical investigation becomes a routine part of ontology development instead of an exceptional task requiring web searches, bespoke scripts, or access to a complete repository clone.

---

# Goals

The project aims to provide a historical view of a version-controlled OBO ontology that is:

* organized around ontology concepts rather than Git commits
* fast to query
* reproducible
* distributable as a generated artifact
* usable without cloning the source ontology's repository
* suitable for both local and hosted use

The emphasis is on supporting day-to-day ontology development, debugging, review, and maintenance.

Rather than asking "Which commit changed this file?", users should be able to ask questions such as:

* When did this term last change?
* What changed for this term over time?
* When was this synonym introduced?
* When did this xref disappear?
* How has this disease classification evolved?
* Which terms changed together?
* What happened between two releases?
* Why does this term look the way it does today?

---

# Non-Goals

`obohog` is **not** intended to:

* replace Git
* replace repository history
* become a second source of truth
* store a mutable copy of the ontology

Git remains authoritative. The history index is a derived representation built solely to improve historical querying.

---

# Design Principles

## Ontology-Centric

The primary unit of navigation should be ontology entities rather than commits.

Users should begin with a MONDO identifier or another ontology concept—not a commit SHA.

## Derived Artifact

The history index is generated from repository history.

The expensive work of processing Git history should occur once during artifact generation rather than every time a user asks a question.

## Independent of Repository History

Historical queries should not depend on the presence of Git history.

The generated artifact should contain sufficient historical information to answer routine ontology questions without requiring users to clone or search the complete repository.

## Deterministic

Historical queries should produce deterministic answers based on a specific version of the generated artifact.

Each source's database should be versioned alongside the underlying ontology's release cadence (or another well-defined cadence) so that historical results are reproducible.

---

# Conceptual Architecture

The system has three logical components, per configured source.

```text
              Source Repository
                        │
                        │
              History Extraction
                        │
                        ▼
         Generated History Database
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
    Local CLI                  Hosted Query Service
```

Only the extraction process needs Git access — and even then not a full clone
of the whole repository. It builds a **blob-filtered clone**
(`git clone --filter=blob:none`: commit graph and trees, no file contents) and
then fetches only the historical contents of the single edited OBO file (as
declared per source in `obohog.toml`) via a **scoped, delta-packed**
`git backfill --sparse` (with the sparse-checkout set to just that file). So
extraction downloads the history of one file and nothing else, in a handful
of batched requests rather than one fetch per version.

All user-facing tools operate against the generated history artifact, never Git.

This separation is fundamental to the design.

---

# History Artifact

The generated history artifact is the central product of the system.

It should be:

* compact
* highly compressible
* portable
* query-efficient
* immutable once generated
* versioned alongside Mondo

The artifact is a small set of **Parquet** files (columnar, heavily compressible,
engine-neutral, archival) queried with **DuckDB**. DuckDB reads the Parquet
locally *and* range-queries it over plain HTTP, so a hosted deployment can be
static file hosting (e.g. GitHub Releases) with no server — clients fetch only the
byte ranges a query touches.

The files share `commit_seq` as a single linear time axis. Updates append new
Parquet part-files rather than rewriting existing ones (see *Build & Update
Model*), so consumers can pull just the delta. The full artifact is estimated at
roughly 100–300 MB, dominated by term snapshots and highly compressible.

See `DESIGN.md` for the concrete schema.

---

# Conceptual Data Model

Rather than organizing history around commits, the system organizes history around ontology entities.

Conceptually:

```text
Git history
        │
        ▼
Structured ontology changes
        │
        ▼
Entity history index
        │
        ▼
Historical queries
```

Each commit contributes historical events associated with ontology entities.

The representation is now decided:

* **Term snapshots are the primitive.** A term's full normalized state is stored
  only on the commits where it changed (detected by content-hashing each term
  frame). Reconstructing a term at any point is "the latest snapshot at or before
  that commit."
* **Clause-level events are derived** by diffing adjacent snapshots of a term:
  `(term_id, commit_seq, predicate, value, add|remove)`. This is the queryable
  spine for "when did this synonym / xref / parent change." A term's creation and
  removal are recoverable from snapshot presence, so they need no dedicated event
  kind.

The important design decision is unchanged: users interact with ontology history
rather than Git history.

---

# Build & Update Model

## Full build

The initial build walks every commit that touched the edited file (~7,500 over
Mondo's history), oldest first, parsing each version and diffing it against the
previous one to emit snapshots and events.

* **Parsing is parallel.** Parsing each ~45 MB version dominates the cost and is
  independent per commit, so the commit range is split into contiguous chunks
  across cores; each chunk parses its own commits plus one "seed" commit from the
  previous chunk so boundary diffs stay correct. Diffing itself is a cheap
  sequential hash-compare.
* **Unparseable commits are skipped and carried forward.** A few historical
  versions cannot be parsed (the OBO parser rejects, or even panics on, some old
  malformed clauses). Such a commit is recorded in a `skipped_commits` table and
  does *not* advance the "previous" state, so its changes fold into the next
  parseable commit rather than crashing the build. Nothing is lost; attribution is
  slightly lumpy across the bad commit.

## Incremental updates

Because Git history is append-only, keeping the artifact current does **not**
require reprocessing history — only the new commits.

* **Self-seeding:** the ontology's full state at the last built commit is
  recovered from the artifact itself (the latest snapshot per term), so an update
  needs no side-car state.
* **Cheap fetch:** `git fetch` for the new commits, then `git backfill --sparse`
  for just the new file blobs.
* **Stable axis:** new commits extend `commit_seq` (`last + 1, +2, …`); existing
  values never change, so published data and cached results stay valid.
* **Append, don't rewrite:** each update writes new Parquet part-files; consumers
  download only the delta.
* **Rewrite guard:** the last built HEAD sha is recorded; if it is no longer an
  ancestor of the new HEAD (history was rewritten), the update falls back to a
  full rebuild.

The steady state is therefore one expensive full build, then perpetual cheap
appends — a typical OBO release adds a handful of commits, seconds of work,
and a few kilobytes of new part-files.

---

# Query Model

Representative queries include:

* Show the complete history of `MONDO:...`.
* Show every modification to xrefs for a term.
* Show when a synonym was introduced or removed.
* Show the history of parent relationships.
* Which commits modified this term?
* Which terms changed together?
* Which ontology entities were affected by this pull request?
* What changed between two releases for this disease?
* Explain how the current state of this term evolved over time.

The design should support both interactive exploration and scripted workflows.

---

# Access Model

The history artifact should support multiple access methods while remaining the single source for historical queries.

Potential interfaces include:

* a command-line interface
* a hosted web application
* a programmatic API

Each interface should operate over the same generated artifact and produce consistent answers.

The web application should expose the same information available through the downloadable artifact rather than maintaining an independent representation.

---

# Future Directions

`obohog` is now source-agnostic in its architecture: `obohog.toml` declares any number of OBO sources, each with its own repo + file. Any version-controlled `.obo` file works today.

Longer term:

* **Other OBO serializations** — shipped for OWL edit files (functional notation, RDF/XML): a source declares `format = "owl"` and each version is converted to OBO via ROBOT before extraction, cached per blob. Remaining: `[Typedef]` (relation) tracking, without which relation-centric ontologies like RO yield little. See DESIGN.md §4 and `OWL-PLAN.md`.
* **Prefix migrations across renames** — for ontologies that changed CURIE prefix at some point (e.g. Mondo's `TBD:X` → `MONDO:X` in 2017), declare the mapping in the source's config so `obohog term MONDO:0000450` transparently includes the pre-rename history. Design in `2026-07-03-note.term-identity-across-renames.md`.
* **Incremental artifact updates** — top of the queue. Right now `source sync` re-clones and rebuilds fully; the plan is to append new part-files instead.

---

# Open Questions

Several of the original open questions are now decided:

* **Granularity of a historical event** — a clause-level add/remove, derived from
  term snapshots (see *Conceptual Data Model*).
* **How changes are represented internally** — normalized OBO clauses (via
  fastobo's own serialization), diffed as multisets.
* **Point-in-time reconstruction** — supported directly: the latest snapshot at or
  before a commit.
* **Release information** — a `releases` table maps each Git tag to the
  file-history commit at or before it, enabling "between two releases" queries.
* **Completeness vs size** — full snapshots on every change keep reconstruction
  trivial; the artifact is an estimated ~100–300 MB and highly compressible. A
  leaner "keyframe + event replay" variant is a known lever if size ever matters
  more than reconstruction simplicity.

Still open, and best informed by real workflows:

* How much contextual information should accompany each event beyond the linked
  commit / PR?
* What indexing strategy best supports interactive queries at scale?
* Which historical questions arise most frequently during ontology development?

These questions should be informed primarily by real debugging and maintenance
workflows rather than implementation convenience.

---

# Vision

`obohog` aims to make an OBO ontology's history as easy to explore as the ontology itself.

Instead of manually traversing commits, GitHub history, and pull requests, developers should be able to ask historical questions directly and receive answers in terms of ontology concepts.

The project is fundamentally about improving the observability of ontology evolution. By transforming repository history into a compact, queryable database, it enables historical investigation to become a routine part of ontology development rather than a specialized form of repository archaeology.

