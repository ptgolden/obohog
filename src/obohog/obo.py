"""Parse an OBO document into normalized, hashable per-term state.

Normalization leans entirely on fastobo: every clause serializes to its canonical
OBO line (``predicate: value``) via ``str(clause)``, so we never hand-maintain a
mapping of clause classes. A term's state is the *set* of those clauses plus a
content hash, which is what lets the extractor detect "did this term change?"
cheaply and diff two versions clause-by-clause.
"""

import hashlib
import io
import re
from collections import Counter
from dataclasses import dataclass

import fastobo

# A stanza header line, e.g. "[Term]\n" or "[Typedef]\n", at the start of a line.
_STANZA_RE = re.compile(rb"(?m)^\[[^\]\n]+\]\n")


@dataclass(frozen=True)
class ParsedValue:
    """A clause value split into its OBO-structural parts.

    ``body`` is everything except the ``{qualifiers}`` block and the ``!`` name
    comment — the parts that carry the clause's semantic identity. ``qualifiers``
    preserves the original order (use ``Counter`` for order-independent
    comparison). ``comment`` is the trailing ``!`` text, ``None`` if absent.

    Invariant: ``body`` + qualifier block + comment recomposes to the
    serialized clause value byte-for-byte (see :func:`decompose_clause`).
    """

    body: str
    qualifiers: tuple[str, ...]
    comment: str | None


@dataclass(frozen=True, order=True)
class Clause:
    """One canonical OBO clause of a term, split into tag and remainder.

    ``predicate`` is the OBO tag (``name``, ``synonym``, ``xref``, ``is_a``,
    ``relationship``, ``subset``, ``def``, ``is_obsolete``, ``replaced_by``, ...);
    ``value`` is the rest of the serialized line. ``parsed`` is ``value``'s
    structural decomposition, captured from the live fastobo clause object at
    parse time. It is a pure function of ``(predicate, value)``, so ordering
    and equality are still decided by those two fields alone — comparisons
    never reach ``parsed`` with unequal values.
    """

    predicate: str
    value: str
    parsed: ParsedValue


@dataclass(frozen=True)
class TermState:
    """A term's normalized content at one point in history."""

    term_id: str
    clauses: tuple[Clause, ...]  # sorted, canonical
    content_hash: str


def clauses_of(frame: fastobo.term.TermFrame) -> tuple[Clause, ...]:
    """Canonical, order-independent clause set for a term frame."""
    out = []
    for clause in frame:
        predicate, _, value = str(clause).partition(": ")
        out.append(Clause(predicate, value, decompose_clause(clause, value)))
    return tuple(sorted(out))


def decompose_clause(clause, value: str) -> ParsedValue:
    """Split a clause's serialized ``value`` into body / qualifiers / comment.

    fastobo gives us the qualifier list and comment as parsed structures.
    Peeling them off ``value`` (which fastobo serializes deterministically)
    leaves the "body" — the value-carrying prefix — intact for every clause
    kind, including ``def:`` whose trailing ``[xref, xref]`` list belongs to
    the body, not the qualifier block.

    A field is recorded only if its marker was actually peeled off the
    tail. fastobo sometimes reports a comment or qualifiers that don't
    sit at the end of the serialization — notably ``comment:`` clauses,
    where ``clause.comment`` is the clause's *value*, not a trailing
    ``!`` comment. Dropping unpeeled fields keeps the invariant that
    ``body`` + qualifier block + comment recomposes to ``value``, byte
    for byte.
    """
    body = value
    comment = clause.comment
    qualifiers = tuple(str(q) for q in (clause.qualifiers or []))
    if comment is not None:
        marker = f" ! {comment}"
        if body.endswith(marker):
            body = body[: -len(marker)]
        else:
            comment = None
    if qualifiers:
        marker = " {" + ", ".join(qualifiers) + "}"
        if body.endswith(marker):
            body = body[: -len(marker)]
        else:
            qualifiers = ()
    return ParsedValue(body=body, qualifiers=qualifiers, comment=comment)


def hash_clauses(clauses: tuple[Clause, ...]) -> str:
    h = hashlib.sha1()
    for clause in clauses:
        h.update(clause.predicate.encode())
        h.update(b"\x00")
        h.update(clause.value.encode())
        h.update(b"\x00")
    return h.hexdigest()


def clause_delta(
    before: tuple[Clause, ...], after: tuple[Clause, ...]
) -> tuple[list[Clause], list[Clause]]:
    """Return ``(added, removed)`` clauses between two states of a term.

    Clauses are compared as a multiset, so an edited single-valued field (a new
    ``name``) and a swapped multi-valued entry (one synonym replaced by another)
    both surface as a removal paired with an addition.
    """
    b, a = Counter(before), Counter(after)
    added = list((a - b).elements())
    removed = list((b - a).elements())
    return added, removed


def split_document(data: bytes) -> tuple[bytes, dict[str, bytes]]:
    """Split an OBO document into a reusable parse context and per-term stanzas.

    Returns ``(context, {term_id: stanza_bytes})`` where ``context`` is the header
    plus every non-``[Term]`` stanza (typedefs, instances) — everything needed to
    parse any single term stanza in isolation. This is a cheap byte-level scan, no
    fastobo, so it lets the extractor find which terms changed without parsing the
    whole file.
    """
    matches = list(_STANZA_RE.finditer(data))
    if not matches:
        return data, {}
    context = [data[: matches[0].start()]]
    terms: dict[str, bytes] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(data)
        stanza = data[match.start() : end]
        if match.group().strip() == b"[Term]":
            term_id = _stanza_id(stanza)
            if term_id is not None:
                terms[term_id] = stanza
                continue
        context.append(stanza)
    return b"".join(context), terms


def _stanza_id(stanza: bytes) -> str | None:
    for line in stanza.split(b"\n"):
        if line.startswith(b"id: "):
            # The id is the first whitespace-delimited token; anything after it is
            # a trailing OBO comment (" ! label"). No id contains whitespace, so
            # this matches fastobo's parsed id exactly.
            tokens = line[4:].split()
            if tokens:
                return tokens[0].decode()
    return None


def stanza_hash(stanza: bytes) -> bytes:
    return hashlib.sha1(stanza).digest()


def parse_stanzas(
    context: bytes, stanzas: dict[str, bytes]
) -> tuple[dict[str, TermState], list[str]]:
    """Parse the given term stanzas (with ``context``), isolating any failures.

    Returns ``(parsed, failed_ids)``. A batch that fails to parse — fastobo may
    raise *or panic* on a malformed historical clause — is bisected until the
    offending stanza is isolated, so one bad term never sinks the whole batch.
    """
    parsed: dict[str, TermState] = {}
    failed: list[str] = []
    _parse_batch(context, stanzas, list(stanzas), parsed, failed)
    return parsed, failed


def _parse_batch(
    context: bytes,
    stanzas: dict[str, bytes],
    ids: list[str],
    parsed: dict[str, TermState],
    failed: list[str],
) -> None:
    if not ids:
        return
    blob = context + b"".join(stanzas[i] for i in ids)
    try:
        doc = fastobo.load(io.BytesIO(blob), threads=1)
        frames = [f for f in doc if isinstance(f, fastobo.term.TermFrame)]
    except BaseException:  # fastobo can panic, not just raise
        if len(ids) == 1:
            failed.append(ids[0])
            return
        mid = len(ids) // 2
        _parse_batch(context, stanzas, ids[:mid], parsed, failed)
        _parse_batch(context, stanzas, ids[mid:], parsed, failed)
        return
    wanted = set(ids)
    for frame in frames:
        term_id = str(frame.id)
        if term_id in wanted:
            clauses = clauses_of(frame)
            parsed[term_id] = TermState(term_id, clauses, hash_clauses(clauses))


@dataclass(frozen=True)
class TermDelta:
    """One term's change at a document version: new state + clause-level diff."""

    term: TermState
    added: list[Clause]
    removed: list[Clause]


@dataclass(frozen=True)
class CommitDelta:
    """The semantic result of advancing a :class:`DocumentState` by one version.

    ``changed`` holds terms whose canonical content changed — including
    brand-new terms, whose delta is ∅ → full clause set. ``removed`` holds
    the last known state of terms that disappeared. ``failed`` records
    stanzas that couldn't be handled at this version, as ``(term_id, kind)``
    with kind ``"ParseError"`` or ``"IdMismatch"``.
    """

    changed: list[TermDelta]
    removed: list[TermState]
    failed: list[tuple[str, str]]


class DocumentState:
    """Evolving parsed state of an OBO document along a version walk.

    Owns the two maps that make diff-scoped parsing work:

    * ``_terms`` — term_id → last successfully parsed :class:`TermState`;
    * ``_raw`` — term_id → hash of the stanza bytes already *seen* (parsed
      **or** failed), so :meth:`apply` re-parses a stanza only when its
      bytes change. A failing stanza lands in ``_raw`` but never in
      ``_terms``: it keeps its last good state and isn't re-attempted until
      its content changes again.
    """

    def __init__(self) -> None:
        self._terms: dict[str, TermState] = {}
        self._raw: dict[str, bytes] = {}

    @classmethod
    def from_blob(cls, blob: bytes) -> "DocumentState":
        """Full state of one document version, for seeding a mid-walk start.

        Per-stanza parse failures are isolated and simply omitted — not
        recorded in ``_raw`` — so a still-failing stanza is re-attempted
        (and re-reported) once at the start of each seeded walk rather
        than silently carried.
        """
        state = cls()
        context, stanzas = split_document(blob)
        parsed, _failed = parse_stanzas(context, stanzas)
        for term_id, term in parsed.items():
            state._terms[term_id] = term
            state._raw[term_id] = stanza_hash(stanzas[term_id])
        return state

    def apply(self, blob: bytes) -> CommitDelta:
        """Advance to this document version; return what changed.

        Only stanzas whose raw bytes differ from the last version are handed
        to fastobo; byte-identical stanzas keep their carried-forward state.
        Applying to an empty state reports every term as created — a change
        from ∅ to its full clause set — which is the story we want in the
        events table.
        """
        context, stanzas = split_document(blob)
        cur_hash = {mid: stanza_hash(s) for mid, s in stanzas.items()}
        changed_ids = [mid for mid in stanzas if cur_hash[mid] != self._raw.get(mid)]
        removed_ids = self._raw.keys() - stanzas.keys()
        parsed, failed_ids = parse_stanzas(
            context, {mid: stanzas[mid] for mid in changed_ids}
        )
        failed = [(term_id, "ParseError") for term_id in failed_ids]
        for term_id in failed_ids:
            self._raw[term_id] = cur_hash[term_id]
        failed_set = set(failed_ids)

        changed: list[TermDelta] = []
        for term_id in changed_ids:
            if term_id in failed_set:
                continue
            term = parsed.get(term_id)
            if term is None:
                # Stanza parsed, but fastobo keyed it under a different id
                # than our text-level scan did; report and move on.
                failed.append((term_id, "IdMismatch"))
                self._raw[term_id] = cur_hash[term_id]
                continue
            before = self._terms.get(term_id)
            self._raw[term_id] = cur_hash[term_id]
            if before is not None and before.content_hash == term.content_hash:
                continue  # bytes changed but canonical content did not
            added, removed = clause_delta(
                before.clauses if before else (), term.clauses
            )
            changed.append(TermDelta(term=term, added=added, removed=removed))
            self._terms[term_id] = term

        removed_terms: list[TermState] = []
        for term_id in removed_ids:
            del self._raw[term_id]
            term = self._terms.pop(term_id, None)
            if term is not None:  # else it only ever failed; nothing to remove
                removed_terms.append(term)
        return CommitDelta(changed=changed, removed=removed_terms, failed=failed)


def parse_terms(data: bytes, threads: int = 1) -> dict[str, TermState]:
    """Parse one OBO document into ``{term_id: TermState}`` for its term frames.

    Non-term frames (typedefs, instances) and the header are ignored — the index
    is about ontology terms.

    ``threads=1`` (the default) uses fastobo's single-threaded parser: it avoids
    CPU oversubscription when many builds run in parallel, and sidesteps the
    threaded parser's habit of *panicking* (rather than raising) on a few
    malformed historical clauses. Callers still guard against parse failure.
    """
    doc = fastobo.load(io.BytesIO(data), threads=threads)
    result: dict[str, TermState] = {}
    for frame in doc:
        if isinstance(frame, fastobo.term.TermFrame):
            term_id = str(frame.id)
            clauses = clauses_of(frame)
            result[term_id] = TermState(term_id, clauses, hash_clauses(clauses))
    return result
