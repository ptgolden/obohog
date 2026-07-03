"""Terminal rendering for the term timeline.

The events table is authoritative — nothing here changes what is recorded. This
module only decides *how* to present a commit's events: pairing an ``add`` and a
``remove`` of the same predicate that describe an edit to the same clause, and
rendering the pair as an inline ``~`` line with intra-value diff highlighting
(git ``--word-diff`` style). Unpaired events render as ``+`` / ``-`` as before.
"""

import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

import fastobo
from rich.text import Text

from .query import Change

PAIR_THRESHOLD = 0.5
DEFAULT_TRUNCATE = 200
ELLIPSIS = "…"

# Tokenizer for the intra-value diff. Each match is one of:
#   * A compound identifier: a word char followed by any run of word chars or
#     the interior-identifier punctuation ``: / . _ -``. This keeps CURIEs
#     (``MONDO:0007739``), URLs (``http://identifiers.org/hgnc/4883``),
#     snake_case names (``has_material_basis_in_germline_mutation_in``), and
#     kboom-style versions (``kboom-pr-1.00/0.77/5.73``) whole. Their parts do
#     not have independent meaning, so we don't want ``SequenceMatcher`` to
#     spuriously align a shared ``:`` or ``_`` in the middle of two different
#     identifiers.
#   * A whitespace run.
#   * Any other single character — structural punctuation like ``{ } [ ] , " = !``
#     stays fine-grained so intra-clause edits (a new evidence code, an added
#     ``source=`` qualifier, a dropped trailing comment) still show precisely.
_TOKEN_RE = re.compile(r"\w[\w:/.\-]*|\s+|[^\w\s]")


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s)


@dataclass(frozen=True)
class Add:
    change: Change

    @property
    def predicate(self) -> str:
        return self.change.predicate


@dataclass(frozen=True)
class Remove:
    change: Change

    @property
    def predicate(self) -> str:
        return self.change.predicate


@dataclass(frozen=True)
class Edit:
    predicate: str
    before: Change  # the removed value
    after: Change   # the added value


Op = Add | Remove | Edit


def pair_events(
    changes: Iterable[Change], threshold: float = PAIR_THRESHOLD
) -> list[Op]:
    """Pair adds/removes within one predicate.

    Two-pass:

    * **Pass 1** — pair by matching parsed **body**. If a remove and an add
      have the same fastobo-parsed body (e.g. both are ``xref: Orphanet:54370
      ...``), they describe edits to the same clause even if their qualifier
      orderings happen to align better with a *different* body's clause.
      This blocks the classic bug where two paired xrefs cross-match by
      lexical similarity because their qualifier text lines up across
      different targets. Ties within a body group go to highest lexical
      similarity.
    * **Pass 2** — greedy lexical similarity for the leftovers, meeting
      ``threshold``. This is where renamed / retyped clauses pair (their
      bodies differ but the text is close).

    Leftover unpaired events fall through as ``Add`` / ``Remove``.
    """
    buckets: dict[str, list[Change]] = defaultdict(list)
    for c in changes:
        buckets[c.predicate].append(c)

    ops: list[Op] = []
    for predicate, group in buckets.items():
        adds = [c for c in group if c.operation == "add"]
        removes = [c for c in group if c.operation == "remove"]

        # Parse each event's body once so both passes can reuse it. ``None``
        # means fastobo couldn't parse — those events skip pass 1 and are
        # matched only in pass 2.
        r_bodies = [_parsed_body(predicate, r.value) for r in removes]
        a_bodies = [_parsed_body(predicate, a.value) for a in adds]

        used_r: set[int] = set()
        used_a: set[int] = set()

        # Pass 1: pair by matching parsed body.
        for i, rb in enumerate(r_bodies):
            if rb is None or i in used_r:
                continue
            candidates = [
                j for j, ab in enumerate(a_bodies)
                if ab == rb and j not in used_a
            ]
            if not candidates:
                continue
            best_j = max(
                candidates,
                key=lambda j: SequenceMatcher(
                    None, removes[i].value, adds[j].value, autojunk=False
                ).ratio(),
            )
            used_r.add(i)
            used_a.add(best_j)
            ops.append(Edit(predicate=predicate, before=removes[i], after=adds[best_j]))

        # Pass 2: greedy lexical similarity for the leftovers.
        scored: list[tuple[float, int, int]] = []
        for i, r in enumerate(removes):
            if i in used_r:
                continue
            for j, a in enumerate(adds):
                if j in used_a:
                    continue
                ratio = SequenceMatcher(None, r.value, a.value, autojunk=False).ratio()
                if ratio >= threshold:
                    scored.append((ratio, i, j))
        scored.sort(reverse=True)
        for _, i, j in scored:
            if i in used_r or j in used_a:
                continue
            used_r.add(i)
            used_a.add(j)
            ops.append(Edit(predicate=predicate, before=removes[i], after=adds[j]))

        for i, r in enumerate(removes):
            if i not in used_r:
                ops.append(Remove(r))
        for j, a in enumerate(adds):
            if j not in used_a:
                ops.append(Add(a))

    ops.sort(key=_sort_key)
    return ops


def _parsed_body(predicate: str, value: str) -> str | None:
    """The fastobo-parsed body of a clause value, or ``None`` if unparseable."""
    pv = parse_clause_value(predicate, value)
    return pv.body if pv is not None else None


def _sort_key(op: Op) -> tuple[str, int, str]:
    """Stable within-commit order: by predicate, then kind, then value."""
    if isinstance(op, Edit):
        return (op.predicate, 0, op.before.value)
    if isinstance(op, Remove):
        return (op.predicate, 1, op.change.value)
    return (op.predicate, 2, op.change.value)


def _truncate(s: str, cap: int | None) -> str:
    if cap is None or len(s) <= cap:
        return s
    return s[: cap - 1] + ELLIPSIS


@dataclass(frozen=True)
class ParsedValue:
    """A clause value split into its OBO-structural parts.

    ``body`` is everything except the ``{qualifiers}`` block and the ``!`` name
    comment — the parts that carry the clause's semantic identity. ``qualifiers``
    preserves the original order (use ``Counter`` for order-independent
    comparison). ``comment`` is the trailing ``!`` text, ``None`` if absent.
    """

    body: str
    qualifiers: tuple[str, ...]
    comment: str | None


_STANZA_TEMPLATE = "format-version: 1.2\n\n[Term]\nid: TMP:0000001\n{tag}: {value}\n"


def parse_clause_value(predicate: str, value: str) -> ParsedValue | None:
    """Parse one OBO clause value using fastobo, returning its structural parts.

    Returns ``None`` when fastobo can't parse the line — some historical
    clauses in the artifact are malformed enough that fastobo rejects (or
    even panics on) them; when that happens we fall back to lexical
    rendering. fastobo already handles the tricky parts (quoted values with
    escapes, ``!`` inside strings, nested brackets), so we don't hand-parse.
    """
    stanza = _STANZA_TEMPLATE.format(tag=predicate, value=value)
    try:
        doc = fastobo.load(io.BytesIO(stanza.encode()))
    except BaseException:  # fastobo can panic, not just raise
        return None
    frames = list(doc)
    if not frames:
        return None
    for clause in frames[0]:
        if clause.raw_tag() == "id":
            continue
        return _clause_to_parsed(clause)
    return None


def _clause_to_parsed(clause) -> ParsedValue:
    """Split ``str(clause)`` into body / qualifiers / comment.

    fastobo gives us the qualifier list and comment as parsed structures.
    Peeling them off ``str(clause)`` (which fastobo serializes deterministically)
    leaves the "body" — the value-carrying prefix — intact for every clause
    kind, including ``def:`` whose trailing ``[xref, xref]`` list belongs to
    the body, not the qualifier block.
    """
    _, _, body = str(clause).partition(": ")
    comment = clause.comment
    qualifiers = tuple(str(q) for q in (clause.qualifiers or []))
    if comment is not None:
        marker = f" ! {comment}"
        if body.endswith(marker):
            body = body[: -len(marker)]
    if qualifiers:
        marker = " {" + ", ".join(qualifiers) + "}"
        if body.endswith(marker):
            body = body[: -len(marker)]
    return ParsedValue(body=body, qualifiers=qualifiers, comment=comment)


def _matches(text: str, query: str, regex: bool, ignore_case: bool) -> bool:
    """Substring or regex match, honoring ``ignore_case`` — mirrors SQL layer."""
    if regex:
        flags = re.IGNORECASE if ignore_case else 0
        return re.search(query, text, flags) is not None
    if ignore_case:
        return query.lower() in text.lower()
    return query in text


def edit_delta_matches(
    edit: Edit,
    query: str,
    regex: bool = False,
    ignore_case: bool = False,
) -> bool:
    """Whether ``query`` appears in the portion of the clause that changed.

    For paired ``Edit`` ops, an add and a remove pair into one ``~`` line and
    only some portion of the clause actually changed. This is what "changed"
    means, precisely:

    * The **body**: token-level symmetric difference of ``before.body`` and
      ``after.body`` (via ``_tokenize`` — the same tokenizer used for the
      word-diff renderer). If the query only appears in tokens that are on
      both sides (kept context inside an otherwise-edited body — e.g. an
      unchanged ``NCIT:C3174`` xref in a synonym evidence list where a
      *different* evidence code was removed), the body doesn't count as a
      match. If the query appears in an added or removed token, it does.
    * The trailing **``!`` comment**: string-level compare (comments are
      short human-readable labels, rarely edited in place).
    * Any **qualifier** in the symmetric difference of the two qualifier
      multisets (i.e. a qualifier that was added or removed, but not one
      present on both sides).

    Kept-unchanged tokens, comment, and qualifiers do **not** count — if
    the query only appears there, the edit's delta doesn't actually
    involve the query.

    Fallback: if either side can't be parsed via fastobo, return ``True``
    (safe default; preserves current behavior on the historical malformed
    clauses fastobo rejects).
    """
    before = parse_clause_value(edit.predicate, edit.before.value)
    after = parse_clause_value(edit.predicate, edit.after.value)
    if before is None or after is None:
        return True

    def check_text(text: str | None) -> bool:
        return text is not None and _matches(text, query, regex, ignore_case)

    if before.body != after.body:
        b_tokens = Counter(_tokenize(before.body))
        a_tokens = Counter(_tokenize(after.body))
        for token in list((b_tokens - a_tokens)) + list((a_tokens - b_tokens)):
            if _matches(token, query, regex, ignore_case):
                return True
    if before.comment != after.comment:
        if check_text(before.comment) or check_text(after.comment):
            return True
    # Qualifier symmetric difference via Counter multiset diff.
    b_counts = Counter(before.qualifiers)
    a_counts = Counter(after.qualifiers)
    only_before = b_counts - a_counts
    only_after = a_counts - b_counts
    for qualifier in list(only_before) + list(only_after):
        if _matches(qualifier, query, regex, ignore_case):
            return True
    return False


def render_op(op: Op, truncate: int | None = DEFAULT_TRUNCATE) -> Text:
    """Render one paired-or-unpaired event as a rich ``Text`` line."""
    if isinstance(op, Add):
        return _render_plain(op.predicate, op.change.value, "+", "bold green", truncate)
    if isinstance(op, Remove):
        return _render_plain(op.predicate, op.change.value, "-", "bold red", truncate)
    if isinstance(op, Edit):
        return _render_edit(op.predicate, op.before.value, op.after.value, truncate)
    raise TypeError(f"unknown op: {op!r}")


def _render_plain(predicate: str, value: str, marker: str, style: str, cap: int | None) -> Text:
    line = Text("    ")
    line.append(f"{marker} ", style=style)
    line.append(f"{predicate}: {_truncate(value, cap)}")
    return line


def _render_edit(predicate: str, before: str, after: str, cap: int | None) -> Text:
    """Render a paired remove/add as one ``~`` line, structure-aware.

    Parses both sides via fastobo into ``(body, qualifiers, ! comment)`` and
    picks a rendering that matches the shape of the change:

    * ``body`` + qualifier set identical, only ``!`` comment differs →
      render shared form plain and the comment change as one bracketed
      edit (no token-level word-diff on the label).
    * ``body`` + comment identical, qualifier multiset identical but ordered
      differently → serialization reshuffle. Render current form, tag
      ``(qualifier order rewritten)`` since the visible content would
      otherwise look unchanged.
    * Qualifier multiset differs (anywhere) → render as a **block**: body +
      comment on the top ``~`` line (word-diffed inline if they changed),
      then each qualifier on its own indented sub-line with a ``-``/``+``/``~``
      marker or dim if kept. Reads like an axiom-annotation diff, not a
      run-together sentence.
    * Everything else (including any case where fastobo can't parse either
      side) → the token-level word-diff fallback.
    """
    b = parse_clause_value(predicate, before)
    a = parse_clause_value(predicate, after)

    if b is not None and a is not None:
        body_same = b.body == a.body
        quals_multiset_same = Counter(b.qualifiers) == Counter(a.qualifiers)
        quals_order_same = b.qualifiers == a.qualifiers
        comment_same = b.comment == a.comment

        if body_same and quals_multiset_same:
            if not comment_same:
                return _render_comment_only(predicate, b, a, cap)
            if not quals_order_same:
                return _render_reorder_only(predicate, a, cap)
        if not quals_multiset_same:
            return _render_qualifier_block(predicate, b, a, cap)

    return _render_token_diff(predicate, before, after, cap)


def _render_comment_only(
    predicate: str, before: ParsedValue, after: ParsedValue, cap: int | None
) -> Text:
    """Only the trailing ``!`` name comment differs.

    Render the shared body + qualifiers plain and the comment change as a
    single bracketed edit — no token-level word-diff on the label itself.
    We used to tag this case (``(referenced term renamed)``) but the tag
    was making an interpretive leap: sometimes the target really was
    renamed elsewhere, sometimes a label was manually added or removed,
    and the reader can see which from the ``[-...-] {+...+}`` marks
    without us projecting a story.
    """
    line = Text("    ")
    line.append("~ ", style="bold yellow")
    line.append(f"{predicate}: ")
    line.append(_truncate(_head(before), cap))
    line.append(" ! ")
    old = _truncate(before.comment or "", cap)
    new = _truncate(after.comment or "", cap)
    if old:
        line.append(f"[-{old}-]", style="red")
    if new:
        line.append(f"{{+{new}+}}", style="green")
    return line


def _render_reorder_only(
    predicate: str, current: ParsedValue, cap: int | None
) -> Text:
    """Qualifier multiset unchanged; only the order was rewritten."""
    line = Text("    ")
    line.append("~ ", style="bold yellow")
    line.append(f"{predicate}: ")
    line.append(_truncate(_head(current), cap))
    if current.comment:
        line.append(f" ! {_truncate(current.comment, cap)}")
    line.append("  (qualifier order rewritten)", style="dim")
    return line


def _head(pv: ParsedValue) -> str:
    """``body [{qualifiers}]`` — everything shown before the ``!`` comment."""
    if not pv.qualifiers:
        return pv.body
    return f"{pv.body} {{{', '.join(pv.qualifiers)}}}"


def _render_qualifier_block(
    predicate: str, before: ParsedValue, after: ParsedValue, cap: int | None
) -> Text:
    """Render a body + comment on the top line, then indent the qualifier diff.

    The qualifier list diffs as a sequence: kept qualifiers render dim as
    context, inserts as ``+``, deletes as ``-``. A ``replace`` opcode gets
    sub-paired by similarity so a qualifier whose value was edited (same
    ``key`` on both sides, different value) shows as one ``~`` line with an
    inline word-diff, rather than a ``-`` / ``+`` pair.
    """
    text = Text("    ")
    text.append("~ ", style="bold yellow")
    text.append(f"{predicate}: ")
    _append_body(text, before.body, after.body, cap)
    _append_comment_tail(text, before.comment, after.comment, cap)

    matcher = SequenceMatcher(None, before.qualifiers, after.qualifiers, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for q in before.qualifiers[i1:i2]:
                # Context line: no marker, indent aligned with the value column
                # of the marked lines. Same convention as git diff's leading
                # space for unchanged context.
                text.append("\n        ")
                text.append(_truncate(q, cap))
        elif tag == "delete":
            for q in before.qualifiers[i1:i2]:
                text.append("\n      ")
                text.append("- ", style="bold red")
                text.append(_truncate(q, cap))
        elif tag == "insert":
            for q in after.qualifiers[j1:j2]:
                text.append("\n      ")
                text.append("+ ", style="bold green")
                text.append(_truncate(q, cap))
        elif tag == "replace":
            removes = list(before.qualifiers[i1:i2])
            adds = list(after.qualifiers[j1:j2])
            pairs, unpaired_r, unpaired_a = _pair_strings(removes, adds)
            for r, a in pairs:
                text.append("\n      ")
                text.append("~ ", style="bold yellow")
                _append_word_diff(text, r, a, cap)
            for q in unpaired_r:
                text.append("\n      ")
                text.append("- ", style="bold red")
                text.append(_truncate(q, cap))
            for q in unpaired_a:
                text.append("\n      ")
                text.append("+ ", style="bold green")
                text.append(_truncate(q, cap))
    return text


def _append_body(text: Text, before: str, after: str, cap: int | None) -> None:
    """Body of the clause: plain if unchanged, word-diffed if changed."""
    if before == after:
        text.append(_truncate(before, cap))
    else:
        _append_word_diff(text, before, after, cap)


def _append_comment_tail(
    text: Text, before: str | None, after: str | None, cap: int | None
) -> None:
    """Trailing ``! comment``: skip if absent both sides, plain if unchanged,
    bracketed pair if changed."""
    if not before and not after:
        return
    text.append(" ! ")
    if before == after:
        text.append(_truncate(after or "", cap))
        return
    if before:
        text.append(f"[-{_truncate(before, cap)}-]", style="red")
    if after:
        text.append(f"{{+{_truncate(after, cap)}+}}", style="green")


def _pair_strings(
    removes: list[str], adds: list[str], threshold: float = 0.4
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Greedy similarity pairing of removed/added strings.

    Used for qualifier sub-pairing inside a ``replace`` opcode: a qualifier
    whose value changed (e.g. ``source="X"`` → ``source="Y"``) usually pairs
    with its highest-similarity counterpart, letting us render it as one
    ``~`` line instead of separate ``-`` / ``+`` lines. Threshold is lower
    than the top-level pair threshold because we've already narrowed to a
    single ``replace`` region and want to catch smaller-similarity same-key
    edits.
    """
    scored: list[tuple[float, int, int]] = []
    for i, r in enumerate(removes):
        for j, a in enumerate(adds):
            ratio = SequenceMatcher(None, r, a, autojunk=False).ratio()
            if ratio >= threshold:
                scored.append((ratio, i, j))
    scored.sort(reverse=True)
    used_r: set[int] = set()
    used_a: set[int] = set()
    pairs: list[tuple[str, str]] = []
    for _, i, j in scored:
        if i in used_r or j in used_a:
            continue
        used_r.add(i)
        used_a.add(j)
        pairs.append((removes[i], adds[j]))
    unpaired_r = [q for i, q in enumerate(removes) if i not in used_r]
    unpaired_a = [q for j, q in enumerate(adds) if j not in used_a]
    return pairs, unpaired_r, unpaired_a


def _append_word_diff(text: Text, before: str, after: str, cap: int | None) -> None:
    """Append the token-level word-diff of ``before`` → ``after`` to ``text``.

    Runs at the **token** level (see ``_TOKEN_RE``) so identifier swaps,
    snake_case edits, and qualifier-membership changes show as whole-token
    edits rather than character shuffles. Uses git's ``--word-diff=plain``
    bracket markers (``[-old-]``, ``{+new+}``) so the diff stays readable
    when piped to a file or a non-color terminal; the markers are
    additionally styled red/green when the console supports it.
    """
    b_tokens = _tokenize(_truncate(before, cap))
    a_tokens = _tokenize(_truncate(after, cap))
    matcher = SequenceMatcher(None, b_tokens, a_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            text.append("".join(b_tokens[i1:i2]))
        elif tag == "delete":
            text.append(f"[-{''.join(b_tokens[i1:i2])}-]", style="red")
        elif tag == "insert":
            text.append(f"{{+{''.join(a_tokens[j1:j2])}+}}", style="green")
        elif tag == "replace":
            text.append(f"[-{''.join(b_tokens[i1:i2])}-]", style="red")
            text.append(f"{{+{''.join(a_tokens[j1:j2])}+}}", style="green")


def _render_token_diff(
    predicate: str, before: str, after: str, cap: int | None
) -> Text:
    """Fallback: token-level word-diff over the raw value strings.

    Used when fastobo can't parse either side, or when there are no
    qualifiers to break out into a block. The output is a single ``~`` line.
    """
    line = Text("    ")
    line.append("~ ", style="bold yellow")
    line.append(f"{predicate}: ")
    _append_word_diff(line, before, after, cap)
    return line
