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
    """Pair adds/removes within one predicate by highest similarity.

    Greedy: score every (remove, add) pair with ``SequenceMatcher.ratio``,
    take the highest scoring pair whose ratio meets ``threshold``, mark both
    used, repeat. Leftover unmatched events fall through as ``Add``/``Remove``.

    The input is expected to be one commit's worth of events, but the function
    only relies on grouping by predicate — it is fine to pass more.
    """
    buckets: dict[str, list[Change]] = defaultdict(list)
    for c in changes:
        buckets[c.predicate].append(c)

    ops: list[Op] = []
    for predicate, group in buckets.items():
        adds = [c for c in group if c.operation == "add"]
        removes = [c for c in group if c.operation == "remove"]

        scored: list[tuple[float, int, int]] = []
        for i, r in enumerate(removes):
            for j, a in enumerate(adds):
                ratio = SequenceMatcher(None, r.value, a.value, autojunk=False).ratio()
                if ratio >= threshold:
                    scored.append((ratio, i, j))
        scored.sort(reverse=True)

        used_r: set[int] = set()
        used_a: set[int] = set()
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
    """Render a paired remove/add as one ``~`` line.

    First tries to parse both sides via fastobo and pick a **structure-aware**
    rendering for cases where lexical word-diff misleads:

    * ``body`` + qualifier set identical, only the ``!`` comment differs →
      the referenced target term was renamed elsewhere; this clause is
      semantically unchanged. The shared form renders plain and only the
      comment change gets marked.
    * ``body`` + comment identical, qualifier multiset identical but ordered
      differently → a pure serialization reshuffle. Render the current form
      and tag it "(qualifier order rewritten)".

    Everything else — including any case where fastobo can't parse either
    side — falls through to the token-level word-diff.
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

    return _render_token_diff(predicate, before, after, cap)


def _render_comment_only(
    predicate: str, before: ParsedValue, after: ParsedValue, cap: int | None
) -> Text:
    """Only the trailing ``!`` name comment differs (target term was renamed)."""
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
    line.append("  (target label)", style="dim")
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


def _render_token_diff(
    predicate: str, before: str, after: str, cap: int | None
) -> Text:
    """Fallback: token-level word-diff over the raw value strings.

    Runs at the **token** level (see ``_TOKEN_RE``) so identifier swaps,
    snake_case edits, and qualifier-membership changes show as whole-token
    edits rather than character shuffles. Uses git's ``--word-diff=plain``
    bracket markers (``[-old-]``, ``{+new+}``) so the diff stays readable
    when piped to a file or a non-color terminal; the markers are
    additionally styled red/green when the console supports it.
    """
    b_tokens = _tokenize(_truncate(before, cap))
    a_tokens = _tokenize(_truncate(after, cap))
    line = Text("    ")
    line.append("~ ", style="bold yellow")
    line.append(f"{predicate}: ")
    matcher = SequenceMatcher(None, b_tokens, a_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            line.append("".join(b_tokens[i1:i2]))
        elif tag == "delete":
            line.append(f"[-{''.join(b_tokens[i1:i2])}-]", style="red")
        elif tag == "insert":
            line.append(f"{{+{''.join(a_tokens[j1:j2])}+}}", style="green")
        elif tag == "replace":
            line.append(f"[-{''.join(b_tokens[i1:i2])}-]", style="red")
            line.append(f"{{+{''.join(a_tokens[j1:j2])}+}}", style="green")
    return line
