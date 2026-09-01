"""Presentation pipeline for the term timeline: pairing, filtering, rendering.

The events table is authoritative — nothing here changes what is recorded. This
module decides *how* to present a commit's events: grouping the query layer's
row stream into (term, commit) units, pairing an ``add`` and a ``remove`` of
the same tag that describe an edit to the same clause, filtering paired
edits by whether the query hit their changed portion, and rendering each op as
an inline ``~`` line with intra-value diff highlighting (git ``--word-diff``
style). Unpaired events render as ``+`` / ``-`` as before.

Everything up to rendering is console-free — the same grouping/pairing/
filtering pipeline serves any consumer of query results (CLI today, HTTP API
later); only the ``render_*`` functions commit to rich ``Text`` output.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby
from cydifflib import SequenceMatcher
from typing import Iterable, Iterator, NamedTuple

from rich.text import Text

from .obo import ParsedValue
from .query import Change, TermChange

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
    def tag(self) -> str:
        return self.change.tag


@dataclass(frozen=True)
class Remove:
    change: Change

    @property
    def tag(self) -> str:
        return self.change.tag


@dataclass(frozen=True)
class Edit:
    tag: str
    before: Change  # the removed value
    after: Change   # the added value


Op = Add | Remove | Edit


def pair_events(
    changes: Iterable[Change], threshold: float = PAIR_THRESHOLD
) -> list[Op]:
    """Pair adds/removes within one tag.

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
        buckets[c.tag].append(c)

    ops: list[Op] = []
    for tag, group in buckets.items():
        adds = [c for c in group if c.operation == "add"]
        removes = [c for c in group if c.operation == "remove"]

        used_r: set[int] = set()
        used_a: set[int] = set()

        # Pass 1: pair by matching parsed body. Each Change carries its
        # value's decomposition from the artifact; ``None`` (possible only
        # on hand-built Changes, e.g. in tests) skips pass 1 and is matched
        # only in pass 2.
        for i, r in enumerate(removes):
            rp = r.parsed
            if rp is None or i in used_r:
                continue
            candidates = [
                j for j, a in enumerate(adds)
                if a.parsed is not None and a.parsed.body == rp.body
                and j not in used_a
            ]
            if not candidates:
                continue
            if len(candidates) == 1:
                # The overwhelmingly common case — and max() would still
                # compute the (expensive) similarity score just to select
                # the only element.
                best_j = candidates[0]
            else:
                best_j = max(
                    candidates,
                    key=lambda j: SequenceMatcher(
                        None, removes[i].value, adds[j].value, autojunk=False
                    ).ratio(),
                )
            used_r.add(i)
            used_a.add(best_j)
            ops.append(Edit(tag=tag, before=removes[i], after=adds[best_j]))

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
            ops.append(Edit(tag=tag, before=removes[i], after=adds[j]))

        for i, r in enumerate(removes):
            if i not in used_r:
                ops.append(Remove(r))
        for j, a in enumerate(adds):
            if j not in used_a:
                ops.append(Add(a))

    ops.sort(key=_sort_key)
    return ops


def _sort_key(op: Op) -> tuple[str, int, str]:
    """Stable within-commit order: by tag, then kind, then value."""
    if isinstance(op, Edit):
        return (op.tag, 0, op.before.value)
    if isinstance(op, Remove):
        return (op.tag, 1, op.change.value)
    return (op.tag, 2, op.change.value)


def _truncate(s: str, cap: int | None) -> str:
    if cap is None or len(s) <= cap:
        return s
    return s[: cap - 1] + ELLIPSIS


def _matches(text: str, query: str, match: str, ignore_case: bool) -> bool:
    """Substring/exact/regex match, honoring ``ignore_case`` — mirrors SQL layer."""
    if match == "regex":
        flags = re.IGNORECASE if ignore_case else 0
        return re.search(query, text, flags) is not None
    if match == "exact":
        if ignore_case:
            return query.lower() == text.lower()
        return query == text
    if ignore_case:
        return query.lower() in text.lower()
    return query in text


def edit_delta_matches(
    edit: Edit,
    query: str,
    match: str = "substring",
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

    ``match="exact"`` compares whole bodies, not tokens: the edit counts
    only if the body itself changed and one side's body is exactly the
    query. A qualifier- or comment-only edit of the queried body is a
    kept-unchanged body — not a match, same philosophy as above.

    Fallback: if either side couldn't be parsed via fastobo, return ``True``
    (safe default; preserves current behavior on the historical malformed
    clauses fastobo rejects).
    """
    before = edit.before.parsed
    after = edit.after.parsed
    if before is None or after is None:
        return True

    def check_text(text: str | None) -> bool:
        return text is not None and _matches(text, query, match, ignore_case)

    if match == "exact":
        if before.body == after.body:
            return False
        return check_text(before.body) or check_text(after.body)

    if before.body != after.body:
        b_tokens = Counter(_tokenize(before.body))
        a_tokens = Counter(_tokenize(after.body))
        for token in list((b_tokens - a_tokens)) + list((a_tokens - b_tokens)):
            if _matches(token, query, match, ignore_case):
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
        if _matches(qualifier, query, match, ignore_case):
            return True
    return False


class PairedCommit(NamedTuple):
    """One (term, commit) group's events, paired into render ops."""

    term_id: str
    name: str | None  # the term's name at this commit, if snapshotted
    head: Change
    ops: list[Op]


def pair_by_term_and_commit(
    events: Iterable[TermChange], order: str = "term"
) -> Iterator[PairedCommit]:
    """Group ``events`` by (term, commit) and pair each group into ops.

    Pairing runs once here; both the search filter and the renderer
    consume the resulting ops, so an ``Edit`` is guaranteed to render
    exactly as it was filtered. ``order`` must name the stream's actual
    sort spine (see :meth:`obohog.query.HistoryDB._iter_term_changes`) so
    groupings are contiguous: ``"term"`` for term-major streams, ``"date"``
    for commit-major ones. Either way each yielded group is one (term,
    commit) pair — only the arrival order differs.

    Lazy: each group is paired as the underlying stream reaches it, so a
    streaming source (:meth:`obohog.query.HistoryDB.iter_search_events`)
    renders its first results long before the full result set has been
    fetched.
    """
    if order == "term":
        key = lambda tc: (tc.term_id, tc.change.commit_seq)
    else:
        key = lambda tc: (tc.change.commit_seq, tc.term_id)
    for _, group in groupby(events, key=key):
        rows = list(group)
        yield PairedCommit(
            rows[0].term_id,
            rows[0].name,
            rows[0].change,
            pair_events([tc.change for tc in rows]),
        )


def filter_ops_by_delta_match(
    ops: list[Op], query: str, match: str, ignore_case: bool
) -> list[Op]:
    """Keep adds/removes; keep edits only if their delta contains the query."""
    return [
        op for op in ops
        if not isinstance(op, Edit)
        or edit_delta_matches(op, query, match, ignore_case)
    ]


def take_sections(
    groups: Iterable[PairedCommit],
    limit: int,
    section_key,
    truncated: list[bool],
) -> Iterator[PairedCommit]:
    """Pass groups through until ``limit`` distinct sections have completed.

    ``section_key`` maps a group to its section identity (term id for
    term-ordered output, commit seq for date-ordered). Stops consuming
    the underlying stream at the section boundary — with a streaming
    source this abandons the query after only a prefix has been fetched.
    Sets ``truncated[0]`` when the limit actually cut something off.
    """
    current = object()
    seen = 0
    for g in groups:
        key = section_key(g)
        if key != current:
            current = key
            seen += 1
            if seen > limit:
                truncated[0] = True
                return
        yield g


class Span(NamedTuple):
    """One run of rendered text with a presentation role, no markup.

    The roles are the full vocabulary any adapter needs: ``same`` renders
    plain, ``del``/``ins`` get the word-diff treatment (bracket markers on
    the terminal, ``<del>``/``<ins>`` in HTML), ``note`` is a dim editorial
    tag (e.g. ``(qualifier order rewritten)``).
    """

    role: str  # "same" | "del" | "ins" | "note"
    text: str


class QualLine(NamedTuple):
    """One indented sub-line of a qualifier block."""

    kind: str  # "context" | "del" | "ins" | "edit"
    spans: list[Span]


class OpView(NamedTuple):
    """A fully-decided rendering of one op, free of any output format.

    ``head`` is the content after ``<marker> <tag>: `` on the top
    line; ``quals`` are the indented qualifier sub-lines (empty except for
    qualifier-block edits). Adapters — the rich terminal renderer below,
    HTML templates — only map roles to markup; every presentation decision
    (dispatch, pairing, truncation, word-diffing) already happened here.
    """

    kind: str  # "add" | "remove" | "edit"
    tag: str
    head: list[Span]
    quals: list[QualLine]


def op_view(op: Op, truncate: int | None = DEFAULT_TRUNCATE) -> OpView:
    """Build the structured rendering of one paired-or-unpaired event."""
    if isinstance(op, Add):
        return OpView(
            "add", op.tag,
            [Span("same", _truncate(op.change.value, truncate))], [],
        )
    if isinstance(op, Remove):
        return OpView(
            "remove", op.tag,
            [Span("same", _truncate(op.change.value, truncate))], [],
        )
    if isinstance(op, Edit):
        return _edit_view(op, truncate)
    raise TypeError(f"unknown op: {op!r}")


def _edit_view(edit: Edit, cap: int | None) -> OpView:
    """View a paired remove/add as one ``~`` op, structure-aware.

    Reads the pairing-time fastobo parses off the ``Edit`` —
    ``(body, qualifiers, ! comment)`` per side — and picks a rendering
    that matches the shape of the change:

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
      marker or as plain context if kept. Reads like an axiom-annotation
      diff, not a run-together sentence.
    * Everything else (including any case where fastobo couldn't parse
      either side) → the token-level word-diff fallback.
    """
    tag = edit.tag
    b = edit.before.parsed
    a = edit.after.parsed

    if b is not None and a is not None:
        body_same = b.body == a.body
        quals_multiset_same = Counter(b.qualifiers) == Counter(a.qualifiers)
        quals_order_same = b.qualifiers == a.qualifiers
        comment_same = b.comment == a.comment

        if body_same and quals_multiset_same:
            if not comment_same:
                return OpView("edit", tag, _comment_only_spans(b, a, cap), [])
            if not quals_order_same:
                return OpView("edit", tag, _reorder_only_spans(a, cap), [])
        if not quals_multiset_same:
            return _qualifier_block_view(tag, b, a, cap)

    return OpView(
        "edit", tag,
        _word_diff_spans(edit.before.value, edit.after.value, cap), [],
    )


def _comment_only_spans(
    before: ParsedValue, after: ParsedValue, cap: int | None
) -> list[Span]:
    """Only the trailing ``!`` name comment differs.

    The shared body + qualifiers render plain and the comment change as a
    single bracketed edit — no token-level word-diff on the label itself.
    We used to tag this case (``(referenced term renamed)``) but the tag
    was making an interpretive leap: sometimes the target really was
    renamed elsewhere, sometimes a label was manually added or removed,
    and the reader can see which from the del/ins marks without us
    projecting a story.
    """
    spans = [Span("same", _truncate(_head(before), cap)), Span("same", " ! ")]
    old = _truncate(before.comment or "", cap)
    new = _truncate(after.comment or "", cap)
    if old:
        spans.append(Span("del", old))
    if new:
        spans.append(Span("ins", new))
    return spans


def _reorder_only_spans(current: ParsedValue, cap: int | None) -> list[Span]:
    """Qualifier multiset unchanged; only the order was rewritten."""
    spans = [Span("same", _truncate(_head(current), cap))]
    if current.comment:
        spans.append(Span("same", f" ! {_truncate(current.comment, cap)}"))
    spans.append(Span("note", "(qualifier order rewritten)"))
    return spans


def _head(pv: ParsedValue) -> str:
    """``body [{qualifiers}]`` — everything shown before the ``!`` comment."""
    if not pv.qualifiers:
        return pv.body
    return f"{pv.body} {{{', '.join(pv.qualifiers)}}}"


def _qual_key(q: str) -> str:
    """The qualifier's key — ``source`` in ``source="DOID:9409"``."""
    return q.split("=", 1)[0]


def _qualifier_block_view(
    tag: str, before: ParsedValue, after: ParsedValue, cap: int | None
) -> OpView:
    """Body + comment on the top line, then the qualifier diff as sub-lines.

    Qualifiers diff as a multiset, grouped by key. Kept qualifiers become
    context lines. Within a key, one removed and one added value pair into
    a single ``~`` line with an inline word-diff — but only when the key is
    single-valued on both sides, where the change really is "this value was
    edited". A repeated key (``source=...``) is a set, and its changes
    render as plain ``-``/``+`` lines: pairing across set members by
    similarity manufactures value edits that never happened
    (``DOID:9409 → MEDGEN:8349`` for what was a drop and an unrelated add).
    """
    head = _body_spans(before.body, after.body, cap)
    head.extend(_comment_tail_spans(before.comment, after.comment, cap))

    kept = Counter(before.qualifiers) & Counter(after.qualifiers)
    removed = Counter(before.qualifiers) - Counter(after.qualifiers)
    added = Counter(after.qualifiers) - Counter(before.qualifiers)
    b_keys = Counter(_qual_key(q) for q in before.qualifiers)
    a_keys = Counter(_qual_key(q) for q in after.qualifiers)

    def take(key: str, source: tuple[str, ...], pool: Counter) -> list[str]:
        """This key's share of ``pool``, in ``source`` order (multiset-aware)."""
        out = []
        for q in source:
            if _qual_key(q) == key and pool[q] > 0:
                pool[q] -= 1
                out.append(q)
        return out

    quals: list[QualLine] = []
    for key in dict.fromkeys(map(_qual_key, (*before.qualifiers, *after.qualifiers))):
        for q in take(key, after.qualifiers, kept):
            quals.append(QualLine("context", [Span("same", _truncate(q, cap))]))
        dels = take(key, before.qualifiers, removed)
        ins = take(key, after.qualifiers, added)
        if dels and ins and b_keys[key] == 1 and a_keys[key] == 1:
            quals.append(QualLine("edit", _word_diff_spans(dels[0], ins[0], cap)))
            continue
        for q in dels:
            quals.append(QualLine("del", [Span("same", _truncate(q, cap))]))
        for q in ins:
            quals.append(QualLine("ins", [Span("same", _truncate(q, cap))]))
    return OpView("edit", tag, head, quals)


def _body_spans(before: str, after: str, cap: int | None) -> list[Span]:
    """Body of the clause: plain if unchanged, word-diffed if changed."""
    if before == after:
        return [Span("same", _truncate(before, cap))]
    return _word_diff_spans(before, after, cap)


def _comment_tail_spans(
    before: str | None, after: str | None, cap: int | None
) -> list[Span]:
    """Trailing ``! comment``: nothing if absent both sides, plain if
    unchanged, del/ins pair if changed."""
    if not before and not after:
        return []
    spans = [Span("same", " ! ")]
    if before == after:
        spans.append(Span("same", _truncate(after or "", cap)))
        return spans
    if before:
        spans.append(Span("del", _truncate(before, cap)))
    if after:
        spans.append(Span("ins", _truncate(after, cap)))
    return spans


def _word_diff_spans(before: str, after: str, cap: int | None) -> list[Span]:
    """The token-level word-diff of ``before`` → ``after`` as spans.

    Runs at the **token** level (see ``_TOKEN_RE``) so identifier swaps,
    snake_case edits, and qualifier-membership changes show as whole-token
    edits rather than character shuffles. Also the fallback for a whole
    edit when fastobo couldn't parse either side.
    """
    b_tokens = _tokenize(_truncate(before, cap))
    a_tokens = _tokenize(_truncate(after, cap))
    spans: list[Span] = []
    matcher = SequenceMatcher(None, b_tokens, a_tokens, autojunk=False)
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            spans.append(Span("same", "".join(b_tokens[i1:i2])))
        elif opcode == "delete":
            spans.append(Span("del", "".join(b_tokens[i1:i2])))
        elif opcode == "insert":
            spans.append(Span("ins", "".join(a_tokens[j1:j2])))
        elif opcode == "replace":
            spans.append(Span("del", "".join(b_tokens[i1:i2])))
            spans.append(Span("ins", "".join(a_tokens[j1:j2])))
    return spans


# ---------------------------------------------------------------------------
# The rich terminal adapter. Everything above is console-free; from here on
# spans and lines become styled ``Text``. del/ins spans use git's
# ``--word-diff=plain`` bracket markers (``[-old-]``, ``{+new+}``) so the
# diff stays readable when piped to a file or a non-color terminal; the
# markers are additionally styled red/green when the console supports it.

_OP_MARKERS = {
    "add": ("+ ", "bold green"),
    "remove": ("- ", "bold red"),
    "edit": ("~ ", "bold yellow"),
}


def render_op(op: Op, truncate: int | None = DEFAULT_TRUNCATE) -> Text:
    """Render one paired-or-unpaired event as a rich ``Text`` line."""
    return render_op_view(op_view(op, truncate))


def render_op_view(view: OpView) -> Text:
    """Map an :class:`OpView` to terminal form: markers, indents, styles."""
    marker, style = _OP_MARKERS[view.kind]
    line = Text("    ")
    line.append(marker, style=style)
    line.append(f"{view.tag}: ")
    _append_spans(line, view.head)
    for ql in view.quals:
        if ql.kind == "context":
            # Context line: no marker, indent aligned with the value column
            # of the marked lines. Same convention as git diff's leading
            # space for unchanged context.
            line.append("\n        ")
        else:
            line.append("\n      ")
            marker, style = _OP_MARKERS[_QUAL_OP_KIND[ql.kind]]
            line.append(marker, style=style)
        _append_spans(line, ql.spans)
    return line


_QUAL_OP_KIND = {"del": "remove", "ins": "add", "edit": "edit"}


def _append_spans(text: Text, spans: list[Span]) -> None:
    for role, s in spans:
        if role == "same":
            text.append(s)
        elif role == "del":
            text.append(f"[-{s}-]", style="red")
        elif role == "ins":
            text.append(f"{{+{s}+}}", style="green")
        elif role == "note":
            text.append(f"  {s}", style="dim")
