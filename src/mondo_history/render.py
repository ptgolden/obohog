"""Terminal rendering for the term timeline.

The events table is authoritative — nothing here changes what is recorded. This
module only decides *how* to present a commit's events: pairing an ``add`` and a
``remove`` of the same predicate that describe an edit to the same clause, and
rendering the pair as an inline ``~`` line with intra-value diff highlighting
(git ``--word-diff`` style). Unpaired events render as ``+`` / ``-`` as before.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

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
    """Render a paired remove/add as one ``~`` line with intra-value diff.

    The diff runs at the **token** level (see ``_TOKEN_RE``), which produces
    readable output on OBO clause values: identifier swaps, snake_case edits,
    and qualifier reorderings show as whole-token deletions and insertions
    rather than character-level shuffles.

    Uses git's ``--word-diff=plain`` bracket markers (``[-old-]``, ``{+new+}``)
    so the diff stays readable when piped to a file or a non-color terminal.
    The markers are additionally styled red/green when the console supports it.
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
