"""Unit tests for the terminal renderer.

Marker syntax (``[-old-]``, ``{+new+}``, indented ``+``/``-``/``~`` sub-lines
on qualifier blocks) is a rendering decision that could change. Tests use
the small helper functions at the top so assertions describe *what* got
deleted, inserted, added, or edited — not the exact byte sequence carrying
that information.
"""

import re

from mondo_history.query import Change
from mondo_history.render import (
    DEFAULT_TRUNCATE,
    edit_delta_matches,
    Add,
    Edit,
    Remove,
    _tokenize,
    pair_events,
    parse_clause_value,
    render_op,
)


def _change(op: str, predicate: str, value: str, seq: int = 1) -> Change:
    return Change(
        commit_seq=seq,
        committed_date="2026-01-01",
        sha="a" * 40,
        author_name="Test Author",
        pr_number=None,
        message="msg",
        operation=op,
        predicate=predicate,
        value=value,
    )


# ---------------------------------------------------------------------------
# Helpers that decode the current marker syntax. Change these if the format
# changes, not every assertion.

_DELETION_RE = re.compile(r"\[-(.+?)-\]")
_INSERTION_RE = re.compile(r"\{\+(.+?)\+\}")


def _deletions(text) -> list[str]:
    return _DELETION_RE.findall(text.plain)


def _insertions(text) -> list[str]:
    return _INSERTION_RE.findall(text.plain)


def _first_line(text) -> str:
    return text.plain.split("\n", 1)[0].strip()


def _sub_lines(text) -> list[str]:
    return [l.strip() for l in text.plain.split("\n")[1:] if l.strip()]


def _kept_qualifiers(text) -> list[str]:
    """Sub-lines with no ``+``/``-``/``~`` marker — kept context qualifiers."""
    return [l for l in _sub_lines(text) if l and l[0] not in "+-~"]


def _added_qualifiers(text) -> list[str]:
    return [l[2:] for l in _sub_lines(text) if l.startswith("+ ")]


def _removed_qualifiers(text) -> list[str]:
    return [l[2:] for l in _sub_lines(text) if l.startswith("- ")]


def _edited_qualifiers(text) -> list[tuple[list[str], list[str]]]:
    """Per ``~`` sub-line, the deleted and inserted spans it carries."""
    out = []
    for line in _sub_lines(text):
        if line.startswith("~ "):
            out.append((_DELETION_RE.findall(line), _INSERTION_RE.findall(line)))
    return out


# ---------------------------------------------------------------------------
# Tokenizer — the intra-value diff quality hinges on this regex keeping
# compound identifiers whole and splitting on structural punctuation.


def test_tokenize_keeps_curies_whole():
    assert _tokenize("MONDO:0012350") == ["MONDO:0012350"]


def test_tokenize_keeps_urls_whole():
    assert _tokenize("http://identifiers.org/hgnc/4883") == [
        "http://identifiers.org/hgnc/4883"
    ]


def test_tokenize_keeps_snake_case_whole():
    assert _tokenize("has_material_basis_in_germline_mutation_in") == [
        "has_material_basis_in_germline_mutation_in"
    ]


def test_tokenize_splits_structural_punctuation():
    # Braces, quotes, equals, comma each become their own token so intra-clause
    # edits (added evidence code, added source= qualifier, etc.) diff cleanly.
    assert _tokenize('{source="X"}') == [
        "{", "source", "=", '"', "X", '"', "}",
    ]


# ---------------------------------------------------------------------------
# pair_events — pairing decisions on structural Add/Remove/Edit objects.


def test_pair_events_case_flip():
    # Two synonyms differing only in case → one Edit.
    changes = [
        _change("add", "synonym", '"Cfh deficiency" RELATED [OMIM:609814]'),
        _change("remove", "synonym", '"Cfh Deficiency" RELATED [OMIM:609814]'),
    ]
    ops = pair_events(changes)
    assert len(ops) == 1
    assert isinstance(ops[0], Edit)
    assert ops[0].before.value == '"Cfh Deficiency" RELATED [OMIM:609814]'
    assert ops[0].after.value == '"Cfh deficiency" RELATED [OMIM:609814]'


def test_pair_events_multiple_pair_best_match():
    # Four adds + four removes on the same predicate: greedy best-first pairs
    # each add with its closest remove.
    changes = [
        _change("remove", "synonym", '"Cfh Deficiency" [OMIM]'),
        _change("remove", "synonym", '"Factor H Deficiency" [OMIM]'),
        _change("remove", "synonym", '"complement FACTOR H DEFICIENCY" [OMIM]'),
        _change("remove", "synonym", '"complement FACTOR H DEFICIENCY; CFHD" [OMIM]'),
        _change("add", "synonym", '"Cfh deficiency" [OMIM]'),
        _change("add", "synonym", '"Factor H deficiency" [OMIM]'),
        _change("add", "synonym", '"complement FACTOR H deficiency; CFHD" [OMIM]'),
        _change("add", "synonym", '"complement factor H deficiency" EXACT [OMIM]'),
    ]
    ops = pair_events(changes)
    edits = [o for o in ops if isinstance(o, Edit)]
    assert len(edits) >= 3
    edit_pairs = {(o.before.value, o.after.value) for o in edits}
    assert ('"Cfh Deficiency" [OMIM]', '"Cfh deficiency" [OMIM]') in edit_pairs
    assert (
        '"Factor H Deficiency" [OMIM]', '"Factor H deficiency" [OMIM]',
    ) in edit_pairs


def test_pair_events_unrelated_stays_split():
    # Two totally different values must not pair — ratio well below threshold.
    changes = [
        _change("add", "xref", "OMIM:123456"),
        _change("remove", "xref", "MESH:C562875"),
    ]
    ops = pair_events(changes)
    assert sorted(type(o).__name__ for o in ops) == ["Add", "Remove"]


def test_pair_events_scoped_to_predicate():
    # An add and remove that would pair by content-similarity are NOT paired
    # if their predicates differ.
    changes = [
        _change("add", "synonym", '"foo"'),
        _change("remove", "xref", '"foo"'),
    ]
    ops = pair_events(changes)
    assert sorted(type(o).__name__ for o in ops) == ["Add", "Remove"]


def test_pair_events_prefers_matching_body_over_lexical_similarity():
    # Regression: at commit 1476, two Orphanet xrefs (different IDs) had their
    # qualifier orderings rewritten. Naive greedy pairing scored the cross
    # pairs higher because their qualifier text happened to align across the
    # different-ID clauses, inventing a non-existent "one xref became
    # another" edit. Same-body pairing wins even when the cross pair scores
    # higher lexically.
    changes = [
        _change("remove", "xref",
                'Orphanet:54370 {source="OMIM:609814", source="MONDO:subClassOf"}'),
        _change("add", "xref",
                'Orphanet:54370 {source="MONDO:subClassOf", source="OMIM:609814"}'),
        _change("remove", "xref",
                'Orphanet:93571 {source="MONDO:directSiblingOf", source="OMIM:609814"}'),
        _change("add", "xref",
                'Orphanet:93571 {source="OMIM:609814", source="MONDO:directSiblingOf"}'),
    ]
    ops = pair_events(changes)
    edits = [o for o in ops if isinstance(o, Edit)]
    assert len(edits) == 2
    for e in edits:
        # Same target CURIE on both sides — no cross-body pairing.
        assert e.before.value.split(" ")[0] == e.after.value.split(" ")[0]


def test_pair_events_add_only_and_remove_only():
    changes = [
        _change("add", "synonym", '"new"'),
        _change("remove", "synonym", '"old"'),
        _change("add", "xref", "OMIM:1"),
    ]
    ops = pair_events(changes)
    # "new" vs "old" too dissimilar to pair.
    assert sorted(type(o).__name__ for o in ops) == ["Add", "Add", "Remove"]


# ---------------------------------------------------------------------------
# parse_clause_value — fastobo-backed split of a value into
# (body, qualifiers, comment).


def test_parse_clause_value_splits_body_qualifiers_comment():
    pv = parse_clause_value(
        "is_a",
        'MONDO:0018013 {source="Orphanet:329918/btnt"} ! non-immunoglobulin-mediated',
    )
    assert pv is not None
    assert pv.body == "MONDO:0018013"
    assert pv.qualifiers == ('source="Orphanet:329918/btnt"',)
    assert pv.comment == "non-immunoglobulin-mediated"


def test_parse_clause_value_def_keeps_xref_list_in_body():
    # For def:, the trailing [xref, xref] list is part of the body, not the
    # qualifier block. It must survive parsing intact.
    pv = parse_clause_value("def", '"Long definition." [OMIM:1, Orphanet:2]')
    assert pv is not None
    assert pv.body == '"Long definition." [OMIM:1, Orphanet:2]'
    assert pv.qualifiers == ()
    assert pv.comment is None


# ---------------------------------------------------------------------------
# Rendering — behavior of the various dispatch branches. Assertions use the
# helper functions above so they describe *what* was rendered, not the
# byte-level marker syntax.


def test_render_op_edit_body_only_uses_token_word_diff_fallback():
    # No qualifiers, no comment; body differs → the token word-diff fallback.
    changes = [
        _change("remove", "xref", "OMIM:1"),
        _change("add", "xref", "OMIM:2"),
    ]
    line = render_op(pair_events(changes)[0])
    assert _deletions(line) == ["OMIM:1"]
    assert _insertions(line) == ["OMIM:2"]
    assert "\n" not in line.plain  # single line — no qualifier block


def test_render_op_edit_comment_only_leaves_shared_form_plain():
    # Same body + qualifiers; only the ! comment differs. Shared form renders
    # plain and the comment change is bracketed as one atomic edit — no
    # token-level word-diff on the label.
    changes = [
        _change(
            "remove", "is_a",
            'MONDO:0018013 {source="Orphanet:329918/btnt"} ! '
            "non-immunoglobulin-mediated membranoproliferative glomerulonephritis",
        ),
        _change(
            "add", "is_a",
            'MONDO:0018013 {source="Orphanet:329918/btnt"} ! '
            "complement 3 glomerulopathy",
        ),
    ]
    line = render_op(pair_events(changes)[0])
    # The body + qualifiers portion is untouched.
    assert 'MONDO:0018013 {source="Orphanet:329918/btnt"}' in line.plain
    # Exactly one deletion and one insertion — the whole old/new comment.
    assert _deletions(line) == [
        "non-immunoglobulin-mediated membranoproliferative glomerulonephritis"
    ]
    assert _insertions(line) == ["complement 3 glomerulopathy"]


def test_render_op_edit_qualifier_reorder_marks_as_tagged_no_op():
    # Same body, same qualifier multiset, different order → serialization
    # reshuffle. Tagged since the visible content would otherwise look
    # unchanged; no [-...-] / {+...+} markers.
    changes = [
        _change(
            "remove", "xref",
            'MESH:C562875 {source="MONDO:ontobio", source="MONDO:equivalentTo"}',
        ),
        _change(
            "add", "xref",
            'MESH:C562875 {source="MONDO:equivalentTo", source="MONDO:ontobio"}',
        ),
    ]
    line = render_op(pair_events(changes)[0])
    assert _deletions(line) == []
    assert _insertions(line) == []
    assert "(qualifier order rewritten)" in line.plain
    # The current form is displayed as context.
    assert "MESH:C562875" in line.plain
    assert "MONDO:ontobio" in line.plain
    assert "MONDO:equivalentTo" in line.plain


def test_render_op_edit_qualifier_added_marks_only_the_new_one():
    # Adding one qualifier: kept qualifier as context, added as "+" sub-line.
    changes = [
        _change("remove", "xref", 'Orphanet:200421 {source="OMIM:609814"}'),
        _change(
            "add", "xref",
            'Orphanet:200421 {source="OMIM:609814", source="MONDO:superClassOf"}',
        ),
    ]
    text = render_op(pair_events(changes)[0])
    assert _first_line(text) == "~ xref: Orphanet:200421"
    assert _kept_qualifiers(text) == ['source="OMIM:609814"']
    assert _added_qualifiers(text) == ['source="MONDO:superClassOf"']
    assert _removed_qualifiers(text) == []
    assert _edited_qualifiers(text) == []


def test_render_op_edit_qualifier_removed_marks_only_the_gone_one():
    # Removing one of two qualifiers: kept as context, removed as "-" sub-line.
    changes = [
        _change(
            "remove", "relationship",
            'has_material_basis_in http://x.example/hgnc/4883 '
            '{source="MONDO:mim2gene_medgen", source="OMIM:609814"} ! CFH',
        ),
        _change(
            "add", "relationship",
            'has_material_basis_in http://x.example/hgnc/4883 '
            '{source="OMIM:609814"} ! CFH',
        ),
    ]
    text = render_op(pair_events(changes)[0])
    top = _first_line(text)
    assert top.startswith("~ relationship: has_material_basis_in")
    assert top.endswith("! CFH")
    assert _kept_qualifiers(text) == ['source="OMIM:609814"']
    assert _removed_qualifiers(text) == ['source="MONDO:mim2gene_medgen"']
    assert _added_qualifiers(text) == []
    assert _edited_qualifiers(text) == []


def test_render_op_edit_qualifier_value_edited_pairs_as_tilde_subline():
    # Same qualifier key on both sides, different value: pair as one ~
    # sub-line with an inline word-diff. Not two separate -/+ lines.
    changes = [
        _change(
            "remove", "is_a",
            'MONDO:0016244 {source="Orphanet:2134/btnt"} ! ahus',
        ),
        _change(
            "add", "is_a",
            'MONDO:0016244 {source="ORDO:2134/btnt"} ! ahus',
        ),
    ]
    text = render_op(pair_events(changes)[0])
    assert _first_line(text) == "~ is_a: MONDO:0016244 ! ahus"
    edits = _edited_qualifiers(text)
    assert edits == [(["Orphanet:2134/btnt"], ["ORDO:2134/btnt"])]
    assert _added_qualifiers(text) == []
    assert _removed_qualifiers(text) == []


def test_render_op_edit_body_diff_plus_qualifier_edit():
    # Both body and qualifier value changed: body word-diffs on the top line,
    # qualifier edit indented as a ~ sub-line.
    changes = [
        _change(
            "remove", "relationship",
            'has_material_basis_in_germline_mutation_in '
            'http://identifiers.org/hgnc/4883 {source="MONDO:mim2gene_medgen"} ! CFH',
        ),
        _change(
            "add", "relationship",
            'disease_has_basis_in_dysfunction_of '
            'http://identifiers.org/hgnc/4883 {source="mim2gene_medgen"} ! CFH',
        ),
    ]
    text = render_op(pair_events(changes)[0])
    # The body word-diff shows the whole snake_case rename swapping wholesale
    # (tokenizer keeps compound identifiers whole — see the _tokenize tests).
    top = _first_line(text)
    assert "has_material_basis_in_germline_mutation_in" in top
    assert "disease_has_basis_in_dysfunction_of" in top
    assert "http://identifiers.org/hgnc/4883" in top
    assert top.endswith("! CFH")
    # And the qualifier edit is a ~ sub-line pairing the two source= forms.
    edits = _edited_qualifiers(text)
    assert edits == [(['MONDO:mim2gene_medgen'], ['mim2gene_medgen'])]


# ---------------------------------------------------------------------------
# Truncation.


def test_truncate_long_value_by_default():
    long = "x" * (DEFAULT_TRUNCATE + 50)
    line = render_op(Add(_change("add", "def", long)))
    assert "…" in line.plain
    assert len(line.plain) < len(long) + 20  # prefix overhead only


def test_full_disables_truncate():
    long = "x" * (DEFAULT_TRUNCATE + 50)
    line = render_op(Add(_change("add", "def", long)), truncate=None)
    assert "…" not in line.plain
    assert long in line.plain


# ---------------------------------------------------------------------------
# edit_delta_matches — the clause-aware search filter. Determines whether a
# paired Edit's *delta* (body-diff, comment-diff, qualifier symmetric
# difference) contains the query. Kept-unchanged qualifiers do not count.


def _edit(before_val: str, after_val: str, predicate: str = "is_a") -> Edit:
    return Edit(
        predicate=predicate,
        before=_change("remove", predicate, before_val),
        after=_change("add", predicate, after_val),
    )


def test_edit_delta_matches_only_in_kept_qualifier_returns_false():
    # Regression from real-artifact commit 4533f68 on MONDO:0011996: body and
    # comment are unchanged; ONCOTREE:CML is present on both sides (a kept
    # qualifier); the only actual delta is "+ source=indirect". Query "CML"
    # does not appear in the delta.
    kept = 'source="ONCOTREE:CML"'
    before = 'MONDO:0020076 {source="EFO:0000339", ' + kept + '} ! neoplasm'
    after = (
        'MONDO:0020076 {source="EFO:0000339", ' + kept
        + ', source="indirect"} ! neoplasm'
    )
    assert edit_delta_matches(_edit(before, after), "CML") is False


def test_edit_delta_matches_added_qualifier_returns_true():
    # Same base, add a new qualifier that contains the query.
    before = 'MONDO:0020076 {source="A"}'
    after = 'MONDO:0020076 {source="A", source="ONCOTREE:CML"}'
    assert edit_delta_matches(_edit(before, after), "CML") is True


def test_edit_delta_matches_removed_qualifier_returns_true():
    # Same base, remove a qualifier that contained the query.
    before = 'MONDO:0020076 {source="A", source="ONCOTREE:CML"}'
    after = 'MONDO:0020076 {source="A"}'
    assert edit_delta_matches(_edit(before, after), "CML") is True


def test_edit_delta_matches_body_change_returns_true():
    # Body changed; the query appears in the new body.
    assert edit_delta_matches(_edit("MONDO:0000001", "MONDO:0000002"), "0002") is True


def test_edit_delta_matches_kept_token_inside_edited_body_returns_false():
    # Real case from `mondo-history search C317`: a synonym's evidence list
    # had CSP2005:2004-1700 removed. NCIT:C3174 stayed put but happens to
    # contain the substring "C317". The commit didn't actually change
    # anything involving C317 — that xref reference was unchanged context
    # inside an edited body.
    before = (
        '"CML" EXACT ABBREVIATION [CSP2005:2004-1700, DOID:8552, '
        'NCIT:C3174, OMIM:608232]'
    )
    after = '"CML" EXACT ABBREVIATION [DOID:8552, NCIT:C3174, OMIM:608232]'
    # Body differs (CSP evidence removed). NCIT:C3174 is present on both
    # sides — a kept token. C317 doesn't appear in the changed tokens.
    assert edit_delta_matches(_edit(before, after, "synonym"), "C317") is False
    # But a query that matches the changed token (or any part of it) hits.
    assert edit_delta_matches(_edit(before, after, "synonym"), "CSP2005") is True


def test_edit_delta_matches_comment_change_returns_true():
    # Body + qualifiers identical; only the ! comment differs and it contains
    # the query.
    before = 'MONDO:0020076 {source="A"} ! old name'
    after = 'MONDO:0020076 {source="A"} ! new CML name'
    assert edit_delta_matches(_edit(before, after), "CML") is True


def test_edit_delta_matches_regex_ignore_case():
    # Regex + ignore-case: query pattern with lowercase word-boundary matches
    # an uppercase CML in the added qualifier.
    before = 'MONDO:0020076 {source="A"}'
    after = 'MONDO:0020076 {source="A", source="ONCOTREE:CML"}'
    assert edit_delta_matches(
        _edit(before, after), r"\bcml\b", regex=True, ignore_case=True
    ) is True


def test_edit_delta_matches_unparseable_falls_through_to_true():
    # If fastobo can't parse either side, we return True as a safe default so
    # historical malformed clauses stay visible in search results.
    weird = 'not a real OBO clause {{{ garbage'
    assert edit_delta_matches(_edit(weird, weird), "anything") is True
