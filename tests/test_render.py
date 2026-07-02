"""Unit tests for the terminal renderer (pairing + intra-value diff)."""

from mondo_history.query import Change
from mondo_history.render import (
    DEFAULT_TRUNCATE,
    Add,
    Edit,
    Remove,
    pair_events,
    parse_clause_value,
    render_op,
)


def _change(op: str, predicate: str, value: str, seq: int = 1) -> Change:
    return Change(
        commit_seq=seq,
        committed_date="2026-01-01",
        sha="a" * 40,
        pr_number=None,
        message="msg",
        operation=op,
        predicate=predicate,
        value=value,
    )


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
    # At least the three obvious case-flip pairs should have been detected.
    assert len(edits) >= 3
    edit_pairs = {(o.before.value, o.after.value) for o in edits}
    assert (
        '"Cfh Deficiency" [OMIM]',
        '"Cfh deficiency" [OMIM]',
    ) in edit_pairs
    assert (
        '"Factor H Deficiency" [OMIM]',
        '"Factor H deficiency" [OMIM]',
    ) in edit_pairs


def test_pair_events_unrelated_stays_split():
    # Two totally different values must not pair — ratio well below threshold.
    changes = [
        _change("add", "xref", "OMIM:123456"),
        _change("remove", "xref", "MESH:C562875"),
    ]
    ops = pair_events(changes)
    kinds = sorted(type(o).__name__ for o in ops)
    assert kinds == ["Add", "Remove"]


def test_pair_events_scoped_to_predicate():
    # An add and remove that would pair by content-similarity are NOT paired
    # if their predicates differ.
    changes = [
        _change("add", "synonym", '"foo"'),
        _change("remove", "xref", '"foo"'),
    ]
    ops = pair_events(changes)
    kinds = sorted(type(o).__name__ for o in ops)
    assert kinds == ["Add", "Remove"]


def test_pair_events_add_only_and_remove_only():
    changes = [
        _change("add", "synonym", '"new"'),
        _change("remove", "synonym", '"old"'),
        _change("add", "xref", "OMIM:1"),
    ]
    ops = pair_events(changes)
    # "new" vs "old" too dissimilar to pair.
    kinds = sorted(type(o).__name__ for o in ops)
    assert kinds == ["Add", "Add", "Remove"]


def test_render_op_add_and_remove_prefixes():
    add = Add(_change("add", "synonym", '"foo"'))
    rem = Remove(_change("remove", "synonym", '"bar"'))
    assert render_op(add).plain.startswith("    + synonym: ")
    assert render_op(rem).plain.startswith("    - synonym: ")


def test_render_op_edit_shows_intra_value_diff():
    changes = [
        _change("remove", "synonym", '"Cfh Deficiency"'),
        _change("add", "synonym", '"Cfh deficiency"'),
    ]
    ops = pair_events(changes)
    assert isinstance(ops[0], Edit)
    line = render_op(ops[0])
    assert line.plain.startswith("    ~ synonym: ")
    # Token-level word-diff: whole token gets replaced, readable without color.
    assert "[-Deficiency-]" in line.plain
    assert "{+deficiency+}" in line.plain
    # Shared context stays plain.
    assert "Cfh " in line.plain


def test_render_op_edit_snake_case_relationship_is_whole_token_swap():
    # A relationship-type rename is a whole-identifier change: the pieces of
    # ``disease_has_basis_in_dysfunction_of`` don't have independent meaning,
    # so trying to align at ``_`` is misleading. The rename must swap
    # wholesale, leaving the untouched URL and qualifier tail in place.
    changes = [
        _change(
            "remove", "relationship",
            "disease_has_basis_in_dysfunction_of http://x.example/hgnc/4883 "
            '{source="mim2gene_medgen"}',
        ),
        _change(
            "add", "relationship",
            "has_material_basis_in_germline_mutation_in http://x.example/hgnc/4883 "
            '{source="mim2gene_medgen"}',
        ),
    ]
    ops = pair_events(changes)
    assert isinstance(ops[0], Edit)
    line = render_op(ops[0])
    assert (
        "[-disease_has_basis_in_dysfunction_of-]"
        "{+has_material_basis_in_germline_mutation_in+}"
    ) in line.plain
    assert "http://x.example/hgnc/4883" in line.plain
    assert '{source="mim2gene_medgen"}' in line.plain


def test_render_op_edit_url_swap_is_whole_token_swap():
    # NCBIGene:3075 and http://identifiers.org/hgnc/4883 share only spurious
    # punctuation (``:``). They must swap wholesale, not letter by letter.
    changes = [
        _change(
            "remove", "relationship",
            "disease_has_basis_in_dysfunction_of NCBIGene:3075 "
            '{source="mim2gene_medgen"}',
        ),
        _change(
            "add", "relationship",
            "disease_has_basis_in_dysfunction_of http://identifiers.org/hgnc/4883 "
            '{source="mim2gene_medgen"}',
        ),
    ]
    ops = pair_events(changes)
    assert isinstance(ops[0], Edit)
    line = render_op(ops[0])
    assert "[-NCBIGene:3075-]{+http://identifiers.org/hgnc/4883+}" in line.plain
    # No character-level bleed like "[-NCBIGene-]{+http+}:[-3075-]".
    assert "[-NCBIGene-]" not in line.plain
    assert "{+http+}" not in line.plain


def test_truncate_long_value_by_default():
    long = "x" * (DEFAULT_TRUNCATE + 50)
    add = Add(_change("add", "def", long))
    line = render_op(add)
    assert "…" in line.plain
    assert len(line.plain) < len(long) + 20  # prefix overhead only


def test_full_disables_truncate():
    long = "x" * (DEFAULT_TRUNCATE + 50)
    add = Add(_change("add", "def", long))
    line = render_op(add, truncate=None)
    assert "…" not in line.plain
    assert long in line.plain


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


def test_render_op_edit_target_label_only():
    # Same is_a target and qualifiers; only the ! label differs (referenced
    # term was renamed elsewhere). This is textually a change but the clause
    # itself is semantically unchanged.
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
    ops = pair_events(changes)
    assert isinstance(ops[0], Edit)
    line = render_op(ops[0])
    # The shared form ("body {qualifiers}") stays plain — no token-diff noise.
    assert 'MONDO:0018013 {source="Orphanet:329918/btnt"}' in line.plain
    # The comment change is bracketed as a whole, not word-diffed piece by piece.
    assert (
        "[-non-immunoglobulin-mediated membranoproliferative glomerulonephritis-]"
        in line.plain
    )
    assert "{+complement 3 glomerulopathy+}" in line.plain
    # And a labeled tag makes the semantics clear.
    assert "(target label)" in line.plain


def test_render_op_edit_qualifier_reorder_is_a_labeled_no_op():
    # Same body, same qualifier multiset, different order → a pure
    # serialization reshuffle. Render as unchanged form with a tag.
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
    ops = pair_events(changes)
    assert isinstance(ops[0], Edit)
    line = render_op(ops[0])
    # No [-...-] / {+...+} markers — the semantics are unchanged.
    assert "[-" not in line.plain
    assert "{+" not in line.plain
    assert "(qualifier order rewritten)" in line.plain
    # The current (post-edit) form is displayed.
    assert 'MESH:C562875' in line.plain
    assert 'MONDO:ontobio' in line.plain
    assert 'MONDO:equivalentTo' in line.plain


def test_render_op_edit_falls_through_when_body_changes():
    # Body differs → structural detectors don't fire; the token word-diff
    # renders normally.
    changes = [
        _change("remove", "xref", "OMIM:1 {source=\"A\"}"),
        _change("add", "xref", "OMIM:2 {source=\"A\"}"),
    ]
    ops = pair_events(changes)
    assert isinstance(ops[0], Edit)
    line = render_op(ops[0])
    # No structural tag; a real word-diff appears.
    assert "(target label)" not in line.plain
    assert "(qualifier order rewritten)" not in line.plain
    assert "[-OMIM:1-]{+OMIM:2+}" in line.plain
