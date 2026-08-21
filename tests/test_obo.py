"""Tests for OBO normalization, hashing, and clause diffing."""

from obohog.obo import (
    clause_delta,
    hash_clauses,
    parse_stanzas,
    parse_terms,
    split_document,
)

HEADER = b"format-version: 1.2\n\n"


def _doc(*stanzas: str) -> bytes:
    return HEADER + "\n".join(stanzas).encode()


TERM_A = "[Term]\nid: MONDO:0000001\nname: disease\n"
TERM_A_SYN = TERM_A + 'synonym: "illness" EXACT []\n'
TERM_B = "[Term]\nid: MONDO:0000002\nname: cancer\n"


def test_parse_terms_indexes_by_id():
    terms = parse_terms(_doc(TERM_A, TERM_B))
    assert set(terms) == {"MONDO:0000001", "MONDO:0000002"}
    assert any(c.predicate == "name" and c.value == "disease"
               for c in terms["MONDO:0000001"].clauses)


def test_hash_is_stable_and_content_sensitive():
    a = parse_terms(_doc(TERM_A))["MONDO:0000001"]
    a_again = parse_terms(_doc(TERM_A))["MONDO:0000001"]
    a_syn = parse_terms(_doc(TERM_A_SYN))["MONDO:0000001"]

    assert a.content_hash == a_again.content_hash
    assert a.content_hash != a_syn.content_hash


def test_source_clause_order_does_not_affect_hash():
    # Same content, clauses written in a different order, must hash identically:
    # normalization canonicalizes clause order so reordering is not a "change".
    ordered = parse_terms(_doc(TERM_A_SYN))["MONDO:0000001"]
    reordered = parse_terms(
        _doc('[Term]\nid: MONDO:0000001\nsynonym: "illness" EXACT []\nname: disease\n')
    )["MONDO:0000001"]
    assert ordered.content_hash == reordered.content_hash
    assert hash_clauses(ordered.clauses) == hash_clauses(reordered.clauses)


def test_split_document_keys_terms_by_id():
    header, terms = split_document(_doc(TERM_A, TERM_B))
    assert set(terms) == {"MONDO:0000001", "MONDO:0000002"}
    assert b"format-version" in header  # header retained as parse context


def test_stanza_id_strips_inline_comment():
    # An OBO "id: X ! label" comment must not become part of the key, or the
    # text-level id won't match fastobo's parsed id (that mismatch crashed a build).
    doc = _doc("[Term]\nid: UBERON:0000002 ! uterine cervix\nname: cervix\n")
    header, terms = split_document(doc)
    assert set(terms) == {"UBERON:0000002"}
    parsed, failed = parse_stanzas(header, terms)
    assert set(parsed) == {"UBERON:0000002"}
    assert failed == []


def test_parse_stanzas_isolates_bad_stanza():
    # One malformed stanza must not sink the batch: it is bisected out, recorded
    # as failed, and the good term still parses.
    good = "[Term]\nid: MONDO:0000001\nname: ok\n"
    bad = '[Term]\nid: MONDO:0000002\nname: b\nsynonym: "x" WRONGSCOPE []\n'
    context, stanzas = split_document(_doc(good, bad))
    parsed, failed = parse_stanzas(context, stanzas)
    assert set(parsed) == {"MONDO:0000001"}
    assert failed == ["MONDO:0000002"]


def test_clause_delta_reports_addition():
    before = parse_terms(_doc(TERM_A))["MONDO:0000001"].clauses
    after = parse_terms(_doc(TERM_A_SYN))["MONDO:0000001"].clauses

    added, removed = clause_delta(before, after)
    assert [c.predicate for c in added] == ["synonym"]
    assert removed == []


def test_clause_delta_reports_edit_as_remove_plus_add():
    before = parse_terms(_doc(TERM_A))["MONDO:0000001"].clauses
    renamed = parse_terms(_doc("[Term]\nid: MONDO:0000001\nname: illness\n"))["MONDO:0000001"].clauses

    added, removed = clause_delta(before, renamed)
    assert [(c.predicate, c.value) for c in added] == [("name", "illness")]
    assert [(c.predicate, c.value) for c in removed] == [("name", "disease")]


def test_clause_decomposition_recomposes_to_value():
    # The stored decomposition must recompose to the serialized value byte
    # for byte — the invariant that lets `value` be cross-checked against
    # (or derived from) the parsed columns.
    doc = _doc(
        "[Term]\n"
        "id: MONDO:0000001\n"
        "name: disease\n"
        "comment: Editor note: see NCIT classification\n"
        'synonym: "illness" EXACT [DOID:4]\n'
        'xref: NCIT:C2991 {source="MONDO:equivalentTo"} ! disease or disorder\n'
        "is_a: MONDO:0000000 ! root\n"
    )
    clauses = parse_terms(doc)["MONDO:0000001"].clauses
    assert len(clauses) == 5
    for c in clauses:
        recomposed = c.parsed.body
        if c.parsed.qualifiers:
            recomposed += " {" + ", ".join(c.parsed.qualifiers) + "}"
        if c.parsed.comment is not None:
            recomposed += " ! " + c.parsed.comment
        assert recomposed == c.value, c.predicate


def test_comment_clause_decomposition_has_no_phantom_comment():
    # fastobo's CommentClause exposes its *value* via `.comment`; the peel
    # must not record it as a trailing `!` comment (nothing was peeled).
    doc = _doc("[Term]\nid: MONDO:0000001\ncomment: check NCIT; see notes\n")
    clauses = parse_terms(doc)["MONDO:0000001"].clauses
    (comment_clause,) = [c for c in clauses if c.predicate == "comment"]
    assert comment_clause.parsed.body == "check NCIT; see notes"
    assert comment_clause.parsed.comment is None
    assert comment_clause.parsed.qualifiers == ()
