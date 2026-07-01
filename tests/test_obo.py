"""Tests for OBO normalization, hashing, and clause diffing."""

from mondo_history.obo import (
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
