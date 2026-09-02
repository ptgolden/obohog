"""Shared fixtures: a tiny, deterministic multi-commit OBO git repository."""

import os
import subprocess
from pathlib import Path

import pytest

HEADER = "format-version: 1.2\n\n"


def _term(term_id: str, *clauses: str) -> str:
    body = "\n".join([f"id: {term_id}", *clauses])
    return f"[Term]\n{body}\n"


def _git(repo: Path, *args: str, date: str | None = None) -> None:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="Test Author",
        GIT_AUTHOR_EMAIL="author@example.org",
        GIT_COMMITTER_NAME="Test Author",
        GIT_COMMITTER_EMAIL="author@example.org",
    )
    if date is not None:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, env=env
    )


def _write(repo: Path, rel: str, content: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def obo_repo(tmp_path: Path) -> Path:
    """A git repo whose OBO file evolves over five commits, including a rename.

    History (oldest first), following the file to its final path ``src/onto.obo``:

    * c0  ``onto.obo``      MONDO:0000001 {name}
    * c1  ``onto.obo``      + synonym on MONDO:0000001
    * c2  ``src/onto.obo``  pure rename (git mv), content unchanged
    * c3  ``src/onto.obo``  + xref on MONDO:0000001
    * c4  ``src/onto.obo``  + new term MONDO:0000002
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    t1_v0 = _term("MONDO:0000001", "name: disease")
    _write(repo, "onto.obo", HEADER + t1_v0)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c0 create", date="2021-01-01T00:00:00+00:00")

    t1_v1 = _term("MONDO:0000001", "name: disease", 'synonym: "illness" EXACT []')
    _write(repo, "onto.obo", HEADER + t1_v1)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c1 add synonym", date="2021-01-02T00:00:00+00:00")

    (repo / "src").mkdir()
    _git(repo, "mv", "onto.obo", "src/onto.obo")
    _git(repo, "commit", "-qm", "c2 rename (#42)", date="2021-01-03T00:00:00+00:00")

    t1_v3 = _term(
        "MONDO:0000001",
        "name: disease",
        'synonym: "illness" EXACT []',
        "xref: DOID:4",
    )
    _write(repo, "src/onto.obo", HEADER + t1_v3)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c3 add xref", date="2021-01-04T00:00:00+00:00")
    _git(repo, "tag", "v1.0")  # release tag on c3

    t2 = _term("MONDO:0000002", "name: cancer")
    _write(repo, "src/onto.obo", HEADER + t1_v3 + "\n" + t2)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c4 add term", date="2021-01-05T00:00:00+00:00")

    return repo


OFN_HEADER = (
    "Prefix(:=<http://example.org/onto#>)\n"
    "Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)\n"
    "Prefix(obo:=<http://purl.obolibrary.org/obo/>)\n"
    "Ontology(<http://example.org/onto.owl>\n"
    "Import(<http://example.org/imports/missing.owl>)\n"
)


def _ofn(*axioms: str) -> str:
    return OFN_HEADER + "\n".join(axioms) + "\n)\n"


@pytest.fixture
def ofn_repo(tmp_path: Path) -> Path:
    """A git repo whose tracked file is OWL functional syntax (``onto.owl``),
    evolving over three commits. Carries an unresolvable ``Import(...)``
    like real ODK edit files, so conversion must strip it.

    * c0  TST:0000001 "test term"
    * c1  + TST:0000002 "second term", subclass of TST:0000001
    * c2  TST:0000001 renamed to "renamed term"
    * c3  + object property TST:9000001 "part of thing", transitive
          (converts to a [Typedef] stanza)
    """
    repo = tmp_path / "ofn-repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    label1 = 'AnnotationAssertion(rdfs:label obo:TST_0000001 "test term")'
    decl1 = "Declaration(Class(obo:TST_0000001))"
    _write(repo, "onto.owl", _ofn(decl1, label1))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c0 create", date="2021-01-01T00:00:00+00:00")

    decl2 = "Declaration(Class(obo:TST_0000002))"
    label2 = 'AnnotationAssertion(rdfs:label obo:TST_0000002 "second term")'
    sub = "SubClassOf(obo:TST_0000002 obo:TST_0000001)"
    _write(repo, "onto.owl", _ofn(decl1, label1, decl2, label2, sub))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c1 add subclass", date="2021-01-02T00:00:00+00:00")

    relabel1 = 'AnnotationAssertion(rdfs:label obo:TST_0000001 "renamed term")'
    _write(repo, "onto.owl", _ofn(decl1, relabel1, decl2, label2, sub))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c2 rename label", date="2021-01-03T00:00:00+00:00")

    decl_p = "Declaration(ObjectProperty(obo:TST_9000001))"
    label_p = 'AnnotationAssertion(rdfs:label obo:TST_9000001 "part of thing")'
    trans = "TransitiveObjectProperty(obo:TST_9000001)"
    _write(
        repo, "onto.owl",
        _ofn(decl1, relabel1, decl2, label2, sub, decl_p, label_p, trans),
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c3 add relation", date="2021-01-04T00:00:00+00:00")

    return repo


@pytest.fixture(scope="session")
def paged_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built artifact wide enough to page over.

    A ``SHARED:`` xref appears on four terms across four commits, so both
    section spines (term for ``order="term"``, commit for ``order="date"``)
    have several sections. Shared by the query-layer cursor tests and the
    service-layer pagination tests.
    """
    from obohog.extract import extract
    from obohog.gitsource import GitSource

    base = tmp_path_factory.mktemp("paged")
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    terms: list[str] = []

    def commit(msg: str, date: str) -> None:
        _write(repo, "onto.obo", HEADER + "\n".join(terms))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", msg, date=date)

    terms.append(_term("MONDO:0000001", "name: alpha", "xref: SHARED:1"))
    commit("c0 alpha", "2021-01-01T00:00:00+00:00")
    terms.append(_term("MONDO:0000002", "name: beta", "xref: SHARED:2"))
    commit("c1 beta", "2021-01-02T00:00:00+00:00")
    terms.append(_term("MONDO:0000003", "name: gamma", "xref: SHARED:3"))
    commit("c2 gamma", "2021-01-03T00:00:00+00:00")
    terms.append(_term("MONDO:0000004", "name: delta", "xref: SHARED:4"))
    commit("c3 delta", "2021-01-04T00:00:00+00:00")
    _git(repo, "tag", "v1.0")

    out = base / "artifact"
    with GitSource(repo) as src:
        extract(src, "onto.obo", out)
    return out


@pytest.fixture(scope="session")
def lifecycle_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built artifact whose clause values come and go across commits.

    Exercises term-set ("has ever / now") membership: a matching value
    removed and never re-added (ever but not now), one removed then
    re-added (both), xrefs with their own lifecycle for intersection
    tests, and a non-MONDO term for namespace narrowing.

    * c0  T1 alpha + synonym "diabetes mellitus"; T2 beta + xref DOID:9
    * c1  T1 synonym removed; T3 gamma created with synonym
          "old diabetes label" + xref DOID:1
    * c2  T3 synonym removed; T4 delta created with synonym "diabetes"
          + xref DOID:2
    * c3  T1 synonym re-added; T4 xref removed; EX:0000001 "diabetes"
          created

    Membership for ``~diabetes``: ever {T1, T3, T4, EX}, now {T1, T4,
    EX}; for ``xref~DOID``: ever {T2, T3, T4}, now {T2, T3}.
    """
    from obohog.extract import extract
    from obohog.gitsource import GitSource

    base = tmp_path_factory.mktemp("lifecycle")
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    t1_syn = 'synonym: "diabetes mellitus" EXACT []'
    t1 = _term("MONDO:0000001", "name: alpha", t1_syn)
    t1_bare = _term("MONDO:0000001", "name: alpha")
    t2 = _term("MONDO:0000002", "name: beta", "xref: DOID:9")
    t3 = _term(
        "MONDO:0000003",
        "name: gamma",
        'synonym: "old diabetes label" EXACT []',
        "xref: DOID:1",
    )
    t3_bare = _term("MONDO:0000003", "name: gamma", "xref: DOID:1")
    t4 = _term(
        "MONDO:0000004",
        "name: delta",
        'synonym: "diabetes" EXACT []',
        "xref: DOID:2",
    )
    t4_bare = _term(
        "MONDO:0000004", "name: delta", 'synonym: "diabetes" EXACT []'
    )
    ex = _term("EX:0000001", "name: diabetes")

    def commit(terms: list[str], msg: str, date: str) -> None:
        _write(repo, "onto.obo", HEADER + "\n".join(terms))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", msg, date=date)

    commit([t1, t2], "c0 seed", "2022-01-01T00:00:00+00:00")
    commit([t1_bare, t2, t3], "c1 strip T1, add T3", "2022-01-02T00:00:00+00:00")
    commit(
        [t1_bare, t2, t3_bare, t4],
        "c2 strip T3, add T4",
        "2022-01-03T00:00:00+00:00",
    )
    commit(
        [t1, t2, t3_bare, t4_bare, ex],
        "c3 restore T1, strip T4 xref, add EX",
        "2022-01-04T00:00:00+00:00",
    )

    out = base / "artifact"
    with GitSource(repo) as src:
        extract(src, "onto.obo", out)
    return out


@pytest.fixture(scope="session")
def requalified_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built artifact where one commit requalifies an xref.

    c1 rewrites T1's UMLS xref qualifier set — ``NCIT:4`` and ``DOID:9``
    out, ``MEDGEN:8`` and ``MONDO:M`` in, ``KEEP:1`` kept — and adds an
    unrelated ``xref: MEDGEN:8`` clause in the same commit. A search for
    ``NCIT`` matches only the *removed* side, so this is the shape that
    demands whole-group streaming: without the add partner the edit
    renders as a fake whole-clause deletion. (Modeled on MONDO:0004782's
    UMLS:C0011848 xref in mondo commit 3a6ab90.)
    """
    from obohog.extract import extract
    from obohog.gitsource import GitSource

    base = tmp_path_factory.mktemp("requalified")
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    def commit(terms: list[str], msg: str, date: str) -> None:
        _write(repo, "onto.obo", HEADER + "\n".join(terms))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", msg, date=date)

    before = _term(
        "MONDO:0000001",
        "name: alpha",
        'xref: UMLS:1 {source="DOID:9", source="KEEP:1", source="NCIT:4"}',
    )
    after = _term(
        "MONDO:0000001",
        "name: alpha",
        'xref: MEDGEN:8',
        'xref: UMLS:1 {source="KEEP:1", source="MEDGEN:8", source="MONDO:M"}',
    )
    commit([before], "c0 seed", "2023-01-01T00:00:00+00:00")
    commit([after], "c1 requalify", "2023-01-02T00:00:00+00:00")

    out = base / "artifact"
    with GitSource(repo) as src:
        extract(src, "onto.obo", out)
    return out


@pytest.fixture
def bad_then_removed_repo(tmp_path: Path) -> Path:
    """A repo where an unparseable term appears, then is removed the next commit.

    Exercises the state/raw divergence: the bad term lands in ``raw`` (so it is
    not re-tried) but never in ``state`` (it never parsed), so removing it must
    not raise.
    """
    repo = tmp_path / "repo_bad"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    good = _term("MONDO:0000001", "name: good")
    bad = _term("MONDO:0000002", "name: bad", 'synonym: "x" WRONGSCOPE []')
    _write(repo, "onto.obo", HEADER + good + "\n" + bad)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c0 good + bad", date="2021-02-01T00:00:00+00:00")

    _write(repo, "onto.obo", HEADER + good)  # bad term removed
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c1 remove bad", date="2021-02-02T00:00:00+00:00")

    return repo


@pytest.fixture(scope="session")
def wide_commit_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built artifact whose single commit touches three terms.

    Exercises the per-commit term caps: date-ordered search sections
    truncating to SECTION_TERM_CAP, and the commit view's term paging.
    """
    from obohog.extract import extract
    from obohog.gitsource import GitSource

    base = tmp_path_factory.mktemp("wide")
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(
        repo,
        "onto.obo",
        HEADER
        + "\n".join(
            _term(f"MONDO:000000{i}", f"name: term {i}") for i in (1, 2, 3)
        ),
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c0 wide", date="2021-03-01T00:00:00+00:00")

    out = base / "artifact"
    with GitSource(repo) as src:
        extract(src, "onto.obo", out)
    return out


@pytest.fixture(scope="session")
def wide_commits_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built artifact with two commits, each touching three terms.

    Exercises the browse consumption cap across a *run* of wide commits:
    a page keeps filling past a capped section instead of ending at the
    first one.
    """
    from obohog.extract import extract
    from obohog.gitsource import GitSource

    base = tmp_path_factory.mktemp("wide2")
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    def commit(names: list[str], msg: str, date: str) -> None:
        _write(
            repo,
            "onto.obo",
            HEADER
            + "\n".join(
                _term(f"MONDO:000000{i}", f"name: {name}")
                for i, name in enumerate(names, start=1)
            ),
        )
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", msg, date=date)

    commit(["alpha", "beta", "gamma"], "c0 wide", "2021-04-01T00:00:00+00:00")
    commit(
        ["alpha two", "beta two", "gamma two"],
        "c1 wide again",
        "2021-04-02T00:00:00+00:00",
    )

    out = base / "artifact"
    with GitSource(repo) as src:
        extract(src, "onto.obo", out)
    return out


@pytest.fixture(scope="session")
def renamed_ns_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A repo whose namespace was renamed wholesale mid-history (TBD → MONDO).

    * c0  TBD:0000001 "alpha" (is_a TBD:0000002), TBD:0000002 "beta"
    * c1  same terms, ids and the is_a reference rewritten to MONDO:;
          alpha also gains a synonym (a real content change alongside
          the rename)
    * c2  MONDO:0000002 gains an xref (ordinary post-rename history)

    Built with ``namespace_map={"TBD": "MONDO"}`` each term should be one
    continuous identity across all three commits.
    """
    repo = tmp_path_factory.mktemp("renamedns") / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    def commit(terms: list[str], msg: str, date: str) -> None:
        _write(repo, "onto.obo", HEADER + "\n".join(terms))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", msg, date=date)

    commit(
        [
            _term("TBD:0000001", "name: alpha", "is_a: TBD:0000002"),
            _term("TBD:0000002", "name: beta"),
        ],
        "c0 born as TBD", "2020-01-01T00:00:00+00:00",
    )
    commit(
        [
            _term(
                "MONDO:0000001", "name: alpha", "is_a: MONDO:0000002",
                'synonym: "first" EXACT []',
            ),
            _term("MONDO:0000002", "name: beta"),
        ],
        "c1 rename TBD to MONDO", "2020-01-02T00:00:00+00:00",
    )
    commit(
        [
            _term(
                "MONDO:0000001", "name: alpha", "is_a: MONDO:0000002",
                'synonym: "first" EXACT []',
            ),
            _term("MONDO:0000002", "name: beta", "xref: DOID:7"),
        ],
        "c2 ordinary edit", "2020-01-03T00:00:00+00:00",
    )
    return repo
