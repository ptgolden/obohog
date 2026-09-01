"""Schema-level invariants for the artifact's row shapes."""

from obohog import model


def test_row_typeddicts_match_schemas():
    # The TypedDict row declarations exist so producers can be type-checked,
    # but only the pa.schema is enforced at write time — this lock keeps the
    # two from drifting (same keys, same order).
    cases = [
        (model.CommitRow, list(model.COMMITS.names)),
        (model.SnapshotRow, list(model.TERM_SNAPSHOTS.names)),
        (model.EventRow, list(model.EVENTS.names)),
        (model.ReleaseRow, list(model.RELEASES.names)),
        (model.SkipRow, list(model.SKIPPED.names)),
        (model.ClauseRow, [f.name for f in model._CLAUSE]),
        (model.BranchCommitRow, [f.name for f in model._BRANCH_COMMIT]),
    ]
    for typed_dict, schema_names in cases:
        assert list(typed_dict.__annotations__) == schema_names, typed_dict.__name__
