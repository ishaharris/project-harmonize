from types import SimpleNamespace

import numpy as np
import pytest

from harmonize.analysis import compare_run
from harmonize.fixtures import build_validation_run


@pytest.mark.native
def test_complete_fixture_panel_and_reference_exclusions(targets, tmp_path, native):
    run = build_validation_run(targets, tmp_path / "validation")
    analysis = compare_run(run, executable=native)
    assert run.complete
    assert run.kind == "validation_fixture"
    assert len(run.artifacts) == 66
    assert {a.kind for a in run.artifacts} == {"validation_fixture"}
    counts = analysis.pairs.comparison_type.value_counts().to_dict()
    assert counts == {"cross_model": 210, "within_model": 120, "experimental_reference": 66}
    assert set(analysis.pairs.status) == {"complete"}
    refs = analysis.pairs[analysis.pairs.comparison_type == "experimental_reference"]
    assert set(refs[refs.target_id == "pdz3"].normalization_length) == {97}
    assert set(refs[refs.target_id == "ramr"].normalization_length) == {191}
    assert not ((analysis.residues.target_id == "pdz3") & (analysis.residues.position == 28)).any()
    assert np.isnan(analysis.residues[analysis.residues.right == "experimental"].right_plddt).all()
    assert all(len(selection) == 3 for selection in analysis.medoids.values())


def test_empty_selection_does_not_restore_medoids():
    from harmonize.plots import structure_view

    analysis = SimpleNamespace(
        run=SimpleNamespace(targets=[SimpleNamespace(id="a")]), medoids={"a": {"m": "m:1"}}
    )
    with pytest.raises(ValueError, match="No structures"):
        structure_view(analysis, "a", labels=[], include_reference=False)
