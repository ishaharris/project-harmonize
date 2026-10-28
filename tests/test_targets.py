from pathlib import Path

import numpy as np
import pytest
import yaml

from harmonize.structures import normalize_prediction, reference_arrays, write_ca_pdb
from harmonize.targets import parse_target

ROOT = Path(__file__).resolve().parents[1]


def test_six_frozen_constructs(targets):
    assert [len(t.sequence) for t in targets] == [95, 194, 91, 98, 124, 159]
    target = {t.id: t for t in targets}
    assert target["gpd"].residues[0].author_number == "15"
    assert target["ramr"].sequence.startswith("MV")
    assert sum(r.observed for r in target["ramr"].residues) == 191
    assert target["dhfr"].sequence[36] == "D"
    pdz = target["pdz3"]
    assert pdz.sequence[27] == "D"
    assert pdz.residues[27].deposited_name == "SNN"
    assert pdz.residues[27].author_number == "332"
    assert pdz.residues[27].observed
    assert not pdz.residues[27].reference_eligible


@pytest.mark.parametrize("target_id", ["ramr", "pdz3"])
def test_parse_deposited_mapping(target_id, tmp_path):
    specs = yaml.safe_load((ROOT / "configs/targets.yaml").read_text())["targets"]
    spec = next(s for s in specs if s["id"] == target_id)
    target = parse_target(
        spec, ROOT / "tests/fixtures" / f"{spec['pdb_id']}.cif", tmp_path / "prepared.json"
    )
    assert len(target.sequence) == spec["expected_length"]
    if target_id == "pdz3":
        assert target.sequence[27] == "D"
        assert not target.residues[27].reference_eligible
    else:
        assert [r.position for r in target.residues if not r.observed] == [1, 193, 194]


def test_reference_b_factors_are_not_confidence(targets):
    for target in targets:
        arrays = reference_arrays(target)
        assert np.isnan(arrays["confidence"]).all()
        assert np.isnan(arrays["raw_confidence"]).all()


def test_fixed_prediction_mapping_and_missing_residues(targets, tmp_path):
    target = targets[0]
    xyz = reference_arrays(target)["xyz"]
    positions = np.arange(2, len(xyz) + 1)
    path = tmp_path / "missing.pdb"
    write_ca_pdb(path, xyz[1:], positions, target.sequence)
    parsed = normalize_prediction(path, target.sequence)
    assert np.isnan(parsed["xyz"][0]).all()
    np.testing.assert_allclose(parsed["xyz"][1:], xyz[1:], atol=0.001)
    bad_sequence = "A" * len(target.sequence)
    with pytest.raises(ValueError, match="sequence"):
        normalize_prediction(path, bad_sequence)


def test_confidence_scale_is_explicit(targets, tmp_path):
    target = targets[0]
    xyz = reference_arrays(target)["xyz"]
    path = tmp_path / "confidence.pdb"
    write_ca_pdb(
        path, xyz, np.arange(1, len(xyz) + 1), target.sequence, confidence=np.full(len(xyz), 0.9)
    )
    unknown = normalize_prediction(path, target.sequence, confidence_scale="unknown")
    assert np.isnan(unknown["confidence"]).all()
    np.testing.assert_allclose(unknown["raw_confidence"], 0.9)
    normalized = normalize_prediction(path, target.sequence, confidence_scale="0-1")
    np.testing.assert_allclose(normalized["confidence"], 90)
    with pytest.raises(ValueError, match="scale"):
        normalize_prediction(
            path, target.sequence, confidence_scale="0-1", plddt=np.full(len(xyz), 90)
        )
