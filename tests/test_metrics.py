import numpy as np
import pytest

from harmonize.metrics import compare_coordinates, parse_tm_score
from harmonize.structures import reference_arrays


@pytest.mark.native
def test_identity_and_rigid_transform(targets, native):
    target = targets[0]
    xyz = reference_arrays(target)["xyz"]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transformed = xyz @ rotation + [25, -15, 3]
    result = compare_coordinates(
        xyz, transformed, target.sequence, normalization_length=len(xyz), executable=native
    )
    assert result["tm_score"] == pytest.approx(1, abs=2e-5)
    assert result["rmsd"] < 1e-4
    assert result["lddt"] == 1


@pytest.mark.native
def test_local_perturbation_and_symmetry(targets, native):
    target = targets[0]
    xyz = reference_arrays(target)["xyz"]
    perturbed = xyz.copy()
    perturbed[30:38] += [5, 0, 0]
    kwargs = {"normalization_length": len(xyz), "executable": native}
    forward = compare_coordinates(xyz, perturbed, target.sequence, **kwargs)
    reverse = compare_coordinates(perturbed, xyz, target.sequence, **kwargs)
    assert 0 < forward["tm_score"] < 1
    assert forward["rmsd"] > 0
    assert 0 < forward["lddt"] < 1
    for metric in ("tm_score", "rmsd", "lddt"):
        assert forward[metric] == pytest.approx(reverse[metric], abs=1e-5)
    assert forward["local_lddt"][30:38].mean() < forward["local_lddt"][:10].mean()


@pytest.mark.native
def test_missing_coordinates_keep_full_length_denominator(targets, native):
    target = targets[0]
    xyz = reference_arrays(target)["xyz"]
    missing = xyz.copy()
    missing[:10] = np.nan
    result = compare_coordinates(
        xyz, missing, target.sequence, normalization_length=len(xyz), executable=native
    )
    expected = (len(xyz) - 10) / len(xyz)
    assert result["coverage"] == pytest.approx(expected)
    assert result["tm_score"] == pytest.approx(expected, abs=2e-5)
    assert result["lddt"] == 1


def test_reject_wrong_normalization():
    with pytest.raises(ValueError, match="normalization"):
        parse_tm_score("TM-score= 0.9 (normalized by user-specified L=94, d0=1.0)", 95)


def test_unscorable_coordinates(native):
    xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    with pytest.raises(ValueError, match="non-collinear"):
        compare_coordinates(xyz, xyz, "AAA", normalization_length=3, executable=native)
