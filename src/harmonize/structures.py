"""Coordinate normalization with explicit sequence correspondence."""

from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure.io import pdb, pdbx

from .targets import AA3

AA1 = {v: k for k, v in AA3.items()}


def read_atoms(path: Path):
    if path.suffix.lower() in (".cif", ".mmcif"):
        return pdbx.get_structure(
            pdbx.CIFFile.read(path), model=1, altloc="occupancy", extra_fields=["b_factor"]
        )
    return pdb.PDBFile.read(path).get_structure(
        model=1, altloc="occupancy", extra_fields=["b_factor"]
    )


def normalize_prediction(path: Path, sequence: str, *, confidence_scale=None, plddt=None):
    atoms = read_atoms(path)
    ca = atoms[(atoms.atom_name == "CA") & np.isin(atoms.res_name, list(AA3))]
    if len(set(ca.chain_id)) != 1 or len(ca) < 3:
        raise ValueError("Expected one protein chain with at least three C-alpha atoms")
    if np.any(ca.ins_code != "") or len(set(ca.res_id.tolist())) != len(ca):
        raise ValueError("Ambiguous prediction residue IDs or insertion codes")
    observed = "".join(AA3[name] for name in ca.res_name)
    positions = ca.res_id.astype(int)
    indexed_match = all(
        1 <= p <= len(sequence) and sequence[p - 1] == aa
        for p, aa in zip(positions, observed, strict=True)
    )
    if not indexed_match:
        # Full-length sequence identity allows safe renumbering of imported chains.
        if observed != sequence:
            raise ValueError("Prediction sequence does not match input at fixed residue positions")
        positions = np.arange(1, len(sequence) + 1)
    if not np.isfinite(ca.coord).all():
        raise ValueError("Prediction contains non-finite coordinates")
    xyz = np.full((len(sequence), 3), np.nan, dtype=float)
    xyz[positions - 1] = ca.coord
    raw_confidence = np.full(len(sequence), np.nan)
    if plddt is not None:
        values = np.asarray(plddt, dtype=float)
        if values.shape != (len(sequence),):
            raise ValueError("Per-residue confidence length differs from input")
        raw_confidence[:] = values
    elif "b_factor" in ca.get_annotation_categories():
        raw_confidence[positions - 1] = ca.b_factor
    confidence = np.full(len(sequence), np.nan)
    if confidence_scale in ("0-1", "0-100"):
        finite = np.isfinite(raw_confidence)
        maximum = 1 if confidence_scale == "0-1" else 100
        if np.any((raw_confidence[finite] < 0) | (raw_confidence[finite] > maximum)):
            raise ValueError("Confidence outside declared provider scale")
        confidence = raw_confidence * (100 if confidence_scale == "0-1" else 1)
    elif confidence_scale not in (None, "unknown"):
        raise ValueError("Unsupported confidence scale")
    return {"xyz": xyz, "confidence": confidence, "raw_confidence": raw_confidence}


def reference_arrays(target):
    xyz = np.array(
        [
            r.ca if r.ca is not None and r.reference_eligible else [np.nan] * 3
            for r in target.residues
        ],
        dtype=float,
    )
    return {
        "xyz": xyz,
        "confidence": np.full(len(xyz), np.nan),
        "raw_confidence": np.full(len(xyz), np.nan),
    }


def ca_atoms(xyz, positions, sequence):
    atoms = struc.AtomArray(len(positions))
    atoms.coord = np.asarray(xyz, dtype=float)
    atoms.chain_id[:] = "A"
    atoms.res_id = np.asarray(positions, dtype=int)
    atoms.res_name = np.array([AA1[sequence[p - 1]] for p in positions])
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    return atoms


def write_ca_pdb(path: Path, xyz, positions, sequence, confidence=None):
    atoms = ca_atoms(xyz, positions, sequence)
    atoms.set_annotation("b_factor", np.zeros(len(atoms)) if confidence is None else confidence)
    structure = pdb.PDBFile()
    structure.set_structure(atoms)
    structure.write(path)


def superpose(left, right):
    """Return right superposed on left, and its reusable affine transform."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (
        len(left) < 3
        or np.linalg.matrix_rank(left - left.mean(0)) < 2
        or np.linalg.matrix_rank(right - right.mean(0)) < 2
    ):
        raise ValueError("At least three non-collinear matched coordinates are required")
    moved, transform = struc.superimpose(left, right)
    distances = np.linalg.norm(left - moved, axis=1)
    return moved, transform, distances
