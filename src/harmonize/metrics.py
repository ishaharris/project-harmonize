"""Fixed-correspondence comparisons; no structural realignment of residues."""

import re
import subprocess
import tempfile
from pathlib import Path

import biotite.structure as struc
import numpy as np

from .structures import ca_atoms, superpose, write_ca_pdb


def verify_usalign(executable: Path) -> str:
    result = subprocess.run([str(executable.resolve()), "-v"], capture_output=True, text=True)
    if "Version 20260819" not in result.stdout:
        raise RuntimeError(
            "Expected project-pinned US-align 20260819; run scripts/bootstrap_native.py"
        )
    return result.stdout


def parse_tm_score(output: str, normalization_length: int) -> float:
    rows = [r for r in output.splitlines() if "TM-score=" in r and "user-specified" in r]
    if len(rows) != 1:
        raise ValueError("US-align did not return exactly one user-normalized TM-score")
    length = re.search(r"L\s*=\s*(\d+)", rows[0])
    score = re.search(r"TM-score=\s*([\d.eE+-]+)", rows[0])
    if length is None or score is None or int(length[1]) != normalization_length:
        raise ValueError("US-align normalization length mismatch")
    value = float(score[1])
    if not np.isfinite(value) or not 0 <= value <= 1.00001:
        raise ValueError("Invalid TM-score")
    return min(value, 1.0)


def tm_score(left, right, positions, sequence, normalization_length, executable, output_dir=None):
    if normalization_length < len(positions):
        raise ValueError("TM-score denominator cannot be smaller than the matched set")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        a, b = directory / "left.pdb", directory / "right.pdb"
        write_ca_pdb(a, left, positions, sequence)
        write_ca_pdb(b, right, positions, sequence)
        scores = []
        # Optimization is approximate. Take the larger directional score for exact symmetry.
        for index, (first, second) in enumerate(((a, b), (b, a))):
            command = [
                str(Path(executable).resolve()),
                str(first),
                str(second),
                "-mol",
                "prot",
                "-TMscore",
                "1",
                "-u",
                str(normalization_length),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
            scores.append(parse_tm_score(result.stdout, normalization_length))
            if output_dir is not None:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                (Path(output_dir) / f"usalign_{index}.txt").write_text(result.stdout)
    return max(scores)


def compare_coordinates(
    left, right, sequence, *, normalization_length, executable, eligible=None, output_dir=None
):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.shape != (len(sequence), 3):
        raise ValueError("Coordinates must be mapped onto the same full input sequence")
    mask = np.isfinite(left).all(1) & np.isfinite(right).all(1)
    if eligible is not None:
        mask &= np.asarray(eligible, dtype=bool)
    positions = np.flatnonzero(mask) + 1
    a, b = left[mask], right[mask]
    _, _, distances = superpose(a, b)
    atoms_a, atoms_b = ca_atoms(a, positions, sequence), ca_atoms(b, positions, sequence)
    lddt_global = float(struc.lddt(atoms_a, atoms_b, symmetric=True))
    lddt_local = struc.lddt(atoms_a, atoms_b, symmetric=True, aggregation="atom")
    return {
        "n_matched": len(positions),
        "coverage": len(positions) / normalization_length,
        "tm_score": tm_score(
            a, b, positions, sequence, normalization_length, executable, output_dir
        ),
        "rmsd": float(np.sqrt(np.mean(distances**2))),
        "lddt": lddt_global if np.isfinite(lddt_global) else None,
        "positions": positions,
        "distances": distances,
        "local_lddt": lddt_local,
    }
