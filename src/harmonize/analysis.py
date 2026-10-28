"""Offline analysis of cached predictions or explicitly labeled validation fixtures."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import compare_coordinates, verify_usalign
from .storage import atomic_json, now, sha256
from .structures import reference_arrays
from .types import PairwiseComparison
from .workflow import artifact_arrays, load_run

RESIDUE_COLUMNS = [
    "target_id",
    "left",
    "right",
    "left_model",
    "right_model",
    "comparison_type",
    "position",
    "author_number",
    "amino_acid",
    "displacement",
    "lddt",
    "left_plddt",
    "right_plddt",
]


@dataclass
class Analysis:
    run: object
    directory: Path
    pairs: pd.DataFrame
    residues: pd.DataFrame
    summary: pd.DataFrame
    confidence: pd.DataFrame
    medoids: dict


def choose_medoids(artifacts, pairs):
    result = {}
    models = sorted({a.model for a in artifacts})
    for model in models:
        candidates = sorted(a.label for a in artifacts if a.model == model)
        scores = {}
        for candidate in candidates:
            subset = pairs[
                (pairs.left_model == model)
                & (pairs.right_model == model)
                & ((pairs.left == candidate) | (pairs.right == candidate))
                & (pairs.status == "complete")
            ]
            scores[candidate] = float(subset.tm_score.mean()) if len(subset) else -1
        # Input is lexically sorted; max retains the first entry on a tie.
        result[model] = max(candidates, key=lambda label: scores[label])
    return result


def compare_run(run="runs/panel", *, output_dir=None, executable=".tools/usalign/USalign"):
    if not hasattr(run, "artifacts"):
        run = load_run(run)
    directory = Path(output_dir or run.directory / "analysis").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    executable = Path(executable).resolve()
    verify_usalign(executable)
    rows, residue_rows, confidence_rows = [], [], []
    medoids = {}
    for target in run.targets:
        artifacts = sorted(
            (a for a in run.artifacts if a.target_id == target.id and a.status == "complete"),
            key=lambda a: a.label,
        )
        arrays = {a.label: artifact_arrays(run, a) for a in artifacts}
        for artifact in artifacts:
            values = arrays[artifact.label]["confidence"]
            finite = np.isfinite(values)
            confidence_rows.append(
                {
                    "target_id": target.id,
                    "sample": artifact.label,
                    "model": artifact.model,
                    "mean_plddt": float(values[finite].mean()) if finite.any() else None,
                    "confidence_coverage": float(finite.mean()),
                    "raw_scale": artifact.confidence_scale,
                    "source": artifact.confidence_source,
                }
            )
        jobs = [
            (
                a.label,
                b.label,
                a.model,
                b.model,
                "within_model" if a.model == b.model else "cross_model",
            )
            for a, b in combinations(artifacts, 2)
        ]
        reference = reference_arrays(target)
        arrays["experimental"] = reference
        jobs += [
            (a.label, "experimental", a.model, "experimental", "experimental_reference")
            for a in artifacts
        ]
        for left, right, left_model, right_model, comparison_type in jobs:
            norm = (
                sum(r.reference_eligible for r in target.residues)
                if right == "experimental"
                else len(target.sequence)
            )
            row = PairwiseComparison(
                target_id=target.id,
                left=left,
                right=right,
                left_model=left_model,
                right_model=right_model,
                comparison_type=comparison_type,
                normalization_length=norm,
            )
            try:
                metrics = compare_coordinates(
                    arrays[left]["xyz"],
                    arrays[right]["xyz"],
                    target.sequence,
                    normalization_length=norm,
                    executable=executable,
                    output_dir=directory
                    / "native"
                    / target.id
                    / f"{left}__{right}".replace(":", "_"),
                )
                for key in ("n_matched", "coverage", "tm_score", "rmsd", "lddt"):
                    setattr(row, key, metrics[key])
                for position, distance, lddt in zip(
                    metrics["positions"], metrics["distances"], metrics["local_lddt"], strict=True
                ):
                    residue = target.residues[position - 1]
                    residue_rows.append(
                        {
                            "target_id": target.id,
                            "left": left,
                            "right": right,
                            "left_model": left_model,
                            "right_model": right_model,
                            "comparison_type": comparison_type,
                            "position": position,
                            "author_number": residue.author_number + residue.insertion_code,
                            "amino_acid": residue.amino_acid,
                            "displacement": float(distance),
                            "lddt": float(lddt),
                            "left_plddt": arrays[left]["confidence"][position - 1],
                            "right_plddt": arrays[right]["confidence"][position - 1],
                        }
                    )
            except (ValueError, RuntimeError) as exc:
                row.status = "unscorable"
                row.error = str(exc)
            rows.append(row.model_dump())
        target_pairs = pd.DataFrame(
            [r for r in rows if r["target_id"] == target.id],
            columns=list(PairwiseComparison.model_fields),
        )
        medoids[target.id] = choose_medoids(artifacts, target_pairs)
    pairs = pd.DataFrame(rows, columns=list(PairwiseComparison.model_fields))
    residues = pd.DataFrame(residue_rows, columns=RESIDUE_COLUMNS)
    confidence = pd.DataFrame(
        confidence_rows,
        columns=[
            "target_id",
            "sample",
            "model",
            "mean_plddt",
            "confidence_coverage",
            "raw_scale",
            "source",
        ],
    )
    groups = ["target_id", "comparison_type", "left_model", "right_model"]
    summaries = []
    for keys, group in pairs[pairs.status == "complete"].groupby(groups, sort=True):
        item = dict(zip(groups, keys, strict=True))
        item["n_pairs"] = len(group)
        for metric in ("tm_score", "rmsd", "lddt", "coverage"):
            for label, quantile in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
                item[f"{metric}_{label}"] = group[metric].quantile(quantile)
        summaries.append(item)
    summary = pd.DataFrame(
        summaries,
        columns=groups
        + ["n_pairs"]
        + [
            f"{metric}_{q}"
            for metric in ("tm_score", "rmsd", "lddt", "coverage")
            for q in ("q25", "median", "q75")
        ],
    )
    for filename, frame in (
        ("pairs.csv", pairs),
        ("residues.csv", residues),
        ("summary.csv", summary),
        ("confidence.csv", confidence),
    ):
        frame.to_csv(directory / filename, index=False)
    atomic_json(directory / "medoids.json", medoids)
    atomic_json(
        directory / "analysis.json",
        {
            "created_at": now(),
            "kind": run.kind,
            "run_complete": run.complete,
            "expected_predictions": run.expected_count,
            "available_predictions": sum(a.status == "complete" for a in run.artifacts),
            "pair_count": int((pairs.comparison_type != "experimental_reference").sum()),
            "reference_comparison_count": int(
                (pairs.comparison_type == "experimental_reference").sum()
            ),
            "usalign_version": "20260819",
            "usalign_binary_sha256": sha256(executable.read_bytes()),
            "tm_protocol": "Fixed input correspondence; max of both optimization directions; full input length for prediction pairs, eligible observed length for references",
            "lddt_protocol": "Symmetric CA contacts, union within 15 A, thresholds 0.5/1/2/4 A; matched coordinates only",
            "rmsd_protocol": "Kabsch over all matched C-alpha positions; no confidence filtering",
            "inputs": {t.id: t.sequence_sha256 for t in run.targets},
            "artifact_hashes": {
                f"{a.target_id}/{a.label}": a.normalized_sha256
                for a in run.artifacts
                if a.status == "complete"
            },
        },
    )
    return Analysis(run, directory, pairs, residues, summary, confidence, medoids)
