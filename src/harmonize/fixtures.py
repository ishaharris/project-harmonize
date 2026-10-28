"""Synthetic validation data: perturbed experimental coordinates, NOT predictions."""

from copy import deepcopy

import numpy as np

from .providers import ProviderOutput
from .structures import reference_arrays, write_ca_pdb
from .types import ModelSpec, Protocol
from .workflow import _record, initialize_run, store_output


def build_validation_run(targets, directory="runs/validation"):
    protocol = Protocol(
        protocol_version="validation-1",
        models={
            name: ModelSpec(
                provider="fixture",
                version="synthetic-coordinate-validation-v1",
                samples=[str(i + 1) for i in range(count)],
                settings={
                    "source": "experimental C-alpha coordinates",
                    "noise_group": index,
                    "seed": 1729,
                },
            )
            for index, (name, count) in enumerate(
                (("fixture_group_a", 5), ("fixture_group_b", 5), ("fixture_singleton", 1))
            )
        },
    )
    run = initialize_run(deepcopy(targets), directory, protocol, kind="validation_fixture")
    for ti, target in enumerate(targets):
        reference = reference_arrays(target)["xyz"]
        present = np.isfinite(reference).all(1)
        positions = np.flatnonzero(present) + 1
        for mi, (model, spec) in enumerate(protocol.models.items()):
            for sample in spec.samples:
                artifact = _record(target, model, spec, sample, kind="validation_fixture")
                if any(
                    a.cache_key == artifact.cache_key and a.status == "complete"
                    for a in run.artifacts
                ):
                    continue
                rng = np.random.default_rng(1729 + 100 * ti + 10 * mi + int(sample))
                coords = reference[present].copy()
                scale = (int(sample) - 1) * 0.08 + mi * 0.3
                coords += rng.normal(scale=scale, size=coords.shape)
                # Controlled local displacement in the middle of the second fixture group.
                if mi == 1:
                    lo, hi = len(coords) // 3, len(coords) // 3 + 8
                    coords[lo:hi] += np.array([2.5, 0.0, 0.0])
                coords += np.array([10 * mi, -5 * int(sample), 3])
                folder = run.directory / "cache" / artifact.cache_key
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / "fixture.pdb"
                write_ca_pdb(path, coords, positions, target.sequence)
                # Fixture confidence is synthetic and intentionally labeled, never experimental B.
                confidence = np.full(len(target.sequence), 90.0 - 10 * mi)
                output = ProviderOutput(
                    path.read_text(),
                    confidence_scale="0-100",
                    plddt=confidence,
                    confidence_source="SYNTHETIC validation confidence; not model pLDDT",
                    provenance={
                        "kind": "validation_fixture",
                        "description": "Perturbed experimental C-alpha coordinates, NOT a model prediction",
                        "seed": 1729 + 100 * ti + 10 * mi + int(sample),
                    },
                )
                store_output(run, target, artifact, output)
    return run
