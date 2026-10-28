"""Resumable prediction collection and explicit import, separate from analysis."""

import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from .providers import ProviderError, ProviderOutput, make_provider, readiness
from .storage import atomic_json, cache_key, now, read_json, sha256
from .structures import normalize_prediction
from .types import PredictionArtifact, Protocol, Run, Target


def environment_provenance():
    packages = {}
    for package in ("project-harmonize", "numpy", "biotite", "httpx", "esm", "pandas"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def save_run(run: Run):
    atomic_json(run.directory / "run.json", run.model_dump(mode="json"))


def load_run(directory="runs/panel") -> Run:
    directory = Path(directory).resolve()
    run = Run.model_validate(read_json(directory / "run.json"))
    run.directory = directory
    for artifact in run.artifacts:
        if artifact.status == "complete":
            validate_artifact(run, artifact)
    return run


def validate_artifact(run, artifact):
    for filename, digest in (
        (artifact.structure_file, artifact.structure_sha256),
        (artifact.normalized_file, artifact.normalized_sha256),
    ):
        if not filename or not digest or sha256((run.directory / filename).read_bytes()) != digest:
            raise ValueError(f"Artifact missing or checksum mismatch: {artifact.label}")


def artifact_arrays(run, artifact):
    validate_artifact(run, artifact)
    with np.load(run.directory / artifact.normalized_file, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def initialize_run(
    targets: list[Target],
    directory="runs/panel",
    protocol="configs/protocol.yaml",
    *,
    kind="prediction",
):
    directory = Path(directory).resolve()
    protocol = (
        protocol
        if isinstance(protocol, Protocol)
        else Protocol.model_validate(yaml.safe_load(Path(protocol).read_text()))
    )
    if (directory / "run.json").exists():
        existing = load_run(directory)

        def identities(items):
            return [(t.id, t.sequence_sha256, t.source_sha256) for t in items]

        if (
            identities(existing.targets) != identities(targets)
            or existing.protocol != protocol
            or existing.kind != kind
        ):
            raise ValueError("Run inputs/protocol differ; use a new run directory")
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    run = Run(
        directory=directory,
        kind=kind,
        created_at=now(),
        targets=targets,
        protocol=protocol,
        provenance=environment_provenance(),
    )
    save_run(run)
    return run


def _record(target, model, spec, sample, *, status="blocked", kind="prediction", error=None):
    return PredictionArtifact(
        target_id=target.id,
        model=model,
        provider=spec.provider,
        model_version=spec.version,
        sample_id=sample,
        sequence_sha256=target.sequence_sha256,
        cache_key=cache_key(target, model, spec, sample),
        status=status,
        kind=kind,
        settings=spec.settings,
        created_at=now(),
        error=error,
    )


def _persist(run, artifact):
    run.artifacts = [
        a
        for a in run.artifacts
        if (a.target_id, a.model, a.sample_id)
        != (artifact.target_id, artifact.model, artifact.sample_id)
    ]
    run.artifacts.append(artifact)
    atomic_json(
        run.directory / "cache" / artifact.cache_key / "artifact.json",
        artifact.model_dump(mode="json"),
    )
    save_run(run)


def store_output(run, target, artifact, output):
    directory = run.directory / "cache" / artifact.cache_key
    directory.mkdir(parents=True, exist_ok=True)
    structure_path = directory / f"structure.{output.format}"
    structure_path.write_text(output.structure)
    normalized = normalize_prediction(
        structure_path,
        target.sequence,
        confidence_scale=output.confidence_scale,
        plddt=output.plddt,
    )
    if output.pae is not None:
        pae = np.asarray(output.pae)
        if pae.shape != (len(target.sequence), len(target.sequence)):
            raise ValueError("PAE shape does not match the input sequence")
        normalized["pae"] = pae
    normalized_path = directory / "normalized.npz"
    np.savez_compressed(normalized_path, **normalized)
    artifact.structure_file = str(structure_path.relative_to(run.directory))
    artifact.structure_sha256 = sha256(structure_path.read_bytes())
    artifact.coordinates_sha256 = sha256(normalized["xyz"].astype("<f8").tobytes())
    artifact.normalized_file = str(normalized_path.relative_to(run.directory))
    artifact.normalized_sha256 = sha256(normalized_path.read_bytes())
    artifact.confidence_source = output.confidence_source
    artifact.confidence_scale = output.confidence_scale
    artifact.provenance.update(output.provenance)
    artifact.request_id = output.provenance.get("request_id", artifact.request_id)
    for candidate in ("response.json", "response.pdb", "response.cif"):
        if (directory / candidate).exists():
            artifact.raw_file = str((directory / candidate).relative_to(run.directory))
            break
    artifact.status = "complete"
    artifact.error = None
    duplicates = [
        a.label
        for a in run.artifacts
        if a.target_id == target.id
        and a.model == artifact.model
        and a.sample_id != artifact.sample_id
        and a.status == "complete"
        and a.coordinates_sha256 == artifact.coordinates_sha256
    ]
    artifact.duplicate_of = min(duplicates) if duplicates else None
    _persist(run, artifact)
    return artifact


def run_predictions(
    targets,
    directory="runs/panel",
    protocol="configs/protocol.yaml",
    *,
    execute=False,
    models=None,
    target_ids=None,
    retry_failed=False,
    provider_factory=make_provider,
):
    """Plan by default; execute=True explicitly submits missing predictions.

    Safe reruns reuse complete artifacts, resume saved NVIDIA requests, and leave
    indeterminate submissions untouched. Failed jobs require retry_failed=True.
    """
    load_dotenv(override=False)
    run = initialize_run(targets, directory, protocol)
    if run.kind != "prediction":
        raise ValueError("Cannot submit inference into a fixture run")
    if models is not None and set(models) - set(run.protocol.models):
        raise ValueError("Unknown requested model")
    providers = {}
    try:
        for target in run.targets:
            for model, spec in run.protocol.models.items():
                for sample in spec.samples:
                    artifact = _record(target, model, spec, sample)
                    cached = run.directory / "cache" / artifact.cache_key / "artifact.json"
                    if cached.exists():
                        artifact = PredictionArtifact.model_validate(read_json(cached))
                        artifact.target_id = target.id
                        if artifact.status == "complete":
                            validate_artifact(run, artifact)
                            _persist(run, artifact)
                            continue
                        if artifact.status == "indeterminate" or (
                            artifact.status == "failed" and not retry_failed
                        ):
                            _persist(run, artifact)
                            continue
                    selected = (models is None or model in models) and (
                        target_ids is None or target.id in target_ids
                    )
                    reason = readiness(spec.provider)
                    if not execute or not selected or reason:
                        if artifact.status != "submitted":
                            artifact.status = "blocked"
                            artifact.error = (
                                reason or "Live inference not requested for this target/model"
                            )
                        _persist(run, artifact)
                        continue
                    directory_path = run.directory / "cache" / artifact.cache_key
                    directory_path.mkdir(parents=True, exist_ok=True)
                    previous = artifact.model_dump()
                    if artifact.status != "submitted":
                        # A process crash after this write must not cause an automatic re-POST.
                        artifact.status = "indeterminate"
                        artifact.error = "Submission started; result not yet persisted"
                    _persist(run, artifact)

                    def checkpoint(status, request_id):
                        artifact.status = status
                        artifact.request_id = request_id
                        artifact.error = None
                        _persist(run, artifact)

                    try:
                        if spec.provider not in providers:
                            providers[spec.provider] = provider_factory(spec.provider)
                        output = providers[spec.provider].predict(
                            target, spec, sample, directory_path, checkpoint, previous
                        )
                        store_output(run, target, artifact, output)
                    except ProviderError as exc:
                        artifact.status = exc.status
                        artifact.error = str(exc)
                        _persist(run, artifact)
                    except Exception as exc:
                        # Do not store arbitrary SDK exception text, which may contain credentials.
                        artifact.status = (
                            "indeterminate"
                            if artifact.raw_file is None
                            and not any(directory_path.glob("response.*"))
                            else "failed"
                        )
                        artifact.error = f"{type(exc).__name__}: provider or output validation failed; inspect cached response"
                        _persist(run, artifact)
    finally:
        for provider in providers.values():
            provider.close()
    return run


def import_prediction(
    run, target_id, model, sample_id, path, *, confidence_scale=None, provenance=None
):
    """Explicitly attach an existing, matching prediction. Never substitute model identity."""
    if run.kind != "prediction":
        raise ValueError("Import real predictions only into prediction runs")
    target = next(t for t in run.targets if t.id == target_id)
    spec = run.protocol.models[model]
    if str(sample_id) not in spec.samples:
        raise ValueError("Sample is not in the run protocol")
    artifact = _record(target, model, spec, str(sample_id))
    if any(a.cache_key == artifact.cache_key and a.status == "complete" for a in run.artifacts):
        raise ValueError("Completed samples are immutable; create a new run to replace one")
    path = Path(path)
    output = ProviderOutput(
        path.read_text(),
        format="cif" if path.suffix.lower() in (".cif", ".mmcif") else "pdb",
        confidence_scale=confidence_scale,
        confidence_source="Explicit imported prediction metadata",
        provenance={
            "imported_from": str(path.resolve()),
            "source_sha256": sha256(path.read_bytes()),
            "declared_model": model,
            **(provenance or {}),
        },
    )
    return store_output(run, target, artifact, output)


def status_table(run):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "target": a.target_id,
                "model": a.model,
                "sample": a.sample_id,
                "status": a.status,
                "kind": a.kind,
                "duplicate_of": a.duplicate_of,
                "detail": a.error,
            }
            for a in run.artifacts
        ]
    )
