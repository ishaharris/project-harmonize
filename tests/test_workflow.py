import numpy as np
import pytest

from harmonize import compare_run
from harmonize.providers import IndeterminateSubmission, ProviderOutput
from harmonize.structures import reference_arrays, write_ca_pdb
from harmonize.types import ModelSpec, Protocol
from harmonize.workflow import (
    artifact_arrays,
    import_prediction,
    initialize_run,
    load_run,
    run_predictions,
)


def protocol(samples=("1", "2")):
    return Protocol(
        protocol_version="test",
        models={
            "esmfold": ModelSpec(
                provider="esmatlas", version="esmfold_v1", samples=list(samples), settings={}
            )
        },
    )


def pdb_text(target, tmp_path):
    xyz = reference_arrays(target)["xyz"]
    mask = np.isfinite(xyz).all(1)
    path = tmp_path / "output.pdb"
    write_ca_pdb(path, xyz[mask], np.flatnonzero(mask) + 1, target.sequence)
    return path.read_text()


def test_cache_and_duplicate_detection(targets, tmp_path):
    text = pdb_text(targets[0], tmp_path)
    calls = []

    class FakeProvider:
        def predict(self, target, spec, sample, directory, checkpoint, previous):
            calls.append(sample)
            return ProviderOutput(text)

        def close(self):
            pass

    kwargs = dict(
        directory=tmp_path / "run",
        protocol=protocol(),
        execute=True,
        provider_factory=lambda _: FakeProvider(),
    )
    run = run_predictions(targets[:1], **kwargs)
    assert run.complete
    assert calls == ["1", "2"]
    assert run.artifacts[1].duplicate_of == "esmfold:1"
    run_predictions(targets[:1], **kwargs)
    assert calls == ["1", "2"]
    assert load_run(tmp_path / "run").complete


def test_uncertain_job_does_not_auto_resubmit(targets, tmp_path):
    calls = []

    class FakeProvider:
        def predict(self, *args):
            calls.append(1)
            raise IndeterminateSubmission("No response")

        def close(self):
            pass

    kwargs = dict(
        directory=tmp_path / "run",
        protocol=protocol(("1",)),
        execute=True,
        provider_factory=lambda _: FakeProvider(),
    )
    run = run_predictions(targets[:1], **kwargs)
    assert run.artifacts[0].status == "indeterminate"
    run_predictions(targets[:1], **kwargs)
    assert len(calls) == 1


def test_default_run_makes_no_inference_calls(targets, tmp_path):
    def forbidden(*args):
        raise AssertionError("Provider must not be constructed")

    run = run_predictions(
        targets, directory=tmp_path / "run", protocol=protocol(), provider_factory=forbidden
    )
    assert not run.complete
    assert len(run.artifacts) == 12


def test_import_and_checksum_integrity(targets, tmp_path):
    path = tmp_path / "import.pdb"
    path.write_text(pdb_text(targets[0], tmp_path))
    run = initialize_run(targets[:1], tmp_path / "run", protocol(("1",)))
    artifact = import_prediction(run, targets[0].id, "esmfold", "1", path)
    assert len(artifact_arrays(run, artifact)["xyz"]) == len(targets[0].sequence)
    (run.directory / artifact.structure_file).write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_run(run.directory)


@pytest.mark.native
def test_incomplete_analysis_has_headers_and_no_predictions(targets, tmp_path, native):
    run = run_predictions(targets[:1], directory=tmp_path / "run", protocol=protocol())
    result = compare_run(run, executable=native)
    assert result.pairs.empty
    assert "tm_score" in result.pairs.columns
    assert (result.directory / "pairs.csv").exists()


def test_changed_protocol_cannot_reuse_run(targets, tmp_path):
    initialize_run(targets[:1], tmp_path / "run", protocol())
    with pytest.raises(ValueError, match="differ"):
        initialize_run(targets[:1], tmp_path / "run", protocol(("1",)))
