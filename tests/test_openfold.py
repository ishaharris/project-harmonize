import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml
from biotite.structure.io.pdbx import CIFCategory, CIFFile

from harmonize.openfold import (
    MSA_URL,
    OPENFOLD_URL,
    OpenFoldProvider,
    PreparationBlocked,
    preparation_directory,
    preparation_status,
    validate_alignments,
)
from harmonize.providers import IndeterminateSubmission, ProviderError
from harmonize.structures import reference_arrays, write_ca_pdb
from harmonize.templates import prepare_template, search_templates
from harmonize.types import Protocol
from harmonize.workflow import load_run, run_predictions

ROOT = Path(__file__).resolve().parents[1]


def add_polymer_sequence(file):
    # The minimal deposited-structure fixtures store this information in the scheme.
    scheme = file.block["pdbx_poly_seq_scheme"]
    file.block["entity_poly_seq"] = CIFCategory(
        {
            "entity_id": scheme["entity_id"].as_array(str),
            "num": scheme["seq_id"].as_array(str),
            "mon_id": scheme["mon_id"].as_array(str),
            "hetero": ["n"] * scheme.row_count,
        }
    )


@pytest.fixture
def spec():
    return Protocol.model_validate(
        yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
    ).models["openfold2"]


@pytest.fixture
def template_text():
    # Test-only backbone scaffolding from experimental CA positions. Not model output.
    file = CIFFile.read(ROOT / "tests/fixtures/3VVX.cif")
    add_polymer_sequence(file)
    file.block["entry"] = CIFCategory({"id": ["3VVX"]})
    atom = file.block["atom_site"]
    values = {key: np.repeat(value.as_array(str), 3) for key, value in atom.items()}
    values["label_atom_id"] = np.tile(["N", "CA", "C"], atom.row_count)
    values["auth_atom_id"] = values["label_atom_id"]
    values["id"] = np.arange(len(values["label_atom_id"])) + 1
    file.block["atom_site"] = CIFCategory(values)
    stream = io.StringIO()
    file.write(stream)
    return stream.getvalue()


def msa_payload(target):
    return {
        "alignments": {
            "uniref30_2302": {
                "a3m": {
                    "format": "a3m",
                    "alignment": f">query\n{target.sequence}\n>hit\n{target.sequence}\n",
                }
            }
        }
    }


def fold_payload(target, tmp_path, model=1):
    xyz = reference_arrays(target)["xyz"]
    present = np.isfinite(xyz).all(1)
    path = tmp_path / "mock.pdb"
    write_ca_pdb(path, xyz[present], np.flatnonzero(present) + 1, target.sequence)
    return {
        "structures_in_ranked_order": [
            {
                "structure": path.read_text(),
                "format": "pdb",
                "model_param_set": model,
                "relaxed": False,
                "plddt": [91.0] * len(target.sequence),
                "predicted_aligned_error": None,
            }
        ],
        "of2_nim_handled_error_message": "no-handled-error",
    }


def public_handler(request, template_text):
    assert "Authorization" not in request.headers
    if request.url.host == "search.rcsb.org":
        query = json.loads(request.url.params["json"])
        assert query["return_type"] == "polymer_entity"
        return httpx.Response(200, json={"result_set": [{"identifier": "3VVX_1", "score": 1.0}]})
    return httpx.Response(200, text=template_text)


def test_template_mapping_and_reference_overlap(targets, spec, template_text, tmp_path):
    target = targets[1]
    converted, info = prepare_template(
        template_text, "1", target.sequence, spec.settings["preparation"]["templates"]
    )
    block = CIFFile.read(io.StringIO(converted)).block
    assert set(block["atom_site"]["label_asym_id"].as_array(str)) == {"A"}
    assert info["identity_to_shorter"] == 1
    assert info["backbone_coverage"] == pytest.approx(191 / 194)
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: public_handler(r, template_text))
    ) as client:
        records = search_templates(
            target, spec.settings["preparation"]["templates"], tmp_path, client
        )
    assert records[0]["same_entry_as_reference"]
    assert records[0]["source_sha256"] != records[0]["sha256"]


def test_modified_template_rejected(targets, spec):
    file = CIFFile.read(ROOT / "tests/fixtures/3K82.cif")
    add_polymer_sequence(file)
    stream = io.StringIO()
    file.write(stream)
    text = stream.getvalue()
    with pytest.raises(ValueError, match="modified"):
        prepare_template(text, "1", targets[3].sequence, spec.settings["preparation"]["templates"])


def test_template_non_a_chain_is_remapped_without_renumbering(targets, spec, template_text):
    file = CIFFile.read(io.StringIO(template_text))
    atom = file.block["atom_site"]
    before = atom["label_seq_id"].as_array(str).copy()
    atom["label_asym_id"] = ["B"] * atom.row_count
    scheme = file.block["pdbx_poly_seq_scheme"]
    scheme["asym_id"] = ["B"] * scheme.row_count
    stream = io.StringIO()
    file.write(stream)
    converted, info = prepare_template(
        stream.getvalue(), "1", targets[1].sequence, spec.settings["preparation"]["templates"]
    )
    result = CIFFile.read(io.StringIO(converted)).block["atom_site"]
    assert info["source_label_chain"] == "B"
    assert set(result["label_asym_id"].as_array(str)) == {"A"}
    np.testing.assert_array_equal(before, result["label_seq_id"].as_array(str))


def test_alignment_correspondence_and_no_homologs(targets):
    target = targets[0]
    data = msa_payload(target)
    _, depths = validate_alignments(data, target.sequence)
    assert depths == {"uniref30_2302": 2}
    with pytest.raises(ProviderError, match="query differs"):
        validate_alignments(data, "A" * len(target.sequence))
    data["alignments"]["uniref30_2302"]["a3m"]["alignment"] = f">query\n{target.sequence}\n"
    with pytest.raises(ProviderError, match="no homolog"):
        validate_alignments(data, target.sequence)


def test_shared_preparation_and_fold_cache(targets, spec, template_text, tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    target, requests = targets[1], []

    def handler(request):
        requests.append(request)
        if request.url.host not in ("health.api.nvidia.com", "api.nvcf.nvidia.com"):
            return public_handler(request, template_text)
        payload = json.loads(request.content)
        if str(request.url) == MSA_URL:
            return httpx.Response(200, json=msa_payload(target))
        assert str(request.url) == OPENFOLD_URL
        assert payload["use_templates"] is True
        assert payload["relax_prediction"] is False
        assert len(payload["explicit_templates"]) == 1
        return httpx.Response(
            200, json=fold_payload(target, tmp_path, payload["selected_models"][0])
        )

    spec = spec.model_copy(update={"samples": ["1", "2"]})
    protocol = Protocol(protocol_version="test-openfold", models={"openfold2": spec})
    kwargs = dict(
        directory=tmp_path / "run",
        protocol=protocol,
        execute=True,
        provider_factory=lambda _: OpenFoldProvider(
            httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    run = run_predictions([target], **kwargs)
    assert run.complete
    assert len(requests) == 5  # one MSA, one RCSB query, one download, two folds
    assert run.artifacts[0].provenance["template_reference_overlap"]
    assert run.artifacts[0].confidence_scale == "0-100"
    assert run.artifacts[1].provenance["checkpoint"] == 2
    assert preparation_status(run)[0]["status"] == "complete"
    run_predictions([target], **kwargs)
    load_run(run.directory)
    assert len(requests) == 5
    prepared = preparation_directory(run.directory, target, spec)
    assert "test-secret" not in "".join(p.read_text() for p in prepared.rglob("*.json"))
    (prepared / "alignments.json").write_text("{}")
    with pytest.raises(ProviderError, match="changed"):
        load_run(run.directory)


def test_async_preparation_and_inference_have_separate_ids(
    targets, spec, template_text, tmp_path, monkeypatch
):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    target, methods, checkpoints = targets[1], [], []

    def handler(request):
        methods.append((request.method, str(request.url)))
        if request.url.host not in ("health.api.nvidia.com", "api.nvcf.nvidia.com"):
            return public_handler(request, template_text)
        if request.method == "POST":
            return httpx.Response(
                202,
                headers={"nvcf-reqid": "msa-job" if str(request.url) == MSA_URL else "fold-job"},
            )
        if request.url.path.endswith("msa-job"):
            return httpx.Response(200, json=msa_payload(target))
        return httpx.Response(200, json=fold_payload(target, tmp_path))

    from harmonize.providers import PendingSubmission

    provider = OpenFoldProvider(httpx.Client(transport=httpx.MockTransport(handler)), max_polls=0)
    directory = tmp_path / "run/cache/sample"
    directory.mkdir(parents=True)
    with pytest.raises(PreparationBlocked):
        provider.predict(target, spec, "1", directory, lambda *args: checkpoints.append(args))
    assert not checkpoints
    with pytest.raises(PendingSubmission):
        provider.predict(target, spec, "1", directory, lambda *args: checkpoints.append(args))
    result = provider.predict(target, spec, "1", directory, lambda *args: checkpoints.append(args))
    assert result.provenance["request_id"] == "fold-job"
    assert all(job in (None, "fold-job") for _, job in checkpoints)
    assert sum(method == "POST" for method, _ in methods) == 2
    provider.close()


def test_uncertain_msa_not_resubmitted(targets, spec, tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("lost response", request=request)

    provider = OpenFoldProvider(httpx.Client(transport=httpx.MockTransport(handler)))
    for _ in range(2):
        with pytest.raises(PreparationBlocked, match="MSA"):
            provider.predict(targets[0], spec, "1", tmp_path / "run/cache/sample", lambda *_: None)
    assert len(calls) == 1
    provider.close()


def test_uncertain_fold_not_resubmitted(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("lost response", request=request)

    provider = OpenFoldProvider(httpx.Client(transport=httpx.MockTransport(handler)))
    for _ in range(2):
        with pytest.raises(IndeterminateSubmission):
            provider._job(OPENFOLD_URL, {"sequence": "AAA"}, tmp_path)
    assert len(calls) == 1
    provider.close()


def test_failed_poll_retry_never_resubmits_accepted_job(tmp_path, monkeypatch):
    from harmonize.providers import PendingSubmission

    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    methods = []

    def handler(request):
        methods.append(request.method)
        if len(methods) == 1:
            return httpx.Response(202, headers={"nvcf-reqid": "accepted"})
        if len(methods) == 2:
            return httpx.Response(403)
        return httpx.Response(200, json={"finished": True})

    provider = OpenFoldProvider(httpx.Client(transport=httpx.MockTransport(handler)), max_polls=0)
    with pytest.raises(PendingSubmission):
        provider._job(OPENFOLD_URL, {"sequence": "AAA"}, tmp_path)
    with pytest.raises(ProviderError):
        provider._job(OPENFOLD_URL, {"sequence": "AAA"}, tmp_path)
    result, _ = provider._job(OPENFOLD_URL, {"sequence": "AAA"}, tmp_path, retry_failed=True)
    assert result == {"finished": True}
    assert methods == ["POST", "GET", "GET"]
    provider.close()


def test_no_templates_does_not_fall_back(targets, spec, tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(204))) as client:
        with pytest.raises(ProviderError, match="fallback is disabled"):
            search_templates(
                targets[0], spec.settings["preparation"]["templates"], tmp_path, client
            )


def test_unprepared_status_is_offline(targets, spec, tmp_path):
    protocol = Protocol(protocol_version="test", models={"openfold2": spec})
    run = run_predictions(targets, directory=tmp_path, protocol=protocol)
    assert all(row["status"] == "not prepared" for row in preparation_status(run))
    assert not list(tmp_path.glob("preparation/*"))


@pytest.mark.parametrize(
    "fault", ["wrong_checkpoint", "ambiguous", "relaxed", "handled_error", "malformed"]
)
def test_openfold_rejects_incompatible_responses(
    targets, spec, template_text, tmp_path, monkeypatch, fault
):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    target = targets[1]

    def handler(request):
        if request.url.host not in ("health.api.nvidia.com", "api.nvcf.nvidia.com"):
            return public_handler(request, template_text)
        if str(request.url) == MSA_URL:
            return httpx.Response(200, json=msa_payload(target))
        response = fold_payload(target, tmp_path)
        record = response["structures_in_ranked_order"][0]
        if fault == "wrong_checkpoint":
            record["model_param_set"] = 5
        elif fault == "ambiguous":
            response["structures_in_ranked_order"].append(record.copy())
        elif fault == "relaxed":
            record["relaxed"] = True
        elif fault == "handled_error":
            response["of2_nim_handled_error_message"] = "template failure"
        else:
            record["format"] = "html"
        return httpx.Response(200, json=response)

    provider = OpenFoldProvider(httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError):
        provider.predict(target, spec, "1", tmp_path / "run/cache/sample", lambda *_: None)
    provider.close()


def test_process_interruption_after_response_recovers_without_post(
    targets, spec, template_text, tmp_path, monkeypatch
):
    import harmonize.workflow as workflow

    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    target, requests = targets[1], []

    def handler(request):
        requests.append(request)
        if request.url.host not in ("health.api.nvidia.com", "api.nvcf.nvidia.com"):
            return public_handler(request, template_text)
        if str(request.url) == MSA_URL:
            return httpx.Response(200, json=msa_payload(target))
        return httpx.Response(200, json=fold_payload(target, tmp_path))

    protocol = Protocol(
        protocol_version="recovery",
        models={"openfold2": spec.model_copy(update={"samples": ["1"]})},
    )
    kwargs = dict(
        directory=tmp_path / "run",
        protocol=protocol,
        execute=True,
        provider_factory=lambda _: OpenFoldProvider(
            httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    original = workflow.store_output

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(workflow, "store_output", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_predictions([target], **kwargs)
    assert len(requests) == 4
    monkeypatch.setattr(workflow, "store_output", original)
    run = run_predictions([target], **kwargs)
    assert run.complete
    assert len(requests) == 4


def test_single_sample_smoke_selection(targets, tmp_path):
    from harmonize.providers import ProviderOutput
    from harmonize.types import ModelSpec

    calls = []

    class FakeProvider:
        def predict(self, target, spec, sample, *args):
            calls.append(sample)
            return ProviderOutput(
                fold_payload(target, tmp_path)["structures_in_ranked_order"][0]["structure"]
            )

        def close(self):
            pass

    protocol = Protocol(
        protocol_version="test",
        models={
            "esmfold": ModelSpec(
                provider="esmatlas", version="test", samples=["1", "2"], settings={}
            )
        },
    )
    run = run_predictions(
        targets[:1],
        directory=tmp_path / "run",
        protocol=protocol,
        execute=True,
        sample_ids=["1"],
        provider_factory=lambda _: FakeProvider(),
    )
    assert calls == ["1"]
    assert not run.complete


def test_gateway_timeout_request_id_is_retained_and_polled(tmp_path, monkeypatch):
    from harmonize.providers import PendingSubmission
    from harmonize.storage import read_json

    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(504, headers={"nvcf-reqid": "recoverable"})
        assert str(request.url) == "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/recoverable"
        return httpx.Response(200, json={"finished": True})

    provider = OpenFoldProvider(httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(PendingSubmission):
        provider._job(OPENFOLD_URL, {"sequence": "AAA"}, tmp_path)
    assert read_json(tmp_path / "job.json")["request_id"] == "recoverable"
    assert read_json(tmp_path / "job.json")["status"] == "submitted"
    result, _ = provider._job(OPENFOLD_URL, {"sequence": "AAA"}, tmp_path)
    assert result == {"finished": True}
    assert [request.method for request in calls] == ["POST", "GET"]
    provider.close()
