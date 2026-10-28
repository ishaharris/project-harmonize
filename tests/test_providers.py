from types import SimpleNamespace

import httpx
import pytest

from harmonize.providers import (
    ESMAtlasProvider,
    IndeterminateSubmission,
    NvidiaProvider,
    ProviderError,
    extract_single_pdb,
    request_http,
)

PDB = "HEADER TEST\nATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\nEND\n"


def test_rate_limit_bounded_retry():
    calls = []
    sleeps = []

    def handler(request):
        calls.append(request)
        return (
            httpx.Response(429, headers={"Retry-After": "1"})
            if len(calls) < 3
            else httpx.Response(200, text=PDB)
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = request_http(client, "POST", "https://example.test/fold", sleep=sleeps.append)
    assert response.status_code == 200
    assert len(calls) == 3
    assert sleeps == [1, 1]


def test_uncertain_submission_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("lost response", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(IndeterminateSubmission):
            request_http(client, "POST", "https://example.test/fold")
    assert len(calls) == 1


def test_async_nvidia_response_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    calls, checkpoints = [], []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(202, headers={"nvcf-reqid": "saved-job"})
        return httpx.Response(200, json={"pdbs": [PDB]})

    provider = NvidiaProvider(
        httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _: None
    )
    result = provider.predict(
        SimpleNamespace(sequence="AAA"),
        SimpleNamespace(settings={}),
        "3",
        tmp_path,
        lambda *x: checkpoints.append(x),
    )
    assert checkpoints == [("submitted", "saved-job")]
    assert result.provenance["checkpoint"] == 3
    assert b'"selected_models":[3]' in calls[0].content
    assert calls[1].url.path.endswith("/saved-job")
    assert "test-secret" not in (tmp_path / "request.json").read_text()
    provider.close()


def test_nvidia_resume_only_polls(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    methods = []

    def handler(request):
        methods.append(request.method)
        return httpx.Response(200, json={"pdbs": [PDB]})

    provider = NvidiaProvider(httpx.Client(transport=httpx.MockTransport(handler)))
    provider.predict(
        SimpleNamespace(sequence="AAA"),
        SimpleNamespace(settings={}),
        "1",
        tmp_path,
        lambda *args: None,
        {"request_id": "existing"},
    )
    assert methods == ["GET"]
    provider.close()


def test_ambiguous_structure_output_rejected():
    with pytest.raises(ProviderError, match="unambiguous"):
        extract_single_pdb({"pdbs": [PDB, PDB]})


def test_esmfold_rejects_html_error(tmp_path):
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text="<html>Unavailable</html>")
        )
    )
    provider = ESMAtlasProvider(client)
    with pytest.raises(ProviderError, match="PDB"):
        provider.predict(SimpleNamespace(sequence="AAA"), None, "1", tmp_path, lambda *args: None)
    provider.close()
