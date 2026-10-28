from types import SimpleNamespace

import numpy as np
import pytest

from harmonize.providers import BiohubProvider


def test_pinned_sdk_configuration_and_honest_cache_status(tmp_path, monkeypatch):
    pytest.importorskip("esm.sdk.forge")
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("BIOHUB_API_TOKEN", "test-only-secret")
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def fold_all_atom(self, request, *, config):
            calls.append(
                {
                    "loops": config.num_loops,
                    "steps": config.num_sampling_steps,
                    "sequence": request.sequences[0].sequence,
                }
            )
            return SimpleNamespace(
                complex=SimpleNamespace(to_mmcif=lambda: "data_test\n"),
                plddt=torch.tensor([0.9, 0.8, 0.7]),
                pae=torch.zeros((3, 3)),
                ptm=0.8,
            )

        def close(self):
            calls.append("closed")

    provider = BiohubProvider(client_factory=FakeClient)
    spec = SimpleNamespace(
        version="esmfold2-2026-05",
        settings={"num_loops": 20, "num_sampling_steps": 100, "include_pae": True},
    )
    output = provider.predict(
        SimpleNamespace(sequence="AAA"), spec, "1", tmp_path, lambda *args: None
    )
    assert calls[0]["max_retry_attempts"] == 1
    assert calls[1] == {"loops": 20, "steps": 100, "sequence": "AAA"}
    assert calls[2] == "closed"
    assert output.confidence_scale == "0-1"
    assert output.provenance["fresh_sample_verified"] is False
    np.testing.assert_allclose(output.plddt, [0.9, 0.8, 0.7])
    assert "test-only-secret" not in (tmp_path / "request.json").read_text()


def test_actual_sdk_method_reaches_transport_without_unsupported_keyword(tmp_path, monkeypatch):
    import json

    import httpx

    forge = pytest.importorskip("esm.sdk.forge")
    monkeypatch.setenv("BIOHUB_API_TOKEN", "test-only-secret")
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "esmfold2-2026-05", "created": "test-time"})

    def factory(**kwargs):
        client = forge.SequenceStructureForgeInferenceClient(**kwargs)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        return client

    parsed = SimpleNamespace(
        complex=SimpleNamespace(to_mmcif=lambda: "data_test\n"), plddt=None, pae=None, ptm=None
    )
    monkeypatch.setattr(
        forge.SequenceStructureForgeInferenceClient,
        "_process_fold_all_atom_response",
        staticmethod(lambda _: parsed),
    )
    spec = SimpleNamespace(
        version="esmfold2-2026-05",
        settings={"num_loops": 20, "num_sampling_steps": 100, "include_pae": True},
    )
    result = BiohubProvider(client_factory=factory).predict(
        SimpleNamespace(sequence="AAA"), spec, "1", tmp_path, lambda *_: None
    )
    assert len(calls) == 1
    assert calls[0]["model"] == spec.version
    assert calls[0]["num_loops"] == 20
    assert calls[0]["num_sampling_steps"] == 100
    assert calls[0]["include_pae"] is True
    assert "disable_cache" not in calls[0]
    assert result.provenance["fresh_sample_verified"] is False
    assert result.provenance["response_model"] == spec.version
    assert np.isnan(result.plddt).all()
    assert (tmp_path / "response.json").exists()


def test_unverified_fresh_ensemble_is_blocked_before_client_construction(tmp_path):
    pytest.importorskip("esm.sdk.forge")
    from harmonize.providers import CacheControlUnverified

    def forbidden(**kwargs):
        raise AssertionError("No client should be created")

    with pytest.raises(CacheControlUnverified):
        BiohubProvider(client_factory=forbidden).predict(None, None, "2", tmp_path, lambda *_: None)
