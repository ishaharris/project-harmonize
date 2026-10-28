from types import SimpleNamespace

import numpy as np
import pytest

from harmonize.providers import BiohubProvider


def test_pinned_sdk_configuration_and_fresh_sampling(tmp_path, monkeypatch):
    pytest.importorskip("esm.sdk.forge")
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("BIOHUB_API_TOKEN", "test-only-secret")
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def fold_all_atom(self, request, *, config, disable_cache):
            calls.append(
                {
                    "disable_cache": disable_cache,
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
    assert calls[1] == {"disable_cache": True, "loops": 20, "steps": 100, "sequence": "AAA"}
    assert calls[2] == "closed"
    assert output.confidence_scale == "0-1"
    np.testing.assert_allclose(output.plddt, [0.9, 0.8, 0.7])
    assert "test-only-secret" not in (tmp_path / "request.json").read_text()
