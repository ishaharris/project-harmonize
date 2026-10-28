import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib"))


@pytest.fixture
def targets():
    from harmonize.targets import load_reference_snapshot

    return load_reference_snapshot(ROOT)


@pytest.fixture
def native():
    executable = ROOT / ".tools/usalign/USalign"
    if not executable.exists():
        pytest.skip("Run scripts/bootstrap_native.py for native scientific tests")
    return executable


@pytest.fixture(autouse=True)
def no_live_network(monkeypatch):
    import httpx

    def blocked(*args, **kwargs):
        raise AssertionError("Offline tests must not make live HTTP requests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
