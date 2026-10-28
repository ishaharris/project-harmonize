"""Provider boundaries. Never retry an ambiguously accepted inference submission."""

import importlib.util
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from .storage import atomic_json

NVIDIA_URL = (
    "https://health.api.nvidia.com/v1/protein-structure/alphafold2/predict-structure-from-sequence"
)
NVIDIA_STATUS_URL = "https://health.api.nvidia.com/v1/status/"
ESMATLAS_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"


class ProviderError(RuntimeError):
    status = "failed"


class IndeterminateSubmission(ProviderError):
    status = "indeterminate"


class PendingSubmission(ProviderError):
    status = "submitted"


@dataclass
class ProviderOutput:
    structure: str
    format: str = "pdb"
    confidence_scale: str | None = None
    confidence_source: str | None = None
    plddt: Any = None
    pae: Any = None
    provenance: dict = field(default_factory=dict)


def readiness(provider):
    if provider in ("nvidia", "nvidia_openfold2") and not os.getenv("NVIDIA_API_KEY"):
        return "NVIDIA_API_KEY is not configured"
    if provider == "biohub":
        if not os.getenv("BIOHUB_API_TOKEN"):
            return "BIOHUB_API_TOKEN is not configured"
        if importlib.util.find_spec("esm") is None:
            return "Install the optional Biohub client: uv sync --extra biohub --group notebook"
    if provider not in ("nvidia", "nvidia_openfold2", "biohub", "esmatlas"):
        return f"Unknown provider: {provider}"
    return None


def request_http(client, method, url, *, sleep=time.sleep, attempts=3, **kwargs):
    """Retry explicit rate rejection and read-only polling; preserve uncertain POSTs."""
    for attempt in range(attempts):
        try:
            response = client.request(method, url, **kwargs)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if method == "POST":
                raise IndeterminateSubmission(
                    "Connection interrupted; submission acceptance is unknown"
                ) from exc
            if attempt == attempts - 1:
                raise PendingSubmission(
                    "Status polling interrupted; resume the saved request ID"
                ) from exc
            sleep(min(2**attempt, 30))
            continue
        if response.status_code == 429:
            if attempt == attempts - 1:
                raise ProviderError("Provider rate limit persisted after bounded retries")
            try:
                delay = max(0, min(float(response.headers.get("Retry-After", "2")), 60))
            except ValueError:
                delay = 2**attempt
            sleep(delay)
            continue
        if response.status_code >= 500:
            if method == "POST":
                raise IndeterminateSubmission(
                    f"HTTP {response.status_code}; inference acceptance is unknown"
                )
            if attempt == attempts - 1:
                raise PendingSubmission(
                    f"Status endpoint HTTP {response.status_code}; resume later"
                )
            sleep(min(2**attempt, 30))
            continue
        if response.status_code not in (200, 202):
            raise ProviderError(
                f"Provider returned HTTP {response.status_code}; inspect provider access or inputs"
            )
        return response
    raise ProviderError("Request exhausted retry budget")


def extract_single_pdb(payload) -> tuple[str, str]:
    """Accept a single PDB leaf; never infer a checkpoint from an ambiguous list."""
    candidates = []

    def walk(value, path):
        if isinstance(value, str) and any(line.startswith("ATOM  ") for line in value.splitlines()):
            candidates.append((value, path))
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload, "response")
    if len(candidates) != 1:
        raise ProviderError(
            f"Expected one unambiguous PDB for one requested checkpoint; found {len(candidates)}"
        )
    return candidates[0]


class NvidiaProvider:
    def __init__(self, client=None, sleep=time.sleep, max_polls=120):
        self.client = client or httpx.Client(timeout=125)
        self.sleep = sleep
        self.max_polls = max_polls

    def predict(self, target, spec, sample, directory, checkpoint, previous=None):
        headers = {
            "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
            "NVCF-POLL-SECONDS": "2",
        }
        request_id = (previous or {}).get("request_id")
        if request_id:
            response = request_http(
                self.client,
                "GET",
                NVIDIA_STATUS_URL + request_id,
                headers=headers,
                sleep=self.sleep,
            )
        else:
            payload = {
                "sequence": target.sequence,
                **spec.settings,
                "selected_models": [int(sample)],
            }
            atomic_json(directory / "request.json", payload)
            response = request_http(
                self.client, "POST", NVIDIA_URL, headers=headers, json=payload, sleep=self.sleep
            )
        for _ in range(self.max_polls):
            if response.status_code != 202:
                break
            request_id = response.headers.get("nvcf-reqid", request_id)
            if not request_id:
                raise IndeterminateSubmission("NVIDIA accepted the job but supplied no request ID")
            checkpoint("submitted", request_id)
            self.sleep(5)
            response = request_http(
                self.client,
                "GET",
                NVIDIA_STATUS_URL + request_id,
                headers=headers,
                sleep=self.sleep,
            )
        if response.status_code == 202:
            raise PendingSubmission("Polling budget exhausted; resume the saved NVIDIA request ID")
        (directory / "response.json").write_bytes(response.content)
        try:
            structure, source_path = extract_single_pdb(response.json())
        except ValueError as exc:
            raise ProviderError("Malformed NVIDIA JSON response") from exc
        return ProviderOutput(
            structure,
            confidence_scale="0-100",
            confidence_source="AlphaFold pLDDT in PDB B-factor",
            provenance={
                "endpoint": NVIDIA_URL,
                "response_path": source_path,
                "request_id": request_id,
                "checkpoint": int(sample),
                "variation": "parameter_set",
                "seed_control": "unavailable",
            },
        )

    def close(self):
        self.client.close()


class ESMAtlasProvider:
    def __init__(self, client=None, sleep=time.sleep):
        self.client = client or httpx.Client(timeout=180)
        self.sleep = sleep

    def predict(self, target, spec, sample, directory, checkpoint, previous=None):
        if len(target.sequence) > 400:
            raise ProviderError("ESM Atlas adapter is limited to <=400 residues")
        atomic_json(
            directory / "request.json", {"sequence": target.sequence, "endpoint": ESMATLAS_URL}
        )
        response = request_http(
            self.client,
            "POST",
            ESMATLAS_URL,
            content=target.sequence,
            headers={"Content-Type": "text/plain"},
            sleep=self.sleep,
        )
        (directory / "response.pdb").write_bytes(response.content)
        if response.status_code != 200 or not any(
            line.startswith("ATOM  ") for line in response.text.splitlines()
        ):
            raise ProviderError("ESM Atlas did not return a PDB structure")
        # Historical endpoint examples use a different B-factor scale from fair-esm.
        # Preserve values without guessing scale from the observed maximum.
        return ProviderOutput(
            response.text,
            confidence_scale="unknown",
            confidence_source="ESM Atlas PDB B-factor; endpoint scale unverified",
            provenance={
                "endpoint": ESMATLAS_URL,
                "variation": "singleton",
                "seed_control": "unavailable",
            },
        )

    def close(self):
        self.client.close()


class BiohubProvider:
    def __init__(self, client_factory=None):
        self.client_factory = client_factory

    def predict(self, target, spec, sample, directory, checkpoint, previous=None):
        from esm.sdk.api import ESMProteinError, FoldingConfig
        from esm.sdk.forge import SequenceStructureForgeInferenceClient
        from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

        factory = self.client_factory or SequenceStructureForgeInferenceClient
        client = factory(
            model=spec.version,
            url="https://biohub.ai",
            token=os.environ["BIOHUB_API_TOKEN"],
            request_timeout=125,
            max_retry_attempts=1,
        )
        request = StructurePredictionInput(
            sequences=[ProteinInput(id="A", sequence=target.sequence)]
        )
        config = FoldingConfig(**spec.settings)
        atomic_json(
            directory / "request.json",
            {
                "sequence": target.sequence,
                "model": spec.version,
                "settings": spec.settings,
                "disable_cache": True,
                "seed": None,
            },
        )
        try:
            result = client.fold_all_atom(request, config=config, disable_cache=True)
        except Exception as exc:
            raise IndeterminateSubmission(
                "Biohub request interrupted; acceptance is unknown"
            ) from exc
        finally:
            client.close()
        if isinstance(result, ESMProteinError):
            code = result.error_code
            if code >= 500:
                raise IndeterminateSubmission(f"Biohub returned {code}; acceptance is unknown")
            raise ProviderError(
                f"Biohub returned {code}; check provider access or request settings"
            )
        if isinstance(result, list):
            raise ProviderError(
                "Unexpected multi-result Biohub response for a single requested sample"
            )
        structure = result.complex.to_mmcif()
        (directory / "response.cif").write_text(structure)
        confidence = None if result.plddt is None else result.plddt.detach().cpu().numpy()
        pae = None if result.pae is None else result.pae.detach().cpu().numpy()
        extra = {"plddt": confidence, "pae": pae}
        np.savez_compressed(
            directory / "raw_confidence.npz", **{k: v for k, v in extra.items() if v is not None}
        )
        return ProviderOutput(
            structure,
            format="cif",
            plddt=confidence,
            pae=pae,
            confidence_scale="0-1",
            confidence_source="Biohub MolecularComplexResult.plddt",
            provenance={
                "endpoint": "https://biohub.ai/api/v1/fold_all_atom",
                "disable_cache": True,
                "variation": "stochastic_sampling",
                "seed_control": "unavailable",
                "ptm": None if result.ptm is None else float(result.ptm),
            },
        )

    def close(self):
        pass


def make_provider(name):
    from .openfold import OpenFoldProvider

    return {
        "nvidia": NvidiaProvider,
        "nvidia_openfold2": OpenFoldProvider,
        "esmatlas": ESMAtlasProvider,
        "biohub": BiohubProvider,
    }[name]()
