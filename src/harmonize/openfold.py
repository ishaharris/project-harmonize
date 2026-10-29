"""Hosted OpenFold2 with independently cached MSA and explicit template preparation."""

import json
import os
import time
from pathlib import Path

import httpx

from .providers import (
    NVIDIA_STATUS_URL,
    IndeterminateSubmission,
    PendingSubmission,
    ProviderError,
    ProviderOutput,
    request_http,
)
from .storage import atomic_json, now, read_json, sha256
from .templates import search_templates

OPENFOLD_URL = "https://health.api.nvidia.com/v1/biology/openfold/openfold2/predict-structure-from-msa-and-template"
MSA_URL = "https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict"
STATUS_URL = NVIDIA_STATUS_URL


class PreparationBlocked(ProviderError):
    status = "blocked"


def preparation_directory(run_directory, target, spec):
    key = sha256(
        json.dumps(
            {
                "schema": 1,
                "sequence": target.sequence_sha256,
                "msa_endpoint": MSA_URL,
                "settings": spec.settings["preparation"],
            },
            sort_keys=True,
        )
    )
    return Path(run_directory) / "preparation" / key


def verify_preparation(directory):
    manifest = read_json(directory / "manifest.json")
    for name, digest in manifest["files"].items():
        path = directory / name
        if not path.exists() or sha256(path.read_bytes()) != digest:
            raise ProviderError("Prepared OpenFold inputs missing or changed; use a new run")
    return manifest


def validate_alignments(payload, sequence):
    alignments = payload.get("alignments")
    if not isinstance(alignments, dict) or not alignments:
        raise ProviderError("MSA search returned no alignments")
    output, depths = {}, {}
    for database, formats in alignments.items():
        # Do not feed the duplicate merged ColabFold entry as another database.
        if database.lower() == "colabfold" and len(alignments) > 1:
            continue
        record = formats.get("a3m", {})
        if record.get("format") != "a3m" or not isinstance(record.get("alignment"), str):
            raise ProviderError("MSA result does not contain an A3M alignment")
        rows = []
        for line in record["alignment"].splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                rows.append("")
            elif rows:
                rows[-1] += line
            else:
                raise ProviderError("Malformed A3M alignment")
        match_columns = ["".join(c for c in row if not c.islower() and c != ".") for row in rows]
        if not match_columns or match_columns[0] != sequence:
            raise ProviderError("MSA query differs from the frozen input sequence")
        if any(
            len(row) != len(sequence) or set(row) - set("ARNDCQEGHILKMFPSTWYVX-")
            for row in match_columns
        ):
            raise ProviderError("MSA rows do not preserve input residue correspondence")
        output[database] = {"a3m": record}
        depths[database] = len(rows)
    if not output or max(depths.values()) < 2:
        raise ProviderError("MSA search found no homolog rows; sequence-only fallback is disabled")
    return output, depths


class OpenFoldProvider:
    def __init__(self, client=None, sleep=time.sleep, max_polls=120):
        self.client = client or httpx.Client(timeout=330)
        self.sleep = sleep
        self.max_polls = max_polls

    def _job(self, url, payload, directory, *, checkpoint=None, retry_failed=False):
        """Each stage owns its durable job ID; an MSA job can never become a fold job."""
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "job.json"
        request_path = directory / "request.json"
        response_path = directory / "response.json"
        state = read_json(state_path) if state_path.exists() else {}
        if request_path.exists() and read_json(request_path) != payload:
            raise ProviderError("Cached NVIDIA request inputs changed; use a new run")
        if state.get("status") == "complete":
            if (
                not response_path.exists()
                or sha256(response_path.read_bytes()) != state["response_sha256"]
            ):
                raise ProviderError("Cached NVIDIA response missing or changed")
            return read_json(response_path), state
        if state.get("status") == "indeterminate":
            raise IndeterminateSubmission(
                "Previous submission acceptance is unknown; inspect provider before retrying"
            )
        if state.get("status") == "failed" and not retry_failed:
            raise ProviderError(
                "Previous NVIDIA stage failed; inspect cached job and use --retry-failed"
            )
        headers = {
            "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
            "NVCF-POLL-SECONDS": "300",
        }
        # Even an explicit retry of failed polling must retain an accepted job's ID.
        request_id = state.get("request_id")
        try:
            if request_id:
                if checkpoint:
                    checkpoint("submitted", request_id)
                response = request_http(
                    self.client, "GET", STATUS_URL + request_id, headers=headers, sleep=self.sleep
                )
            else:
                atomic_json(request_path, payload)
                state = {"status": "indeterminate", "endpoint": url, "started_at": now()}
                atomic_json(state_path, state)
                if checkpoint:
                    checkpoint("indeterminate", None)
                response = request_http(
                    self.client, "POST", url, json=payload, headers=headers, sleep=self.sleep
                )
                # Save even a synchronous request identifier, when supplied.
                request_id = response.headers.get("nvcf-reqid")
            polls = 0
            while response.status_code == 202:
                request_id = response.headers.get("nvcf-reqid", request_id)
                if not request_id:
                    raise IndeterminateSubmission("NVIDIA accepted a job without a request ID")
                state.update(status="submitted", request_id=request_id)
                atomic_json(state_path, state)
                if checkpoint:
                    checkpoint("submitted", request_id)
                if polls >= self.max_polls:
                    raise PendingSubmission("Polling budget exhausted; resume the saved request ID")
                self.sleep(5)
                response = request_http(
                    self.client, "GET", STATUS_URL + request_id, headers=headers, sleep=self.sleep
                )
                polls += 1
            response_path.write_bytes(response.content)
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Expected object")
            except ValueError as exc:
                raise ProviderError("Malformed NVIDIA JSON response") from exc
            state.update(
                status="complete",
                request_id=request_id,
                completed_at=now(),
                response_sha256=sha256(response.content),
            )
            atomic_json(state_path, state)
            return data, state
        except ProviderError as exc:
            if exc.response_body is not None:
                error_path = directory / "error-response.txt"
                error_path.write_bytes(exc.response_body)
                state.update(
                    http_status=exc.http_status,
                    error_response_file=error_path.name,
                    error_response_sha256=sha256(exc.response_body),
                )
            if exc.request_id and not request_id:
                state.update(status="submitted", request_id=exc.request_id, error=str(exc))
                atomic_json(state_path, state)
                if checkpoint:
                    checkpoint("submitted", exc.request_id)
                raise PendingSubmission(
                    "NVIDIA error response supplied a request ID; poll the saved request without resubmitting"
                ) from exc
            state.update(status=exc.status, error=str(exc))
            if request_id:
                state["request_id"] = request_id
            atomic_json(state_path, state)
            raise

    def prepare(self, target, spec, run_directory, *, retry_failed=False):
        directory = preparation_directory(run_directory, target, spec)
        if (directory / "manifest.json").exists():
            return directory, verify_preparation(directory)
        recipe = spec.settings["preparation"]
        payload = {"sequence": target.sequence, **recipe["msa"]}
        try:
            response, msa_job = self._job(
                MSA_URL, payload, directory / "msa", retry_failed=retry_failed
            )
        except (PendingSubmission, IndeterminateSubmission) as exc:
            # This is preparation, not an accepted structure prediction.
            raise PreparationBlocked(f"MSA preparation: {exc}") from exc
        alignments, depths = validate_alignments(response, target.sequence)
        atomic_json(directory / "alignments.json", alignments)
        templates = search_templates(
            target, recipe["templates"], directory / "templates", self.client, sleep=self.sleep
        )
        files = {
            str(path.relative_to(directory)): sha256(path.read_bytes())
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_at": now(),
            "sequence_sha256": target.sequence_sha256,
            "settings": recipe,
            "msa_endpoint": MSA_URL,
            "msa_job": msa_job,
            "msa_depths": depths,
            "templates": templates,
            "files": files,
            "template_search": "RCSB experimental polymer sequence search; MMseqs2",
            "template_search_date_cutoff": None,
            "reference_overlap_policy": "allowed and explicitly annotated; not a held-out benchmark",
        }
        atomic_json(directory / "manifest.json", manifest)
        return directory, manifest

    def predict(self, target, spec, sample, directory, checkpoint, previous=None):
        if sample not in {"1", "2", "3", "4", "5"}:
            raise ProviderError("OpenFold checkpoint must be 1–5")
        if (
            spec.settings.get("use_templates") is not True
            or spec.settings.get("relax_prediction") is not False
        ):
            raise ProviderError("This protocol requires templates enabled and relaxation disabled")
        run_directory = directory.parent.parent
        prepared, manifest = self.prepare(
            target, spec, run_directory, retry_failed=(previous or {}).get("status") == "failed"
        )
        if manifest["sequence_sha256"] != target.sequence_sha256:
            raise ProviderError("Prepared inputs belong to another sequence")
        templates = [
            {
                "name": item["identifier"],
                "format": "mmcif",
                "rank": -1,
                "structure": (prepared / "templates" / item["file"]).read_text(),
            }
            for item in manifest["templates"]
        ]
        payload = {
            "sequence": target.sequence,
            "input_id": f"{target.id}-openfold2-{sample}",
            "selected_models": [int(sample)],
            "relax_prediction": False,
            "use_templates": True,
            "explicit_templates": templates,
            "alignments": read_json(prepared / "alignments.json"),
        }
        data, job = self._job(
            OPENFOLD_URL,
            payload,
            directory,
            checkpoint=checkpoint,
            retry_failed=(previous or {}).get("status") == "failed",
        )
        if data.get("of2_nim_handled_error_message") not in (None, "no-handled-error"):
            raise ProviderError(
                "OpenFold reported a handled inference error; inspect cached response"
            )
        records = data.get("structures_in_ranked_order")
        if not isinstance(records, list) or len(records) != 1:
            raise ProviderError("Expected exactly one OpenFold result for one selected checkpoint")
        record = records[0]
        if record.get("model_param_set") not in (None, int(sample)):
            raise ProviderError("OpenFold returned the wrong parameter set")
        if record.get("relaxed") is True:
            raise ProviderError(
                "OpenFold returned relaxed coordinates despite relaxation being disabled"
            )
        structure = record.get("structure")
        fmt = record.get("format")
        if not isinstance(structure, str) or not structure or fmt not in ("pdb", "mmcif"):
            raise ProviderError("Malformed OpenFold structure record")
        plddt = record.get("plddt")
        return ProviderOutput(
            structure,
            format="cif" if fmt == "mmcif" else "pdb",
            plddt=plddt,
            pae=record.get("predicted_aligned_error"),
            confidence_scale="0-100" if plddt is not None else "unknown",
            confidence_source="OpenFold JSON plddt"
            if plddt is not None
            else "OpenFold structure B-factor; response scale unverified",
            provenance={
                "endpoint": OPENFOLD_URL,
                "request_id": job.get("request_id"),
                "checkpoint": int(sample),
                "variation": "parameter_set",
                "seed_control": "unavailable",
                "implementation": "OpenFold2",
                "weights": "AlphaFold 2.3.2 per NVIDIA model card",
                "deployed_version": "not exposed by hosted request",
                "preparation_file": str((prepared / "manifest.json").relative_to(run_directory)),
                "preparation_sha256": sha256((prepared / "manifest.json").read_bytes()),
                "msa_depths": manifest["msa_depths"],
                "template_ids": [item["identifier"] for item in manifest["templates"]],
                "template_reference_overlap": any(
                    item["same_entry_as_reference"] for item in manifest["templates"]
                ),
                "templates_supplied": True,
                "template_features_expected": int(sample) in (1, 2),
                "ptm": record.get("ptm_score"),
                "provider_metrics": data.get("metrics"),
            },
        )

    def close(self):
        self.client.close()


def preparation_status(run):
    """Offline notebook view; neither searches nor submits jobs."""
    rows = []
    spec = run.protocol.models.get("openfold2")
    if spec is None:
        return rows
    for target in run.targets:
        directory = preparation_directory(run.directory, target, spec)
        row = {
            "target": target.id,
            "status": "not prepared",
            "templates": "",
            "reference_overlap": False,
        }
        if (directory / "manifest.json").exists():
            manifest = verify_preparation(directory)
            row.update(
                status="complete",
                templates=", ".join(t["identifier"] for t in manifest["templates"]),
                reference_overlap=any(t["same_entry_as_reference"] for t in manifest["templates"]),
            )
        elif (directory / "msa/job.json").exists():
            state = read_json(directory / "msa/job.json")
            row["status"] = (
                "templates pending" if state["status"] == "complete" else f"MSA {state['status']}"
            )
        rows.append(row)
    return rows
