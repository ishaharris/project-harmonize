"""Serializable scientific records. Coordinates are always in angstroms."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Residue(Record):
    position: int
    amino_acid: str
    author_number: str
    insertion_code: str = ""
    deposited_name: str
    observed: bool
    modified: bool
    reference_eligible: bool
    ca: list[float] | None = None


class Target(Record):
    id: str
    name: str
    pdb_id: str
    chain: str
    sequence: str
    sequence_sha256: str
    source_url: str
    source_sha256: str
    source_file: str
    prepared_file: str
    notes: str
    residues: list[Residue]
    overrides: list[dict[str, Any]] = Field(default_factory=list)
    ligands: list[str] = Field(default_factory=list)


class ModelSpec(Record):
    provider: str
    version: str
    samples: list[str]
    settings: dict[str, Any]


class Protocol(Record):
    protocol_version: str
    models: dict[str, ModelSpec]


class PredictionArtifact(Record):
    target_id: str
    model: str
    provider: str
    model_version: str
    sample_id: str
    sequence_sha256: str
    cache_key: str
    status: Literal["complete", "blocked", "failed", "submitted", "indeterminate"]
    kind: Literal["prediction", "validation_fixture"] = "prediction"
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    structure_file: str | None = None
    structure_sha256: str | None = None
    coordinates_sha256: str | None = None
    normalized_file: str | None = None
    normalized_sha256: str | None = None
    raw_file: str | None = None
    confidence_source: str | None = None
    confidence_scale: str | None = None
    request_id: str | None = None
    seed: int | None = None
    error: str | None = None
    duplicate_of: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.model}:{self.sample_id}"


class Run(Record):
    schema_version: int = 1
    directory: Path
    kind: Literal["prediction", "validation_fixture"] = "prediction"
    created_at: str
    targets: list[Target]
    protocol: Protocol
    artifacts: list[PredictionArtifact] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def expected_count(self) -> int:
        return len(self.targets) * sum(len(spec.samples) for spec in self.protocol.models.values())

    @property
    def complete(self) -> bool:
        return len([a for a in self.artifacts if a.status == "complete"]) == self.expected_count


class PairwiseComparison(Record):
    target_id: str
    left: str
    right: str
    left_model: str
    right_model: str
    comparison_type: Literal["within_model", "cross_model", "experimental_reference"]
    n_matched: int = 0
    normalization_length: int
    coverage: float = 0
    tm_score: float | None = None
    rmsd: float | None = None
    lddt: float | None = None
    status: str = "complete"
    error: str | None = None
