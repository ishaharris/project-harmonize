"""Atomic, strict JSON storage and hashes for reproducible artifacts."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        name = stream.name
    os.replace(name, path)


def read_json(path: Path):
    return json.loads(path.read_text())


def cache_key(target, model: str, spec, sample: str) -> str:
    return sha256(
        json.dumps(
            {
                "schema": 1,
                "sequence": target.sequence_sha256,
                "model": model,
                "provider": spec.provider,
                "version": spec.version,
                "settings": spec.settings,
                "sample": sample,
            },
            sort_keys=True,
        )
    )
