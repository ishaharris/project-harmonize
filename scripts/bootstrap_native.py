"""Download verified US-align source and build a project-local executable."""

import hashlib
import io
import json
import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "configs/native-tools.json").read_text())["usalign"]
    directory = root / ".tools/usalign"
    executable = directory / "USalign"
    if executable.exists():
        result = subprocess.run([str(executable), "-v"], capture_output=True, text=True)
        if (
            f"Version {spec['version']}" in result.stdout
            and (directory / "provenance.json").exists()
        ):
            print(f"Already installed: {executable}")
            return
        if f"Version {spec['version']}" not in result.stdout:
            raise RuntimeError("Existing US-align does not match the pinned version")
    directory.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(spec["archive_url"], timeout=60) as response:
        archive = response.read()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != spec["archive_sha256"]:
        raise RuntimeError(f"US-align archive checksum mismatch: {digest}")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        bundle.extractall(directory, filter="data")
    source = directory / f"USalign-{spec['commit']}"
    subprocess.run(
        [os.environ.get("CXX", "c++"), "-O3", "-o", str(executable), "USalign.cpp"],
        cwd=source,
        check=True,
    )
    banner = subprocess.run([str(executable), "-v"], capture_output=True, text=True)
    if f"Version {spec['version']}" not in banner.stdout:
        raise RuntimeError("Built US-align version mismatch")
    (directory / "provenance.json").write_text(
        json.dumps(
            {**spec, "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}, indent=2
        )
        + "\n"
    )
    print(f"Built and verified US-align {spec['version']}: {executable}")


if __name__ == "__main__":
    main()
