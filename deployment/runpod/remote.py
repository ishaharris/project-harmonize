"""Standalone remote supervisor (Python standard library only). No credentials required."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/harmonize")
AA = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
    )
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def call(args, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def capture(args):
    return subprocess.check_output(list(map(str, args)), text=True).strip()


def download(spec, path):
    if path.exists() and digest(path) == spec["sha256"]:
        return
    temp = path.with_suffix(".download")
    urllib.request.urlretrieve(spec["url"], temp)
    if digest(temp) != spec["sha256"]:
        raise RuntimeError("Toolchain download checksum mismatch: " + path.name)
    temp.replace(path)


def bootstrap(job):
    tools = ROOT / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    pins = json.loads((job / "toolchain.json").read_text())
    if os.uname().machine != "x86_64":
        raise RuntimeError("This deployment requires Linux x86_64")
    # Test namespaces before large downloads; container root alone is insufficient.
    call(["unshare", "--user", "--map-root-user", "true"])
    mount = capture(["findmnt", "-n", "-o", "TARGET", "-T", str(ROOT)])
    if mount == "/":
        raise RuntimeError(
            "/workspace must be backed by persistent storage, not the container root"
        )
    if shutil.disk_usage(ROOT).free < 50 * 1024**3:
        raise RuntimeError("At least 50 GiB free persistent storage is required for initial setup")
    gpu = capture(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    if os.geteuid() != 0:
        raise RuntimeError("Bootstrap requires root inside the RunPod pod to install Apptainer")
    call(["apt-get", "update"])
    call(
        [
            "apt-get",
            "install",
            "-y",
            "git",
            "curl",
            "ca-certificates",
            "squashfs-tools",
            "fuse3",
            "uidmap",
            "procps",
        ]
    )
    for name in ("java", "nextflow", "apptainer"):
        download(pins[name], tools / name)
    java = tools / "jdk-21.0.9+10"
    if not java.exists():
        call(["tar", "-xzf", tools / "java", "-C", tools])
    (tools / "nextflow").chmod(0o755)
    if not shutil.which("apptainer") or "1.4.3" not in capture(["apptainer", "--version"]):
        shutil.copyfile(tools / "apptainer", tools / "apptainer.deb")
        call(["apt-get", "install", "-y", str(tools / "apptainer.deb")])
    os.environ.update(
        JAVA_HOME=str(java), PATH=str(java / "bin") + ":" + str(tools) + ":" + os.environ["PATH"]
    )
    os.environ.update(
        NXF_HOME=str(ROOT / "nextflow"),
        NXF_APPTAINER_CACHEDIR=str(ROOT / "images"),
        APPTAINER_CACHEDIR=str(ROOT / "apptainer-cache"),
    )
    pipeline = tools / ("proteinfold-" + pins["pipeline_commit"])
    if not pipeline.exists():
        call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                pins["pipeline_release"],
                "https://github.com/nf-core/proteinfold.git",
                pipeline,
            ]
        )
    if capture(["git", "-C", pipeline, "rev-parse", "HEAD"]) != pins["pipeline_commit"]:
        raise RuntimeError("Pipeline commit differs from lock")
    images = ROOT / "images"
    images.mkdir(exist_ok=True)
    image_info = {}
    for mode, probe in [
        ("colabfold", 'import jax; assert any(d.platform == "gpu" for d in jax.devices())'),
        ("esmfold", "import torch; assert torch.cuda.is_available()"),
    ]:
        image = images / (mode + "-2.0.0.sif")
        source = "docker://quay.io/nf-core/proteinfold_" + mode + ":2.0.0"
        if not image.exists():
            call(["apptainer", "pull", image, source])
        call(["apptainer", "exec", "--nv", "--userns", image, "python", "-c", probe])
        image_info[mode] = {
            "source": source,
            "sif_sha256": digest(image),
            "inspect": capture(["apptainer", "inspect", "--json", image]),
        }
    previous = ROOT / "images.lock.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if any(old[m]["sif_sha256"] != image_info[m]["sif_sha256"] for m in image_info):
            raise RuntimeError("Cached container image changed; use a new deployment directory")
    else:
        write(previous, image_info)
    write(
        job / "deployment.json",
        {
            "gpu": gpu,
            "mount": mount,
            "pins": pins,
            "images": image_info,
            "java": capture(["java", "--version"]),
            "nextflow": capture(["nextflow", "-version"]),
            "apptainer": capture(["apptainer", "--version"]),
        },
    )
    return pipeline


def pdb_sequence(path):
    rows = [
        line
        for line in path.read_text().splitlines()
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA"
    ]
    if not rows or len({line[21] for line in rows}) != 1:
        raise ValueError("Expected one protein chain: " + str(path))
    if len({line[22:27] for line in rows}) != len(rows):
        raise ValueError("Duplicate residue coordinates: " + str(path))
    for line in rows:
        if not all(
            math.isfinite(float(line[a:b])) for a, b in [(30, 38), (38, 46), (46, 54), (60, 66)]
        ):
            raise ValueError("Nonfinite coordinate/confidence")
        if not 0 <= float(line[60:66]) <= 100:
            raise ValueError("Confidence outside 0–100")
    return "".join(AA[line[17:20]] for line in rows)


def inspect_outputs(out, targets, modes, allow_partial=False):
    records = []
    for target in targets:
        tid, sequence = target["id"], target["sequence"]
        if "colabfold" in modes:
            directory = out / "colabfold" / tid
            pdbs = list(directory.rglob("*_unrelaxed_rank_*_model_*_seed_*.pdb"))
            by_model = {}
            for pdb in pdbs:
                match = re.search(r"_model_([1-5])_seed_(\d+)\.pdb$", pdb.name)
                if not match or int(match[2]) != 0 or match[1] in by_model:
                    raise ValueError("Ambiguous checkpoint/seed outputs for " + tid)
                by_model[match[1]] = pdb
            if not allow_partial and set(by_model) != set("12345"):
                raise ValueError("Expected five checkpoints for " + tid)
            # Selected hit rows, plus materialized template coordinates; flag no-template runs.
            hits = [p for p in directory.rglob("*.m8") if p.read_text().strip()]
            templates = [
                p
                for p in directory.rglob("*")
                if p.is_file() and p.suffix in (".cif", ".gz") and ".cif" in p.name
            ]
            if not hits or not templates:
                raise ValueError(
                    "Missing selected template hits or template coordinates for " + tid
                )
            for model, pdb in by_model.items():
                scores = pdb.with_name(
                    pdb.name.replace("_unrelaxed_", "_scores_").replace(".pdb", ".json")
                )
                data = json.loads(scores.read_text())
                if len(data.get("plddt", [])) != len(sequence) or not all(
                    math.isfinite(x) and 0 <= x <= 100 for x in data["plddt"]
                ):
                    raise ValueError("Invalid ColabFold confidence for " + tid)
                if "pae" in data and (
                    len(data["pae"]) != len(sequence)
                    or any(len(row) != len(sequence) for row in data["pae"])
                    or any(
                        not math.isfinite(value) or value < 0
                        for row in data["pae"]
                        for value in row
                    )
                ):
                    raise ValueError("Invalid PAE dimensions")
                if pdb_sequence(pdb) != sequence:
                    raise ValueError("Prediction sequence mismatch for " + tid)
                records.append(
                    {
                        "target": tid,
                        "model": "colabfold",
                        "sample": model,
                        "pdb": str(pdb.relative_to(out)),
                        "scores": str(scores.relative_to(out)),
                    }
                )
        if "esmfold" in modes:
            pdb = out / "esmfold/top_ranked_structures" / (tid + ".pdb")
            if pdb_sequence(pdb) != sequence:
                raise ValueError("Prediction sequence mismatch for " + tid)
            records.append(
                {"target": tid, "model": "esmfold", "sample": "1", "pdb": str(pdb.relative_to(out))}
            )
    return records


def cached_weights(mode):
    directory = ROOT / "databases" / mode
    if mode == "esmfold":
        names = [
            "esmfold_3B_v1.pt",
            "esm2_t36_3B_UR50D.pt",
            "esm2_t36_3B_UR50D-contact-regression.pt",
        ]
        if all((directory / "params" / name).is_file() for name in names):
            return {
                "esmfold_db": str(directory),
                "esmfold_params_path": str(directory / "params/*.pt"),
            }
    if mode == "colabfold":
        for parent in sorted({p.parent for p in directory.rglob("*.npz")}):
            if all(
                any(
                    (parent / name).is_file()
                    for name in (f"params_model_{i}.npz", f"params_model_{i}_ptm.npz")
                )
                for i in range(1, 6)
            ):
                return {
                    "colabfold_db": str(directory),
                    "colabfold_alphafold2_params_path": str(parent / "*"),
                }
    return {}


def resource_limits():
    cpus = max(1, min(8, os.cpu_count() or 1))
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal:")
            )
        )
        // 1024**2
    )
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.exists() and limit.read_text().strip().isdigit():
        memory = min(memory, int(limit.read_text()) // 1024**3)
    if memory < 12:
        raise RuntimeError("At least 12 GiB RAM is required; 32 GiB or more recommended")
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        amount, period = quota.read_text().split()
        if amount.isdigit():
            cpus = max(1, min(cpus, int(amount) // int(period)))
    return cpus, memory


def execute(job, resume):
    manifest = json.loads((job / "manifest.json").read_text())
    state = {
        "status": "preflight",
        "pid": os.getpid(),
        "started_at": time.time(),
        "run_id": manifest["run_id"],
    }
    write(job / "state.json", state)
    try:
        with (ROOT / "setup.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            pipeline = bootstrap(job)
        modes = manifest["params"]["mode"].split(",")
        cpus, memory = resource_limits()
        host_arg = (
            (" --host-url " + manifest["params"]["msa_server_url"])
            if manifest["params"].get("msa_server_url")
            else ""
        )
        config = f"""process.executor = 'local'
process.maxForks = 1
executor.queueSize = 1
process.cpus = {cpus}
process.memory = '{max(4, min(36, memory - 4))} GB'
apptainer.runOptions = '--nv --userns'
process {{
withLabel: 'process_.*' {{ cpus = {cpus}; memory = '{max(4, min(36, memory - 4))} GB' }}
withName: '.*PREPARE_ESMFOLD_DBS:ARIA2_.*' {{
    publishDir = [path: '{ROOT}/databases/esmfold/params', mode: 'copy', pattern: '*.pt']
}}
withName: '.*ARIA2_COLABFOLD_PARAMS:UNTAR' {{
    publishDir = [path: '{ROOT}/databases/colabfold/params', mode: 'copy']
}}
withName: 'COLABFOLD_BATCH' {{
    container = '{ROOT}/images/colabfold-2.0.0.sif'
    ext.args = '--templates --num-models 5 --num-seeds 1 --random-seed 0 --save-all{host_arg}'
}}
withName: 'RUN_ESMFOLD' {{
    container = '{ROOT}/images/esmfold-2.0.0.sif'
    containerOptions = '--nv --userns'
}}
}}
"""
        (job / "runpod.config").write_text(config)
        targets = manifest["targets"]
        smoke = next((t for t in targets if t["id"] == "gpd"), targets[0])
        for stage, members in [("smoke", [smoke]), ("panel", [t for t in targets if t != smoke])]:
            if not members:
                continue
            folder = job / stage
            folder.mkdir(exist_ok=True)
            sample = folder / "samplesheet.csv"
            if not sample.exists():
                with sample.open("w") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["id", "fasta"])
                    for target in members:
                        writer.writerow(
                            [target["id"], str(job / "inputs" / (target["id"] + ".fasta"))]
                        )
            params = {**manifest["params"], "input": str(sample), "outdir": str(folder / "results")}
            for mode in modes:
                if mode + "_db" not in params:
                    params.update(cached_weights(mode))
                if mode not in ("colabfold", "esmfold") and not Path(params[mode + "_db"]).is_dir():
                    raise RuntimeError("Missing remote database directory for " + mode)
            if not (folder / "params.json").exists():
                write(folder / "params.json", params)
            # Existing parameters and input mtimes stay frozen for Nextflow -resume.
            params = json.loads((folder / "params.json").read_text())
            state.update(status=stage, updated_at=time.time())
            write(job / "state.json", state)
            command = [
                "nextflow",
                "run",
                str(pipeline),
                "-profile",
                "apptainer",
                "-c",
                str(job / "runpod.config"),
                "-params-file",
                str(folder / "params.json"),
                "-work-dir",
                str(folder / "work"),
                "-with-report",
                f"report-{int(time.time())}.html",
                "-with-trace",
                f"trace-{int(time.time())}.txt",
                "-with-timeline",
                f"timeline-{int(time.time())}.html",
            ]
            if resume and (folder / ".nextflow").exists():
                command.append("-resume")
            process = subprocess.Popen(list(map(str, command)), cwd=folder)
            state["nextflow_pid"] = process.pid
            write(job / "state.json", state)
            code = process.wait()
            if code:
                raise RuntimeError(
                    f"Nextflow {stage} exited {code}; see runner.log and .nextflow.log"
                )
            sessions = [p.name for p in (folder / ".nextflow/cache").glob("*") if p.is_dir()]
            state.setdefault("nextflow_sessions", {})[stage] = sessions
            records = inspect_outputs(folder / "results", members, modes)
            write(folder / "validated.json", records)
            weights = {
                str(p.relative_to(ROOT)): digest(p)
                for p in (ROOT / "databases").rglob("*")
                if p.is_file() and p.suffix in (".npz", ".pt")
            }
            write(job / "weights.json", weights)
        state.update(status="complete", completed_at=time.time())
    except Exception as exc:
        state.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", completed_at=time.time()
        )
        raise
    finally:
        write(job / "state.json", state)


def snapshot(job):
    paths = []
    for path in job.rglob("*"):
        rel = path.relative_to(job)
        if (
            path.is_file()
            and not path.is_symlink()
            and not any(
                p
                in (
                    "work",
                    "DBs",
                    ".nextflow",
                    "snapshot.tar.gz",
                    "checksums.json",
                    "launch.lock",
                    "worker.lock",
                )
                for p in rel.parts
            )
        ):
            paths.append(path)
    checksums = {str(p.relative_to(job)): digest(p) for p in paths}
    write(job / "checksums.json", checksums)
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(job)), recursive=False)
        archive.add(job / "checksums.json", arcname="checksums.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "worker", "status", "snapshot"])
    parser.add_argument("job", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    job = args.job.resolve()
    if args.action == "snapshot":
        snapshot(job)
        return
    state_path = job / "state.json"
    if args.action == "status":
        state = (
            json.loads(state_path.read_text()) if state_path.exists() else {"status": "not_started"}
        )
        if state.get("status") in ("starting", "preflight", "smoke", "panel"):
            try:
                command = Path(f"/proc/{state['pid']}/cmdline").read_bytes()
                state["alive"] = str(job).encode() in command and b"worker" in command
            except (FileNotFoundError, KeyError):
                state["alive"] = False
            if not state["alive"] and state.get("nextflow_pid"):
                try:
                    command = Path(f"/proc/{state['nextflow_pid']}/cmdline").read_bytes()
                    state["alive"] = str(job).encode() in command and b"nextflow" in command
                    state["orphaned_nextflow"] = state["alive"]
                except FileNotFoundError:
                    pass
            if not state["alive"]:
                state["status"] = "interrupted"
        print(json.dumps(state))
        return
    with (job / ("launch.lock" if args.action == "start" else "worker.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "worker":
            with (job / "launch.lock").open("a") as launch_lock:
                fcntl.flock(launch_lock, fcntl.LOCK_EX)
            with (ROOT / "gpu.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                execute(job, args.resume)
        else:
            state = json.loads(capture([sys.executable, __file__, "status", job]))
            if state.get("alive") or state["status"] == "complete":
                print(json.dumps(state))
                return
            if state["status"] != "not_started" and not args.resume:
                raise RuntimeError("Existing attempt: use resume, not a fresh launch")
            with (job / "runner.log").open("ab") as log:
                command = [sys.executable, __file__, "worker", str(job)] + (
                    ["--resume"] if args.resume else []
                )
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            state = {"status": "starting", "pid": process.pid}
            write(state_path, state)
            print(json.dumps(state))


if __name__ == "__main__":
    main()
