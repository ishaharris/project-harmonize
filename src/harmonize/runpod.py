"""SSH orchestration for nf-core/proteinfold. Analysis and status never launch inference."""

import argparse
import csv
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from .providers import ProviderOutput
from .storage import atomic_json, now, read_json, sha256
from .targets import AA3, load_reference_snapshot
from .types import ModelSpec, Protocol, Residue, Target
from .workflow import _persist, _record, initialize_run, load_run, store_output

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/workspace/harmonize"
DEFAULT_PARAMS = {
    "mode": "colabfold,esmfold",
    "use_gpu": True,
    "use_msa_server": True,
    "colabfold_use_templates": True,
    "colabfold_use_amber": False,
    "colabfold_model_preset": "alphafold2_ptm",
    "colabfold_num_recycles": 3,
    "esmfold_model_preset": "monomer",
    "esmfold_num_recycles": 4,
    "save_intermediates": True,
}
MODES = {
    "colabfold",
    "esmfold",
    "alphafold2",
    "alphafold3",
    "boltz",
    "helixfold3",
    "rosettafold2na",
    "rosettafold_all_atom",
}


class SSH:
    def __init__(self, host, port, identity=None):
        if not re.fullmatch(r"(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*", host):
            raise ValueError("Use a hostname/IP or SSH alias, optionally user@host")
        if not 1 <= int(port) <= 65535:
            raise ValueError("Invalid SSH port")
        sockets = Path("/tmp") / f"harmonize-ssh-{os.getuid()}"
        sockets.mkdir(mode=0o700, exist_ok=True)
        if sockets.is_symlink() or sockets.stat().st_uid != os.getuid():
            raise ValueError("Unsafe SSH control directory")
        sockets.chmod(0o700)
        self.args = [
            "ssh",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=600",
            "-o",
            "ControlPath=" + str(sockets / "%C"),
            "-T",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(port),
        ]
        if identity:
            key = Path(identity).expanduser().resolve()
            if not key.is_file():
                raise ValueError("SSH identity file does not exist")
            self.args += ["-i", str(key), "-o", "IdentitiesOnly=yes"]
        self.args.append(host)

    def command(self, args):
        return self.args + [shlex.join([str(a) for a in args])]

    def call(self, args, **kwargs):
        return subprocess.run(self.command(args), check=True, **kwargs)

    def text(self, args):
        return subprocess.check_output(self.command(args), text=True)

    def status(self, remote):
        return json.loads(self.text(["python3", remote + "/remote.py", "status", remote]))


def fasta_entries(path):
    entries, identifier, sequence = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if identifier is not None:
                entries.append((identifier, "".join(sequence)))
            identifier, sequence = line[1:].split()[0], []
        elif line.strip():
            if identifier is None:
                raise ValueError("FASTA sequence before first header")
            sequence.append(line.strip().upper())
    if identifier is not None:
        entries.append((identifier, "".join(sequence)))
    if not entries:
        raise ValueError("Empty FASTA")
    return entries


def input_targets(fasta=None, samplesheet=None):
    if not fasta and not samplesheet:
        return load_reference_snapshot(ROOT)
    entries = []
    if fasta:
        entries = fasta_entries(fasta)
    else:
        sheet = Path(samplesheet).resolve()
        with sheet.open() as stream:
            for row in csv.DictReader(stream):
                path = Path(row["fasta"])
                if not path.is_absolute():
                    path = sheet.parent / path
                records = fasta_entries(path)
                if len(records) != 1:
                    raise ValueError("Each samplesheet FASTA must contain one monomer")
                entries.append((row["id"], records[0][1]))
    result, seen = [], set()
    known = {t.id: t for t in load_reference_snapshot(ROOT)}
    for identifier, sequence in entries:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier) or identifier in seen:
            raise ValueError("Duplicate or unsafe FASTA identifier: " + identifier)
        if len(sequence) < 3 or set(sequence) - set(AA3.values()):
            raise ValueError("Expected a monomer sequence of standard amino acids: " + identifier)
        seen.add(identifier)
        if identifier in known and sequence == known[identifier].sequence:
            result.append(known[identifier])
            continue
        result.append(
            Target(
                id=identifier,
                name=identifier,
                pdb_id="",
                chain="",
                sequence=sequence,
                sequence_sha256=sha256(sequence),
                source_url="",
                source_sha256=sha256(sequence),
                source_file="",
                prepared_file="",
                notes="User FASTA; no experimental reference supplied",
                residues=[
                    Residue(
                        position=i,
                        amino_acid=aa,
                        author_number=str(i),
                        insertion_code="",
                        deposited_name="",
                        observed=False,
                        modified=False,
                        reference_eligible=False,
                    )
                    for i, aa in enumerate(sequence, 1)
                ],
            )
        )
    return result


def parameters(path=None):
    import yaml

    supplied = (yaml.safe_load(Path(path).read_text()) or {}) if path else {}
    if not isinstance(supplied, dict):
        raise ValueError("Parameters must be a mapping")
    params = {**DEFAULT_PARAMS, **(supplied or {})}
    if params.get("msa_server_url") and not re.fullmatch(
        r"https?://[A-Za-z0-9.-]+(?::[0-9]+)?(?:/[A-Za-z0-9_/-]*)?", params["msa_server_url"]
    ):
        raise ValueError(
            "msa_server_url must be an HTTP(S) base URL without credentials or query parameters"
        )
    modes = params["mode"].split(",")
    if not modes or set(modes) - MODES or len(modes) != len(set(modes)):
        raise ValueError("Unsupported or duplicate nf-core modes")
    for key in ("use_gpu", "save_intermediates", "use_msa_server", "colabfold_use_templates"):
        if params.get(key) is not True:
            raise ValueError("This launcher requires " + key + "=true")
    if (
        params["colabfold_use_amber"] is not False
        or params["colabfold_model_preset"] != "alphafold2_ptm"
    ):
        raise ValueError("Harmonize requires unrelaxed alphafold2_ptm ColabFold outputs")
    for mode in set(modes) - {"colabfold", "esmfold"}:
        if not params.get(mode + "_db"):
            raise ValueError(
                "Additional modes require an explicit remote database path: " + mode + "_db"
            )
    for key in ("input", "outdir", "db"):
        if key in supplied:
            raise ValueError(key + " is managed by the launcher; use mode-specific database paths")
    return params


def prepare(targets, params):
    pins = read_json(ROOT / "deployment/runpod/toolchain.json")
    implementation = {
        p.name: sha256(p.read_bytes())
        for p in (ROOT / "deployment/runpod").glob("*")
        if p.is_file()
    }
    identity = {
        "targets": [t.model_dump(mode="json") for t in targets],
        "params": params,
        "pins": pins,
        "implementation": implementation,
    }
    run_id = "proteinfold-" + sha256(json.dumps(identity, sort_keys=True))[:16]
    directory = ROOT / "runs/runpod" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "inputs").mkdir(exist_ok=True)
    for target in targets:
        (bundle / "inputs" / (target.id + ".fasta")).write_text(
            f">{target.id}\n{target.sequence}\n"
        )
    manifest = {**identity, "run_id": run_id}
    atomic_json(bundle / "manifest.json", manifest)
    for name in ("remote.py", "toolchain.json"):
        shutil.copyfile(ROOT / "deployment/runpod" / name, bundle / name)
    atomic_json(directory / "launch.json", {"run_id": run_id, "remote": REMOTE + "/runs/" + run_id})
    return directory


def upload(ssh, directory, remote):
    # Only explicitly staged public inputs/scripts are transferred, never the repository or .env.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for p in sorted((directory / "bundle").rglob("*")):
            if p.is_file():
                archive.add(p, arcname=str(p.relative_to(directory / "bundle")))
    ssh.call(["mkdir", "-p", remote])
    check = 'from pathlib import Path; import sys; p=Path(sys.argv[1]); print((p/"manifest.json").read_text() if (p/"manifest.json").exists() else "")'
    existing = ssh.text(["python3", "-c", check, remote]).strip()
    if existing:
        if json.loads(existing) != read_json(directory / "bundle/manifest.json"):
            raise ValueError("Remote run identity differs; refusing overwrite")
        return
    ssh.call(["tar", "-xzf", "-", "-C", remote], input=buffer.getvalue())


def unpack_snapshot(archive_path, destination):
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError("Unexpected archive link or device")
            if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError("Unsafe archive member")
        archive.extractall(destination, filter="data")
    checksums = read_json(destination / "checksums.json")
    for name, expected in checksums.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe checksum path")
        path = destination / name
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError("Downloaded artifact checksum mismatch: " + name)
    actual = {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()}
    if actual != set(checksums) | {"checksums.json"}:
        raise ValueError("Unlisted files in snapshot")


def collect(ssh, directory, remote):
    state = ssh.status(remote)
    if state.get("alive"):
        raise RuntimeError(
            "Job is still writing results; use status/logs, then fetch after it exits"
        )
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        temp = Path(temp)
        archive = temp / "snapshot.tar.gz"
        with archive.open("wb") as stream:
            ssh.call(["python3", remote + "/remote.py", "snapshot", remote], stdout=stream)
        contents = temp / "contents"
        contents.mkdir()
        unpack_snapshot(archive, contents)
        destination = directory / (
            "download-" + sha256((contents / "checksums.json").read_bytes())[:12]
        )
        if not destination.exists():
            shutil.move(contents, destination)
        atomic_json(directory / "download.json", {"directory": str(destination), "state": state})
    return destination


def import_results(directory, downloaded):
    manifest = read_json(directory / "bundle/manifest.json")
    targets = [Target.model_validate(t) for t in manifest["targets"]]
    deployment = read_json(downloaded / "deployment.json")
    version = "nf-core/proteinfold 2.0.0; deployment " + sha256(
        json.dumps(deployment, sort_keys=True)
    )
    models = {}
    modes = manifest["params"]["mode"].split(",")
    for mode in ("colabfold", "esmfold"):
        if mode in modes:
            models[mode] = ModelSpec(
                provider="nfcore_import",
                version=version,
                samples=list("12345") if mode == "colabfold" else ["1"],
                settings=manifest["params"],
            )
    old = (
        load_run(ROOT / "runs/panel-openfold2")
        if (ROOT / "runs/panel-openfold2/run.json").exists()
        else None
    )
    if old:
        models["esmfold2"] = old.protocol.models["esmfold2"]
    protocol = Protocol(protocol_version="3-nfcore-runpod-" + manifest["run_id"], models=models)
    run = initialize_run(targets, directory / "harmonize", protocol)
    errors = []
    # Inspect individual targets so a malformed target cannot hide other valid successes.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "harmonize_remote_validation", ROOT / "deployment/runpod/remote.py"
    )
    remote_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote_module)
    for target in targets:
        for mode in set(modes) & {"colabfold", "esmfold"}:
            found = False
            for stage in ("smoke", "panel"):
                out = downloaded / stage / "results"
                try:
                    records = remote_module.inspect_outputs(
                        out,
                        [{"id": target.id, "sequence": target.sequence}],
                        [mode],
                        allow_partial=True,
                    )
                except (ValueError, KeyError, FileNotFoundError) as exc:
                    if (out / mode / target.id).exists() or (
                        out / mode / "top_ranked_structures" / (target.id + ".pdb")
                    ).exists():
                        errors.append(f"{target.id}/{mode}: {exc}")
                    continue
                for rec in records:
                    sample = rec["sample"]
                    if any(
                        a.target_id == target.id
                        and a.model == mode
                        and a.sample_id == sample
                        and a.status == "complete"
                        for a in run.artifacts
                    ):
                        continue
                    pdb = out / rec["pdb"]
                    score = read_json(out / rec["scores"]) if rec.get("scores") else {}
                    artifact = _record(target, mode, models[mode], sample)
                    output = ProviderOutput(
                        pdb.read_text(),
                        confidence_scale="0-100",
                        confidence_source="ColabFold JSON pLDDT"
                        if mode == "colabfold"
                        else "ESMFold v1 PDB pLDDT",
                        plddt=score.get("plddt"),
                        pae=score.get("pae"),
                        provenance={
                            "pipeline": manifest["pins"],
                            "deployment": deployment,
                            "source_file": str(pdb),
                            "source_sha256": sha256(pdb.read_bytes()),
                            "raw_directory": str(out / mode),
                            "checkpoint": sample,
                            "seed": 0 if mode == "colabfold" else None,
                            "variation": "parameter_set" if mode == "colabfold" else "singleton",
                            "template_policy": "ColabFold public-server selection; reference overlap possible; not held out",
                        },
                    )
                    try:
                        store_output(run, target, artifact, output)
                    except ValueError as exc:
                        artifact.status = "failed"
                        artifact.error = str(exc)
                        _persist(run, artifact)
                        errors.append(f"{target.id}/{mode}/{sample}: {exc}")
                found = True
                break
            if not found:
                errors.append(f"{target.id}/{mode}: no fully validated output set")
    if old:
        for artifact in old.artifacts:
            if artifact.model != "esmfold2" or artifact.status != "complete":
                continue
            matches = [
                t
                for t in targets
                if t.id == artifact.target_id and t.sequence_sha256 == artifact.sequence_sha256
            ]
            if not matches or any(a.cache_key == artifact.cache_key for a in run.artifacts):
                continue
            source = old.directory / "cache" / artifact.cache_key
            dest = run.directory / "cache" / artifact.cache_key
            shutil.copytree(source, dest, dirs_exist_ok=True)
            copied = artifact.model_copy(deep=True)
            copied.provenance["carried_from_run"] = str(old.directory)
            _persist(run, copied)
    for target in targets:
        for model, model_spec in models.items():
            for sample in model_spec.samples:
                if not any(
                    a.target_id == target.id and a.model == model and a.sample_id == sample
                    for a in run.artifacts
                ):
                    _persist(
                        run,
                        _record(
                            target,
                            model,
                            model_spec,
                            sample,
                            error="Not collected; ESMFold2 remains a separate provider"
                            if model == "esmfold2"
                            else "Missing or invalid nf-core output",
                        ),
                    )
    atomic_json(
        directory / "import.json",
        {
            "at": now(),
            "run": str(run.directory),
            "errors": errors,
            "available": sum(a.status == "complete" for a in run.artifacts),
        },
    )
    return run, errors


def report(run, destination):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    destination.mkdir(exist_ok=True, parents=True)
    label = (
        "VALIDATION FIXTURES — NOT MODEL PREDICTIONS"
        if run.kind == "validation_fixture"
        else "nf-core predictions"
    )
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib"))
    os.environ.setdefault("IPYTHONDIR", str(ROOT / ".tools/ipython"))
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                f"# Harmonize — {label}\nMissing samples remain unavailable. Reference comparisons are contextual, not a held-out benchmark."
            ),
            nbformat.v4.new_code_cell(
                "from pathlib import Path\nfrom IPython.display import display\nimport matplotlib.pyplot as plt\nfrom harmonize import load_run, compare_run\nfrom harmonize.workflow import status_table\nfrom harmonize.plots import plot_overview, plot_distributions, plot_residues, interactive_viewer\n"
                + f"run = load_run({str(run.directory)!r})\ndisplay(status_table(run))\nanalysis = compare_run(run, executable=Path({str(ROOT / '.tools/usalign/USalign')!r}))\ndisplay(analysis.summary)"
            ),
            nbformat.v4.new_code_cell(
                'display(plot_overview(analysis, "tm_score"))\nfor target in run.targets:\n    if any(a.target_id == target.id and a.status == "complete" for a in run.artifacts):\n        display(plot_residues(analysis, target.id))\n        display(plot_distributions(analysis, target.id))\nplt.close("all")'
            ),
        ]
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "from IPython.display import HTML\nfrom harmonize.plots import structure_view\n"
            + f"viewer_js = Path({str(ROOT / 'assets/3Dmol-min.js')!r}).read_text()\n"
            + 'display(HTML("<script>" + viewer_js + "</script>"))\ndisplay(interactive_viewer(analysis))\n'
            + "for target in run.targets:\n"
            + '    if any(a.target_id == target.id and a.status == "complete" for a in run.artifacts):\n'
            + "        overlay = structure_view(analysis, target.id)\n"
            + '        (analysis.directory / f"overlay_{target.id}.html").write_text("<!doctype html><script>" + viewer_js + "</script>" + overlay._make_html())'
        )
    )
    NotebookClient(
        notebook, timeout=600, kernel_name="harmonize", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    nbformat.write(notebook, destination / "prediction.ipynb")
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (destination / "prediction.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "resume", "status", "logs", "fetch"])
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--identity")
    parser.add_argument("--run-id")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--fasta")
    inputs.add_argument("--samplesheet")
    parser.add_argument("--params")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.action != "run" and any((args.fasta, args.samplesheet, args.params)):
        parser.error("Input changes require a new run")
    if args.action == "run":
        directory = prepare(input_targets(args.fasta, args.samplesheet), parameters(args.params))
    else:
        if not args.run_id or not re.fullmatch(r"proteinfold-[0-9a-f]{16}", args.run_id):
            parser.error("Provide --run-id proteinfold-... from the launch output")
        directory = ROOT / "runs/runpod" / args.run_id
    launch = read_json(directory / "launch.json")
    saved = (
        read_json(directory / "connection.json") if (directory / "connection.json").exists() else {}
    )
    host = args.host or saved.get("host") or input("RunPod full SSH user@host: ").strip()
    port = args.port or saved.get("port") or int(input("SSH port [22]: ").strip() or "22")
    identity = args.identity or saved.get("identity")
    if not identity and not os.getenv("SSH_AUTH_SOCK"):
        identity = input("Private key path (stays on this computer): ").strip()
        if not identity:
            parser.error("Provide an SSH key path or load the key into ssh-agent")
    ssh = SSH(host, port, identity)
    atomic_json(directory / "connection.json", {"host": host, "port": port, "identity": identity})
    (directory / "connection.json").chmod(0o600)
    remote = launch["remote"]
    print("Run ID:", launch["run_id"], flush=True)
    if args.action == "status":
        print(json.dumps(ssh.status(remote), indent=2))
        return
    if args.action == "logs":
        ssh.call(["tail", "-n", "100", "-f", remote + "/runner.log"])
        return
    if args.action in ("run", "resume"):
        upload(ssh, directory, remote)
        command = ["python3", remote + "/remote.py", "start", remote]
        if args.action == "resume":
            command.append("--resume")
        print(ssh.text(command))
        if args.detach:
            print("Remote job continues. Use status/logs/fetch with --run-id", launch["run_id"])
            return
        offset = 0
        while True:
            state = ssh.status(remote)
            log = ssh.text(["tail", "-c", "+" + str(offset + 1), remote + "/runner.log"])
            print(log, end="", flush=True)
            offset += len(log.encode())
            if not state.get("alive"):
                break
            time.sleep(10)
    downloaded = collect(ssh, directory, remote)
    errors = []
    if (downloaded / "deployment.json").exists():
        run, errors = import_results(directory, downloaded)
        if any(a.status == "complete" for a in run.artifacts):
            report(run, ROOT / "reports/runpod" / launch["run_id"])
        print("Import:", directory / "import.json")
        if errors:
            print("\n".join(errors))
    state = read_json(directory / "download.json")["state"]
    print("Remote state:", state)
    print(
        "RunPod GPU billing continues until you stop the pod. No automatic termination was performed."
    )
    if state["status"] != "complete" or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
