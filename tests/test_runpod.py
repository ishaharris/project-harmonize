"""Offline orchestration tests. Coordinates below are validation fixtures, not predictions."""

import importlib.util
import io
import json
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

from harmonize import runpod
from harmonize.storage import atomic_json, sha256
from harmonize.structures import reference_arrays, write_ca_pdb

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "runpod_remote_test", ROOT / "deployment/runpod/remote.py"
)
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


def outputs(path, target):
    """Test-only copies of experimental CA coordinates with synthetic confidence."""
    raw = path / "colabfold" / target.id / "raw"
    raw.mkdir(parents=True)
    (raw / "hits.m8").write_text("query\t1c5e_A\t100\n")
    (raw / "1c5e.cif").write_text("data_validation_fixture\n")
    for model in range(1, 6):
        pdb = (
            raw
            / f"{target.id}_unrelaxed_rank_{6 - model:03d}_alphafold2_ptm_model_{model}_seed_000.pdb"
        )
        write_ca_pdb(
            pdb,
            reference_arrays(target)["xyz"],
            np.arange(1, len(target.sequence) + 1),
            target.sequence,
        )
        score = pdb.with_name(pdb.name.replace("_unrelaxed_", "_scores_").replace(".pdb", ".json"))
        atomic_json(
            score,
            {
                "plddt": [81.0] * len(target.sequence),
                "pae": np.zeros((len(target.sequence), len(target.sequence))).tolist(),
            },
        )
    esm = path / "esmfold/top_ranked_structures" / (target.id + ".pdb")
    esm.parent.mkdir(parents=True)
    shutil.copyfile(pdb, esm)
    return raw


def test_ssh_arguments_are_not_shell_evaluated(tmp_path):
    key = tmp_path / "key with spaces"
    key.touch()
    ssh = runpod.SSH("root@example.org", 2222, str(key))
    command = ssh.command(["echo", "$(do-not-execute); spaces"])
    assert str(key) in command
    assert shlex.split(command[-1]) == ["echo", "$(do-not-execute); spaces"]
    assert "ForwardAgent=no" in command
    assert not any("StrictHostKeyChecking=no" in x for x in command)
    for host in ("-oProxyCommand=evil", "root@host;bad", "root@host $(bad)"):
        with pytest.raises(ValueError):
            runpod.SSH(host, 22)
    with pytest.raises(ValueError):
        runpod.SSH("example.org", 0)


def test_input_identity_and_no_invented_reference(tmp_path, targets):
    fasta = tmp_path / "sequences with spaces.fa"
    fasta.write_text(f">gpd\n{targets[0].sequence}\n>new\nACDEFGHIK\n")
    records = runpod.input_targets(fasta=fasta)
    assert records[0] == targets[0]
    assert not any(r.reference_eligible for r in records[1].residues)
    fasta.write_text(">x\nACD\n>x\nACD\n")
    with pytest.raises(ValueError, match="Duplicate"):
        runpod.input_targets(fasta=fasta)


def test_protocol_changes_make_new_run(tmp_path, targets, monkeypatch):
    shutil.copytree(ROOT / "deployment", tmp_path / "deployment")
    monkeypatch.setattr(runpod, "ROOT", tmp_path)
    one = runpod.prepare(targets[:1], runpod.DEFAULT_PARAMS)
    assert one == runpod.prepare(targets[:1], runpod.DEFAULT_PARAMS)
    two = runpod.prepare(targets[:1], {**runpod.DEFAULT_PARAMS, "colabfold_num_recycles": 4})
    assert one != two
    assert not list((one / "bundle").rglob(".env"))


def test_parameters_refuse_silent_protocol_change(tmp_path):
    path = tmp_path / "params.json"
    for settings in (
        {"colabfold_use_templates": False},
        {"colabfold_use_amber": True},
        {"mode": "alphafold3"},
    ):
        atomic_json(path, settings)
        with pytest.raises(ValueError):
            runpod.parameters(path)


def test_checkpoint_not_rank_and_smoke_completeness(tmp_path, targets):
    t = targets[0]
    raw = outputs(tmp_path, t)
    records = remote.inspect_outputs(tmp_path, [t.model_dump()], ["colabfold", "esmfold"])
    assert len(records) == 6
    model1 = next(r for r in records if r["model"] == "colabfold" and r["sample"] == "1")
    assert "rank_005" in model1["pdb"]
    (tmp_path / model1["pdb"]).unlink()
    with pytest.raises(ValueError, match="five checkpoints"):
        remote.inspect_outputs(tmp_path, [t.model_dump()], ["colabfold"])
    assert (
        len(remote.inspect_outputs(tmp_path, [t.model_dump()], ["colabfold"], allow_partial=True))
        == 4
    )
    (raw / "hits.m8").write_text("")
    with pytest.raises(ValueError, match="template"):
        remote.inspect_outputs(tmp_path, [t.model_dump()], ["colabfold"], allow_partial=True)


def test_sequence_mismatch_is_not_accepted(tmp_path, targets):
    t = targets[0]
    outputs(tmp_path, t)
    changed = {**t.model_dump(), "sequence": "A" * len(t.sequence)}
    with pytest.raises(ValueError, match="sequence mismatch"):
        remote.inspect_outputs(tmp_path, [changed], ["esmfold"])


def test_import_keeps_pae_and_raw_confidence(tmp_path, targets, monkeypatch):
    shutil.copytree(ROOT / "deployment", tmp_path / "deployment")
    monkeypatch.setattr(runpod, "ROOT", tmp_path)
    directory = runpod.prepare(targets[:1], runpod.DEFAULT_PARAMS)
    downloaded = tmp_path / "download"
    outputs(downloaded / "smoke/results", targets[0])
    atomic_json(downloaded / "deployment.json", {"images": "validation-fixture-only"})
    run, errors = runpod.import_results(directory, downloaded)
    assert not errors
    assert len(run.artifacts) == 6
    first = next(a for a in run.artifacts if a.model == "colabfold")
    with np.load(run.directory / first.normalized_file) as data:
        assert data["pae"].shape == (95, 95)
        assert np.all(data["confidence"] == 81)
    original_hashes = [a.structure_sha256 for a in run.artifacts]
    run, _ = runpod.import_results(directory, downloaded)
    assert [a.structure_sha256 for a in run.artifacts] == original_hashes
    assert targets[0].residues[0].ca is not None  # Reference records remain untouched.


def make_archive(path, payload, checksum=None):
    with tarfile.open(path, "w:gz") as archive:
        files = {
            "result.txt": payload,
            "checksums.json": json.dumps({"result.txt": checksum or sha256(payload)}).encode(),
        }
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def test_transfer_checksums_and_truncated_archive(tmp_path):
    archive = tmp_path / "download.tar.gz"
    make_archive(archive, b"result")
    runpod.unpack_snapshot(archive, tmp_path / "valid")
    make_archive(archive, b"changed", sha256(b"original"))
    with pytest.raises(ValueError, match="checksum"):
        runpod.unpack_snapshot(archive, tmp_path / "invalid")
    archive.write_bytes(b"partial")
    with pytest.raises(tarfile.ReadError):
        runpod.unpack_snapshot(archive, tmp_path / "truncated")


def test_no_fetch_while_remote_writes(tmp_path):
    class LiveSSH:
        def status(self, _):
            return {"status": "panel", "alive": True}

    with pytest.raises(RuntimeError, match="still writing"):
        runpod.collect(LiveSSH(), tmp_path, "/workspace/job")


def test_namespace_failure_is_persisted_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    atomic_json(tmp_path / "manifest.json", {"run_id": "test"})

    def blocked(_):
        raise RuntimeError("unshare: Operation not permitted")

    monkeypatch.setattr(remote, "bootstrap", blocked)
    with pytest.raises(RuntimeError, match="unshare"):
        remote.execute(tmp_path, False)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "failed"
    assert "Operation not permitted" in state["error"]


def test_dead_job_status_is_interrupted(tmp_path):
    atomic_json(tmp_path / "state.json", {"status": "panel", "pid": 999999999})
    value = subprocess.check_output(
        ["python3", str(ROOT / "deployment/runpod/remote.py"), "status", str(tmp_path)], text=True
    )
    assert json.loads(value)["status"] == "interrupted"


def test_completed_job_is_not_relaunched(tmp_path):
    atomic_json(tmp_path / "state.json", {"status": "complete"})
    value = subprocess.check_output(
        ["python3", str(ROOT / "deployment/runpod/remote.py"), "start", str(tmp_path)], text=True
    )
    assert json.loads(value)["status"] == "complete"
    assert not (tmp_path / "runner.log").exists()


def test_partial_weight_download_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    params = tmp_path / "databases/esmfold/params"
    params.mkdir(parents=True)
    (params / "esmfold_3B_v1.pt").touch()
    assert remote.cached_weights("esmfold") == {}
    for name in ("esm2_t36_3B_UR50D.pt", "esm2_t36_3B_UR50D-contact-regression.pt"):
        (params / name).touch()
    assert remote.cached_weights("esmfold")["esmfold_db"] == str(params.parent)


def test_missing_reference_has_no_experimental_comparisons(tmp_path, targets, native, monkeypatch):
    from harmonize.analysis import compare_run

    shutil.copytree(ROOT / "deployment", tmp_path / "deployment")
    monkeypatch.setattr(runpod, "ROOT", tmp_path)
    target = targets[0].model_copy(deep=True)
    target.pdb_id = ""
    for residue in target.residues:
        residue.reference_eligible = False
    directory = runpod.prepare([target], runpod.DEFAULT_PARAMS)
    downloaded = tmp_path / "download"
    outputs(downloaded / "smoke/results", targets[0])
    atomic_json(downloaded / "deployment.json", {"images": "validation-fixture-only"})
    run, errors = runpod.import_results(directory, downloaded)
    assert not errors
    analysis = compare_run(run, executable=native)
    assert len(analysis.pairs) == 15
    assert not (analysis.pairs.comparison_type == "experimental_reference").any()


def test_unsafe_archive_member_refused(tmp_path):
    path = tmp_path / "unsafe.tar"
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        runpod.unpack_snapshot(path, tmp_path / "output")


def test_resume_keeps_nextflow_inputs_and_params_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    monkeypatch.setattr(remote, "bootstrap", lambda _: tmp_path / "pipeline")
    monkeypatch.setattr(remote, "resource_limits", lambda: (8, 32))
    monkeypatch.setattr(remote, "inspect_outputs", lambda *args: [])
    atomic_json(
        tmp_path / "manifest.json",
        {
            "run_id": "test",
            "targets": [{"id": "gpd", "sequence": "ACD"}],
            "params": runpod.DEFAULT_PARAMS,
        },
    )
    commands = []

    class Completed:
        pid = 9999999

        def __init__(self, args, cwd):
            commands.append(args)
            (cwd / ".nextflow/cache/session-id").mkdir(parents=True, exist_ok=True)

        def wait(self):
            return 0

    monkeypatch.setattr(remote.subprocess, "Popen", Completed)
    remote.execute(tmp_path, False)
    sample = tmp_path / "smoke/samplesheet.csv"
    params = tmp_path / "smoke/params.json"
    before = sample.stat().st_mtime_ns, params.stat().st_mtime_ns, params.read_bytes()
    monkeypatch.setattr(remote, "cached_weights", lambda _: {"esmfold_db": "/new-cache"})
    remote.execute(tmp_path, True)
    assert before == (sample.stat().st_mtime_ns, params.stat().st_mtime_ns, params.read_bytes())
    assert "-resume" not in commands[0] and "-resume" in commands[1]
    assert json.loads((tmp_path / "state.json").read_text())["nextflow_sessions"]["smoke"] == [
        "session-id"
    ]
