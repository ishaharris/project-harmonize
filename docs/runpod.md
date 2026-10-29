# Run nf-core/proteinfold on RunPod

This runs the full pinned **nf-core/proteinfold 2.0.0** workflow over SSH. The default
modes are ColabFold and original ESMFold. Biohub ESMFold2 remains separate.
No NVIDIA hosted inference endpoint is used.

## Create the pod

Choose a Linux x86_64 Ubuntu 22.04/24.04 pod with Python 3, root shell access,
one RTX 4090 (24 GB), at least 8 vCPUs/32 GB RAM, and a network volume mounted at
`/workspace`. Allow 150–200 GB persistent storage for both inference images, model
weights, Nextflow work files, and outputs. Keep at least 30 GB container disk.
Larger future proteins may require more GPU memory and storage.

Enable **full SSH with a public IP**, add your SSH public key in RunPod, and copy the
host and mapped SSH port from the Connect panel. The private key stays on your Mac.
Encrypted keys can be loaded into `ssh-agent` first.

**Container compatibility is not assumed.** The launcher checks user namespaces and
Apptainer GPU execution before model-weight downloads. Some RunPod hosts may block
these capabilities even with root inside the pod. If `unshare` or Apptainer returns
`Operation not permitted`, changing the container image alone may not resolve the
host restriction. Use a host that supports unprivileged Apptainer; do not disable
security checks or switch to unsupported Conda execution. This limitation still
requires validation on your actual pod.

## Start

From this repository:

```bash
uv run python scripts/runpod.py run
```

Answer the host, port, and key-path prompts. Alternatively:

```bash
uv run python scripts/runpod.py run --host root@YOUR_IP --port YOUR_PORT --identity ~/.ssh/runpod_key
```

The launcher verifies the pinned Java/Nextflow/Apptainer downloads, installs the
pipeline, tests the GPU inside both inference containers, uploads the frozen panel,
and starts gpD. It validates five ColabFold checkpoints plus one ESMFold prediction,
including templates and residue correspondence, before running the other five proteins.
Templates are enabled; relaxation is disabled; the ColabFold seed is 0. Raw outputs
are retained. The original project environment remains Python 3.12.12; inference
uses the upstream containers' own Python environments.

The terminal displays a `proteinfold-...` run ID. The remote job survives SSH loss.
With an active terminal the launcher fetches results, imports them into a new Harmonize
run, and executes an offline comparison notebook automatically. Add `--detach` to
return immediately, then use `fetch` after completion.

```bash
uv run python scripts/runpod.py status --run-id proteinfold-YOUR_ID
uv run python scripts/runpod.py logs --run-id proteinfold-YOUR_ID
uv run python scripts/runpod.py resume --run-id proteinfold-YOUR_ID
uv run python scripts/runpod.py fetch --run-id proteinfold-YOUR_ID
```

Connection settings are saved in the ignored local run directory. Supply `--host`
and `--port` again if moving the persistent volume to another pod. Keep SSH host-key
verification enabled and verify any changed host key through your normal SSH workflow.

`resume` checks existing processes before starting anything and uses Nextflow's
cache. Do not delete the remote run's `.nextflow` or `work` directories while you
need resume support. A completed remote job is not relaunched. Downloads are staged
and checksum-verified before import. Fetching while the job is writing is refused;
failed/interrupted runs can be fetched to retain available successes.

## More proteins and models

```bash
uv run python scripts/runpod.py run --fasta /path/to/proteins.fasta
uv run python scripts/runpod.py run --samplesheet /path/to/samplesheet.csv --params /path/to/params.yaml
```

Each input is a monomer. FASTA identifiers must be unique letters, numbers, `_`, or
`-`. Samplesheets have `id,fasta` columns and one sequence per local FASTA. Relative
paths resolve against the samplesheet directory. Multimers and RNA are not supported
by this launcher's input adapter.

A parameter file may select other proteinfold modes and their **existing remote**
mode-specific database paths, such as `alphafold2_db`. Additional modes run through
the upstream workflow; automatic scientific import currently covers ColabFold and
ESMFold only. Their other raw outputs and nf-core reports remain downloadable.
The launcher never automatically downloads the large sequence databases for extra
modes. Initial validated settings require GPU execution, public MSA access, templates,
unrelaxed `alphafold2_ptm`, and retained intermediates. Configuration changes receive
a new run ID rather than overwriting existing results.

Use the public MSA server for small batches with serial requests. For large-scale
production, supply `msa_server_url` pointing to your own MMseqs2 API server rather
than overloading the public service.

## Outputs and billing

- Remote: `/workspace/harmonize` contains tools, image/weight caches, and run directories.
- Local: `runs/runpod/<ID>/` contains launch records, verified downloads, imports, and
  the separate `harmonize/` analysis run.
- Reports: `reports/runpod/<ID>/prediction.ipynb` and `.html`; metrics, figures, and
  structure overlays are in the analysis run directory.

Custom sequences have no experimental reference unless they exactly match a frozen
panel identifier and sequence. They receive prediction comparisons without invented
reference metrics. Cached Biohub results are carried forward only for matching
sequence identities; missing ESMFold2 samples stay explicitly missing.

**Stop the pod yourself when finished.** This tool neither creates nor terminates
pods and cannot stop RunPod billing. A network volume survives termination; ordinary
pod volume storage does not. Keep downloaded results before deleting any storage.

## Validation status

Offline tests exercise orchestration and scientific import using explicitly synthetic
validation fixtures. Live acceptance is pending an actual RunPod SSH connection:
container/GPU preflight, six gpD outputs, reconnect/resume, and the full 36-output panel
must all succeed before this deployment is considered live-validated.
