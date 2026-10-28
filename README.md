# Project Harmonize

Compare structural predictions from OpenFold2 (AlphaFold2 weights), ESMFold, and standard ESMFold2 for
gpD, RamR, FNIII-10, PDZ3, RNase A, and DHFR. The notebook shows global and local
agreement, within-model variability, confidence, and separate experimental-reference comparisons.

**Current state: offline-ready. No live predictions have been submitted.** The executed
validation report uses explicitly labeled synthetic perturbations of experimental coordinates.
The real panel remains incomplete until its 66 requested structures are available.

## Open the notebook

The environment and `Python (Harmonize)` kernel are installed locally in `.venv`.

```bash
uv run --group notebook jupyter lab notebooks/01_concordance.ipynb
```

The notebook defaults to `MODE = "validation"`, `RUN_LIVE = False`. Validation group names
are intentionally different from model names. Set `MODE = "prediction"` to inspect only
genuine cached or imported predictions; this does not itself submit inference.

Executed copies and browser-readable reports are in `reports/validation.ipynb`,
`reports/validation.html`, `reports/prediction.ipynb`, and `reports/prediction.html`.
These generated files are ignored by Git. Interactive sample controls require Jupyter;
standalone structure overlays are exported into each run's `analysis/` folder.

## Reproduce setup

Install `uv` and a C++ compiler, then run:

```bash
make setup
make check
make validate
```

Python is pinned to **3.12.12**, and `uv.lock` locks all Python dependencies including
the optional Biohub SDK. `setup` creates a project environment, builds checksum-verified
US-align **20260819**, registers the kernel inside that environment, and installs local
pre-commit checks. The existing system US-align is not used.

The shipped 3Dmol 2.5.5 JavaScript bundle permits offline structure rendering. Its source
URL, checksum, and license are in `assets/`. Initial dependency/native-tool setup needs
network access; analysis and both notebook modes with `RUN_LIVE = False` need no inference API.

See [environment details](docs/environment.md) for isolated cache settings and native-tool provenance.

## Frozen constructs

| Target | PDB chain | Input length | Eligible reference Cα positions |
|---|---|---:|---:|
| gpD | [1C5E:A](https://www.rcsb.org/structure/1C5E) | 95 | 95 |
| RamR | [3VVX:A](https://www.rcsb.org/structure/3VVX) | 194 | 191 |
| FNIII-10 | [1FNA:A](https://www.rcsb.org/structure/1FNA) | 91 | 91 |
| PDZ3 | [3K82:A](https://www.rcsb.org/structure/3K82) | 98 | 97 |
| RNase A | [7RSA:A](https://www.rcsb.org/structure/7RSA) | 124 | 124 |
| DHFR | [1RF7:A](https://www.rcsb.org/structure/1RF7) | 159 | 159 |

Deposited polymer boundaries are preserved, including RamR expression-tag residues and
DHFR Asp37. PDZ3 input position 28 is restored to native **Asp332**; the experimental SNN
residue at this position is excluded from reference metrics. The [PDB paper summary](https://www.rcsb.org/structure/3K82)
describes this modification and its potential conformational effects.

`inputs/targets.fasta` contains the exact prediction inputs. `inputs/sources.lock.json`
records downloaded mmCIF and sequence hashes. `inputs/references.json` is a portable,
frozen experimental Cα snapshot with residue mappings, allowing offline use from a fresh clone.
`prepare_targets()` downloads/caches the full source mmCIFs and rejects changed frozen inputs.

## Collect real predictions later

1. Copy `.env.example` to `.env` and set `NVIDIA_API_KEY` and `BIOHUB_API_TOKEN` locally.
2. Install the optional client with `uv sync --frozen --group notebook --extra biohub`.
3. Check readiness without submitting jobs: `uv run python scripts/predict.py`.
4. Smoke-test each provider on one target before the full panel:

```bash
uv run --extra biohub python scripts/predict.py --execute --target gpd --model openfold2 --sample 1
uv run --extra biohub python scripts/predict.py --execute --target gpd --model esmfold2 --sample 1
uv run python scripts/predict.py --execute --target gpd --model esmfold
# After checking the smoke results:
uv run --extra biohub python scripts/predict.py --execute
```

Get the NVIDIA key from [OpenFold2](https://build.nvidia.com/openfold/openfold2), and the Biohub
token from its [developer console](https://biohub.ai/developer-console/api-keys). The NVIDIA key
is also used for MSA search. RCSB template search needs no key. Do not use the deprecated AF2 page.

OpenFold2 is requested one checkpoint per submission to preserve unambiguous sample identity.
NVIDIA's model card specifies **AlphaFold 2.3.2 weights**; this is recorded as OpenFold2 with
AF2 weights, rather than independently trained OpenFold weights. Relaxation is disabled.
The first smoke command submits one checkpoint, preceded by one shared MSA/template preparation.
The full run reuses successful results and still requests **66 structures** in total.

This protocol replaces JackHMMer/HHsearch with NVIDIA MMseqs2/ColabFold MSA search and a separate
RCSB template search. Templates remain enabled. All five checkpoints receive the same frozen
inputs; standard presets 1–2 use template features, while 3–5 do not. See the
[OpenFold2 input protocol](docs/openfold2.md) for selection thresholds and provenance.
The new run is `runs/panel-openfold2`; the historical `runs/panel` is preserved, along with
`configs/protocol-alphafold2-legacy.yaml`. Protocols and caches are never silently relabeled.
ESMFold2 uses `esmfold2-2026-05`, 20 loops, 100 steps, and five uncached stochastic samples.
ESMFold has one sample because the legacy API does not expose ensemble controls.

Provider adapters are fixture-tested; credentialed live inference remains unverified.
The legacy ESM Atlas endpoint may be unavailable. Its PDB B-factor values are preserved,
but their scale is left **unknown**, rather than inferred from the observed maximum.
OpenFold2 JSON pLDDT is 0–100; absent JSON confidence leaves structure B-factors unnormalized.
The pinned Biohub SDK's result pLDDT is 0–1 and is converted explicitly.
Missing pLDDT/PAE remains missing.

Completed samples are immutable and checksum-verified. Reruns reuse cache entries. NVIDIA
request IDs are persisted before polling. Timeouts or ambiguous submission failures become
`indeterminate` and are never blindly resubmitted; inspect the provider before creating a
replacement run. Explicitly failed jobs can be retried with `--retry-failed`.
Do not run two writers against the same run directory concurrently.

An existing prediction can be imported without substituting model identity:

```python
from harmonize import load_run, import_prediction

run = load_run("runs/panel-openfold2")
import_prediction(
    run,
    "gpd",
    "esmfold",
    "1",
    "path/to/esmfold_gpd.pdb",
    confidence_scale="0-100",  # Only if established by the producing software.
    provenance={"checkpoint": "esmfold_v1", "source": "describe how it was generated"},
)
```

Import requires the same sequence. Partial predictions must retain input residue numbering;
full-length exact sequence matches may be renumbered safely. Ambiguous chains/insertions fail.

## Analysis outputs and interpretation

`compare_run(load_run("runs/panel-openfold2"))` makes no API calls. It writes pair and per-residue CSVs,
summary quantiles, confidence summaries, medoid selections, native US-align output, and provenance.

- **TM-score:** fixed input-residue correspondence, full input-length normalization for
  prediction pairs. Both optimization directions are evaluated; their maximum removes numerical
  order dependence without changing residue correspondence.
- **Cα RMSD/displacement:** Kabsch fit over matched coordinates, with no confidence filtering.
- **Symmetric Cα lDDT:** union of contacts within 15 Å, using 0.5/1/2/4 Å thresholds.
- **Reference scores:** only observed, unmodified positions; a fixed eligible-reference denominator.

All 11 outputs produce 55 unordered prediction pairs per protein: 20 within-model and 35
cross-model. Experimental comparisons are separate. Coverage accompanies scores; missing
coordinates can inflate matched-only RMSD/lDDT agreement. Exact coordinate duplicates are flagged.
Medoids maximize mean within-model TM-score, with lexical sample ID breaking ties.

Agreement is not proof of correctness. AF2 and the ESM models have different information inputs;
AF2 checkpoint variation and ESMFold2 stochastic variation are not matched experiments. Old PDB
references may overlap model training/templates, and ligands/assembly context can affect structure.
No composite ranking, significance test, or claim of independent pairwise replicates is produced.

## Development

```bash
make check
uv run --group notebook python scripts/build_notebook.py
uv run ruff format notebooks/01_concordance.ipynb
make validate
```

Pre-commit runs Ruff and strips notebook outputs. Offline tests prohibit live HTTP traffic and
cover residue mapping, geometry, missing data, provider responses, cache/recovery, and full-panel
fixture counts. CI installs dependencies, builds the pinned native tool, and executes both safe
notebook modes. No Git remote or publication is configured.
