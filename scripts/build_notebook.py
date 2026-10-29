"""Generate the clean, tracked notebook. Executed copies belong in reports/."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text, **metadata):
    compile(text, "<notebook cell>", "exec")
    cells.append(nbf.v4.new_code_cell(text, metadata=metadata))


markdown("""# Project Harmonize · structural concordance

Compare OpenFold2 (AlphaFold2 weights), ESMFold, and ESMFold2 on six frozen PDB-derived constructs.

**The default mode is offline validation.** Its structures are perturbed experimental coordinates,
with synthetic confidence values, and are **not model predictions**. Real prediction slots are
tracked separately. Switching to `prediction` displays only genuine cached/imported predictions.
No inference is submitted unless `RUN_LIVE = True` in prediction mode.

The experimental structures provide context; this is not a held-out benchmark of accuracy.
""")
code(
    'MODE = "validation"  # "validation" or "prediction"\nRUN_LIVE = False\nTARGET_ID = "pdz3"',
    tags=["parameters"],
)
code("""import os
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs/targets.yaml").exists())
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".tools/cache"))

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Markdown, display

from harmonize import compare_run, run_predictions
from harmonize.fixtures import build_validation_run
from harmonize.openfold import preparation_status
from harmonize.plots import interactive_viewer, plot_distributions, plot_overview, plot_residues, structure_view
from harmonize.targets import load_reference_snapshot
from harmonize.workflow import artifact_arrays, status_table

if MODE not in {"validation", "prediction"} or (MODE == "validation" and RUN_LIVE):
    raise ValueError("Select a valid mode; live inference is allowed only in prediction mode")
""")
markdown("""## Frozen inputs

Sequences come from deposited polymer records, including unresolved residues. RamR keeps its
expression-tag residues; DHFR keeps deposited Asp37. PDZ3 input position 28 is explicitly restored
to native Asp332. Its experimental succinimide is excluded from reference scores.

`inputs/targets.fasta`, `inputs/sources.lock.json`, and the frozen reference coordinate snapshot
make this notebook runnable without downloading structures or calling inference services.
""")
code("""targets = load_reference_snapshot(ROOT)
display(pd.DataFrame([{
    "protein": t.name, "reference": f"{t.pdb_id}:{t.chain}", "input_length": len(t.sequence),
    "eligible_reference_residues": sum(r.reference_eligible for r in t.residues),
    "ligands": ", ".join(t.ligands), "notes": t.notes,
} for t in targets]))
display(pd.DataFrame([{"protein": t.name, "sequence": t.sequence, "sha256": t.sequence_sha256} for t in targets]))
""")
markdown("""## Real prediction readiness and collection

The protocol requests 5 OpenFold2/AF2 parameter sets, 5 fresh standard ESMFold2 samples, and 1 ESMFold
prediction per protein. NVIDIA's OpenFold2 uses AlphaFold2 weights. Its inputs are prepared by
NVIDIA MMseqs2/ColabFold MSA search and a separate RCSB experimental-template search. This
replaces the deprecated AF2 endpoint and the original JackHMMer/HHsearch workflow. Templates
remain enabled; standard parameter sets 1–2 use template features, while 3–5 are template-free.
The ESM models use sequence alone. Checkpoint variation differs from ESMFold2 sampling variation.

**Live smoke status (2026-10-29):** gpD has validated ESMFold and standard ESMFold2 predictions.
NVIDIA MSA and template preparation now succeed after fixing the worker wait. OpenFold2
inference returns HTTP 500 with a CUDA illegal-memory-access error, reproduced on one retry.
No OpenFold2 structure is available; earlier attempts and raw server errors are preserved.
The pinned Biohub SDK and
current endpoint schema do not expose the cache-bypass control assumed in the original plan;
ESMFold2 samples 2–5 are blocked pending verification, and sample 1 is not labeled verified fresh.

Up to four template chains pass the documented selection policy; source structures, chain
mapping, scores, and checksums are retained. Exact reference entries are allowed and flagged.
The table below is an offline view of preparation status. No missing template is silently omitted.

With `RUN_LIVE = False`, this cell only prepares or loads the manifest. Credentials stay in `.env`.
Indeterminate submissions are retained for inspection; saved NVIDIA job IDs resume polling.
""")
code("""panel = run_predictions(targets, ROOT / "runs/panel-openfold2", ROOT / "configs/protocol.yaml", execute=RUN_LIVE)
panel_status = status_table(panel)
display(panel_status.groupby(["model", "status"], dropna=False).size().rename("samples").reset_index())
display(panel_status[["model", "status", "detail"]].drop_duplicates())
display(pd.DataFrame(preparation_status(panel)))
display(Markdown(f"**Real prediction run: {'complete' if panel.complete else 'INCOMPLETE'}** · "
                 f"{sum(a.status == 'complete' for a in panel.artifacts)}/{panel.expected_count} structures available."))
""")
markdown("""## Offline concordance analysis

TM-score uses fixed input-residue correspondence and full input length; it never searches for a
different residue alignment. RMSD and displacement use Kabsch superposition. Symmetric Cα lDDT
uses contacts within 15 Å in either structure and 0.5/1/2/4 Å distance thresholds.

Missing coordinates reduce reported coverage. Low-confidence residues remain in primary results.
Pairwise distributions are descriptive: pairs sharing a prediction are not independent replicates.
""")
code("""run = build_validation_run(targets, ROOT / "runs/validation") if MODE == "validation" else panel
if MODE == "validation":
    display(Markdown("### VALIDATION FIXTURES — NOT MODEL PREDICTIONS\\n"
                     "The following figures test geometry, data flow, and presentation only. "
                     "Groups A/B/singleton are synthetic; their values say nothing about AF2 or ESM performance."))
analysis = compare_run(run, executable=ROOT / ".tools/usalign/USalign")
display(analysis.pairs.groupby(["comparison_type", "status"]).size().rename("pairs").reset_index())
display(analysis.summary)
""")
code("""for metric in ("tm_score", "lddt"):
    figure = plot_overview(analysis, metric)
    display(figure)
    plt.close(figure)
""")
markdown("""## Local disagreement and within-model variation

Select `TARGET_ID` in the parameter cell to change the detailed example. The local plots show
median disagreement and interquartile ranges across cross-model pairs. Confidence lines belong
to ensemble medoids. Unknown confidence scales are left unnormalized and remain available in
the raw cache; experimental B-factors are never presented as pLDDT.
""")
code("""for figure in (plot_distributions(analysis, TARGET_ID), plot_residues(analysis, TARGET_ID)):
    display(figure)
    plt.close(figure)
display(analysis.confidence[analysis.confidence.target_id == TARGET_ID])
""")
markdown("""## Inspect structures

Choose proteins and individual samples below. Defaults are within-model medoids, with lexical
sample ID breaking ties. All traces use the same input residue mapping and are fitted onto the
first selected structure. Gaps in the Cα trace are preserved. The reference can be toggled.

Interactive controls require a running Jupyter kernel. A standalone overlay is also exported for
the selected target; the local 3Dmol bundle supplies browser rendering without inference calls.
""")
code("""viewer_bundle = ROOT / "assets/3Dmol-min.js"
if viewer_bundle.exists():
    display(HTML("<script>" + viewer_bundle.read_text() + "</script>"))
display(interactive_viewer(analysis))
overlay = structure_view(analysis, TARGET_ID)
overlay_html = overlay._make_html()
banner = "VALIDATION FIXTURES — NOT MODEL PREDICTIONS" if MODE == "validation" else "Cached model predictions"
script = "<script>" + viewer_bundle.read_text() + "</script>" if viewer_bundle.exists() else ""
(analysis.directory / f"overlay_{TARGET_ID}.html").write_text(
    "<!doctype html><html><head><meta charset='utf-8'></head><body>"
    + f"<h2>{banner}</h2>" + script + overlay_html + "</body></html>"
)
""")
markdown("""## Experimental-reference context

These comparisons are separate from model concordance. Their TM-score denominator is the
fixed number of eligible observed reference residues. Modified/unresolved reference residues
are excluded. Bound ligands, oligomerization, and the PDZ3 modification can affect conformation;
agreement with these old structures is not evidence of generalization to unseen targets.
""")
code("""reference_pairs = analysis.pairs[analysis.pairs.comparison_type == "experimental_reference"]
display(reference_pairs[["target_id", "left", "n_matched", "normalization_length", "coverage", "tm_score", "rmsd", "lddt", "status"]])
figure = plot_distributions(analysis, TARGET_ID, reference=True)
display(figure)
plt.close(figure)
""")
markdown("""## Available PAE and exports

PAE is retained when supplied. Missing PAE is not synthesized. Exported CSVs contain raw pairwise
and per-residue values as well as descriptive summaries. All six proteins receive PNG/SVG figures.
The run and analysis manifests record versions, hashes, parameters, availability, and provenance.
""")
code("""pae_available = False
for artifact in run.artifacts:
    if artifact.target_id != TARGET_ID or artifact.status != "complete":
        continue
    data = artifact_arrays(run, artifact)
    if "pae" in data:
        pae_available = True
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(data["pae"], origin="lower", cmap="viridis")
        fig.colorbar(image, ax=ax, label="Predicted aligned error (Å)")
        ax.set_title(f"{TARGET_ID} · {artifact.label}")
        ax.set_xlabel("Input residue index (0-based)")
        ax.set_ylabel("Input residue index (0-based)")
        display(fig)
        plt.close(fig)
if not pae_available:
    print("No PAE supplied for this target in the selected data mode.")

for target in targets:
    for figure in (plot_residues(analysis, target.id), plot_distributions(analysis, target.id),
                   plot_distributions(analysis, target.id, reference=True)):
        plt.close(figure)
print(f"Data mode: {run.kind}; real prediction panel complete: {panel.complete}")
print(f"Exports: {analysis.directory}")
""")
notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (Harmonize)",
            "language": "python",
            "name": "harmonize",
        },
        "language_info": {"name": "python", "version": "3.12.12"},
    },
)
path = ROOT / "notebooks/01_concordance.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, path)
print(path)
