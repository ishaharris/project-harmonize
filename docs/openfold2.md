# OpenFold2 input protocol

The hosted NVIDIA AlphaFold2 endpoint is deprecated. Protocol `2-openfold2-templates` uses
[NVIDIA OpenFold2](https://build.nvidia.com/openfold/openfold2), whose
[model card](https://docs.api.nvidia.com/nim/reference/openfold-openfold2) lists AlphaFold
2.3.2 weights. The exact deployed build is not selectable. It is not a new independently
trained model family, and is not OpenFold3. The original AF2 protocol is archived separately.

## Preparation

Each sequence has one immutable preparation shared across checkpoints 1–5:

1. [NVIDIA MSA search](https://docs.api.nvidia.com/nim/reference/colabfold-msa-search-infer)
   uses MMseqs2, `search_type=colabfold`, all available databases, e-value 0.0001, and A3M output.
   Server defaults control depth; returned database names and row counts are recorded. This
   replaces the originally planned JackHMMer search. The duplicate merged `colabfold` entry is
   omitted when individual database alignments are available. Query identity and match-column
   lengths must match the frozen construct exactly. Query-only results block inference.
2. [RCSB sequence search](https://search.rcsb.org/) searches experimental polymer entities,
   with identity cutoff 0.90 and e-value cutoff 0.0001. Inspect at most 20 hits ranked by returned
   search score, breaking ties by entity ID. Select at most four distinct PDB entries.
3. Download each candidate's mmCIF and check the selected polymer locally with global BLOSUM62
   alignment, gap open/extension -10/-1. Require 90% identity relative to the shorter sequence,
   70% query coverage, and 70% observed N/CA/C backbone coverage. Reject modified/unsupported
   polymers. Choose the best-covered chain (lexical chain ID breaks ties), retain its deposited
   sequence numbering, and rename its label/author chain to A in the submitted template.
4. Preserve original and submitted mmCIFs, search responses, selection/rejection reasons, chain
   mappings, timestamps, and SHA-256 checksums. No eligible templates means an explicit failure;
   there is no automatic template-free fallback.

The 90% threshold and chain-A preprocessing reflect the service's
[direct template requirements](https://docs.nvidia.com/nim/bionemo/openfold2/latest/template-processing.html).
The server subsequently aligns templates with Kalign, so it may still reject a candidate.
This explicit near-homolog selection is not the old HHsearch template protocol.

There is no date cutoff or exclusion of the supplied reference PDB entries. If a selected
template shares the reference entry, the manifest and notebook flag it. Closely related templates
can also reveal the experimental conformation even when the entry differs. Reference scores
therefore remain contextual comparisons, not a held-out accuracy benchmark. PDZ3's input Asp332
override is unchanged; modified template polymers, including the SNN-bearing construct, are rejected.

## Prediction and recovery

Five separate requests select parameter sets 1–5 with `use_templates=true` and relaxation off.
The same explicit templates are supplied each time; standard AF2/OpenFold presets 1–2 use
template features and 3–5 are template-free. These are five checkpoint outputs, not five
independent stochastic samples. ESMFold2 remains five stochastic samples and ESMFold a singleton.
Total counts remain 66 structures and 55 prediction pairs per target.

MSA and folding stages own separate durable job records. A resumed MSA request ID cannot be
mistaken for a folding request ID. Bounded polling and rate-limit retries retain partial results.
Ambiguous POST outcomes are never automatically resubmitted; definite failures require
`--retry-failed`. Public search/download GETs have bounded safe retries. Preparation is frozen
on first success and checked when predictions are loaded. Change the recipe in a new run to
refresh it; never edit cached inputs in place. Only one writer may use a run directory.

OpenFold JSON pLDDT is retained on its documented 0–100 scale. PAE is saved when returned. If
JSON confidence is absent, structure B-factors remain raw with an unknown scale. Endpoint
responses, parameter identities, templates, MSA hashes and provider metrics are retained.

The CLI and notebook use `runs/panel-openfold2`. `load_run`, `compare_run`, and
`preparation_status` make no network calls. Merely opening the notebook does not prepare inputs.
The separate source-data smoke checks in `data/prepared/template-smoke` are not inference results.

Credentialed NVIDIA OpenFold2 and MSA inference still require a live smoke test. Documentation
availability is not proof of account access or hosted-schema compatibility. Any access or input
failure remains explicit; there is no fallback to another model or paid provider.
