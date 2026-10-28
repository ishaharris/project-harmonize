# Project Harmonize

- Use the project Python 3.12.12 environment and committed uv.lock.
- Keep inference separate from analysis. Reading or comparing a run must never submit jobs.
- Do not replace missing predictions with fixtures or other model families.
- Validation fixtures are transformed experimental coordinates, never model predictions.
- Preserve input residue correspondence, full-length TM normalization, and coverage.
- Keep experimental B-factors separate from predicted confidence.
- Preserve the explicit PDZ3 Asp332 override and reference exclusion.
- Never log credentials. Keep raw structures, runs, reports, and environments out of Git.
- Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest`.
- Strip tracked notebook outputs. Executed notebooks belong in reports/.
- Use project-local pinned US-align, not an arbitrary executable on PATH.
