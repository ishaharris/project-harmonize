"""Explicit prediction entry point; default is a no-inference readiness report."""

import argparse
from pathlib import Path

from harmonize.targets import load_reference_snapshot
from harmonize.workflow import run_predictions, status_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Submit missing predictions")
    parser.add_argument("--model", action="append", choices=["alphafold2", "esmfold", "esmfold2"])
    parser.add_argument(
        "--target", action="append", choices=["gpd", "ramr", "fniii10", "pdz3", "rnasea", "dhfr"]
    )
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = run_predictions(
        load_reference_snapshot(root),
        root / "runs/panel",
        root / "configs/protocol.yaml",
        execute=args.execute,
        models=args.model,
        target_ids=args.target,
        retry_failed=args.retry_failed,
    )
    print(status_table(run).to_string(index=False))
    print(
        f"Complete: {run.complete}; available: {sum(a.status == 'complete' for a in run.artifacts)}/{run.expected_count}"
    )


if __name__ == "__main__":
    main()
