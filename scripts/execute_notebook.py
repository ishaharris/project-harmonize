"""Execute a safe notebook copy; never enable inference from this export command."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validation", "prediction"], default="validation")
    parser.add_argument("--target", choices=["gpd", "ramr", "fniii10", "pdz3", "rnasea", "dhfr"])
    args = parser.parse_args()
    target = args.target or ("gpd" if args.mode == "prediction" else "pdz3")
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".tools/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".tools/cache"))
    os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".tools/jupyter-runtime"))
    notebook = nbformat.read(root / "notebooks/01_concordance.ipynb", as_version=4)
    for cell in notebook.cells:
        if "parameters" in cell.metadata.get("tags", []):
            cell.source = f'MODE = "{args.mode}"\nRUN_LIVE = False\nTARGET_ID = "{target}"'
    NotebookClient(
        notebook, timeout=600, kernel_name="harmonize", resources={"metadata": {"path": str(root)}}
    ).execute()
    output = root / "reports"
    output.mkdir(exist_ok=True)
    nbformat.write(notebook, output / f"{args.mode}.ipynb")
    exporter = HTMLExporter()
    html, _ = exporter.from_notebook_node(notebook)
    (output / f"{args.mode}.html").write_text(html)
    print(f"Executed offline: {output / (args.mode + '.ipynb')}")


if __name__ == "__main__":
    main()
