.PHONY: setup check notebook validate readiness

setup:
	uv sync --frozen --group notebook
	uv run python scripts/bootstrap_native.py
	uv run python -m ipykernel install --sys-prefix --name harmonize --display-name 'Python (Harmonize)'
	uv run pre-commit install

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

notebook:
	uv run --group notebook jupyter lab notebooks/01_concordance.ipynb

validate:
	uv run --group notebook python scripts/execute_notebook.py --mode validation
	uv run --group notebook python scripts/execute_notebook.py --mode prediction

readiness:
	uv run python scripts/predict.py
