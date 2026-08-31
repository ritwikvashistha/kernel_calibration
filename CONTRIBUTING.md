# Contributing

Thanks for your interest in improving `kernel_calibration` (KiTE)! This guide covers
the development workflow.

## Development setup

```bash
git clone https://github.com/ritwikvashistha/kernel_calibration.git
cd kernel_calibration
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs the package in editable mode along with the test and lint tooling
(`pytest`, `ruff`, `build`, `twine`, `matplotlib`).

## Running the checks

```bash
pytest                                       # run the test suite
ruff check kernel_calibration/ tests/ docs/  # lint
ruff format kernel_calibration/ tests/ docs/ # auto-format
```

CI runs the same lint and tests on Python 3.9–3.12, so please make sure they pass
locally before opening a pull request. `ruff format --check` must be clean.

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve       # live preview at http://127.0.0.1:8000
```

API pages are generated from the NumPy-style docstrings via `mkdocstrings`, so
document new public functions in that style. The README figures are reproducible
with `python docs/generate_figures.py`.

## Building the package

```bash
python -m build
twine check dist/*
```

## Pull requests

- Keep changes focused; add tests for new behaviour.
- Add a note under "Unreleased" in [`CHANGELOG.md`](CHANGELOG.md).
- Public functions should have NumPy-style docstrings and type hints.

## Reporting issues

Please open issues at <https://github.com/ritwikvashistha/kernel_calibration/issues> with a minimal
reproducible example where possible.
