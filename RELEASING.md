# Releasing

This project is **not yet on PyPI**. Everything is prepared so that publishing is a
short, well-defined next step. This file is the checklist.

## One-time PyPI setup (Trusted Publishing — no tokens)

1. Create the project owner account on [PyPI](https://pypi.org) (and, recommended,
   [TestPyPI](https://test.pypi.org)).
2. On PyPI, add a **pending trusted publisher** so the first release can create the
   project without a token:
   - PyPI → *Your projects* → *Publishing* → *Add a pending publisher*.
   - Owner: `ritwikvashistha`, Repository: `kernel_calibration`,
     Workflow: `publish.yml`, Environment: `pypi`.
3. In the GitHub repo, create an **environment** named `pypi`
   (Settings → Environments). The [`publish.yml`](.github/workflows/publish.yml)
   workflow targets it and uses OIDC (`id-token: write`) — no secrets required.

## Cutting a release

1. Make sure `main` is green (CI passing).
2. Bump the version in **two** places:
   - `pyproject.toml` → `version`
   - `CITATION.cff` → `version`
3. Update [`CHANGELOG.md`](CHANGELOG.md): move items from *Unreleased* into a new
   dated version section.
4. Commit, then tag and push:
   ```bash
   git commit -am "Release vX.Y.Z"
   git tag vX.Y.Z
   git push && git push --tags
   ```
5. **Dry run to TestPyPI** (optional but recommended):
   ```bash
   python -m build
   twine check dist/*
   twine upload --repository testpypi dist/*
   # verify in a clean venv:
   pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ kernel_calibration
   ```
6. **Publish to PyPI**: create a **GitHub Release** for the tag. That triggers
   [`publish.yml`](.github/workflows/publish.yml), which builds and uploads via
   Trusted Publishing. (Manual alternative: `twine upload dist/*`.)

## After the first release

- **Zenodo DOI**: enable the GitHub–Zenodo integration for the repo, then the
  GitHub Release mints an archival DOI. Add the DOI badge to `README.md`.
- Add the **PyPI version** badge to `README.md`:
  `[![PyPI](https://img.shields.io/pypi/v/kernel_calibration.svg)](https://pypi.org/project/kernel_calibration/)`

## Building locally

```bash
python -m build          # sdist + wheel into dist/
twine check dist/*       # validate metadata
```
