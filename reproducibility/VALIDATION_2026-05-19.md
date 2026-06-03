# Validation Summary — 2026-05-19

This file records the local validation run after moving claims and results under `experiments/` and limiting figure outputs to PNG files.

## Environment

- Date: 2026-05-19
- Python: 3.12.3
- Node: 22.x

## Checks

| Status | Check | Command / evidence |
| --- | --- | --- |
| passed | Experiment result sync | `python scripts/sync_results.py` |
| passed | Table export | `python scripts/export_tables.py`; CSV files written under `experiments/results/tables/`. |
| passed | Figure output | `python scripts/write_figures.py`; 12 PNG files written under `experiments/results/figures/`. Matplotlib emitted a non-blocking `Axes3D` import warning; these figures do not use 3D projection. |
| passed | Python syntax | `python -m py_compile scripts/sync_results.py scripts/export_tables.py scripts/write_figures.py backend/src/uavsim/experiment_runner.py backend/src/uavsim/server.py` |
| passed | Backend tests | `cd backend && ../.venv/bin/python -m pytest -q`; 41 passed. |
| passed | Ruff | `cd backend && ../.venv/bin/ruff check .`; all checks passed. |
| passed | Pyright | `cd backend && ../.venv/bin/pyright src tests`; 0 errors, 0 warnings, 0 informations. |
| passed | Frontend type check | `npm run lint`; TypeScript no-emit check passed. |
| passed | Frontend build | `npm run build`; Vite build passed with the existing chunk-size warning for the Three.js bundle. |
| passed | Markdown local links | local link checker for root/docs/experiments/reproducibility/backend README files. |
| passed | Tool-residue scan | No tool-residue references found outside excluded dependency/build/cache folders. |
| passed | Whitespace check | `git diff --check`. |
| passed | PNG-only figure folder | `find experiments/results/figures -type f ! -name '*.png' -print`; no files reported. |
| passed | Legacy path/name scan | no legacy result paths or old script-name references after manifest refresh. |
| passed | Experiment manifest | `sha256sum -c reproducibility/MANIFEST.sha256`. |

## Notes

- Raw benchmark results now live under `experiments/results/raw/`.
- Compact result data, tables, and figures now live under `experiments/results/`.
- Figure outputs are PNG only.
- Claims are documented under `experiments/claims/`.
