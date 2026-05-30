# Reproducibility Guide

This document defines the files and commands needed to inspect the simulation code and reproduce the stored experiment result surfaces.

## Required sharing set

### Code and runtime

- `backend/src/uavsim/**/*.py`
- `backend/configs/aircraft/*.yaml`
- `backend/pyproject.toml`, `backend/pyrightconfig.json`, `backend/tests/*.py`
- `src/**`, `public/**`, `package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `playwright.config.ts`
- `docs/**`

### Experiment package

- `experiments/claims/**`
- `experiments/results/raw/full-duration/**`
- `experiments/results/raw/coarse-20x50/**`
- `experiments/results/data/**`
- `experiments/results/tables/*.csv`
- `experiments/results/figures/*.png`
- `reproducibility/MANIFEST.sha256`
- `reproducibility/VALIDATION_2026-05-19.md`

### Do not share

- `.env`, real API keys, tokens, private credentials
- `.venv/`, `node_modules/`, `dist/`, caches, `*.egg-info/`, local intermediates
- third-party literature PDFs and private upload bundles
- private notebooks/workbooks/notes unless explicitly cleared

## Source-of-truth contract

Raw benchmark outputs live under:

```text
experiments/results/raw/full-duration/
experiments/results/raw/coarse-20x50/
```

The compact data package used by table and figure scripts lives under:

```text
experiments/results/data/
```

After changing raw benchmark outputs, refresh compact data, tables, figures, and hashes:

```bash
python scripts/sync_results.py
python scripts/export_tables.py
python scripts/write_figures.py
find experiments reproducibility scripts README.md REPRODUCIBILITY.md NOTICE CITATION.cff requirements-figures.txt backend/README.md backend/pyproject.toml backend/src/uavsim/experiment_runner.py backend/src/uavsim/server.py docs -type f \
  ! -path '*/__pycache__/*' \
  ! -path 'reproducibility/MANIFEST.sha256' \
  -print0 | sort -z | xargs -0 sha256sum > reproducibility/MANIFEST.sha256
sha256sum -c reproducibility/MANIFEST.sha256
```

## Environment

Tested local versions during preparation:

```text
Python: 3.12.3
Node:   22.x
```

The backend declares Python `>=3.10`; numeric results can vary slightly across NumPy/SciPy/Python versions.

## Backend quality gate

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e './backend[dev,experiments]'
cd backend
python -m py_compile $(find src tests -name '*.py' -print)
ruff check .
pyright src tests
python -m pytest -q
```

## Frontend quality gate

```bash
npm install
npm run lint
npm run build
npm run smoke:browser
```

`npm run smoke:browser` starts Vite through Playwright and uses a dummy backend URL for browser smoke testing.

## Fast smoke benchmark

Use a temporary output directory for a quick environment sanity check:

```bash
. .venv/bin/activate
cd backend
python -m uavsim.experiment_runner benchmark \
  --output-dir /tmp/uavsim-benchmark-smoke \
  --max-scenarios 1 \
  --duration 2 \
  --step-log-stride 10
```

This checks the benchmark pipeline but does not replace the stored full result sets.

## Full result refresh

```bash
. .venv/bin/activate
cd backend
python -m uavsim.experiment_runner benchmark \
  --output-dir ../experiments/results/raw/full-duration \
  --step-log-stride 10

python -m uavsim.experiment_runner benchmark \
  --output-dir ../experiments/results/raw/coarse-20x50 \
  --seed-count 50 \
  --duration 8 \
  --step-log-stride 0 \
  --substeps 1 \
  --sample-time 0.02
```

Then refresh compact result surfaces:

```bash
cd ..
python scripts/sync_results.py
python scripts/export_tables.py
python scripts/write_figures.py
```

## Metric dictionary

| Metric / CSV column | Implementation source | Interpretation boundary |
| --- | --- | --- |
| `rms_reference_error_m` | `experiment_runner.py` | Horizontal reference/path RMS; not a component ablation. |
| `rms_altitude_error_m` | `experiment_runner.py` | RMS of commanded altitude error. |
| `rms_airspeed_error_mps` | `experiment_runner.py` | RMS of runtime airspeed command error. |
| `control_energy_integral` | `experiment_runner.py` | Runtime actuator-command activity index; not physical energy. |
| `safety_time_fraction` | `experiment_runner.py` | Empirical threshold fraction; not a formal safety certificate. |
| `max_abs_load_factor_nz` | `server.py`, `experiment_runner.py` | Maximum absolute logged load-factor proxy. |
| `residual_active_fraction` | residual supervisors | Fraction of samples with nonzero residual activity. |
| `shield_active_fraction`, `mean_hjb_*`, `mean_candidate_count` | `sharq_hjb.py` | Diagnostics; not proof of HJB optimality or safety. |

## Scenario contract

The full-duration benchmark uses 20 matched scenario definitions for all three controller modes. Scenario fields include profile, seed, duration, target altitude, circle diameter/airspeed/direction, straight length, lookahead, steady wind N/E/D, body gust u/v/w, and turbulence standard deviation. See:

- `backend/src/uavsim/experiment_runner.py` (`REFERENCE_BENCHMARK_SCENARIOS`)
- `experiments/results/raw/full-duration/scenario_catalog.json`
- `experiments/results/data/scenario_catalog.json`

## Claim boundaries to preserve

- The HJB residual supervisor is HJB/Bellman-inspired and finite-action; it does not solve a continuous HJB PDE.
- The CLF/CBF-style filter is a finite-candidate risk filter; it is not a formal CBF-QP certificate.
- The simulation runtime is software-in-the-loop only; no flight, wind-tunnel, HIL, or certification evidence is included.
- The 3D frontend is a visualization/demo layer, not aerodynamic evidence.
