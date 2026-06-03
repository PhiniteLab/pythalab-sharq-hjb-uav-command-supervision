# PythaLab UAVSim Backend

**ver0.0.5 — Cinematic Fight Mode and smoothed video capture** backend package for PythaLab UAV 6-DOF Digital Twin.

This package exposes:

- `uavsim.server:app` — FastAPI application.
- `uavsim-backend` — console script entrypoint.
- `/health` — health endpoint.
- `/ws/uav-digital-twin` — live telemetry WebSocket.

## What the backend simulates

The backend runs a 12-state NED/body-frame rigid-body MAV model:

```text
x = [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
```

It combines aerodynamic force/moment model, propulsion model, steady wind/gust/turbulence, MATLAB/Simulink-style fixed-gain autopilot, airspeed gain scheduling, compact waypoint/orbit guidance, optional bounded tabular Q-learning guidance-residual supervision, optional HJB residual command supervision with finite-action risk filtering, `fight_mode` cinematic dogfight reference generation and FastAPI WebSocket telemetry.

It does **not** simulate real SAAB Gripen aerodynamics, certified/optimal RL flight control, flexible-wing FEM/MCK dynamics, landing, certified flight control safety, or certified aerobatics.

## Active ver0.0.5 missions

### Compact waypoint/orbit profiles

All live profiles except `fight_mode` use the compact mission template:

1. runway start from rest,
2. smooth takeoff speed command,
3. climb to `200 m`,
4. fly a `200 m` straight waypoint segment,
5. enter a `200 m` diameter orbit at about `25 m/s`.

### Fight Mode

`fight_mode` uses a separate cinematic S-turn/dogfight reference after takeoff and climb.

Important backend details:

- reference is tied to aircraft path progress plus lead distance, not only wall-clock time;
- altitude command is rate-limited for climb/dive smoothness;
- minimum throttle floor preserves energy during visual dogfight manoeuvres;
- target speed is raised to keep the aircraft from collapsing to very low airspeed;
- frontend may visually amplify roll/inverted moments for recording.

## Run

```bash
cd backend
../.venv/bin/python -m pip install -e '.[dev]'
../.venv/bin/python -m uavsim.server
```

or:

```bash
cd backend
../.venv/bin/uavsim-backend
```

Endpoints:

```text
GET http://localhost:8000/health
WS  ws://localhost:8000/ws/uav-digital-twin
```

## Validation

```bash
cd backend
../.venv/bin/python -m py_compile $(find src tests -name '*.py' -print)
../.venv/bin/ruff check .
../.venv/bin/pyright src tests
../.venv/bin/python -m pytest -q
```

Current backend suite expectation: `41 passed`.

Offline experiment CLI:

```bash
cd backend
../.venv/bin/python -m uavsim.experiment_runner compare --duration 60 --summary-csv experiments/tmp/uavsim/compare.csv
../.venv/bin/python -m uavsim.experiment_runner compare --all-methods --duration 60 --summary-csv experiments/tmp/uavsim/compare_all.csv
../.venv/bin/python -m uavsim.experiment_runner train --duration 60 --checkpoint-out experiments/tmp/uavsim/q_policy.json
../.venv/bin/python -m uavsim.experiment_runner eval --duration 60 --checkpoint-in experiments/tmp/uavsim/q_policy.json
../.venv/bin/python -m uavsim.experiment_runner benchmark --output-dir ../experiments/results/raw/full-duration --step-log-stride 10
../.venv/bin/python -m uavsim.experiment_runner benchmark --output-dir ../experiments/results/raw/coarse-20x50 --seed-count 50 --duration 8 --step-log-stride 0 --substeps 1 --sample-time 0.02
```

`benchmark` writes baseline / baseline+Q / HJB residual comparisons over the reference scenario set. `--seed-count` creates sequential seeds for short-duration coarse sweeps.
