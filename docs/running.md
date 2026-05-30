# Running

This document explains how to run the frontend, backend, live WebSocket demo, and offline benchmark CLI.

## Frontend development server

```bash
npm install
npm run dev
```

Default URL:

```text
http://localhost:3000
```

The frontend contains a local backend lifecycle plugin. Pressing **Start** in the UI can launch the Python backend if no backend is already reachable on port `8000`.

## Manual backend server

```bash
cd backend
../.venv/bin/python -m pip install -e '.[dev]'
../.venv/bin/python -m uavsim.server
```

Health check:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

WebSocket endpoint:

```text
ws://localhost:8000/ws/uav-digital-twin
```

## Live controller modes

| Mode | Description |
| --- | --- |
| `fixed_matlab_autopilot` | Nominal guidance command goes directly to the fixed/gain-scheduled autopilot. |
| `online_q_learning` | Tabular Q-learning selects bounded residuals on airspeed, altitude, and heading commands. |
| `sharq_hjb` | The HJB residual supervisor scores and filters the same residuals using value/advantage/risk diagnostics. |

The residual methods modify only:

```text
[Va_c, h_c, chi_c]
```

They do not directly command throttle, elevator, aileron, or rudder.

## Offline experiment CLI

The backend runner can compare controllers, train/evaluate residual policies, and write CSV/JSONL outputs.

```bash
cd backend
../.venv/bin/python -m uavsim.experiment_runner compare \
  --profile loiter_orbit \
  --duration 60 \
  --summary-csv experiments/tmp/uavsim/compare.csv

../.venv/bin/python -m uavsim.experiment_runner train \
  --profile loiter_orbit \
  --duration 60 \
  --checkpoint-out experiments/tmp/uavsim/q_policy.json \
  --summary-csv experiments/tmp/uavsim/train.csv \
  --step-jsonl experiments/tmp/uavsim/train_steps.jsonl

../.venv/bin/python -m uavsim.experiment_runner eval \
  --profile loiter_orbit \
  --duration 60 \
  --checkpoint-in experiments/tmp/uavsim/q_policy.json \
  --summary-csv experiments/tmp/uavsim/eval.csv

../.venv/bin/python -m uavsim.experiment_runner batch \
  --profiles loiter_orbit,racetrack,fight_mode \
  --seeds 23341,23342,23343 \
  --duration 60 \
  --summary-csv experiments/tmp/uavsim/batch.csv
```

The `train` command updates a Q-table and can save JSON/NPZ checkpoints. The `eval` command loads a checkpoint and evaluates with exploration disabled. The `compare` and `batch` commands run controller comparisons with shared runtime code paths.

## Stored benchmark regeneration

The repository also contains stored benchmark result directories. To write similarly shaped outputs, use the `benchmark` command:

```bash
cd backend
../.venv/bin/python -m uavsim.experiment_runner benchmark \
  --output-dir ../experiments/results/raw/full-duration \
  --step-log-stride 10

../.venv/bin/python -m uavsim.experiment_runner benchmark \
  --output-dir ../experiments/results/raw/coarse-20x50 \
  --seed-count 50 \
  --duration 8 \
  --step-log-stride 0 \
  --substeps 1 \
  --sample-time 0.02
```

These commands run the same backend plant, guidance, autopilot, actuator model, and residual-supervisor code used by the live WebSocket server.

## UI modes

### Operator

Use Operator mode for debugging and algorithm inspection. It shows controls, instruments, mission telemetry, system status, and charts.

### Cinematic

Use Cinematic mode for visual inspection. It keeps the main 3D scene large and uses a guided camera with a compact HUD.

## Profile options

| UI label | Backend profile | Task |
| --- | --- | --- |
| Fight Mode | `fight_mode` | progress-tied S-turn/climb-dive reference with visual augmentation |
| 200m Circle | `runway_takeoff_accel_200` | takeoff, climb, straight segment, compact orbit |
| Climb Circle | `takeoff_climbout_200` | compact mission |
| Tight Circle | `high_speed_climb_s_turn_200` | compact mission |
| Hold Circle | `straight_climb_altitude_hold` | compact mission |
| 8 Compact | `figure_eight` | compact mission |
| Race 200m | `racetrack` | compact mission |
| Loiter 200m | `loiter_orbit` | compact mission |

## Wind and disturbance presets

| UI preset | Backend effect | Purpose |
| --- | --- | --- |
| Calm | zero wind/gust/turbulence | baseline sanity check |
| Headwind | negative north steady wind | energy and airspeed stress |
| Tailwind | positive north steady wind | low relative airspeed stress |
| Crosswind | east steady wind | lateral tracking stress |
| Turbulence | Dryden-like gust scale | noisy disturbance stress |
| Gust | body-axis gust | short disturbance impulse |

## Vite backend lifecycle API

| Endpoint | Method | Function |
| --- | --- | --- |
| `/api/backend/status` | GET | Returns backend state, pid, message, and recent log. |
| `/api/backend/start` | POST | Starts backend if needed, or reuses an existing process on port `8000`. |
| `/api/backend/stop` | POST | Stops the backend process launched by Vite. |

## Manual live smoke checklist

1. Run `npm run dev`.
2. Choose `Fight Mode` or a compact profile in Operator mode.
3. Select `fixed_matlab_autopilot`, `online_q_learning`, or `sharq_hjb`.
4. Press **Start**.
5. Confirm live telemetry turns active.
6. Confirm the 3D aircraft, reference trace, and charts update.
7. Switch to Cinematic mode and check that camera/trace motion remains smooth.
