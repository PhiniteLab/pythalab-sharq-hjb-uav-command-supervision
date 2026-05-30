# HJB-Inspired Residual UAV Command Supervision

This repository contains a simulation-only fixed-wing UAV command-supervision system. It combines:

- a 12-state rigid-body UAV simulator,
- a fixed/gain-scheduled autopilot,
- a bounded tabular Q-learning residual supervisor,
- an HJB residual supervisor,
- a React/Three.js live visualization frontend,
- benchmark outputs and reproducibility utilities.

The central design choice is simple: **the learning methods do not replace the low-level autopilot**. The autopilot remains the actuator-facing controller. Q-learning and the HJB residual supervisor only add small bounded residuals to the commanded airspeed, altitude, and heading references.

```text
mission/profile generator
  -> nominal command r = [Va_c, h_c, chi_c]
  -> optional residual Δr
  -> adjusted command r + Δr
  -> fixed/gain-scheduled autopilot
  -> actuator model
  -> 12-state UAV dynamics
```

## Scope boundary

This code is for simulation and algorithm inspection. It does **not** claim real-flight validation, hardware-in-the-loop validation, certified flight safety, real SAAB Gripen aerodynamics, a continuous high-dimensional HJB PDE solution, or a formal CBF/HJ safety certificate. The 3D aircraft mesh is a visual asset only; it does not affect the backend physics or benchmark results.

## Canonical experiment paths

The canonical public experiment package is `experiments/`. Claims, raw results, compact data, tables, and figures all live under this directory. No alternate result root is used in this release.

## Repository layout

| Path | Role |
| --- | --- |
| `backend/src/uavsim/` | Python UAV simulation, autopilot, guidance, residual supervisors, WebSocket server, benchmark CLI. |
| `backend/tests/` | Backend tests for dynamics helpers, controllers, server frames, persistence, and benchmark outputs. |
| `src/` | React/TypeScript frontend, telemetry store, WebSocket contracts, 3D scene, charts. |
| `public/` | Static frontend assets, including visual-only STL meshes. |
| `docs/` | English documentation for setup, running, architecture, backend, frontend, API, testing, and development. |
| `experiments/claims/` | Plain-language claim statements linked to code and result surfaces. |
| `experiments/results/raw/` | Raw CSV/JSON/JSONL result sets produced by the backend runner. |
| `experiments/results/data/` | Compact result data used by table and figure scripts. |
| `experiments/results/tables/` | CSV table outputs. |
| `experiments/results/figures/` | PNG figure outputs only. |
| `scripts/` | Utility scripts that copy, summarize, and write benchmark experiment outputs. |
| `reproducibility/` | Experiment hash manifest and recorded validation summary. |

## Quick start: backend checks

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e './backend[dev]'

cd backend
python -m py_compile $(find src tests -name '*.py' -print)
ruff check .
pyright src tests
python -m pytest -q
```

Expected backend test suite for this release: **41 passed**.

## Quick start: live demo

In one shell:

```bash
. .venv/bin/activate
cd backend
python -m uavsim.server
```

In another shell:

```bash
npm install
npm run dev
```

Open:

```text
UI:        http://localhost:3000
Health:    http://localhost:8000/health
WebSocket: ws://localhost:8000/ws/uav-digital-twin
```

The frontend supports two display modes:

- `operator` — instruments, controller/profile controls, telemetry panels, and charts;
- `cinematic` — large 3D view, compact HUD, and guided chase camera.

Available profiles include `fight_mode`, `runway_takeoff_accel_200`, `takeoff_climbout_200`, `high_speed_climb_s_turn_200`, `straight_climb_altitude_hold`, `figure_eight`, `racetrack`, and `loiter_orbit`.

## Method overview

### 1. Baseline autopilot

`fixed_matlab_autopilot` runs the nominal mission guidance and fixed/gain-scheduled autopilot only.

The command vector is:

```text
r = [Va_c, h_c, chi_c]
```

where:

- `Va_c` is commanded airspeed,
- `h_c` is commanded altitude,
- `chi_c` is commanded heading/course angle.

The autopilot converts these references into throttle, elevator, aileron, and rudder commands. The actuator model applies limits, lag, and saturation before the rigid-body dynamics step is integrated.

### 2. Tabular Q residual supervisor

`online_q_learning` keeps the same autopilot but inserts a small residual before the autopilot:

```text
r_supervised = clip(r + Δr)
```

The action set is finite and command-level:

```text
Δr = [ΔVa_c, Δh_c, Δchi_c]
A = {
  no-op,
  +2 m/s airspeed, -2 m/s airspeed,
  +10 m altitude, -10 m altitude,
  +3 deg heading, -3 deg heading
}
```

The supervisor observes tracking and stress signals, discretizes them into a tabular state, chooses a valid residual action, applies the residual to the command vector, and updates the Q-table from the next-step reward.

The state encoder uses seven factors:

1. airspeed error,
2. altitude error,
3. horizontal reference error,
4. straight-path cross-track error,
5. orbit radial error,
6. wind/turbulence stress,
7. low-energy condition flag.

The reward penalizes tracking error, actuator saturation, envelope risk, and unnecessary residual activity. Residuals are gated: in easy conditions only the no-op action is allowed, so the baseline autopilot is preserved.

### 3. HJB residual supervisor

`sharq_hjb` uses the same command-level residual action set and the same baseline autopilot. It adds an interpretable HJB/Bellman-inspired candidate evaluator before selecting the residual.

For each candidate residual, the supervisor computes a normalized feature vector:

```text
z = [
  airspeed_error,
  altitude_error,
  horizontal_reference_error,
  lateral_path_error,
  wind_turbulence_stress,
  actuator_saturation_stress,
  load_factor_stress
]
```

It then estimates how each residual would change those features over a short horizon. The value score combines:

- a transparent quadratic value over normalized tracking/stress features,
- a small semi-discrete Bellman/HJB value table over the same feature abstraction,
- a residual action cost,
- a Hamiltonian-style advantage relative to the no-op action,
- a CLF/CBF-style finite-action risk filter that filters candidates predicted to grow value or load/saturation risk too much.

The no-op action is always available. Under selected high-disturbance conditions, the HJB residual supervisor can fall back to the tabular Q selection path. This keeps the method bounded and inspectable while preserving the same actuator-facing autopilot.

## Runtime algorithm

At each backend simulation step:

1. The active profile produces nominal airspeed, altitude, and heading references.
2. Wind, gust, turbulence, and guidance errors are computed.
3. If the controller mode is fixed, the nominal command is passed directly to the autopilot.
4. If the controller mode is Q-learning or HJB residual, the residual supervisor selects a bounded command residual.
5. The adjusted command is clipped to the allowed command envelope.
6. The fixed/gain-scheduled autopilot computes actuator commands.
7. Actuator limits and lags are applied.
8. The 12-state rigid-body dynamics are advanced with RK4 integration.
9. The backend emits telemetry and diagnostics over WebSocket.
10. The frontend maps the frame into charts and the 3D scene.

## Controller modes

| Mode | What changes? | What stays fixed? |
| --- | --- | --- |
| `fixed_matlab_autopilot` | No residual; nominal guidance only. | Plant, actuator model, autopilot. |
| `online_q_learning` | Adds tabular Q residuals to `[Va_c, h_c, chi_c]`. | Plant, actuator model, autopilot. |
| `sharq_hjb` | Scores and filters the same residuals with HJB/Bellman-inspired diagnostics. | Plant, actuator model, autopilot. |

## Benchmark outputs

The stored benchmark outputs compare the three controller modes on challenging reference and wind/gust/turbulence cases. Main scalar metrics include:

- RMS horizontal reference error,
- RMS altitude error,
- RMS airspeed error,
- actuator-command activity index,
- saturation time fraction,
- empirical safety-threshold time fraction,
- maximum absolute load factor,
- residual active fraction,
- Q-table update count,
- HJB residual value/advantage/risk-filter diagnostics.

Metric boundaries:

- `control_energy_integral` is an actuator-command activity index, not physical fuel or battery energy.
- `safety_time_fraction` is an empirical simulator threshold fraction, not certification evidence.
- The HJB residual supervisor is finite-action and HJB/Bellman-inspired; it is not a continuous HJB PDE solver.

## More documentation

Start with [`docs/README.md`](docs/README.md), then read:

1. [`docs/setup.md`](docs/setup.md)
2. [`docs/running.md`](docs/running.md)
3. [`docs/architecture.md`](docs/architecture.md)
4. [`docs/backend.md`](docs/backend.md)
5. [`docs/api-websocket.md`](docs/api-websocket.md)
6. [`docs/testing.md`](docs/testing.md)

## License and third-party assets

Code is Apache-2.0 unless noted otherwise. See `NOTICE` for data/result licensing notes and third-party asset warnings. SAAB Gripen/Ro3code STL meshes are included as visual-only frontend assets. Their documented provenance is the Ro3code `aircraft_3d_animation` project, specifically the upstream `import_stl_model/SAAB-Gripen/` asset set. They are not aerodynamic evidence and do not affect backend physics or benchmark results. The frontend falls back to a simple placeholder aircraft when meshes are absent or fail to load.
