# Documentation Index

This directory explains the UAV command-supervision system in plain English. The goal is to describe what the code does, how the controllers are connected, and how to run and validate the repository.

## Recommended reading order

1. [`setup.md`](setup.md) — install Python and Node dependencies.
2. [`running.md`](running.md) — run the backend, frontend, and offline benchmark CLI.
3. [`architecture.md`](architecture.md) — understand the runtime data flow and controller loop.
4. [`backend.md`](backend.md) — map the Python modules and residual-supervision algorithms.
5. [`api-websocket.md`](api-websocket.md) — inspect command messages and telemetry frames.
6. [`frontend.md`](frontend.md) — understand the React/Three.js visualization layer.
7. [`high-speed-140.md`](high-speed-140.md) — read the gain-schedule and high-speed mission notes.
8. [`testing.md`](testing.md) — run the quality gates and smoke checks.
9. [`development.md`](development.md) — use the safe change workflow.

## System picture

```text
React/Vite frontend
  ├─ operator UI: controls, instruments, telemetry, charts
  ├─ cinematic UI: large 3D scene and compact HUD
  ├─ WebSocket client: ws://localhost:8000/ws/uav-digital-twin
  └─ Three.js aircraft/terrain/path visualization
        │
        │ JSON command / simulation_frame telemetry
        ▼
FastAPI backend (`uavsim.server`)
  ├─ mission/profile reference generator
  ├─ optional residual supervisor: Q-learning or HJB residual
  ├─ fixed/gain-scheduled autopilot
  ├─ actuator limits and lag
  ├─ wind/gust/turbulence model
  └─ 12-state rigid-body RK4 dynamics
```

## Main idea

The repository studies **command supervision**, not direct actuator learning.

The nominal guidance command is:

```text
r = [Va_c, h_c, chi_c]
```

The residual supervisors can add only one small residual from a finite action set:

```text
Δr = [ΔVa_c, Δh_c, Δchi_c]
```

The adjusted command is clipped and then passed to the same fixed autopilot. This keeps the learning layer above the actuator-facing controller and makes every action inspectable.

## Controller modes

- `fixed_matlab_autopilot` — nominal guidance and fixed/gain-scheduled autopilot only.
- `online_q_learning` — tabular Q-learning chooses bounded command residuals.
- `sharq_hjb` — HJB/Bellman-inspired candidate scoring and finite-action risk filtering chooses bounded command residuals.

## Important boundaries

- The backend is a simulator, not a certified flight controller.
- The 3D aircraft mesh is visual only and is not used by the physics model.
- The HJB residual supervisor uses a finite residual-action abstraction and a semi-discrete value surrogate; it does not solve a continuous high-dimensional HJB PDE.
- Safety metrics are simulator diagnostics, not certification guarantees.
