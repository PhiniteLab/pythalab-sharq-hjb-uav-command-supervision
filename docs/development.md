# Development

Use this workflow to keep changes small, testable, and aligned with the simulator architecture.

## Safe workflow

1. Read the relevant source files before editing; do not trust stale documentation alone.
2. Keep each change small and bounded.
3. If dynamics, autopilot, API frames, residual algorithms, frontend mapping, trace, or camera behavior changes, update the matching documentation in the same change.
4. Run focused validation first, then broader validation when the change is non-trivial.
5. If validation fails, fix it or report it clearly. Do not claim completion with failing checks.

## Common change areas

| Need | Files |
| --- | --- |
| Dynamics / forces | `backend/src/uavsim/dynamics.py`, `forces_moments.py`, backend tests |
| Autopilot | `backend/src/uavsim/autopilot.py`, `server.py`, backend tests |
| Compact mission guidance | `backend/src/uavsim/server.py`, `guidance.py`, `backend/tests/test_uavsim_server.py` |
| Fight Mode guidance | `server.py`, `test_uavsim_server.py`, frontend profile UI, docs |
| Tabular Q residual logic | `backend/src/uavsim/q_learning.py`, tests, API docs if metrics change |
| HJB residual logic | `backend/src/uavsim/sharq_hjb.py`, tests, API docs if metrics change |
| WebSocket frame | `server.py`, `backendMessages.ts`, `telemetryTypes.ts`, scene mapping, docs |
| Frontend trace/performance | `SimpleFlightScene.tsx`, `TrajectoryCharts.tsx`, `simulationStore.ts`, `docs/frontend.md` |
| Cinematic camera/UI | `SimpleFlightScene.tsx`, `simulationStore.ts`, `styles/global.css` |

## Required validation commands

Backend:

```bash
cd backend
../.venv/bin/python -m py_compile $(find src tests -name '*.py' -print)
../.venv/bin/ruff check .
../.venv/bin/pyright src tests
../.venv/bin/python -m pytest -q
```

Frontend:

```bash
npm run lint
npm run build
```

Browser smoke:

```bash
npm run smoke:browser
```

## Residual-supervisor rule

When changing Q-learning or HJB residual logic, preserve these invariants unless you intentionally document and test a new contract:

- the fixed autopilot remains actuator-facing;
- residuals modify only `[Va_c, h_c, chi_c]`;
- the no-op action remains available;
- command outputs are clipped to the allowed command envelope;
- high load/saturation risk can restrict residual choices;
- telemetry exposes enough metrics to audit why an action was selected.

## Fight Mode rule

When changing Fight Mode, inspect:

- `Va` and `Va_c`,
- throttle floor and throttle collapse,
- altitude command rate,
- heading command rate,
- horizontal reference error,
- visual bank direction from heading error,
- browser camera/trace snapping.

## Frontend performance rule

Do not reintroduce unbounded allocation in render loops. Prefer bounded telemetry history, stable `BufferGeometry`, `InstancedMesh`, explicit disposal, downsampled chart data, and render-side smoothing for 10 Hz telemetry.
