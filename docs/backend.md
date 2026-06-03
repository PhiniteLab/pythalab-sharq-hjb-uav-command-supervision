# Backend

The backend package lives under `backend/src/uavsim`. It provides the simulator, autopilot, guidance logic, residual supervisors, WebSocket server, and offline benchmark runner.

## Entrypoints

```bash
cd backend
../.venv/bin/python -m uavsim.server
../.venv/bin/uavsim-backend
```

Offline CLI:

```bash
cd backend
../.venv/bin/python -m uavsim.experiment_runner --help
```

## Module map

| File | Responsibility |
| --- | --- |
| `server.py` | FastAPI app, WebSocket loop, runtime state, command handling, profile guidance, residual supervisor wiring, frame construction. |
| `autopilot.py` | Fixed-gain cascaded autopilot and altitude state machine. |
| `gain_schedule.py` | Airspeed-indexed gain bank and interpolation. |
| `parameters.py` | Physical constants, aerodynamic coefficients, trim/gain fields, sample time, actuator limits. |
| `dynamics.py` | 12-state rigid-body derivatives and RK4 integration. |
| `forces_moments.py` | Air data, gravity, aerodynamic, control, and propulsion forces/moments. |
| `wind.py` | Steady wind, body gusts, and Dryden-like turbulence generation. |
| `guidance.py` | Straight-path and orbit guidance helpers. |
| `commands.py` | Legacy command profile helpers. |
| `actuators.py` | Actuator limits, saturation, and lag behavior. |
| `q_learning.py` | Tabular Q-learning residual supervisor. |
| `sharq_hjb.py` | HJB residual supervisor. |
| `tecs.py` | Energy-related helper signals used by residual modes and diagnostics. |
| `experiment_runner.py` | Compare/train/eval/batch/benchmark CLI and CSV/JSONL persistence. |

## Simulation state

The main rigid-body state is:

```text
x = [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
```

where position is NED, body velocities are `u/v/w`, Euler angles are `phi/theta/psi`, and body rates are `p/q/r`.

## Baseline controller

The baseline controller is a fixed/gain-scheduled autopilot. It tracks:

```text
[Va_c, h_c, chi_c]
```

and produces actuator commands for throttle, elevator, aileron, and rudder. The residual supervisors do not replace this autopilot; they only adjust its references before the autopilot runs.

## Tabular Q residual algorithm

`TabularQLearningSupervisor` uses a finite action set:

```text
0: no-op
1: +2 m/s airspeed command
2: -2 m/s airspeed command
3: +10 m altitude command
4: -10 m altitude command
5: +3 deg heading command
6: -3 deg heading command
```

At each supervised step it:

1. measures airspeed, altitude, reference, lateral path, wind/turbulence, saturation, and load-factor context;
2. converts those signals into a seven-part discrete state;
3. builds a valid action list from hard-condition and safety-risk gates;
4. uses epsilon-greedy selection during training, or greedy selection when frozen;
5. applies the residual to `[Va_c, h_c, chi_c]` and clips the command envelope;
6. computes a reward from next-step tracking error, saturation, load risk, and residual cost;
7. updates the Q-table with a standard one-step temporal-difference update when training is enabled.

Easy conditions usually allow only no-op. This prevents the residual layer from changing nominal flight unnecessarily.

## HJB residual algorithm

The HJB residual supervisor inherits the same Q-table, action set, checkpoint format, and reward structure, then adds HJB/Bellman-inspired candidate scoring.

For each valid candidate action it:

1. builds a normalized feature vector containing tracking errors, lateral error, wind/turbulence stress, saturation stress, and load-factor stress;
2. predicts a short-horizon feature transition for the candidate residual;
3. evaluates a transparent value function made from a quadratic feature value plus a semi-discrete Bellman/HJB value-table surrogate;
4. computes a Hamiltonian-style advantage relative to no-op;
5. applies a CLF/CBF-style finite-action risk filter that can reject candidates with excessive predicted value growth or load/saturation risk;
6. scores remaining candidates using Q-value, HJB advantage, stage cost, and risk penalties;
7. keeps no-op available as the safe fallback.

The implementation is intentionally inspectable: telemetry includes selected action, HJB value, HJB advantage, stage cost, `shield_active` status, candidate count, load factor, and safety-risk score.

## Frame construction

`build_simulation_frame()` emits post-step state, air data, reference position/error, command targets, actuator commands, saturation metrics, wind/gust fields, residual metrics, and compatibility fields used by the frontend.

Display yaw is wrapped to a stable range:

```text
yaw_display = atan2(sin(psi), cos(psi))
```

## Validation

```bash
cd backend
../.venv/bin/python -m py_compile $(find src tests -name '*.py' -print)
../.venv/bin/ruff check .
../.venv/bin/pyright src tests
../.venv/bin/python -m pytest -q
```

Expected result: `41 passed`.
