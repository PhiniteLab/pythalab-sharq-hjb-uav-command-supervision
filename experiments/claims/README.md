# Claims

This file states the repository claims in code-facing language and links each claim to the implementation and result surfaces. The claims are intentionally limited to simulation behavior.

## Claim 1: residual supervision preserves the fixed autopilot

The residual methods do not command actuators directly. They adjust only:

```text
[Va_c, h_c, chi_c]
```

The fixed/gain-scheduled autopilot remains the actuator-facing controller.

Code links:

- `backend/src/uavsim/server.py`
- `backend/src/uavsim/autopilot.py`
- `backend/src/uavsim/q_learning.py`
- `backend/src/uavsim/sharq_hjb.py`

## Claim 2: the residual action space is finite and bounded

The residual action set contains no-op plus small airspeed, altitude, and heading command changes:

```text
no-op
+/- 2 m/s airspeed command
+/- 10 m altitude command
+/- 3 deg heading command
```

The adjusted command is clipped before it reaches the autopilot.

Code links:

- `backend/src/uavsim/q_learning.py`
- `backend/src/uavsim/sharq_hjb.py`

## Claim 3: the three controller modes share the same plant and runtime path

The baseline, tabular Q residual mode, and HJB residual mode run through the same backend runtime, wind model, actuator model, autopilot, force/moment model, and 12-state dynamics.

Code links:

- `backend/src/uavsim/server.py`
- `backend/src/uavsim/experiment_runner.py`
- `backend/src/uavsim/dynamics.py`
- `backend/src/uavsim/forces_moments.py`

## Claim 4: HJB residual mode adds inspectable value and risk-filter diagnostics

The HJB residual supervisor scores finite candidate residuals with normalized tracking/stress features, a semi-discrete Bellman/HJB value surrogate, Hamiltonian-style advantage, and a CLF/CBF-style finite-action risk filter. Telemetry records the selected action and diagnostics.

Code links:

- `backend/src/uavsim/sharq_hjb.py`
- `docs/api-websocket.md`

Result links:

- `experiments/results/data/`
- `experiments/results/tables/`
- `experiments/results/figures/`

## Claim 5: result metrics are simulator diagnostics

The stored results report simulator metrics such as reference error, altitude error, airspeed error, actuator-command activity, saturation fraction, load factor, residual activity, and HJB residual diagnostics.

Boundaries:

- `control_energy_integral` is an actuator-command activity index, not physical fuel or battery energy.
- `safety_time_fraction` is an empirical simulator threshold fraction, not certification evidence.
- The HJB residual supervisor uses a finite residual-action abstraction; it is not a continuous high-dimensional HJB PDE solver.
