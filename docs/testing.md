# Testing

This repository has frontend checks, backend checks, browser smoke checks, and small runtime diagnostics. Use the smallest check that covers your change, then run broader checks before sharing changes.

## Frontend checks

```bash
npm run lint
npm run build
npm run smoke:browser
```

`npm run smoke:browser` runs Playwright against the Vite dev server. It verifies that:

- the UI loads,
- the backend lifecycle status endpoint is reachable without pressing **Start**,
- telemetry history remains bounded during burst updates.

If Playwright browsers are not installed yet:

```bash
npx playwright install chromium
```

## Backend checks

```bash
cd backend
../.venv/bin/python -m py_compile $(find src tests -name '*.py' -print)
../.venv/bin/ruff check .
../.venv/bin/pyright src tests
../.venv/bin/python -m pytest -q
```

Expected pytest result:

```text
41 passed
```

## What the backend tests cover

`backend/tests/test_uavsim_core.py` covers:

- core dynamics/forces/autopilot helper smoke tests,
- atmosphere, actuator, guidance, TECS, Q-learning, and HJB residual unit checks,
- Q-table JSON/NPZ checkpoint round trips,
- experiment runner CSV/JSONL and benchmark persistence behavior.

`backend/tests/test_uavsim_server.py` covers:

- health endpoint,
- profile command coverage,
- effect clamping,
- WebSocket command validation,
- runtime reset/profile state,
- post-step telemetry coherence,
- compact mission smoke,
- Fight Mode finite reference behavior,
- residual controller mode frame checks.

## Method-specific checks

When changing `q_learning.py` or `sharq_hjb.py`, inspect at least:

- selected `action_index`,
- whether no-op remains available,
- `residual_active`,
- `hard_condition_score`,
- `safety_risk_score`,
- reward and TD error,
- Q-table update count,
- HJB residual `hjb_value`, `hjb_advantage`, `hjb_stage_cost`, `shield_active`, and `candidate_count`.

## Fight Mode backend diagnostic snippet

Use this when checking airspeed collapse, reference runaway, or command-rate problems:

```bash
cd backend
../.venv/bin/python - <<'PYCODE'
from uavsim.server import Runtime, build_simulation_frame, FIGHT_MODE_PROFILE
rt = Runtime.create(profile=FIGHT_MODE_PROFILE)
info = rt.step()
for i in range(int(120 / rt.P.Ts) - 1):
    info = rt.step()
    if i % int(5 / rt.P.Ts) == 0:
        f = build_simulation_frame(rt=rt, info=info, episode=1, scenario=rt.profile, controller='fixed_matlab_autopilot')
        print(
            round(rt.t, 1),
            'Va', round(f['uav_state']['airspeed'], 1),
            'Va_c', round(f['reference_state']['target_airspeed'], 1),
            'h', round(f['uav_state']['position'][2], 1),
            'h_c', round(f['reference_state']['target_altitude'], 1),
            'thr', round(f['control_state']['throttle'], 2),
            'href', round(f['reference_state']['horizontal_reference_error'], 1),
        )
PYCODE
```

## Browser smoke checklist

1. Run `npm run dev`.
2. Choose `Fight Mode` or a compact profile.
3. Start the backend.
4. Check that live telemetry starts.
5. Switch to Cinematic mode.
6. Confirm airspeed does not collapse abruptly.
7. Confirm camera, trace, runway, and terrain do not visibly snap at telemetry rate.
8. Return to Operator mode and confirm Mission, Telemetry, and System panels render.
