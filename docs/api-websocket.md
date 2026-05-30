# WebSocket API

The backend exposes a small HTTP health endpoint and one WebSocket stream. The WebSocket accepts control commands from the frontend and emits `simulation_frame` telemetry messages.

## Endpoints

```text
GET  /health
WS   /ws/uav-digital-twin
```

Health response:

```json
{"status":"ok"}
```

## Command message

The frontend sends JSON commands. Invalid command, profile, controller, or effect values are rejected without corrupting runtime state.

```json
{
  "command": "start",
  "episode": 1,
  "scenario": "fight_mode",
  "profile": "fight_mode",
  "controller": "fixed_matlab_autopilot",
  "effects": {
    "steady_wind_n": 0,
    "steady_wind_e": 0,
    "steady_wind_d": 0,
    "gust_body_u": 0,
    "gust_body_v": 0,
    "gust_body_w": 0,
    "turbulence_std": 0
  }
}
```

Supported commands:

```text
start
pause
stop
reset
configure
```

Supported controller labels:

```text
fixed_matlab_autopilot
fixed_matlab_baseline
baseline
online_q_learning
baseline_q
sharq_hjb
baseline_sharq_hjb
```

Controller behavior:

```text
fixed_matlab_autopilot -> nominal guidance + fixed/gain-scheduled autopilot
online_q_learning      -> tabular Q residuals on [Va_c, h_c, chi_c]
sharq_hjb              -> HJB/Bellman-scored and risk-filtered residuals on [Va_c, h_c, chi_c]
```

Supported profiles:

```text
runway_takeoff_accel_200
takeoff_climbout_200
high_speed_climb_s_turn_200
straight_climb_altitude_hold
figure_eight
racetrack
loiter_orbit
fight_mode
```

`fight_mode` uses a progress-tied S-turn/climb-dive reference. Other profiles use compact waypoint/orbit references.

## Effect clamp ranges

| Effect | Range | Meaning |
| --- | --- | --- |
| `steady_wind_n` | `[-10, 10]` | inertial NED north wind, m/s |
| `steady_wind_e` | `[-10, 10]` | inertial NED east wind, m/s |
| `steady_wind_d` | `[-4, 4]` | inertial NED down wind, m/s |
| `gust_body_u` | `[-4, 4]` | body x gust, m/s |
| `gust_body_v` | `[-4, 4]` | body y gust, m/s |
| `gust_body_w` | `[-4, 4]` | body z gust, m/s |
| `turbulence_std` | `[0, 2]` | Dryden-like gust scale |

Non-finite values such as `NaN` and `inf` are ignored.

## Simulation frame rate and schema

The backend emits `simulation_frame` messages at about 10 Hz while simulating internally at a smaller time step. The top-level `schema_version` is currently `1`. Increment it only for intentional frame-contract changes and update the TypeScript contracts/tests at the same time.

## Important field mapping

| Backend field | Frontend field or use |
| --- | --- |
| `schema_version` | contract guard in `SimulationFrameMessage` |
| `timestamp` | UI/store time basis |
| `profile` | active trajectory profile |
| `uav_state.position[0]` | north/forward position |
| `uav_state.position[1]` | east/lateral position |
| `uav_state.position[2]` | altitude after NED-to-UI mapping |
| `uav_state.attitude[0]` | roll angle |
| `uav_state.attitude[1]` | pitch angle |
| `uav_state.attitude[2]` | yaw angle |
| `reference_state.reference_position_n` | reference north position |
| `reference_state.reference_position_e` | reference east position |
| `reference_state.reference_altitude` | reference altitude |
| `reference_state.target_airspeed` | `Va_c` command after guidance/residual logic |
| `reference_state.target_altitude` | `h_c` command after guidance/residual logic |
| `reference_state.target_heading_deg` | `chi_c` command after guidance/residual logic |
| `control_state.throttle` | applied throttle command |
| `control_state.actuator_commanded` | pre-lag/pre-application actuator commands |
| `aero_state.load_factor_nz` | normal load-factor diagnostic |
| `rl_metrics.*` | residual-supervisor diagnostics |

## Residual-supervisor diagnostics

When the fixed controller is active, `rl_metrics.enabled` is false. When Q-learning or the HJB residual mode is active, `rl_metrics` reports the selected residual and learning diagnostics.

Common Q/residual fields:

- `method`
- `action_index`
- `reward`
- `episode_return`
- `td_error`
- `epsilon`
- `explored`
- `q_state`
- `q_value`
- `max_next_q`
- `updates`
- `residual_active`
- `hard_condition_score`
- `load_factor_nz`
- `safety_risk_score`

HJB residual-specific fields:

- `hjb_value`
- `hjb_advantage`
- `hjb_stage_cost`
- `shield_active`
- `candidate_count`

These diagnostics are emitted so the selected residual is auditable from telemetry.

## Display stability

Backend display yaw is wrapped to `[-180°, 180°]`. The frontend additionally smooths world pose, trace pose, visual aircraft attitude, and camera state. This is visual smoothing only; it does not affect backend control or simulation state.

## Contract maintenance checklist

When changing `simulation_frame`, update these surfaces together:

1. `backend/src/uavsim/server.py` frame construction.
2. `src/simulation/contracts/backendMessages.ts` TypeScript contract.
3. `src/simulation/contracts/telemetryTypes.ts` UI telemetry shape if mapped fields change.
4. `backend/tests/test_uavsim_server.py` schema and runtime tests.
5. This document.
