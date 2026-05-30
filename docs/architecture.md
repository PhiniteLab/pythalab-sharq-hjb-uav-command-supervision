# Architecture

The repository has two runtime layers: a Python simulation backend and a React/Three.js frontend. The backend owns the physics, controllers, residual supervisors, and telemetry contract. The frontend sends commands and visualizes the telemetry.

## Layer diagram

```text
Browser / React UI
  ├─ profile and controller controls
  ├─ wind/gust/turbulence effect controls
  ├─ operator instruments and charts
  ├─ cinematic camera and HUD
  ├─ Three.js visual aircraft shell
  └─ BackendTelemetryBridge
        │
        │ JSON command messages
        ▼
FastAPI backend (`uavsim.server`)
  ├─ WebSocket command handler
  ├─ Runtime mission state
  ├─ compact waypoint/orbit guidance
  ├─ Fight Mode progress-tied reference
  ├─ residual supervisor: none / Q / HJB residual
  ├─ gain-scheduled fixed autopilot
  ├─ actuator model
  ├─ wind and turbulence model
  ├─ force/moment model
  └─ 12-state RK4 integrator
        │
        │ simulation_frame telemetry
        ▼
Frontend telemetry store and 3D rendering
```

## Runtime data flow

1. The user chooses a profile, controller mode, and environment effects.
2. `BackendTelemetryBridge` sends a JSON command over WebSocket.
3. The backend validates the command, profile, controller label, and effect ranges.
4. `Runtime.step()` builds the nominal command vector:

   ```text
   r = [Va_c, h_c, chi_c]
   ```

5. If residual supervision is active, the Q-learning or HJB residual mode selects one bounded residual:

   ```text
   r_supervised = clip(r + Δr)
   ```

6. The fixed/gain-scheduled autopilot tracks the resulting command.
7. The actuator model applies saturation, rate/lag behavior, and command limits.
8. The force/moment model and 12-state dynamics advance the aircraft state.
9. The backend emits a `simulation_frame` at about 10 Hz.
10. The frontend stores the frame, updates charts, and smooths the 3D render at browser frame rate.

## Coordinate systems

Backend state uses NED coordinates:

```text
pn: north position, positive north
pe: east position, positive east
pd: down position, positive down
h = -pd
```

Frontend Three.js mapping:

```text
inertial/body forward -> scene z
lateral/east          -> scene x
altitude              -> scene y
```

## Guidance profiles

Most live profiles use a compact mission template:

```text
runway start
  -> climb to about 200 m
  -> fly a short straight segment
  -> enter a 200 m-class orbit or related compact path
```

`fight_mode` uses a separate high-energy reference. The reference is tied to aircraft progress instead of only elapsed time:

```text
current_progress = dot(position - anchor, heading_unit)
reference        = path(current_progress + reference_lead)
lookahead        = path(current_progress + lookahead_lead)
heading_c        = line-of-sight(current_position, lookahead)
```

This prevents the reference from running far ahead when the simulated aircraft is slow or disturbed.

## Residual-supervision placement

The residual supervisors sit between guidance and autopilot:

```text
guidance command -> residual supervisor -> fixed autopilot -> actuators -> dynamics
```

They do not command elevator, aileron, rudder, or throttle directly. Their only authority is the small finite residual action set on airspeed, altitude, and heading commands.

## Frontend smoothing model

The backend emits telemetry near 10 Hz. The browser renders near 60 Hz. To avoid visual snapping, the frontend smooths:

- world pose,
- reference and actual trace pose,
- aircraft visual attitude through quaternion interpolation,
- cinematic camera position and field of view,
- chart history through bounded/downsampled arrays.

The smoothing is visual only. It does not feed back into backend physics or controller decisions.

## Known boundaries

- The STL aircraft is a visual shell, not the physics model.
- Fight Mode is a visualization/reference stress profile, not real air combat.
- The actuator-facing controller is still the fixed/gain-scheduled autopilot.
- Residual supervision is finite-action command shaping, not direct actuator RL.
