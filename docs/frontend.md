# Frontend

The frontend is a Vite + React + TypeScript + Three.js application. It visualizes backend telemetry and provides operator controls for profiles, controller modes, and environment effects.

## Main files

| File | Responsibility |
| --- | --- |
| `src/components/scene/SimpleFlightScene.tsx` | 3D scene, WebSocket bridge, operator/cinematic UI, profile/effect controls, smoothing. |
| `src/components/scene/TrajectoryCharts.tsx` | Downsampled telemetry charts. |
| `src/state/simulationStore.ts` | Zustand store for telemetry history, backend state, and UI mode. |
| `src/simulation/contracts/backendMessages.ts` | TypeScript representation of backend command/frame messages. |
| `src/simulation/contracts/telemetryTypes.ts` | UI telemetry data shape. |
| `vite.config.ts` | Vite config and local backend lifecycle plugin. |

## UI modes

### Operator

Operator mode shows controls and diagnostics:

- profile and controller selection,
- wind/gust/turbulence presets,
- flight instruments,
- mission/reference telemetry,
- system/backend status,
- bounded telemetry charts.

Use this mode when inspecting the algorithm and frame diagnostics.

### Cinematic

Cinematic mode emphasizes the 3D scene:

- larger visual view,
- compact HUD,
- guided chase camera,
- reduced side-panel clutter.

Use this mode when visually checking whether the simulated aircraft, reference trace, and camera remain stable.

## 3D scene components

| Component | Role |
| --- | --- |
| `FlightSimWorld` | Ground, runway, terrain, clouds, trees, and world pose mapping. |
| `GripenModel` | Visual-only STL aircraft shell and control-surface animation. |
| `GripenLoadingPlaceholder` | Simple fallback aircraft if STL loading fails. |
| `AircraftVisualAttitude` | Quaternion-smoothed aircraft attitude. |
| `ReferencePathTrace` | Reference and actual path traces. |
| `StableTraceLine` | Persistent `BufferGeometry` trace renderer. |
| `CinematicCameraRig` | Guided camera in cinematic mode. |
| `BackendTelemetryBridge` | WebSocket command and telemetry path. |

## Telemetry mapping

The frontend receives backend `simulation_frame` messages and converts them into `TelemetryPoint` values for rendering and charts. The backend remains the source of truth. Frontend smoothing does not feed back into the simulator.

Main mapped quantities:

- position and altitude,
- roll/pitch/yaw,
- target and reference positions,
- target airspeed/altitude/heading,
- actuator commands and throttle,
- wind/gust/turbulence diagnostics,
- Q-learning or HJB residual metrics.

## Residual method display

When `online_q_learning` or `sharq_hjb` is selected, the UI can display residual diagnostics coming from `rl_metrics`:

- whether a residual is active,
- selected action index,
- Q state/value/update count,
- reward and TD error,
- hard-condition score,
- HJB residual value, advantage, stage cost, `shield_active` status, and candidate count.

This makes the method observable during live runs.

## Fight Mode visualization

`fight_mode` uses backend reference data plus frontend visual smoothing:

- bank direction is derived from target-heading error,
- climb/dive visual pitch is derived from altitude-command error,
- aircraft attitude uses quaternion slerp,
- the camera follows faster than in normal profiles but remains smoothed,
- world and trace pose are smoothed to avoid 10 Hz telemetry snapping.

The extra visual effects do not change the physics model or controller decisions.

## Performance rules

The scene is designed for bounded memory use:

- keep telemetry history bounded,
- keep chart data downsampled,
- avoid unbounded arrays in render loops,
- reuse `BufferGeometry` and typed arrays where possible,
- dispose Three.js resources when replacing them,
- prefer `InstancedMesh` for repeated terrain/cloud/tree elements.

Run:

```bash
npm run lint
npm run build
```
