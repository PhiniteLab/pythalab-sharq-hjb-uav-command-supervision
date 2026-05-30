# Speed, Energy, and Gain Schedule Notes

This document explains how airspeed-dependent gains and high-energy profiles are handled in the simulator.

## Active runtime speed range

The backend uses `GainSchedule` to select/interpolate autopilot gains as airspeed changes. The main schedule points are:

```text
25, 35, 50, 75, 100, 125, 150, 175, 200 m/s
```

The autopilot remains the actuator-facing controller across this range. Residual methods only change command references before the autopilot.

## Compact mission speeds

| Phase | Target speed behavior |
| --- | --- |
| Takeoff/climb | smooth `25 -> 90 m/s` ramp |
| Straight segment | about `70 m/s` |
| Circle/orbit | about `25 m/s` |

## Fight Mode speed and energy decisions

`fight_mode` is a high-energy reference profile used to stress the runtime and visualization. Its backend logic includes:

- elevated base speed target,
- bounded sinusoidal speed variation,
- progress-tied reference instead of open-loop time-only reference,
- altitude command rate limit,
- heading command rate limit,
- throttle floor to reduce zero-throttle energy collapse.

These choices keep the simulation finite and prevent the reference from racing too far ahead of the aircraft during disturbed or low-energy conditions.

## Energy-aware residual behavior

The residual action set includes airspeed and altitude adjustments. In low-energy or high-disturbance conditions:

- `+2 m/s` airspeed residual can ask the autopilot for more energy,
- `-10 m` altitude residual can reduce climb demand and help recovery,
- climb and lateral residuals are restricted when load/saturation risk is high,
- no-op remains available so the supervisor can preserve the baseline.

## Compact waypoint/orbit task

All live profiles except `fight_mode` follow the compact mission style:

```text
runway start at rest
  -> climb to about 200 m
  -> fly a short straight segment
  -> enter a compact orbit or related path
```

## Fight Mode task

```text
runway start at rest
  -> climb to about 240 m
  -> progress-tied S-turn reference
  -> climb/dive altitude corridor
  -> visual-only bank/inverted augmentation in the frontend
```

## Boundaries

- The visual aircraft shell is not the physics model.
- Fight Mode is a simulation stress/visualization profile, not certified aerobatics.
- The residual supervisors do not bypass the fixed/gain-scheduled autopilot.
