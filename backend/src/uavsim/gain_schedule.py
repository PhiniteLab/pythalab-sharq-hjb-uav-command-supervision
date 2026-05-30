"""Airspeed-scheduled autopilot gain bank.

The Simulink-era controller was designed around one trim point.  The live
backend now operates from runway roll through 200 m/s, so runtime code uses a
small set of linearized trim/gain points and linearly interpolates only the
controller/trim feed-forward fields.  The rigid-body physical parameters remain
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import numpy as np

from .parameters import UAVParameters, build_default_parameters

GAIN_SCHEDULE_SPEEDS: Final[tuple[float, ...]] = (25.0, 35.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0)
SCHEDULED_SCALAR_FIELDS: Final[tuple[str, ...]] = (
    "Va",
    "Va0",
    "roll_kp",
    "roll_kd",
    "roll_ki",
    "heading_kp",
    "heading_kd",
    "heading_ki",
    "beta_kp",
    "beta_kd",
    "beta_ki",
    "pitch_kp",
    "pitch_kd",
    "pitch_ki",
    "K_theta_DC",
    "altitude_kp",
    "altitude_kd",
    "altitude_ki",
    "airspeed_pitch_kp",
    "airspeed_pitch_kd",
    "airspeed_pitch_ki",
    "airspeed_throttle_kp",
    "airspeed_throttle_kd",
    "airspeed_throttle_ki",
    "sideslip_kp",
    "sideslip_kd",
    "sideslip_ki",
)


@dataclass(frozen=True)
class GainSchedule:
    speeds: tuple[float, ...]
    points: tuple[UAVParameters, ...]

    def apply_to(self, target: UAVParameters, airspeed_mps: float) -> UAVParameters:
        """Copy/interpolate scheduled gains and trim feed-forward into ``target``."""
        speed = float(np.clip(airspeed_mps, self.speeds[0], self.speeds[-1]))
        hi = int(np.searchsorted(self.speeds, speed, side="right"))
        if hi <= 0:
            lo = hi = 0
            frac = 0.0
        elif hi >= len(self.speeds):
            lo = hi = len(self.speeds) - 1
            frac = 0.0
        else:
            lo = hi - 1
            span = self.speeds[hi] - self.speeds[lo]
            frac = 0.0 if span <= 0.0 else (speed - self.speeds[lo]) / span

        p0 = self.points[lo]
        p1 = self.points[hi]
        for field_name in SCHEDULED_SCALAR_FIELDS:
            v0 = float(getattr(p0, field_name))
            v1 = float(getattr(p1, field_name))
            setattr(target, field_name, v0 + frac * (v1 - v0))
        target.u_trim = p0.u_trim + frac * (p1.u_trim - p0.u_trim)
        target.x_trim = p0.x_trim + frac * (p1.x_trim - p0.x_trim)
        return target


@lru_cache(maxsize=4)
def build_gain_schedule(speeds: tuple[float, ...] = GAIN_SCHEDULE_SPEEDS) -> GainSchedule:
    """Build and cache linearized trim/gain points from low-speed control to 200 m/s.

    The 0 m/s runway start is not a meaningful aerodynamic trim point, so the
    scheduler holds the first valid low-speed trim (25 m/s) until the aircraft
    accelerates into the aerodynamic control range.
    """
    points = tuple(build_default_parameters(compute_trim_and_gains=True, exact_source=True, Va=Va) for Va in speeds)
    return GainSchedule(tuple(float(v) for v in speeds), points)
