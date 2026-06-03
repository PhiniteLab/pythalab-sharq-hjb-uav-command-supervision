"""FastAPI WebSocket bridge exposing the converted UAV Simulink autopilot.

The bridge wraps :mod:`uavsim` (a direct Python port of the MATLAB/Simulink
``mavsim_auto`` autopilot) and streams telemetry frames to the React/Three.js
frontend that visualises the SAAB Gripen model.

The fixed-gain Simulink-style autopilot remains the inner-loop baseline.  The
optional ``online_q_learning`` and ``sharq_hjb`` modes add bounded
guidance-residual supervisors with TECS/L1-style diagnostics for experiments.
Flexible-wing/MCK fields are still compatibility placeholders.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from json import JSONDecodeError
from math import atan2, cos, degrees, isfinite, pi, sin, sqrt
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from uavsim import build_default_parameters
from uavsim.actuators import ActuatorState
from uavsim.atmosphere import sample_atmosphere
from uavsim.autopilot import Autopilot
from uavsim.dynamics import mav_derivatives, mav_derivatives_from_forces, rk4_step
from uavsim.forces_moments import air_data, forces_moments
from uavsim.gain_schedule import GainSchedule, build_gain_schedule
from uavsim.guidance import orbit_guidance, straight_path_guidance
from uavsim.parameters import UAVParameters
from uavsim.q_learning import QLearningMetrics, TabularQLearningSupervisor
from uavsim.sharq_hjb import SHARQHJBResidualSupervisor
from uavsim.tecs import TECSController, TECSState
from uavsim.wind import DrydenWind

logger = logging.getLogger(__name__)

# asyncio.TimeoutError is an alias for the built-in TimeoutError on Python 3.11+
# but they are distinct on 3.10 (our declared floor). Catch both.
_WAIT_FOR_TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    TimeoutError,
)

# ---------------------------------------------------------------------------
# Mission / trajectory profiles
# ---------------------------------------------------------------------------

CRUISE_ALTITUDE_M = 100.0
CRUISE_AIRSPEED_MPS = 200.0
TAKEOFF_ACCEL_TIME_S = 40.0
TAKEOFF_CLIMB_RATE_MPS = 4.0
MISSION_AREA_SIZE_M = 10_000.0
COMPACT_TARGET_ALTITUDE_M = 200.0
COMPACT_STRAIGHT_M = 200.0
COMPACT_TAKEOFF_AIRSPEED_MPS = 90.0
COMPACT_STRAIGHT_AIRSPEED_MPS = 70.0
COMPACT_CIRCLE_DIAMETER_M = 200.0
COMPACT_CIRCLE_AIRSPEED_MPS = 25.0
FIGHT_MODE_PROFILE = "fight_mode"
FIGHT_MODE_TARGET_ALTITUDE_M = 240.0
FIGHT_MODE_BASE_AIRSPEED_MPS = 112.0
FIGHT_MODE_LOOKAHEAD_S = 1.6
FIGHT_MODE_REFERENCE_LEAD_M = 120.0
FIGHT_MODE_LOOKAHEAD_LEAD_M = 260.0
FIGHT_MODE_ALTITUDE_RATE_LIMIT_MPS = 9.0
FIGHT_MODE_THROTTLE_FLOOR = 0.34
WAYPOINT_LOOKAHEAD_M = 80.0
CIRCLE_LOOKAHEAD_RAD = 0.25
GUIDANCE_SPEED_RATE_LIMIT_MPS2 = 12.0
WEBSOCKET_FRAME_INTERVAL_S = 0.10
SIMULATION_FRAME_SCHEMA_VERSION = 1
FIXED_CONTROLLER_MODE = "fixed_matlab_autopilot"
Q_LEARNING_CONTROLLER_MODE = "online_q_learning"
SHARQ_HJB_CONTROLLER_MODE = "sharq_hjb"
RESIDUAL_CONTROLLER_MODES = {Q_LEARNING_CONTROLLER_MODE, SHARQ_HJB_CONTROLLER_MODE}

TRAJECTORY_PROFILES = {
    "runway_takeoff_accel_200",
    "takeoff_climbout_200",
    "high_speed_climb_s_turn_200",
    "straight_climb_altitude_hold",
    "figure_eight",
    "racetrack",
    "loiter_orbit",
    FIGHT_MODE_PROFILE,
}
# Every selectable live route is now a short visible mission profile: runway
# start, climb to 200 m, 200 m straight segment, then a 200 m diameter circle.
TAKEOFF_PROFILES = set(TRAJECTORY_PROFILES)

CIRCLE_PROFILE_CONFIG: dict[str, tuple[float, float, float, int]] = {
    # profile: (target climb altitude, circle diameter, circle airspeed, turn direction)
    "runway_takeoff_accel_200": (COMPACT_TARGET_ALTITUDE_M, COMPACT_CIRCLE_DIAMETER_M, COMPACT_CIRCLE_AIRSPEED_MPS, 1),
    "takeoff_climbout_200": (COMPACT_TARGET_ALTITUDE_M, COMPACT_CIRCLE_DIAMETER_M, COMPACT_CIRCLE_AIRSPEED_MPS, 1),
    "high_speed_climb_s_turn_200": (COMPACT_TARGET_ALTITUDE_M, COMPACT_CIRCLE_DIAMETER_M, COMPACT_CIRCLE_AIRSPEED_MPS, 1),
    "straight_climb_altitude_hold": (COMPACT_TARGET_ALTITUDE_M, COMPACT_CIRCLE_DIAMETER_M, COMPACT_CIRCLE_AIRSPEED_MPS, 1),
    "figure_eight": (COMPACT_TARGET_ALTITUDE_M, COMPACT_CIRCLE_DIAMETER_M, COMPACT_CIRCLE_AIRSPEED_MPS, 1),
    "racetrack": (COMPACT_TARGET_ALTITUDE_M, COMPACT_CIRCLE_DIAMETER_M, COMPACT_CIRCLE_AIRSPEED_MPS, 1),
    "loiter_orbit": (COMPACT_TARGET_ALTITUDE_M, COMPACT_CIRCLE_DIAMETER_M, COMPACT_CIRCLE_AIRSPEED_MPS, 1),
    # Cinematic dogfight/aerobatic reference. The backend still uses the
    # fixed-gain autopilot; the frontend adds a visual-only roll augmentation
    # so recordings can show inverted/barrel-roll moments without claiming
    # certified aerobatic control authority.
    FIGHT_MODE_PROFILE: (FIGHT_MODE_TARGET_ALTITUDE_M, 480.0, FIGHT_MODE_BASE_AIRSPEED_MPS, 1),
}


def takeoff_speed_command(t: float) -> float:
    """Smooth low-speed takeoff command used after a 0 m/s runway start."""
    tau = float(np.clip(t / TAKEOFF_ACCEL_TIME_S, 0.0, 1.0))
    smooth = tau * tau * (3.0 - 2.0 * tau)
    return 25.0 + (COMPACT_TAKEOFF_AIRSPEED_MPS - 25.0) * smooth


def takeoff_altitude_command(t: float, target_altitude_m: float = 300.0) -> float:
    if t < 4.0:
        return 0.0
    return min(target_altitude_m, TAKEOFF_CLIMB_RATE_MPS * (t - 4.0))


def circle_heading_command(elapsed_s: float, diameter_m: float, airspeed_mps: float, direction: int = 1) -> float:
    """Heading sweep whose kinematic radius corresponds to the requested circle.

    A 200 m circle is intentionally a visual/autopilot stress test.  The speed
    is held near 25 m/s so the coordinated-turn bank demand stays inside the
    controller's 45° roll-command envelope.
    """
    radius_m = max(diameter_m / 2.0, 1.0)
    heading_rad = float(direction) * (airspeed_mps / radius_m) * max(elapsed_s, 0.0)
    return degrees(heading_rad)


def mission_circle_commands(profile: str, t: float) -> tuple[float, float, float]:
    target_altitude_m, _, _, _ = CIRCLE_PROFILE_CONFIG[profile]
    Va_c = takeoff_speed_command(t)
    h_c = takeoff_altitude_command(t, target_altitude_m)
    chi_c = 0.0
    return Va_c, h_c, chi_c


def reference_commands(profile: str, t: float) -> np.ndarray:
    """Return ``[Va_c, h_c, heading_c_deg]`` for the named reference profile.

    The third slot keeps the historical ``chi_c``/course variable name used by
    the source Simulink model, but the active autopilot closes the loop on yaw
    heading ``psi``. WebSocket frames therefore expose it as
    ``target_heading_deg`` rather than a ground-track/course target.
    """
    if profile in CIRCLE_PROFILE_CONFIG:
        Va_c, h_c, chi_c = mission_circle_commands(profile, t)
    else:
        Va_c = takeoff_speed_command(t)
        h_c = takeoff_altitude_command(t, COMPACT_TARGET_ALTITUDE_M)
        chi_c = 0.0
    return np.array([Va_c, h_c, chi_c], dtype=float)


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to [-π, π] for telemetry/display stability."""
    return atan2(sin(angle_rad), cos(angle_rad))


# ---------------------------------------------------------------------------
# Wind / effect presets
# ---------------------------------------------------------------------------

@dataclass
class WindConfig:
    """Mutable wind/turbulence configuration applied by the frontend."""

    steady_n: float = 0.0
    steady_e: float = 0.0
    steady_d: float = 0.0
    gust_body_u: float = 0.0
    gust_body_v: float = 0.0
    gust_body_w: float = 0.0
    turbulence_std: float = 0.0
    dryden: DrydenWind | None = field(default=None, repr=False)

    def vector(self) -> np.ndarray:
        # Steady wind expressed in inertial NED, gust in body frame.
        if self.dryden is not None and self.turbulence_std > 1e-6:
            gust = self.dryden.gust() * self.turbulence_std
        else:
            gust = np.zeros(3)
        gust = gust + np.array([self.gust_body_u, self.gust_body_v, self.gust_body_w], dtype=float)
        return np.array(
            [self.steady_n, self.steady_e, self.steady_d, gust[0], gust[1], gust[2]],
            dtype=float,
        )


def apply_effects(wind: WindConfig, effects: dict[str, Any]) -> None:
    """Clamp and copy operator-selectable wind/gust settings."""

    allowed: dict[str, tuple[float, float, str]] = {
        "steady_wind_n": (-10.0, 10.0, "steady_n"),
        "steady_wind_e": (-10.0, 10.0, "steady_e"),
        "steady_wind_d": (-4.0, 4.0, "steady_d"),
        "gust_body_u": (-4.0, 4.0, "gust_body_u"),
        "gust_body_v": (-4.0, 4.0, "gust_body_v"),
        "gust_body_w": (-4.0, 4.0, "gust_body_w"),
        "turbulence_std": (0.0, 2.0, "turbulence_std"),
    }
    for key, (lo, hi, attr) in allowed.items():
        if key in effects:
            try:
                value = float(effects[key])
            except (TypeError, ValueError):
                continue
            if not isfinite(value):
                continue
            setattr(wind, attr, float(np.clip(value, lo, hi)))


# ---------------------------------------------------------------------------
# Simulation runtime
# ---------------------------------------------------------------------------

@dataclass
class Runtime:
    P: UAVParameters
    autopilot: Autopilot
    wind: WindConfig
    x: np.ndarray
    gain_schedule: GainSchedule
    actuators: ActuatorState = field(default_factory=ActuatorState)
    tecs: TECSController = field(default_factory=TECSController)
    q_learner: TabularQLearningSupervisor = field(default_factory=TabularQLearningSupervisor)
    sharq_hjb_learner: SHARQHJBResidualSupervisor = field(default_factory=SHARQHJBResidualSupervisor)
    t: float = 0.0
    profile: str = "loiter_orbit"
    controller_mode: str = "fixed_matlab_autopilot"
    substeps: int = 5
    initial_altitude_m: float = CRUISE_ALTITUDE_M
    reference_n: float = 0.0
    reference_e: float = 0.0
    reference_circle_started: bool = False
    mission_straight_start_n: float | None = None
    mission_straight_start_e: float | None = None
    mission_straight_start_time_s: float | None = None
    mission_straight_unit_n: float = 1.0
    mission_straight_unit_e: float = 0.0
    mission_circle_elapsed_s: float = 0.0
    mission_circle_heading0_deg: float = 0.0
    mission_circle_center_n: float | None = None
    mission_circle_center_e: float | None = None
    mission_circle_phase_rad: float = 0.0
    mission_speed_command_mps: float = 25.0
    mission_profile_config_override: tuple[float, float, float, int] | None = None
    mission_straight_length_m: float = COMPACT_STRAIGHT_M
    mission_waypoint_lookahead_m: float = WAYPOINT_LOOKAHEAD_M
    mission_heading_command_deg: float | None = None
    mission_altitude_command_m: float | None = None
    guidance_cross_track_error_m: float = 0.0
    guidance_radial_error_m: float = 0.0
    guidance_lookahead_m: float = 0.0
    guidance_bearing_error_rad: float = 0.0
    guidance_lateral_accel_mps2: float = 0.0
    guidance_roll_command_rad: float = 0.0
    previous_load_factor_nz: float = 1.0
    fight_mode_start_time_s: float | None = None
    fight_mode_anchor_n: float = 0.0
    fight_mode_anchor_e: float = 0.0
    fight_mode_heading_rad: float = 0.0

    @classmethod
    def create(cls, profile: str = "loiter_orbit") -> "Runtime":
        active_profile = profile if profile in TRAJECTORY_PROFILES else "loiter_orbit"
        P = build_default_parameters(compute_trim_and_gains=True, exact_source=True, Va=CRUISE_AIRSPEED_MPS)
        schedule = build_gain_schedule()
        x0 = P.initial_state().copy()
        x0[2] = -CRUISE_ALTITUDE_M
        if active_profile in TAKEOFF_PROFILES:
            x0[:] = 0.0
            x0[2] = 0.0
        wind = WindConfig(dryden=DrydenWind(P, enabled=True))
        rt = cls(
            P=P,
            autopilot=Autopilot(P),
            wind=wind,
            x=x0,
            gain_schedule=schedule,
            profile=active_profile,
            initial_altitude_m=0.0 if active_profile in TAKEOFF_PROFILES else CRUISE_ALTITUDE_M,
        )
        return rt

    def reset(self) -> None:
        self.autopilot.reset()
        self.actuators.reset()
        self.tecs.reset()
        self.q_learner.reset()
        self.sharq_hjb_learner.reset()
        if self.profile in TAKEOFF_PROFILES:
            self.x = np.zeros(12, dtype=float)
            self.x[2] = 0.0
        else:
            self.x = self.P.initial_state().copy()
            self.x[2] = -self.initial_altitude_m
        self.t = 0.0
        self.reference_n = 0.0
        self.reference_e = 0.0
        self.reference_circle_started = False
        self.mission_straight_start_n = None
        self.mission_straight_start_e = None
        self.mission_straight_start_time_s = None
        self.mission_straight_unit_n = 1.0
        self.mission_straight_unit_e = 0.0
        self.mission_circle_elapsed_s = 0.0
        self.mission_circle_heading0_deg = 0.0
        self.mission_circle_center_n = None
        self.mission_circle_center_e = None
        self.mission_circle_phase_rad = 0.0
        self.mission_speed_command_mps = 25.0
        self.mission_straight_length_m = max(float(self.mission_straight_length_m), 1.0)
        self.mission_waypoint_lookahead_m = max(float(self.mission_waypoint_lookahead_m), 1.0)
        self.mission_heading_command_deg = None
        self.mission_altitude_command_m = None
        self.guidance_cross_track_error_m = 0.0
        self.guidance_radial_error_m = 0.0
        self.guidance_lookahead_m = 0.0
        self.guidance_bearing_error_rad = 0.0
        self.guidance_lateral_accel_mps2 = 0.0
        self.guidance_roll_command_rad = 0.0
        self.previous_load_factor_nz = 1.0
        self.fight_mode_start_time_s = None
        self.fight_mode_anchor_n = 0.0
        self.fight_mode_anchor_e = 0.0
        self.fight_mode_heading_rad = 0.0
        if self.wind.dryden is not None:
            self.wind.dryden.reset()

    def active_circle_profile_config(self) -> tuple[float, float, float, int]:
        """Return the current mission reference envelope.

        Live UI profiles use ``CIRCLE_PROFILE_CONFIG``.  Offline experiment
        benchmarks can install a per-runtime override so the same controller
        code is exercised under richer reference families without changing the
        default browser experience.
        """

        if self.mission_profile_config_override is not None:
            target_altitude_m, circle_diameter_m, circle_airspeed_mps, direction = self.mission_profile_config_override
            return (
                float(target_altitude_m),
                float(max(circle_diameter_m, 1.0)),
                float(max(circle_airspeed_mps, 1.0)),
                1 if int(direction) >= 0 else -1,
            )
        target_altitude_m, circle_diameter_m, circle_airspeed_mps, direction = CIRCLE_PROFILE_CONFIG.get(
            self.profile,
            (COMPACT_TARGET_ALTITUDE_M, COMPACT_CIRCLE_DIAMETER_M, COMPACT_CIRCLE_AIRSPEED_MPS, 1),
        )
        return float(target_altitude_m), float(circle_diameter_m), float(circle_airspeed_mps), int(direction)

    def _rate_limited_airspeed_command(self, target_mps: float) -> float:
        """Slew outer-loop airspeed commands to avoid mission-phase steps."""
        if self.t <= 0.0:
            self.mission_speed_command_mps = float(target_mps)
            return float(target_mps)
        max_delta = GUIDANCE_SPEED_RATE_LIMIT_MPS2 * self.P.Ts
        delta = float(np.clip(target_mps - self.mission_speed_command_mps, -max_delta, max_delta))
        self.mission_speed_command_mps += delta
        return self.mission_speed_command_mps

    def _line_of_sight_heading_deg(self, target_n: float, target_e: float) -> float:
        """Return waypoint line-of-sight heading from current position."""
        dn = target_n - float(self.x[0])
        de = target_e - float(self.x[1])
        if abs(dn) < 1e-9 and abs(de) < 1e-9:
            return degrees(float(self.x[8]))
        return degrees(atan2(de, dn))

    def _rate_limited_heading_command(self, target_deg: float, max_rate_deg_s: float = 70.0) -> float:
        """Slew a heading command through the shortest angular path."""
        target = ((float(target_deg) + 180.0) % 360.0) - 180.0
        if self.mission_heading_command_deg is None or self.t <= 0.0:
            self.mission_heading_command_deg = target
            return target
        current = self.mission_heading_command_deg
        error = ((target - current + 180.0) % 360.0) - 180.0
        max_delta = max_rate_deg_s * self.P.Ts
        current = ((current + float(np.clip(error, -max_delta, max_delta)) + 180.0) % 360.0) - 180.0
        self.mission_heading_command_deg = current
        return current

    def _rate_limited_altitude_command(self, target_m: float, max_rate_mps: float) -> float:
        """Slew altitude commands so dogfight dives/climbs do not cause energy snaps."""
        target = float(target_m)
        if self.mission_altitude_command_m is None or self.t <= 0.0:
            self.mission_altitude_command_m = target
            return target
        max_delta = max_rate_mps * self.P.Ts
        delta = float(np.clip(target - self.mission_altitude_command_m, -max_delta, max_delta))
        self.mission_altitude_command_m += delta
        return self.mission_altitude_command_m

    def _fight_reference_point(self, progress_m: float) -> tuple[float, float, float]:
        """Cinematic dogfight reference point in inertial N/E/altitude.

        The shape is a bounded Lissajous/S-turn corridor ahead of the runway:
        it creates aggressive-looking reference curvature and altitude changes
        while keeping the fixed-gain autopilot inside a finite smoke envelope.
        """
        forward_m = max(float(progress_m), 0.0)
        t = forward_m / max(FIGHT_MODE_BASE_AIRSPEED_MPS, 1.0)
        lateral_m = 150.0 * sin(0.34 * t) + 38.0 * sin(0.78 * t)
        altitude_m = FIGHT_MODE_TARGET_ALTITUDE_M + 82.0 * sin(0.22 * t) + 34.0 * sin(0.49 * t)
        c = cos(self.fight_mode_heading_rad)
        s = sin(self.fight_mode_heading_rad)
        n = self.fight_mode_anchor_n + forward_m * c - lateral_m * s
        e = self.fight_mode_anchor_e + forward_m * s + lateral_m * c
        return n, e, float(np.clip(altitude_m, 120.0, 390.0))

    def fight_mode_commands(self) -> np.ndarray:
        """High-energy cinematic reference for the Fight Mode profile."""
        altitude = -float(self.x[2])
        if altitude < FIGHT_MODE_TARGET_ALTITUDE_M - 12.0 and self.fight_mode_start_time_s is None:
            self.reference_n = float(self.x[0])
            self.reference_e = float(self.x[1])
            Va_c = self._rate_limited_airspeed_command(takeoff_speed_command(self.t))
            return np.array([Va_c, FIGHT_MODE_TARGET_ALTITUDE_M, 0.0], dtype=float)

        if self.fight_mode_start_time_s is None:
            self.fight_mode_start_time_s = self.t
            self.fight_mode_anchor_n = float(self.x[0])
            self.fight_mode_anchor_e = float(self.x[1])
            self.fight_mode_heading_rad = float(self.x[8])
            self.reference_circle_started = True
            self.mission_circle_elapsed_s = 0.0
            self.mission_circle_center_n = self.fight_mode_anchor_n
            self.mission_circle_center_e = self.fight_mode_anchor_e

        c = cos(self.fight_mode_heading_rad)
        s = sin(self.fight_mode_heading_rad)
        current_progress = max(
            (float(self.x[0]) - self.fight_mode_anchor_n) * c + (float(self.x[1]) - self.fight_mode_anchor_e) * s,
            0.0,
        )
        ref_n, ref_e, ref_h = self._fight_reference_point(current_progress + FIGHT_MODE_REFERENCE_LEAD_M)
        lookahead_n, lookahead_e, _ = self._fight_reference_point(current_progress + FIGHT_MODE_LOOKAHEAD_LEAD_M)
        self.reference_n = ref_n
        self.reference_e = ref_e
        raw_heading_c = self._line_of_sight_heading_deg(lookahead_n, lookahead_e)
        heading_c = self._rate_limited_heading_command(raw_heading_c, max_rate_deg_s=58.0)
        path_t = (current_progress + FIGHT_MODE_REFERENCE_LEAD_M) / max(FIGHT_MODE_BASE_AIRSPEED_MPS, 1.0)
        speed_target = FIGHT_MODE_BASE_AIRSPEED_MPS + 16.0 * (0.5 + 0.5 * sin(0.18 * path_t))
        Va_c = self._rate_limited_airspeed_command(speed_target)
        h_c = self._rate_limited_altitude_command(ref_h, FIGHT_MODE_ALTITUDE_RATE_LIMIT_MPS)
        return np.array([Va_c, h_c, heading_c], dtype=float)

    def mission_commands(self) -> np.ndarray:
        """Waypoint-followed mission: climb, straight segment, then compact orbit.

        The low-level autopilot still accepts ``[airspeed, altitude, heading]``.
        This method is the outer guidance loop: it creates moving geometric
        waypoints/reference points from the live aircraft position instead of
        replaying an open-loop heading sweep.  The reference line streamed to
        the frontend is therefore the same path the controller is trying to
        follow.
        """
        if self.profile == FIGHT_MODE_PROFILE:
            return self.fight_mode_commands()

        target_altitude_m, circle_diameter_m, circle_airspeed_mps, direction = self.active_circle_profile_config()
        altitude = -float(self.x[2])
        if altitude < target_altitude_m - 5.0 and self.mission_straight_start_n is None:
            self.reference_n = float(self.x[0])
            self.reference_e = float(self.x[1])
            Va_c = self._rate_limited_airspeed_command(takeoff_speed_command(self.t))
            return np.array([Va_c, target_altitude_m, 0.0], dtype=float)

        if self.mission_straight_start_n is None:
            self.mission_straight_start_n = float(self.x[0])
            self.mission_straight_start_e = float(self.x[1])
            self.mission_straight_start_time_s = self.t
            self.mission_straight_unit_n = cos(float(self.x[8]))
            self.mission_straight_unit_e = sin(float(self.x[8]))
        straight_start_n = self.mission_straight_start_n
        straight_start_e = self.mission_straight_start_e
        if straight_start_n is None or straight_start_e is None:  # pragma: no cover - defensive narrowing
            straight_start_n = float(self.x[0])
            straight_start_e = float(self.x[1])

        straight_dn = float(self.x[0]) - straight_start_n
        straight_de = float(self.x[1]) - straight_start_e
        straight_progress = max(
            straight_dn * self.mission_straight_unit_n + straight_de * self.mission_straight_unit_e,
            0.0,
        )
        straight_length_m = max(float(self.mission_straight_length_m), 1.0)
        reference_progress = min(straight_progress, straight_length_m)
        self.reference_n = straight_start_n + reference_progress * self.mission_straight_unit_n
        self.reference_e = straight_start_e + reference_progress * self.mission_straight_unit_e
        straight_elapsed_s = self.t - (self.mission_straight_start_time_s if self.mission_straight_start_time_s is not None else self.t)
        if not self.reference_circle_started and (straight_progress < straight_length_m or straight_elapsed_s < 8.0):
            lookahead_progress = min(straight_progress + self.mission_waypoint_lookahead_m, straight_length_m)
            guidance = straight_path_guidance(
                pn=float(self.x[0]),
                pe=float(self.x[1]),
                start_n=straight_start_n,
                start_e=straight_start_e,
                unit_n=self.mission_straight_unit_n,
                unit_e=self.mission_straight_unit_e,
                lookahead_m=max(lookahead_progress - straight_progress, 1.0),
                ground_speed_mps=max(float(self.x[3]), 1.0),
            )
            self.guidance_cross_track_error_m = guidance.cross_track_error_m
            self.guidance_radial_error_m = 0.0
            self.guidance_lookahead_m = guidance.lookahead_m
            self.guidance_bearing_error_rad = guidance.bearing_error_rad
            self.guidance_lateral_accel_mps2 = guidance.lateral_accel_mps2
            self.guidance_roll_command_rad = guidance.roll_command_rad
            heading_c = guidance.heading_deg
            Va_c = self._rate_limited_airspeed_command(COMPACT_STRAIGHT_AIRSPEED_MPS)
            return np.array([Va_c, target_altitude_m, heading_c], dtype=float)

        if not self.reference_circle_started:
            self.reference_circle_started = True
            self.mission_circle_elapsed_s = 0.0
            self.mission_circle_heading0_deg = degrees(float(self.x[8]))
            radius_m = max(circle_diameter_m / 2.0, 1.0)
            heading_rad = float(self.x[8])
            turn_dir = 1.0 if direction >= 0 else -1.0
            # Center the orbit on the commanded turn side of the aircraft.
            self.mission_circle_center_n = float(self.x[0]) - turn_dir * radius_m * sin(heading_rad)
            self.mission_circle_center_e = float(self.x[1]) + turn_dir * radius_m * cos(heading_rad)
            self.mission_circle_phase_rad = atan2(
                float(self.x[1]) - self.mission_circle_center_e,
                float(self.x[0]) - self.mission_circle_center_n,
            )

        radius_m = max(circle_diameter_m / 2.0, 1.0)
        center_n = self.mission_circle_center_n
        center_e = self.mission_circle_center_e
        if center_n is None or center_e is None:  # pragma: no cover - defensive recovery
            center_n = float(self.x[0])
            center_e = float(self.x[1]) + radius_m
            self.mission_circle_center_n = center_n
            self.mission_circle_center_e = center_e
        radial_n = float(self.x[0]) - center_n
        radial_e = float(self.x[1]) - center_e
        if abs(radial_n) < 1e-9 and abs(radial_e) < 1e-9:
            phase = self.mission_circle_phase_rad
        else:
            phase = atan2(radial_e, radial_n)
            self.mission_circle_phase_rad = phase
        guidance = orbit_guidance(
            pn=float(self.x[0]),
            pe=float(self.x[1]),
            center_n=center_n,
            center_e=center_e,
            radius_m=radius_m,
            direction=direction,
            ground_speed_mps=max(float(self.x[3]), 1.0),
        )
        self.reference_n = guidance.reference_n
        self.reference_e = guidance.reference_e
        self.guidance_cross_track_error_m = 0.0
        self.guidance_radial_error_m = guidance.radial_error_m
        self.guidance_lookahead_m = guidance.lookahead_m
        self.guidance_bearing_error_rad = guidance.bearing_error_rad
        self.guidance_lateral_accel_mps2 = guidance.lateral_accel_mps2
        self.guidance_roll_command_rad = guidance.roll_command_rad
        heading_c = guidance.heading_deg
        Va_c = self._rate_limited_airspeed_command(circle_airspeed_mps)
        return np.array([Va_c, target_altitude_m, heading_c], dtype=float)

    def step(self) -> dict[str, Any]:
        """Advance the autopilot + dynamics by one autopilot sample period."""
        P = self.P
        Ts = P.Ts
        commands = self.mission_commands() if self.profile in TAKEOFF_PROFILES else reference_commands(self.profile, self.t)
        wind_vec = self.wind.vector()
        Va, _, _, _ = air_data(self.x, wind_vec, P, exact_source=True)
        altitude = -float(self.x[2])
        wind_speed = sqrt(float(wind_vec[0]) ** 2 + float(wind_vec[1]) ** 2 + float(wind_vec[2]) ** 2 + float(wind_vec[3]) ** 2 + float(wind_vec[4]) ** 2 + float(wind_vec[5]) ** 2)
        reference_error = sqrt((float(self.x[0]) - self.reference_n) ** 2 + (float(self.x[1]) - self.reference_e) ** 2)
        q_metrics = QLearningMetrics(enabled=False, method="fixed_baseline")
        if self.controller_mode == Q_LEARNING_CONTROLLER_MODE:
            commands, q_metrics = self.q_learner.begin_step(
                commands,
                airspeed_error=float(commands[0]) - Va,
                altitude_error=float(commands[1]) - altitude,
                reference_error=reference_error,
                cross_track_error=self.guidance_cross_track_error_m,
                radial_error=self.guidance_radial_error_m,
                wind_speed=wind_speed,
                turbulence_std=self.wind.turbulence_std,
                saturation_ratio=self.actuators.saturation_ratio(),
                load_factor_nz=self.previous_load_factor_nz,
            )
        elif self.controller_mode == SHARQ_HJB_CONTROLLER_MODE:
            commands, q_metrics = self.sharq_hjb_learner.begin_step(
                commands,
                airspeed_error=float(commands[0]) - Va,
                altitude_error=float(commands[1]) - altitude,
                reference_error=reference_error,
                cross_track_error=self.guidance_cross_track_error_m,
                radial_error=self.guidance_radial_error_m,
                wind_speed=wind_speed,
                turbulence_std=self.wind.turbulence_std,
                saturation_ratio=self.actuators.saturation_ratio(),
                load_factor_nz=self.previous_load_factor_nz,
            )
        self.gain_schedule.apply_to(P, max(Va, 25.0))
        disturbance_energy = wind_speed + 2.0 * float(self.wind.turbulence_std)
        tecs_active = self.controller_mode in RESIDUAL_CONTROLLER_MODES and (
            bool(q_metrics.residual_active) or disturbance_energy >= 4.0
        )
        if tecs_active:
            tecs_state = self.tecs.update(
                altitude_m=altitude,
                airspeed_mps=Va,
                target_altitude_m=float(commands[1]),
                target_airspeed_mps=float(commands[0]),
                trim_throttle=float(P.u_trim[3]),
                trim_pitch_rad=float(P.x_trim[7]),
            )
            pitch_override = tecs_state.pitch_command_rad
        else:
            tecs_state = TECSState()
            pitch_override = None
        y_ap = self.autopilot.update(self.x, commands, self.t, measured_airspeed=Va, pitch_command_override_rad=pitch_override)
        delta_cmd = y_ap[:4]
        delta_cmd = delta_cmd.copy()
        if tecs_active:
            delta_cmd[3] = max(float(delta_cmd[3]), tecs_state.throttle_command)
            delta_cmd[0] = float(delta_cmd[0]) + tecs_state.elevator_bias_rad
        if self.profile == FIGHT_MODE_PROFILE and self.fight_mode_start_time_s is not None:
            # Preserve energy during cinematic vertical manoeuvres. The fixed
            # altitude/throttle state machine can otherwise cut throttle to
            # zero during commanded dives, causing visible speed collapses.
            delta_cmd[3] = max(float(delta_cmd[3]), FIGHT_MODE_THROTTLE_FLOOR)
        delta = self.actuators.update(delta_cmd, Ts)
        x_command = y_ap[4:16]

        # RK4 sub-step integration with zero-order-hold actuator/wind.
        substeps = max(1, int(self.substeps))
        dt = Ts / substeps

        def _f(_tt: float, xx: np.ndarray) -> np.ndarray:
            return mav_derivatives(xx, delta, wind_vec, P, exact_source=True)

        x_next = self.x.copy()
        for j in range(substeps):
            tj = self.t + j * dt
            x_next = rk4_step(_f, x_next, tj, dt)
            if not np.all(np.isfinite(x_next)):
                raise FloatingPointError(f"Non-finite state at t={tj:.3f}: {x_next}")
        if self.profile in TAKEOFF_PROFILES and x_next[2] > 0.0 and x_next[3] < P.takeoff_liftoff_speed:
            # Simple runway contact model for the pre-rotation roll: keep the
            # aircraft on the runway and prevent the free 6-DOF body from
            # falling below ground before aerodynamic lift is available.
            x_next[2] = 0.0
            x_next[5] = min(0.0, x_next[5])
            x_next[6] = 0.0
            x_next[7] = max(0.0, min(x_next[7], 3.0 * pi / 180.0))
            x_next[9:12] = 0.0
        self.x = x_next
        self.t += Ts
        if self.reference_circle_started:
            self.mission_circle_elapsed_s += Ts
        reference_position = np.array([self.reference_n, self.reference_e, float(commands[1])], dtype=float)

        # Build frames from one coherent sample: after integration, recompute
        # state-dependent air data / forces / derivatives using the held wind
        # and actuator command that produced the new state. This avoids mixing
        # post-step position/attitude with pre-step airspeed/AoA in WebSocket
        # telemetry consumed by the UI and experiment logs.
        Va_post, alpha_post, beta_post, wind_body_post = air_data(self.x, wind_vec, P, exact_source=True)
        fm_post = forces_moments(self.x, delta, wind_vec, P, exact_source=True)
        dx_post = mav_derivatives_from_forces(self.x, fm_post[:3], fm_post[3:], P, exact_source=True)
        load_factor_nz = float(np.clip(1.0 - dx_post[5] / P.gravity, -8.0, 8.0))
        atmosphere = sample_atmosphere(P.rho, -float(self.x[2]), Va_post)
        saturation_ratio = self.actuators.saturation_ratio()
        if self.controller_mode == Q_LEARNING_CONTROLLER_MODE:
            q_metrics = self.q_learner.end_step(
                airspeed_error=float(commands[0]) - Va_post,
                altitude_error=float(commands[1]) + float(self.x[2]),
                reference_error=sqrt((float(self.x[0]) - self.reference_n) ** 2 + (float(self.x[1]) - self.reference_e) ** 2),
                saturation_ratio=saturation_ratio,
                load_factor_nz=load_factor_nz,
                cross_track_error=self.guidance_cross_track_error_m,
                radial_error=self.guidance_radial_error_m,
                wind_speed=wind_speed,
                turbulence_std=self.wind.turbulence_std,
                altitude_m=-float(self.x[2]),
                time_s=self.t,
            )
        elif self.controller_mode == SHARQ_HJB_CONTROLLER_MODE:
            q_metrics = self.sharq_hjb_learner.end_step(
                airspeed_error=float(commands[0]) - Va_post,
                altitude_error=float(commands[1]) + float(self.x[2]),
                reference_error=sqrt((float(self.x[0]) - self.reference_n) ** 2 + (float(self.x[1]) - self.reference_e) ** 2),
                saturation_ratio=saturation_ratio,
                load_factor_nz=load_factor_nz,
                cross_track_error=self.guidance_cross_track_error_m,
                radial_error=self.guidance_radial_error_m,
                wind_speed=wind_speed,
                turbulence_std=self.wind.turbulence_std,
                altitude_m=-float(self.x[2]),
                time_s=self.t,
            )
        self.previous_load_factor_nz = load_factor_nz

        return {
            "commands": commands,
            "delta": delta,
            "delta_cmd": delta_cmd,
            "x_command": x_command,
            "wind": wind_vec,
            "forces_moments": fm_post,
            "derivatives": dx_post,
            "Va": float(Va_post),
            "alpha": float(alpha_post),
            "beta": float(beta_post),
            "wind_body": wind_body_post.astype(float),
            "load_factor_nz": load_factor_nz,
            "atmosphere": atmosphere,
            "tecs": tecs_state,
            "q_learning": q_metrics,
            "guidance_cross_track_error_m": self.guidance_cross_track_error_m,
            "guidance_radial_error_m": self.guidance_radial_error_m,
            "guidance_lookahead_m": self.guidance_lookahead_m,
            "guidance_bearing_error_rad": self.guidance_bearing_error_rad,
            "guidance_lateral_accel_mps2": self.guidance_lateral_accel_mps2,
            "guidance_roll_command_rad": self.guidance_roll_command_rad,
            "altitude_state": int(self.autopilot.altitude_state if self.autopilot.altitude_state is not None else -1),
            "reference_position": reference_position,
        }


# ---------------------------------------------------------------------------
# Frame construction (matches the existing frontend contract)
# ---------------------------------------------------------------------------

def _flight_phase(altitude_m: float, altitude_state: int) -> str:
    if altitude_state == 1 or altitude_m <= 1.5:
        return "ground_roll"
    return "airborne"


def _flight_mode_label(altitude_state: int) -> str:
    return {
        1: "takeoff",
        2: "climb",
        3: "descend",
        4: "altitude_hold",
    }.get(altitude_state, "airborne")


def build_simulation_frame(
    *,
    rt: Runtime,
    info: dict[str, Any],
    episode: int,
    scenario: str,
    controller: str,
) -> dict[str, Any]:
    x = rt.x
    pn, pe, pd = float(x[0]), float(x[1]), float(x[2])
    phi, theta, psi = float(x[6]), float(x[7]), float(x[8])
    display_psi = wrap_angle_rad(psi)
    q = float(x[10])
    altitude = -pd
    delta = info["delta"]
    delta_cmd = info.get("delta_cmd", delta)
    delta_e, delta_a, delta_r, delta_t = (float(delta[0]), float(delta[1]), float(delta[2]), float(delta[3]))
    delta_e_cmd, delta_a_cmd, delta_r_cmd, delta_t_cmd = (
        float(delta_cmd[0]),
        float(delta_cmd[1]),
        float(delta_cmd[2]),
        float(delta_cmd[3]),
    )
    Va = info["Va"]
    atmosphere = info.get("atmosphere", sample_atmosphere(rt.P.rho, altitude, Va))
    tecs_state = info.get("tecs", TECSState())
    q_metrics = info.get("q_learning", QLearningMetrics(enabled=False))
    x_command = info["x_command"]
    target_altitude = float(-x_command[2])
    target_airspeed = float(x_command[3])
    target_pitch_rad = float(x_command[7])
    target_roll_rad = float(x_command[6])
    target_heading_rad = float(x_command[8])
    reference_position = np.asarray(info.get("reference_position", np.array([pn, pe, target_altitude])), dtype=float)
    reference_n = float(reference_position[0])
    reference_e = float(reference_position[1])
    reference_altitude = float(reference_position[2])
    horizontal_reference_error = sqrt((pn - reference_n) ** 2 + (pe - reference_e) ** 2)
    distance_to_reference = sqrt(horizontal_reference_error ** 2 + (altitude - reference_altitude) ** 2)
    _, circle_diameter_m, circle_airspeed_mps, circle_direction = rt.active_circle_profile_config()
    circle_start_time_s = max(0.0, rt.t - rt.mission_circle_elapsed_s) if rt.reference_circle_started else 0.0
    wind_body = info["wind_body"]
    wind_body_vec = [float(wind_body[0]), float(wind_body[1]), float(wind_body[2])]
    wind_speed = sqrt(sum(c * c for c in wind_body_vec))
    wind_direction = [c / wind_speed for c in wind_body_vec] if wind_speed > 1e-6 else [0.0, 0.0, 0.0]
    altitude_state = info["altitude_state"]
    flight_mode = _flight_mode_label(altitude_state)
    flight_phase = _flight_phase(altitude, altitude_state)

    # Elevon / aileron mapping for the visual model.
    elevator_deg = degrees(delta_e)
    rudder_deg = degrees(delta_r)
    aileron_deg = degrees(delta_a)
    left_aileron_deg = aileron_deg
    right_aileron_deg = -aileron_deg
    actuator_commands = [
        left_aileron_deg,
        right_aileron_deg,
        elevator_deg,
        rudder_deg,
        delta_t * 100.0,
    ]
    saturation_ratio = min(
        1.0,
        max(
            abs(delta_e) / (45.0 * pi / 180.0),
            abs(delta_a) / (45.0 * pi / 180.0),
            abs(delta_r) / (30.0 * pi / 180.0),
            abs(delta_t),
        ),
    )
    roll_command_saturation_ratio = min(1.0, abs(target_roll_rad) / (45.0 * pi / 180.0))
    guidance_saturation_ratio = max(saturation_ratio, roll_command_saturation_ratio)

    # Flight-path angle from body velocity + Euler (inertial up positive).
    u, v, w = float(x[3]), float(x[4]), float(x[5])
    cphi, sphi = cos(phi), sin(phi)
    ctheta, stheta = cos(theta), sin(theta)
    cpsi, spsi = cos(psi), sin(psi)
    vn = ctheta * cpsi * u + (sphi * stheta * cpsi - cphi * spsi) * v + (cphi * stheta * cpsi + sphi * spsi) * w
    ve = ctheta * spsi * u + (sphi * stheta * spsi + cphi * cpsi) * v + (cphi * stheta * spsi - sphi * cpsi) * w
    vd = -stheta * u + sphi * ctheta * v + cphi * ctheta * w
    horizontal_speed = max(sqrt(vn * vn + ve * ve), 1e-6)
    flight_path_angle_deg = degrees(atan2(-vd, horizontal_speed))

    return {
        "type": "simulation_frame",
        "schema_version": SIMULATION_FRAME_SCHEMA_VERSION,
        "timestamp": rt.t,
        "episode": episode,
        "controller": controller,
        "scenario": scenario,
        "profile": rt.profile,
        "uav_state": {
            "position": [pn, pe, altitude],
            "attitude": [degrees(phi), degrees(theta), degrees(display_psi)],
            "pitch_rate_deg_s": degrees(q),
            "airspeed": Va,
            "mach": atmosphere.mach,
            "flight_mode": flight_mode,
        },
        "reference_state": {
            "autopilot": controller,
            "target_altitude": target_altitude,
            "altitude_error": target_altitude - altitude,
            "target_airspeed": target_airspeed,
            "airspeed_error": target_airspeed - Va,
            "target_pitch_deg": degrees(target_pitch_rad),
            "target_roll_deg": degrees(target_roll_rad),
            "target_heading_deg": degrees(target_heading_rad),
            "target_lateral_offset": 0.0,
            "target_throttle": delta_t,
            "reference_position_n": reference_n,
            "reference_position_e": reference_e,
            "reference_altitude": reference_altitude,
            "horizontal_reference_error": horizontal_reference_error,
            "distance_to_reference": distance_to_reference,
            "distance_error": distance_to_reference,
            "mission_area_size_m": MISSION_AREA_SIZE_M,
            "circle_diameter_m": circle_diameter_m,
            "circle_radius_m": circle_diameter_m / 2.0,
            "circle_airspeed_mps": circle_airspeed_mps,
            "circle_direction": circle_direction,
            "circle_start_time_s": circle_start_time_s,
            "trajectory_profile": rt.profile,
        },
        # Rigid-body model: no flexible-wing data. Frontend tolerates zeros.
        "wing_state": {
            "tip_deflection": 0.0,
            "max_deflection": 0.0,
            "average_strain": 0.0,
            "twist_angle": 0.0,
            "node_displacements": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "aero_state": {
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "wind_body": wind_body_vec,
            "turbulence_intensity": float(min(1.0, rt.wind.turbulence_std)),
            "gust_level": float(
                sqrt(
                    rt.wind.gust_body_u ** 2
                    + rt.wind.gust_body_v ** 2
                    + rt.wind.gust_body_w ** 2
                )
            ),
            "angle_of_attack_deg": degrees(float(info["alpha"])),
            "sideslip_deg": degrees(float(info["beta"])),
            "flight_path_angle_deg": flight_path_angle_deg,
            "load_factor_nz": float(info.get("load_factor_nz", 1.0)),
            "density_kg_m3": atmosphere.density_kg_m3,
            "dynamic_pressure_pa": atmosphere.dynamic_pressure_pa,
            "speed_of_sound_mps": atmosphere.speed_of_sound_mps,
        },
        "control_state": {
            "control_energy": delta_e * delta_e + delta_a * delta_a + delta_r * delta_r + delta_t * 0.1,
            "actuator_commands": actuator_commands,
            "actuator_commanded": [
                degrees(delta_a_cmd),
                -degrees(delta_a_cmd),
                degrees(delta_e_cmd),
                degrees(delta_r_cmd),
                delta_t_cmd * 100.0,
            ],
            "saturation_ratio": saturation_ratio,
            "roll_command_saturation_ratio": roll_command_saturation_ratio,
            "guidance_saturation_ratio": guidance_saturation_ratio,
            "aileron_deflection": {"left": left_aileron_deg, "right": right_aileron_deg},
            "elevator_deflection": elevator_deg,
            "rudder_deflection": rudder_deg,
            "flap_deflection": {"left": 0.0, "right": 0.0},
            "spoiler_deployment": {"left": 0.0, "right": 0.0},
            "throttle": delta_t,
            "flight_phase": flight_phase,
            "tecs_total_energy_error": float(tecs_state.total_energy_error),
            "tecs_balance_energy_error": float(tecs_state.balance_energy_error),
            "tecs_throttle_command": float(tecs_state.throttle_command),
            "tecs_pitch_command_rad": float(tecs_state.pitch_command_rad),
            "tecs_elevator_bias_rad": float(tecs_state.elevator_bias_rad),
        },
        # Fixed autopilot modes report zero diagnostics; online_q_learning
        # fills these with bounded tabular residual-controller metrics.
        "rl_metrics": {
            "method": str(q_metrics.method),
            "reward": float(q_metrics.reward),
            "episode_return": float(q_metrics.episode_return),
            "td_error": float(q_metrics.td_error),
            "policy_entropy": float(q_metrics.policy_entropy),
            "safety_violations": int(q_metrics.safety_violations),
            "action_index": int(q_metrics.action_index),
            "enabled": bool(q_metrics.enabled),
            "epsilon": float(q_metrics.epsilon),
            "explored": bool(q_metrics.explored),
            "q_state": list(q_metrics.q_state) if q_metrics.q_state is not None else None,
            "q_value": float(q_metrics.q_value),
            "max_next_q": float(q_metrics.max_next_q),
            "updates": int(q_metrics.updates),
            "residual_active": bool(q_metrics.residual_active),
            "hard_condition_score": float(q_metrics.hard_condition_score),
            "hjb_value": float(q_metrics.hjb_value),
            "hjb_advantage": float(q_metrics.hjb_advantage),
            "hjb_stage_cost": float(q_metrics.hjb_stage_cost),
            "shield_active": bool(q_metrics.shield_active),
            "candidate_count": int(q_metrics.candidate_count),
            "load_factor_nz": float(q_metrics.load_factor_nz),
            "safety_risk_score": float(q_metrics.safety_risk_score),
        },
        "guidance_state": {
            "cross_track_error_m": float(info.get("guidance_cross_track_error_m", 0.0)),
            "radial_error_m": float(info.get("guidance_radial_error_m", 0.0)),
            "lookahead_m": float(info.get("guidance_lookahead_m", 0.0)),
            "bearing_error_rad": float(info.get("guidance_bearing_error_rad", 0.0)),
            "lateral_accel_mps2": float(info.get("guidance_lateral_accel_mps2", 0.0)),
            "roll_command_rad": float(info.get("guidance_roll_command_rad", 0.0)),
        },
    }


# ---------------------------------------------------------------------------
# FastAPI app + WebSocket handler
# ---------------------------------------------------------------------------

app = FastAPI(title="UAV Simulink Autopilot Backend", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _handle_command(rt: Runtime, command: Any, running: bool = True) -> tuple[bool, int, str, str]:
    """Mutate the runtime according to a frontend command. Returns updated flags.

    Non-object JSON payloads are valid WebSocket messages but invalid commands;
    reject them explicitly so one malformed client frame cannot tear down the
    stream. Configuration commands preserve the current run/pause state.
    """
    if not isinstance(command, dict):
        raise ValueError("WebSocket command must be a JSON object")

    episode = int(command.get("episode", 1))
    controller_requested = str(command.get("controller", "fixed_matlab_autopilot"))
    controller_aliases = {
        "fixed_matlab_autopilot": FIXED_CONTROLLER_MODE,
        "fixed_matlab_baseline": FIXED_CONTROLLER_MODE,
        "baseline": FIXED_CONTROLLER_MODE,
        "online_q_learning": Q_LEARNING_CONTROLLER_MODE,
        "baseline_q": Q_LEARNING_CONTROLLER_MODE,
        "baseline+q": Q_LEARNING_CONTROLLER_MODE,
        "sharq_hjb": SHARQ_HJB_CONTROLLER_MODE,
        "baseline_sharq_hjb": SHARQ_HJB_CONTROLLER_MODE,
        "baseline+sharq-hjb": SHARQ_HJB_CONTROLLER_MODE,
    }
    if controller_requested not in controller_aliases:
        raise ValueError(f"Unsupported controller label: {controller_requested}")
    controller = controller_aliases[controller_requested]

    command_name = str(command.get("command", "start"))
    if command_name not in {"configure", "pause", "reset", "start", "stop"}:
        raise ValueError(f"Unsupported WebSocket command: {command_name}")

    profile = command.get("profile")
    next_profile = rt.profile
    if profile is not None:
        if not isinstance(profile, str) or profile not in TRAJECTORY_PROFILES:
            raise ValueError(f"Unsupported trajectory profile: {profile}")
        next_profile = profile
    profile_changed = next_profile != rt.profile
    scenario_requested = command.get("scenario")
    if scenario_requested is not None and scenario_requested != next_profile:
        raise ValueError(f"Scenario must match active profile: {scenario_requested} != {next_profile}")
    scenario = next_profile

    effects = command.get("effects")

    # All schema/provenance checks above must pass before mutating runtime
    # state. This keeps rejected WebSocket commands from partially changing
    # profile or wind settings.
    rt.profile = next_profile
    rt.controller_mode = controller
    if isinstance(effects, dict):
        apply_effects(rt.wind, effects)

    if command_name == "reset":
        rt.reset()
        running = True
    elif profile_changed:
        # A trajectory profile also defines the initial condition envelope.
        # Switching into a runway-takeoff mission from a live cruise state must
        # not reuse the old 200 m/s trim state, otherwise the first takeoff
        # frame starts airborne/high-speed instead of at rest on the runway.
        rt.reset()
    elif command_name == "start":
        running = True
    elif command_name in {"pause", "stop"}:
        running = False
    return running, episode, scenario, controller


@app.websocket("/ws/uav-digital-twin")
async def uav_digital_twin_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    rt = Runtime.create()
    episode = 1
    scenario = rt.profile
    controller = "fixed_matlab_autopilot"
    running = True
    last_frame_time = -float("inf")
    try:
        while True:
            try:
                command = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                running, episode, scenario, controller = _handle_command(rt, command, running)
            except _WAIT_FOR_TIMEOUT_ERRORS:
                pass
            except JSONDecodeError:
                logger.warning("Received non-JSON websocket payload; ignoring frame.")
            except (TypeError, ValueError) as exc:
                logger.warning("Rejected websocket command (%s); continuing stream.", exc)

            if running:
                try:
                    info = rt.step()
                    if rt.t - last_frame_time >= WEBSOCKET_FRAME_INTERVAL_S:
                        frame = build_simulation_frame(
                            rt=rt,
                            info=info,
                            episode=episode,
                            scenario=scenario,
                            controller=controller,
                        )
                        await websocket.send_json(frame)
                        last_frame_time = rt.t
                except (WebSocketDisconnect, ConnectionError):
                    raise
                except Exception:
                    logger.exception("Simulation step failed; pausing stream.")
                    running = False
            await asyncio.sleep(rt.P.Ts)
    except WebSocketDisconnect:
        return


def main() -> None:
    uvicorn.run("uavsim.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
