"""Wind-aware path-following helpers for straight and orbit references."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, pi, radians, sin, sqrt, tanh


def wrap_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


@dataclass(frozen=True)
class GuidanceCommand:
    heading_deg: float
    reference_n: float
    reference_e: float
    cross_track_error_m: float = 0.0
    radial_error_m: float = 0.0
    lookahead_m: float = 0.0
    bearing_error_rad: float = 0.0
    lateral_accel_mps2: float = 0.0
    roll_command_rad: float = 0.0


def _wrap_rad(angle_rad: float) -> float:
    return (float(angle_rad) + pi) % (2.0 * pi) - pi


def _roll_from_lateral_accel(lateral_accel_mps2: float, gravity: float = 9.80665) -> float:
    return max(-45.0 * pi / 180.0, min(45.0 * pi / 180.0, atan2(float(lateral_accel_mps2), gravity)))


def straight_path_guidance(
    *,
    pn: float,
    pe: float,
    start_n: float,
    start_e: float,
    unit_n: float,
    unit_e: float,
    lookahead_m: float,
    chi_inf_deg: float = 45.0,
    k_path: float = 0.015,
    ground_speed_mps: float = 25.0,
) -> GuidanceCommand:
    dn = float(pn) - float(start_n)
    de = float(pe) - float(start_e)
    along = max(dn * unit_n + de * unit_e, 0.0)
    e_py = -unit_e * dn + unit_n * de
    ref_progress = along
    lookahead_progress = along + max(float(lookahead_m), 1.0)
    ref_n = start_n + ref_progress * unit_n
    ref_e = start_e + ref_progress * unit_e
    lookahead_n = start_n + lookahead_progress * unit_n
    lookahead_e = start_e + lookahead_progress * unit_e
    base = degrees(atan2(lookahead_e - pe, lookahead_n - pn))
    correction = chi_inf_deg * (2.0 / pi) * atan2(k_path * e_py, 1.0)
    heading = wrap_deg(base - correction)
    path_course = atan2(unit_e, unit_n)
    eta = _wrap_rad(radians(heading) - path_course)
    l1_distance = max(float(lookahead_m), 1.0)
    lateral_accel = 2.0 * max(float(ground_speed_mps), 0.0) ** 2 / l1_distance * sin(eta)
    return GuidanceCommand(
        heading,
        ref_n,
        ref_e,
        float(e_py),
        0.0,
        l1_distance,
        float(eta),
        float(lateral_accel),
        _roll_from_lateral_accel(lateral_accel),
    )


def orbit_guidance(
    *,
    pn: float,
    pe: float,
    center_n: float,
    center_e: float,
    radius_m: float,
    direction: int,
    k_orbit: float = 4.0,
    ground_speed_mps: float = 25.0,
) -> GuidanceCommand:
    radius = max(float(radius_m), 1.0)
    dn = float(pn) - float(center_n)
    de = float(pe) - float(center_e)
    dist = max(sqrt(dn * dn + de * de), 1e-9)
    phase = atan2(de, dn)
    turn = 1.0 if direction >= 0 else -1.0
    radial_error = dist - radius
    course = phase + turn * (pi / 2.0 + atan2(k_orbit * radial_error, radius))
    ref_n = center_n + radius * cos(phase)
    ref_e = center_e + radius * sin(phase)
    l1_distance = max(0.5 * radius, 1.0)
    # L1-style centripetal demand with radial-error damping; bounded tanh keeps
    # extreme transients from producing unrealistic roll telemetry.
    radial_term = tanh(radial_error / radius)
    lateral_accel = turn * max(float(ground_speed_mps), 0.0) ** 2 / radius * (1.0 + radial_term)
    tangent_course = phase + turn * pi / 2.0
    eta = _wrap_rad(course - tangent_course)
    return GuidanceCommand(
        wrap_deg(degrees(course)),
        ref_n,
        ref_e,
        0.0,
        float(radial_error),
        l1_distance,
        float(eta),
        float(lateral_accel),
        _roll_from_lateral_accel(lateral_accel),
    )
