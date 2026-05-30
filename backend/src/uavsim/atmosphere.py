"""Atmosphere utilities for runtime aerodynamic scaling and telemetry.

The legacy MAV model stores a nominal sea-level density in ``UAVParameters``.
These helpers preserve that value at sea level and scale it with a simple ISA
ratio so existing trim/gain assumptions remain close while altitude-dependent
qbar/Mach telemetry becomes available.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


T0_K = 288.15
LAPSE_K_PER_M = 0.0065
GAS_R_AIR = 287.05
GAMMA_AIR = 1.4
GRAVITY = 9.80665


@dataclass(frozen=True)
class AtmosphereSample:
    altitude_m: float
    density_kg_m3: float
    speed_of_sound_mps: float
    mach: float
    dynamic_pressure_pa: float


def isa_temperature_k(altitude_m: float) -> float:
    """Troposphere temperature with a lower clamp for numerical robustness."""
    h = max(float(altitude_m), 0.0)
    return max(216.65, T0_K - LAPSE_K_PER_M * h)


def density_ratio(altitude_m: float) -> float:
    """ISA troposphere density ratio relative to sea level."""
    h = max(float(altitude_m), 0.0)
    if h <= 11_000.0:
        theta = isa_temperature_k(h) / T0_K
        return theta ** (GRAVITY / (GAS_R_AIR * LAPSE_K_PER_M) - 1.0)
    # Sufficient bounded continuation for this simulator's low-altitude use.
    theta_11 = isa_temperature_k(11_000.0) / T0_K
    rho_11 = theta_11 ** (GRAVITY / (GAS_R_AIR * LAPSE_K_PER_M) - 1.0)
    return rho_11 * 0.999 ** (h - 11_000.0)


def density_at_altitude(nominal_sea_level_density: float, altitude_m: float) -> float:
    return max(0.05, float(nominal_sea_level_density) * density_ratio(altitude_m))


def speed_of_sound(altitude_m: float) -> float:
    return sqrt(GAMMA_AIR * GAS_R_AIR * isa_temperature_k(altitude_m))


def sample_atmosphere(nominal_sea_level_density: float, altitude_m: float, airspeed_mps: float) -> AtmosphereSample:
    rho = density_at_altitude(nominal_sea_level_density, altitude_m)
    a = speed_of_sound(altitude_m)
    va = max(float(airspeed_mps), 0.0)
    return AtmosphereSample(
        altitude_m=max(float(altitude_m), 0.0),
        density_kg_m3=rho,
        speed_of_sound_mps=a,
        mach=va / max(a, 1e-9),
        dynamic_pressure_pa=0.5 * rho * va * va,
    )
