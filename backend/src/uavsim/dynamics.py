r"""Six-degree-of-freedom MAV dynamics.

Both ``exact_source=True`` and ``exact_source=False`` now use the textbook
(Beard & McLain) equations of motion. The legacy ``exact_source=True`` path
used to mirror typos from ``mav_dynamics.c`` (altitude derivative used ``w``
twice and the inertia coefficients :math:`\Gamma_3`, :math:`\Gamma_5`,
:math:`\Gamma_6` were inconsistent). The flag is kept for API compatibility.
"""
from __future__ import annotations
import numpy as np
from math import sin, cos, tan

from .parameters import UAVParameters
from .forces_moments import forces_moments


def gamma_constants(P: UAVParameters, *, exact_source: bool = True) -> dict[str, float]:
    r"""Return the eight :math:`\Gamma` coefficients used by the body-rate EOM.

    Reference: Beard & McLain, *Small Unmanned Aircraft*, equations (3.13).
    The ``exact_source`` flag is kept for API compatibility; both branches now
    return the same mathematically-correct values.
    """
    _ = exact_source
    Jx, Jy, Jz, Jxz = P.Jx, P.Jy, P.Jz, P.Jxz
    Gamma = Jx * Jz - Jxz * Jxz
    return {
        "Gamma": Gamma,
        "Gamma1": Jxz * (Jx - Jy + Jz) / Gamma,
        "Gamma2": (Jz * (Jz - Jy) + Jxz * Jxz) / Gamma,
        "Gamma3": Jz / Gamma,
        "Gamma4": Jxz / Gamma,
        "Gamma5": (Jz - Jx) / Jy,
        "Gamma6": Jxz / Jy,
        "Gamma7": ((Jx - Jy) * Jx + Jxz * Jxz) / Gamma,
        "Gamma8": Jx / Gamma,
    }


def mav_derivatives_from_forces(
    x: np.ndarray,
    force: np.ndarray,
    torque: np.ndarray,
    P: UAVParameters,
    *,
    exact_source: bool = True,
) -> np.ndarray:
    """Derivative of 12-state MAV model for supplied body force/moment."""
    x = np.asarray(x, dtype=float).reshape(12)
    force = np.asarray(force, dtype=float).reshape(3)
    torque = np.asarray(torque, dtype=float).reshape(3)

    u, v, w = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    Gamma = gamma_constants(P, exact_source=exact_source)

    dx = np.zeros(12, dtype=float)
    dx[0] = (
        cos(theta) * cos(psi) * u
        + (sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)) * v
        + (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) * w
    )
    dx[1] = (
        cos(theta) * sin(psi) * u
        + (sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)) * v
        + (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)) * w
    )
    # Altitude derivative: third row of R_b^v applied to body velocity.
    # The legacy ``exact_source=True`` branch erroneously used ``w`` for both
    # the body-y and body-z contributions; that typo is now removed.
    dx[2] = -sin(theta) * u + sin(phi) * cos(theta) * v + cos(phi) * cos(theta) * w

    dx[3] = r * v - q * w + force[0] / P.mass
    dx[4] = p * w - r * u + force[1] / P.mass
    dx[5] = q * u - p * v + force[2] / P.mass

    dx[6] = p + sin(phi) * tan(theta) * q + cos(phi) * tan(theta) * r
    dx[7] = cos(phi) * q - sin(phi) * r
    dx[8] = (sin(phi) / cos(theta)) * q + (cos(phi) / cos(theta)) * r

    l, m, n = torque
    # Body angular acceleration: Beard & McLain eq. (3.14). The cross-term in
    # ``dx[10]`` must use Γ_6 = J_xz / J_y, not Γ_4 = J_xz / Γ.
    dx[9] = Gamma["Gamma1"] * p * q - Gamma["Gamma2"] * q * r + Gamma["Gamma3"] * l + Gamma["Gamma4"] * n
    dx[10] = Gamma["Gamma5"] * p * r - Gamma["Gamma6"] * (p * p - r * r) + m / P.Jy
    dx[11] = Gamma["Gamma7"] * p * q - Gamma["Gamma1"] * q * r + Gamma["Gamma4"] * l + Gamma["Gamma8"] * n
    return dx


def mav_derivatives(
    x: np.ndarray,
    delta: np.ndarray,
    wind: np.ndarray | None,
    P: UAVParameters,
    *,
    exact_source: bool = True,
) -> np.ndarray:
    fm = forces_moments(x, delta, wind, P, exact_source=exact_source)
    return mav_derivatives_from_forces(x, fm[:3], fm[3:], P, exact_source=exact_source)


def rk4_step(f, x: np.ndarray, t: float, dt: float) -> np.ndarray:
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
