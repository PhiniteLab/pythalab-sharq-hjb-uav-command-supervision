"""Python replacement for Simulink's ``trim`` call in ``compute_trim.m``."""
from __future__ import annotations
import numpy as np
from math import sin, sqrt, atan2
from scipy.optimize import least_squares

from .parameters import UAVParameters
from .dynamics import mav_derivatives


def _pack_to_state(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map optimizer vector to state/input.

    Optimized entries are u, v, w, phi, theta, p, q, r, delta_e, delta_a,
    delta_r, delta_t. Position and yaw are fixed as in the initial guess used
    by ``compute_trim.m``.
    """
    z = np.asarray(z, dtype=float)
    x = np.zeros(12, dtype=float)
    x[3:6] = z[0:3]
    x[6] = z[3]
    x[7] = z[4]
    x[8] = 0.0
    x[9:12] = z[5:8]
    delta = z[8:12]
    return x, delta


def compute_trim(
    P: UAVParameters,
    Va: float,
    gamma: float,
    R: float,
    *,
    exact_source: bool = True,
    max_nfev: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute trim state and input with a least-squares analogue of MATLAB trim.

    This matches the constraints in ``compute_trim.m``:
    derivatives ``pdot[3:12]`` are constrained, output airspeed and beta are
    constrained, while ``pn``, ``pe`` and ``psi`` are fixed to zero.
    """
    psidot = Va / R if R != 0.0 else 0.0
    dx0 = np.array(
        [
            0.0,
            0.0,
            -Va * sin(gamma),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            psidot,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    wind = np.zeros(6, dtype=float)

    # Initial guess mirrors compute_trim.m: x0=[..., Va,0,0,0,gamma,...], u0=[0,0,0,1].
    z0 = np.array([Va, 0.0, 0.0, 0.0, gamma, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)

    # Reasonable bounds prevent optimizer from finding nonphysical, high-angle solutions.
    lb = np.array(
        [
            0.1,
            -5 * Va,
            -5 * Va,
            -np.pi / 2,
            -np.pi / 2,
            -20,
            -20,
            -20,
            -45.0 * np.pi / 180.0,
            -45.0 * np.pi / 180.0,
            -30.0 * np.pi / 180.0,
            0.0,
        ],
        dtype=float,
    )
    ub = np.array(
        [
            5 * Va,
            5 * Va,
            5 * Va,
            np.pi / 2,
            np.pi / 2,
            20,
            20,
            20,
            45.0 * np.pi / 180.0,
            45.0 * np.pi / 180.0,
            30.0 * np.pi / 180.0,
            1.0,
        ],
        dtype=float,
    )

    # Scale residuals so derivative constraints dominate but airspeed and beta
    # are still enforced.
    def residual(z: np.ndarray) -> np.ndarray:
        x, delta = _pack_to_state(z)
        dx = mav_derivatives(x, delta, wind, P, exact_source=exact_source)
        u, v, w = x[3], x[4], x[5]
        Va_here = sqrt(max(u * u + v * v + w * w, 0.0))
        beta = atan2(v, sqrt(max(u * u + w * w, 0.0)))
        # idx in MATLAB is [3;4;...;12] in one-based indexing.
        r = []
        r.extend((dx[2:12] - dx0[2:12]).tolist())
        r.append(5.0 * (Va_here - Va))
        r.append(20.0 * beta)
        return np.asarray(r, dtype=float)

    res = least_squares(residual, z0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=max_nfev)
    x_trim, u_trim = _pack_to_state(res.x)
    info = {
        "success": bool(res.success),
        "cost": float(res.cost),
        "optimality": float(res.optimality),
        "message": res.message,
        "residual_norm": float(np.linalg.norm(res.fun)),
    }
    info["feasible"] = bool(info["success"] and info["residual_norm"] < 1e-6)
    return x_trim, u_trim, info
