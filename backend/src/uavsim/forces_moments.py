"""Forces and moments for the converted UAV model.

The function ``forces_moments`` is a line-by-line Python analogue of
``forces_moments.m``.  The default ``exact_source=True`` intentionally keeps
several source-level quirks so that Python output follows the uploaded
Simulink/MATLAB source.  Set ``exact_source=False`` to use the textbook wind
transformation.
"""
from __future__ import annotations
import numpy as np
from math import sin, cos, sqrt, atan2, exp

from .parameters import UAVParameters
from .rotations import R_v_to_b
from .atmosphere import density_at_altitude


def _safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    if abs(den) < 1e-12:
        return fallback
    return num / den


def air_data(x: np.ndarray, wind: np.ndarray, P: UAVParameters, *, exact_source: bool = True) -> tuple[float, float, float, np.ndarray]:
    """Return ``Va, alpha, beta, wind_body`` for the current state.

    ``wind`` is ordered as in the Simulink model:
    ``[w_ns, w_es, w_ds, u_wg, v_wg, w_wg]``.
    """
    x = np.asarray(x, dtype=float).reshape(12)
    wind = np.asarray(wind, dtype=float).reshape(6)
    u, v, w = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    w_ns, w_es, w_ds, u_wg, v_wg, w_wg = wind
    R = R_v_to_b(phi, theta, psi)

    # Textbook wind decomposition (Beard & McLain eq. 4.3): rotate steady NED
    # wind into the body frame and add the body-frame gust. The legacy
    # ``exact_source=True`` path rotated the body-gust vector and then added
    # the body-gust *again* — a bug that injected spurious wind whenever the
    # operator dialled in any non-zero gust; the corrected expression is now
    # used regardless of ``exact_source``.
    _ = exact_source
    body = R @ np.array([w_ns, w_es, w_ds], dtype=float) + np.array([u_wg, v_wg, w_wg], dtype=float)

    ur, vr, wr = np.array([u, v, w], dtype=float) - body
    Va = sqrt(max(ur * ur + vr * vr + wr * wr, 0.0))
    alpha = atan2(wr, ur)
    beta = atan2(vr, sqrt(max(ur * ur + wr * wr, 0.0)))
    return Va, alpha, beta, body


def lift_drag_coefficients(alpha: float, P: UAVParameters) -> tuple[float, float, float]:
    """Return CL, CD, sigma using the same blending model as the MATLAB file."""
    ca = cos(alpha)
    sa = sin(alpha)
    # Guard the exponentials for extreme alpha during numerical optimization.
    a1 = max(min(-P.M * (alpha - P.alpha0), 700.0), -700.0)
    a2 = max(min(P.M * (alpha + P.alpha0), 700.0), -700.0)
    tmp1 = exp(a1)
    tmp2 = exp(a2)
    sigma = (1.0 + tmp1 + tmp2) / ((1.0 + tmp1) * (1.0 + tmp2))
    CL = (1.0 - sigma) * (P.C_L_0 + P.C_L_alpha * alpha)
    CD = P.C_D_0 + (1.0 - sigma) * P.epsilon * (P.C_L_0 + P.C_L_alpha * alpha) ** 2
    if alpha >= 0.0:
        CL += sigma * 2.0 * sa * sa * ca
        CD += sigma * 2.0 * sa * sa * sa
    else:
        CL -= sigma * 2.0 * sa * sa * ca
        CD -= sigma * 2.0 * sa * sa * sa
    return CL, CD, sigma


def forces_moments(
    x: np.ndarray,
    delta: np.ndarray,
    wind: np.ndarray | None,
    P: UAVParameters,
    *,
    exact_source: bool = True,
) -> np.ndarray:
    """Compute total force and moment vector ``[Fx,Fy,Fz,l,m,n]``.

    Parameters use the same order as the uploaded MATLAB model:
    ``delta = [delta_e, delta_a, delta_r, delta_t]``.
    """
    x = np.asarray(x, dtype=float).reshape(12)
    delta = np.asarray(delta, dtype=float).reshape(4)
    if wind is None:
        wind = np.zeros(6)
    else:
        wind = np.asarray(wind, dtype=float).reshape(6)

    u, v, w = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    delta_e, delta_a, delta_r, delta_t = delta

    Va, alpha, beta, _wind_body = air_data(x, wind, P, exact_source=exact_source)
    rho = density_at_altitude(P.rho, -float(x[2]))
    qbar = 0.5 * rho * Va * Va
    ca = cos(alpha)
    sa = sin(alpha)
    CL, CD, _sigma = lift_drag_coefficients(alpha, P)

    # Gravitational forces
    Force = np.zeros(3, dtype=float)
    Torque = np.zeros(3, dtype=float)
    Force[0] = -P.mass * P.gravity * sin(theta)
    Force[1] = P.mass * P.gravity * cos(theta) * sin(phi)
    Force[2] = P.mass * P.gravity * cos(theta) * cos(phi)

    inv_2Va = 1.0 / (2.0 * Va) if Va > 1e-12 else 0.0

    # Aerodynamic forces
    Force[0] += qbar * P.S_wing * (-CD * ca + CL * sa)
    Force[0] += qbar * P.S_wing * (-P.C_D_q * ca + P.C_L_q * sa) * P.c * q * inv_2Va

    Force[1] += qbar * P.S_wing * (P.C_Y_0 + P.C_Y_beta * beta)
    Force[1] += qbar * P.S_wing * (P.C_Y_p * p + P.C_Y_r * r) * P.b * inv_2Va

    Force[2] += qbar * P.S_wing * (-CD * sa - CL * ca)
    Force[2] += qbar * P.S_wing * (-P.C_D_q * sa - P.C_L_q * ca) * P.c * q * inv_2Va

    # Aerodynamic torques
    Torque[0] = qbar * P.S_wing * P.b * (P.C_ell_0 + P.C_ell_beta * beta)
    Torque[0] += qbar * P.S_wing * P.b * (P.C_ell_p * p + P.C_ell_r * r) * P.b * inv_2Va

    Torque[1] = qbar * P.S_wing * P.c * (P.C_M_0 + P.C_M_alpha * alpha)
    Torque[1] += qbar * P.S_wing * P.c * P.C_M_q * P.c * q * inv_2Va

    Torque[2] = qbar * P.S_wing * P.b * (P.C_n_0 + P.C_n_beta * beta)
    Torque[2] += qbar * P.S_wing * P.b * (P.C_n_p * p + P.C_n_r * r) * P.b * inv_2Va

    # Control forces
    Force[0] += qbar * P.S_wing * (-P.C_D_delta_e * ca + P.C_L_delta_e * sa) * delta_e
    Force[1] += qbar * P.S_wing * (P.C_Y_delta_a * delta_a + P.C_Y_delta_r * delta_r)
    Force[2] += qbar * P.S_wing * (-P.C_D_delta_e * sa - P.C_L_delta_e * ca) * delta_e

    # Control torques
    Torque[0] += qbar * P.S_wing * P.b * (P.C_ell_delta_a * delta_a + P.C_ell_delta_r * delta_r)
    Torque[1] += qbar * P.S_wing * P.c * P.C_M_delta_e * delta_e
    Torque[2] += qbar * P.S_wing * P.b * (P.C_n_delta_a * delta_a + P.C_n_delta_r * delta_r)

    # Propulsion force
    motor_temp = P.k_motor * P.k_motor * delta_t * delta_t - Va * Va
    Force[0] += 0.5 * rho * P.S_prop * P.C_prop * motor_temp

    return np.concatenate([Force, Torque])
