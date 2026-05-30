"""Transfer-function constants equivalent to ``compute_tf_model.m``."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from math import sqrt, atan2

from .parameters import UAVParameters


@dataclass
class TransferFunctionData:
    # Coefficients used by the autopilot design
    a_phi1: float
    a_phi2: float
    a_theta1: float
    a_theta2: float
    a_theta3: float
    a_V1: float
    a_V2: float
    a_V3: float
    a_beta1: float
    a_beta2: float
    Va_trim: float
    alpha_trim: float
    theta_trim: float

    # Numerator/denominator arrays mirroring MATLAB tf objects where useful
    T_phi_delta_a: tuple[np.ndarray, np.ndarray]
    T_chi_phi: tuple[np.ndarray, np.ndarray]
    T_theta_delta_e: tuple[np.ndarray, np.ndarray]
    T_h_theta: tuple[np.ndarray, np.ndarray]
    T_h_Va: tuple[np.ndarray, np.ndarray]
    T_Va_delta_t: tuple[np.ndarray, np.ndarray]
    T_Va_theta: tuple[np.ndarray, np.ndarray]
    T_v_delta_r: tuple[np.ndarray, np.ndarray]


def compute_tf_model(x_trim: np.ndarray, u_trim: np.ndarray, P: UAVParameters) -> TransferFunctionData:
    """Direct translation of ``compute_tf_model.m``.

    The MATLAB file hardcodes many aerodynamic constants rather than reading all
    from ``P``. Here the same numerical values are used through ``P`` for
    consistency with ``param.m``.
    """
    x_trim = np.asarray(x_trim, dtype=float).reshape(12)
    u_trim = np.asarray(u_trim, dtype=float).reshape(4)

    Gamma = P.Jx * P.Jz - P.Jxz ** 2
    Gamma3 = P.Jz / Gamma
    Gamma4 = P.Jxz / Gamma

    Va_trim = sqrt(x_trim[3] ** 2 + x_trim[4] ** 2 + x_trim[5] ** 2)
    alpha_trim = atan2(x_trim[5], x_trim[3])
    theta_trim = x_trim[7]

    C_D_alpha = 2.0 * P.epsilon * (P.C_L_0 + P.C_L_alpha * alpha_trim) * P.C_L_alpha
    C_p_p = Gamma3 * P.C_ell_p + Gamma4 * P.C_n_p
    C_p_delta_a = Gamma3 * P.C_ell_delta_a + Gamma4 * P.C_n_delta_a

    a_phi1 = -0.5 * P.rho * Va_trim ** 2 * P.S_wing * P.b * C_p_p * P.b / 2.0 / Va_trim
    a_phi2 = 0.5 * P.rho * Va_trim ** 2 * P.S_wing * P.b * C_p_delta_a
    a_theta1 = -P.rho * Va_trim ** 2 * P.c * P.S_wing / 2.0 / P.Jy * P.C_M_q * P.c / 2.0 / Va_trim
    a_theta2 = -P.rho * Va_trim ** 2 * P.c * P.S_wing / 2.0 / P.Jy * P.C_M_alpha
    a_theta3 = P.rho * Va_trim ** 2 * P.c * P.S_wing / 2.0 / P.Jy * P.C_M_delta_e
    a_V1 = (
        P.rho
        * Va_trim
        * P.S_wing
        / P.mass
        * (P.C_D_0 + C_D_alpha * alpha_trim + P.C_D_delta_e * u_trim[0])
        + P.rho * P.S_prop / P.mass * P.C_prop * Va_trim
    )
    a_V2 = P.rho * P.S_prop / P.mass * P.C_prop * P.k_motor ** 2 * u_trim[3]
    a_V3 = P.gravity * np.cos(theta_trim - alpha_trim)

    a_beta1 = (P.rho * P.Va * P.S_wing) / 2.0 / P.mass * P.C_Y_beta
    a_beta2 = (P.rho * P.Va ** 2 * P.S_wing) / 2.0 / P.mass * P.C_Y_delta_r

    return TransferFunctionData(
        a_phi1=a_phi1,
        a_phi2=a_phi2,
        a_theta1=a_theta1,
        a_theta2=a_theta2,
        a_theta3=a_theta3,
        a_V1=a_V1,
        a_V2=a_V2,
        a_V3=a_V3,
        a_beta1=a_beta1,
        a_beta2=a_beta2,
        Va_trim=Va_trim,
        alpha_trim=alpha_trim,
        theta_trim=theta_trim,
        T_phi_delta_a=(np.array([a_phi2]), np.array([1.0, a_phi1, 0.0])),
        T_chi_phi=(np.array([P.gravity / Va_trim]), np.array([1.0, 0.0])),
        T_theta_delta_e=(np.array([a_theta3]), np.array([1.0, a_theta1, a_theta2])),
        T_h_theta=(np.array([Va_trim]), np.array([1.0, 0.0])),
        T_h_Va=(np.array([theta_trim]), np.array([1.0, 0.0])),
        T_Va_delta_t=(np.array([a_V2]), np.array([1.0, a_V1])),
        T_Va_theta=(np.array([-a_V3]), np.array([1.0, a_V1])),
        T_v_delta_r=(np.array([a_beta2]), np.array([1.0, -a_beta1])),
    )
