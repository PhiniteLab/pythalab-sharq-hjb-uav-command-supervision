"""Parameters for the converted UAV Simulink model.

This module is a direct Python translation of ``param.m`` where practical.
The trim calculation is implemented in Python because MATLAB/Simulink's
``trim`` command is not available in a normal Python environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class UAVParameters:
    # Physical constants
    gravity: float = 9.8

    # Airframe physical parameters
    mass: float = 1.56
    Jx: float = 0.1147
    Jy: float = 0.0576
    Jz: float = 0.1712
    Jxz: float = 0.0015

    # Aerodynamic coefficients
    M: float = 50.0
    epsilon: float = 0.1592
    alpha0: float = 0.4712
    rho: float = 1.2682
    c: float = 0.3302
    b: float = 1.4224
    S_wing: float = 0.2589
    S_prop: float = 0.0314
    # High-speed motor constant. With Va=high-speed this gives a level-flight
    # trim throttle of ~0.65 and enough thrust margin for gust/path transients.
    k_motor: float = 240.0
    C_L_0: float = 0.28
    C_L_alpha: float = 3.45
    C_L_q: float = 0.0
    C_L_delta_e: float = -0.36
    C_D_0: float = 0.03
    C_D_q: float = 0.0
    C_D_delta_e: float = 0.0
    C_M_0: float = 0.0
    C_M_alpha: float = -0.38
    C_M_q: float = -3.6
    C_M_delta_e: float = -0.5
    C_Y_0: float = 0.0
    C_Y_beta: float = -0.98
    C_Y_p: float = -0.26
    C_Y_r: float = 0.0
    C_Y_delta_a: float = 0.0
    C_Y_delta_r: float = -0.17
    C_ell_0: float = 0.0
    C_ell_beta: float = -0.12
    C_ell_p: float = -0.26
    C_ell_r: float = 0.14
    C_ell_delta_a: float = 0.08
    C_ell_delta_r: float = 0.105
    C_n_0: float = 0.0
    C_n_beta: float = 0.25
    C_n_p: float = 0.022
    C_n_r: float = -0.35
    C_n_delta_a: float = 0.06
    C_n_delta_r: float = -0.032
    C_prop: float = 1.0

    # Wind parameters
    wind_n: float = 0.0
    wind_e: float = 0.0
    wind_d: float = 0.0
    L_wx: float = 1250.0
    L_wy: float = 1750.0
    L_wz: float = 1750.0
    sigma_wx: float = 1.0
    sigma_wy: float = 1.0
    sigma_wz: float = 1.0
    Va0: float = 10.0

    # Autopilot sample rate
    Ts: float = 0.01
    tau: float = 5.0

    # Desired airspeed used by param.m before trim/gain computation
    Va: float = 140.0

    # Initial conditions; these are overwritten by trim in build_default_parameters.
    pn0: float = 0.0
    pe0: float = 0.0
    pd0: float = 0.0
    u0: float = 140.0
    v0: float = 0.0
    w0: float = 0.0
    phi0: float = 0.0
    theta0: float = 0.0
    psi0: float = 0.0
    p0: float = 0.0
    q0: float = 0.0
    r0: float = 0.0

    # Trim state/input are filled by build_default_parameters.
    x_trim: np.ndarray = field(default_factory=lambda: np.zeros(12))
    u_trim: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=float))

    # Autopilot zones
    altitude_take_off_zone: float = 10.0
    altitude_hold_zone: float = 10.0
    takeoff_rotation_speed: float = 65.0
    takeoff_liftoff_speed: float = 78.0
    takeoff_climbout_altitude: float = 25.0

    # Autopilot gains; filled by compute_gains.
    roll_kp: float = 0.0
    roll_kd: float = 0.0
    roll_ki: float = 0.0
    heading_kp: float = 0.0
    heading_kd: float = 0.0
    heading_ki: float = 0.0
    beta_kp: float = 0.0
    beta_kd: float = 0.0
    beta_ki: float = 0.0
    pitch_kp: float = 0.0
    pitch_kd: float = 0.0
    pitch_ki: float = 0.0
    K_theta_DC: float = 1.0
    altitude_kp: float = 0.0
    altitude_kd: float = 0.0
    altitude_ki: float = 0.0
    airspeed_pitch_kp: float = 0.0
    airspeed_pitch_kd: float = 0.0
    airspeed_pitch_ki: float = 0.0
    airspeed_throttle_kp: float = 0.0
    airspeed_throttle_kd: float = 0.0
    airspeed_throttle_ki: float = 0.0
    sideslip_kp: float = 0.1
    sideslip_kd: float = -0.5
    sideslip_ki: float = 0.0

    def initial_state(self) -> np.ndarray:
        return np.array(
            [
                self.pn0,
                self.pe0,
                self.pd0,
                self.u0,
                self.v0,
                self.w0,
                self.phi0,
                self.theta0,
                self.psi0,
                self.p0,
                self.q0,
                self.r0,
            ],
            dtype=float,
        )


def build_default_parameters(
    *,
    compute_trim_and_gains: bool = True,
    exact_source: bool = True,
    Va: float = 140.0,
    gamma: float = 0.0,
    R: float = 0.0,
    initial_altitude_m: float = 100.0,
    aircraft_config: str = "small_mav",
) -> UAVParameters:
    """Create parameters equivalent to running ``param.m``.

    Parameters
    ----------
    compute_trim_and_gains:
        If True, compute trim, transfer-function coefficients, and autopilot gains.
        If False, use simple untrimmed initial conditions and zero gains.
    exact_source:
        If True, use source-compatible dynamics/force implementation for trim. If
        False, use corrected textbook equations for trim. The default is True to
        match the Simulink/C/MATLAB files as closely as possible.
    Va, gamma, R:
        Same interpretation as in ``param.m``/``compute_trim.m``.
    """
    from .aircraft_config import parameters_from_config

    P = parameters_from_config(aircraft_config)
    P.Va = Va
    P.Va0 = Va
    P.u0 = Va
    P.pd0 = -float(initial_altitude_m)
    if not compute_trim_and_gains:
        P.x_trim = P.initial_state()
        P.u_trim = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return P

    from .trim import compute_trim
    from .transfer_functions import compute_tf_model
    from .gains import compute_gains

    x_trim, u_trim, info = compute_trim(P, Va=Va, gamma=gamma, R=R, exact_source=exact_source)
    if not info.get("feasible", False):
        raise ValueError(
            f"Trim infeasible for Va={Va:.3f} m/s "
            f"(residual_norm={info.get('residual_norm')}, throttle={u_trim[3]:.3f})"
        )
    P.x_trim = x_trim
    P.u_trim = u_trim

    P.pn0 = 0.0
    P.pe0 = 0.0
    P.pd0 = -float(initial_altitude_m)
    P.u0 = float(x_trim[3])
    P.v0 = float(x_trim[4])
    P.w0 = float(x_trim[5])
    P.phi0 = float(x_trim[6])
    P.theta0 = float(x_trim[7])
    P.psi0 = float(x_trim[8])
    P.p0 = float(x_trim[9])
    P.q0 = float(x_trim[10])
    P.r0 = float(x_trim[11])

    tf = compute_tf_model(x_trim, u_trim, P)
    compute_gains(P, tf)
    return P
