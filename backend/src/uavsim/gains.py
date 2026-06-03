"""Autopilot gain computation equivalent to ``computeGains.m``."""
from __future__ import annotations
from math import sqrt, pi

from .parameters import UAVParameters
from .transfer_functions import TransferFunctionData


def compute_gains(P: UAVParameters, tf: TransferFunctionData) -> UAVParameters:
    # Roll loop
    a_phi2 = tf.a_phi2
    a_phi1 = tf.a_phi1
    delta_a_max = 45.0 * pi / 180.0
    phi_max = 15.0 * pi / 180.0
    zeta_roll = 0.707
    wn_roll_raw = 1.1 * sqrt(a_phi2 * delta_a_max * sqrt(1.0 - zeta_roll ** 2) / phi_max)
    # The continuous-time source formula scales with dynamic pressure. At
    # high-speed it asks for a ~90 rad/s roll loop, far above the 100 Hz sampled
    # autopilot's useful bandwidth and it causes actuator chatter/saturation.
    # Keep the legacy low-speed bandwidth, but cap the scheduled roll loop for
    # the new high-speed trim region.
    wn_roll = min(wn_roll_raw, 8.5)
    P.roll_kp = wn_roll ** 2 / a_phi2
    P.roll_kd = max(0.0, (2.0 * zeta_roll * wn_roll - a_phi1) / a_phi2)
    P.roll_ki = 0.1 * min(1.0, 13.0 / max(P.Va, 1e-6))

    # Heading loop
    zeta_heading = 0.707
    # High-speed operation makes the heading plant (chi_dot ~= g/Va*phi)
    # slower and the roll loop bandwidth grows with dynamic pressure.  A raw
    # wn_roll/10 schedule therefore produces very large heading PI gains at
    # high-speed, driving phi_c into saturation and winding up the integrator.
    # Cap the outer-loop bandwidth by speed while preserving the legacy
    # low-speed design near the original 13 m/s trim point.
    wn_heading_nominal = wn_roll / 10.0
    wn_heading_limit = min(0.85, max(0.18, 11.9 / max(P.Va, 1e-6)))
    wn_heading = min(wn_heading_nominal, wn_heading_limit)
    P.heading_kp = 2.0 * zeta_heading * wn_heading * P.Va / P.gravity
    P.heading_ki = wn_heading ** 2 * P.Va / P.gravity
    P.heading_kd = 0.0

    # Sideslip hold block in computeGains.m computes beta gains, but the final
    # autopilot uses P.sideslip_* constants below.
    a_beta2 = tf.a_beta2
    a_beta1 = tf.T_v_delta_r[1][1]  # MATLAB: den(2) from tf([a_beta2],[1,-a_beta1])
    delta_r_max = 20.0 * pi / 180.0
    vr_max = 3.0
    zeta_beta = 0.707
    P.beta_kp = delta_r_max / vr_max
    # MATLAB: wn_beta = (a_beta2*P.beta_kp+a_beta1)/2/zeta_beta;
    _wn_beta = (a_beta2 * P.beta_kp + a_beta1) / 2.0 / zeta_beta
    P.beta_ki = 0.0
    P.beta_kd = 0.0

    # Pitch loop
    a_theta1 = tf.a_theta1
    a_theta2 = tf.a_theta2
    a_theta3 = tf.a_theta3
    delta_e_max = 45.0 * pi / 180.0
    theta_max = 15.0 * pi / 180.0
    zeta_pitch = 0.707
    wn_pitch = sqrt(abs(a_theta3) * delta_e_max * sqrt(1.0 - zeta_pitch ** 2) / theta_max)
    P.pitch_kp = (wn_pitch ** 2 - a_theta2) / a_theta3
    P.pitch_kd = (2.0 * zeta_pitch * wn_pitch - a_theta1) / a_theta3
    P.pitch_ki = 0.0
    P.K_theta_DC = P.pitch_kp * a_theta3 / (a_theta2 + P.pitch_kp * a_theta3)

    # Altitude loop
    zeta_altitude = 0.707
    wn_altitude = wn_pitch / 30.0
    P.altitude_kp = 2.0 * zeta_altitude * wn_altitude / P.K_theta_DC / P.Va
    P.altitude_ki = wn_altitude ** 2 / P.K_theta_DC / P.Va
    P.altitude_kd = 0.0

    # Airspeed using pitch
    a_V1 = tf.a_V1
    zeta_airspeed_pitch = 0.707
    wn_airspeed_pitch = wn_pitch / 10.0
    P.airspeed_pitch_kp = (a_V1 - 2.0 * zeta_airspeed_pitch * wn_airspeed_pitch) / P.K_theta_DC / P.gravity
    P.airspeed_pitch_ki = -wn_airspeed_pitch ** 2 / P.K_theta_DC / P.gravity
    P.airspeed_pitch_kd = 0.0

    # Airspeed using throttle
    a_Vt1 = tf.a_V1
    a_Vt2 = tf.a_V2
    zeta_airspeed_throttle = 0.707
    wn_airspeed_throttle = 5.0
    # Directly preserves computeGains.m: denominator uses a_Vt1, not a_Vt2.
    airspeed_throttle_kp = (2.0 * zeta_airspeed_throttle * wn_airspeed_throttle - a_Vt1) / a_Vt1
    # At the high-speed trim point the propulsive control derivative is large and
    # the raw source gain turns small airspeed errors into full 0/1 throttle
    # toggles. Keep the computed low-speed gain, but limit high-speed throttle
    # proportional action around the trim feed-forward.
    if P.Va >= 50.0:
        airspeed_throttle_kp = min(airspeed_throttle_kp, 0.12)
    P.airspeed_throttle_kp = airspeed_throttle_kp
    P.airspeed_throttle_ki = wn_airspeed_throttle ** 2 / a_Vt2
    P.airspeed_throttle_kd = 0.0

    # Final sideslip constants used by autopilot.m.
    P.sideslip_kp = 0.1 * min(1.0, 13.0 / max(P.Va, 1e-6))
    P.sideslip_kd = -0.5 * min(1.0, 13.0 / max(P.Va, 1e-6))
    P.sideslip_ki = 0.0
    return P
