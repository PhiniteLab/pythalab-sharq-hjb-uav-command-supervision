"""Autopilot converted from ``autopilot.m``."""
from __future__ import annotations
from dataclasses import dataclass, field
from math import isfinite, pi, sqrt
import numpy as np

from .parameters import UAVParameters


def sat(value: float, up_limit: float, low_limit: float) -> float:
    if value > up_limit:
        return up_limit
    if value < low_limit:
        return low_limit
    return value


def wrap_to_pi(angle: float) -> float:
    """Wrap ``angle`` (radians) into the closed interval :math:`[-\\pi, \\pi]`.

    Used to prevent the heading-error discontinuity that would otherwise cause
    a 2π step at :math:`\\psi = \\pm\\pi`, wind up the integrator, and produce
    a hard aileron command (the visible "controller jumping" behaviour).
    """
    return (angle + pi) % (2.0 * pi) - pi


def _should_undo_integrator(unsat: float, up_limit: float, low_limit: float, integral_step_output: float) -> bool:
    """Return True when an integral update drives farther into saturation."""
    return (unsat > up_limit and integral_step_output > 0.0) or (
        unsat < low_limit and integral_step_output < 0.0
    )


@dataclass
class PIState:
    integrator: float = 0.0
    error_d1: float = 0.0

    def reset(self) -> None:
        self.integrator = 0.0
        self.error_d1 = 0.0


@dataclass
class PIDState:
    integrator: float = 0.0
    differentiator: float = 0.0
    differentiator_d1: float = 0.0
    error_d1: float = 0.0

    def reset(self) -> None:
        self.integrator = 0.0
        self.differentiator = 0.0
        self.differentiator_d1 = 0.0
        self.error_d1 = 0.0


@dataclass
class Autopilot:
    P: UAVParameters
    altitude_state: int | None = None
    initialize_integrator: int = 1
    heading_state: PIState = field(default_factory=PIState)
    roll_state: PIState = field(default_factory=PIState)
    pitch_state: PIState = field(default_factory=PIState)
    airspeed_pitch_state: PIDState = field(default_factory=PIDState)
    airspeed_throttle_state: PIDState = field(default_factory=PIDState)
    altitude_hold_state: PIDState = field(default_factory=PIDState)
    sideslip_state: PIState = field(default_factory=PIState)
    # Last commanded pitch angle, used for bumpless transfer between the
    # longitudinal state machine’s branches (climb / descend / hold). Without
    # this preload the PID that owns ``theta_c`` would re-initialise to a value
    # set only by its proportional gain, producing a θ_c step that propagates
    # to δ_e and triggers a visible pitch jump.
    last_theta_c: float = 0.0

    def reset(self) -> None:
        self.altitude_state = None
        self.initialize_integrator = 1
        self.heading_state.reset()
        self.roll_state.reset()
        self.pitch_state.reset()
        self.airspeed_pitch_state.reset()
        self.airspeed_throttle_state.reset()
        self.altitude_hold_state.reset()
        self.sideslip_state.reset()
        self.last_theta_c = 0.0

    def update(
        self,
        x: np.ndarray,
        commands: np.ndarray,
        t: float,
        measured_airspeed: float | None = None,
        pitch_command_override_rad: float | None = None,
    ) -> np.ndarray:
        """Return ``[delta; x_command; xhat]`` as in ``autopilot.m``.

        ``commands = [Va_c, h_c, heading_c_degrees]``. The legacy source
        variable was named ``chi_c``, but this controller tracks yaw heading
        ``psi`` rather than ground-track course.
        ``measured_airspeed`` may be supplied by the runtime when wind is
        present; if omitted, fall back to the legacy body-speed magnitude.
        """
        P = self.P
        x = np.asarray(x, dtype=float).reshape(12)
        commands = np.asarray(commands, dtype=float).reshape(3)

        pn, pe, pd = x[0], x[1], x[2]
        u, v, w = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        p, q, r = x[9], x[10], x[11]
        h = -pd
        Va_c = commands[0]
        h_c = commands[1]
        chi_c = pi / 180.0 * commands[2]
        body_speed = sqrt(u * u + v * v + w * w)
        Va = float(measured_airspeed) if measured_airspeed is not None and isfinite(measured_airspeed) else body_speed

        reset_flag = 1 if self.altitude_state is None or t == 0 else 0

        # Lateral autopilot
        delta_r = self.coordinated_turn_hold(v, reset_flag)
        phi_c = self.heading_hold(chi_c, psi, r, reset_flag)
        delta_a = self.roll_hold(phi_c, phi, p, reset_flag)

        # Longitudinal state machine
        if self.altitude_state is None or t == 0:
            if h <= P.altitude_take_off_zone:
                self.altitude_state = 1
            elif h <= h_c - P.altitude_hold_zone:
                self.altitude_state = 2
            elif h >= h_c + P.altitude_hold_zone:
                self.altitude_state = 3
            else:
                self.altitude_state = 4
            self.initialize_integrator = 1

        if self.altitude_state == 1:  # runway roll / rotation / liftoff
            speed_ratio = sat(Va / max(P.takeoff_rotation_speed, 1e-6), 1.0, 0.0)
            delta_t = sat(0.10 + 0.0030 * Va + 0.0015 * max(Va_c - Va, 0.0), 0.72, 0.10)
            if Va < P.takeoff_rotation_speed:
                theta_c = (1.0 + 2.0 * speed_ratio) * pi / 180.0
            else:
                rotate_ratio = sat((Va - P.takeoff_rotation_speed) / 35.0, 1.0, 0.0)
                theta_c = (5.0 + 7.0 * rotate_ratio) * pi / 180.0
            if h >= P.takeoff_climbout_altitude or (h >= P.altitude_take_off_zone and Va >= P.takeoff_liftoff_speed):
                self._prepare_bumpless_transfer(theta_c, next_state=2)
                self.altitude_state = 2
            else:
                self.initialize_integrator = 0

        elif self.altitude_state == 2:  # climb / acceleration zone
            speed_error = Va_c - Va
            delta_t = sat(P.u_trim[3] + 0.06 + 0.004 * speed_error, 1.0, 0.18)
            theta_alt = self.altitude_hold(h_c, h, self.initialize_integrator)
            speed_pitch_cap_deg = 14.0 if speed_error <= 5.0 else max(3.0, 13.0 - 0.16 * speed_error)
            theta_c = sat(theta_alt, speed_pitch_cap_deg * pi / 180.0, -4.0 * pi / 180.0)
            # Once near the commanded altitude, hand over to altitude-hold even
            # when the aircraft is faster than the new target.  Keeping the
            # climb branch active on negative speed error pins throttle to its
            # climb minimum and prevents compact waypoint/orbit missions from
            # decelerating before the turn.
            if h >= h_c - P.altitude_hold_zone and speed_error <= 8.0:
                self._prepare_bumpless_transfer(theta_c, next_state=4)
                self.altitude_state = 4
            elif h <= -1.0 and Va < P.takeoff_rotation_speed:
                self._prepare_bumpless_transfer(theta_c, next_state=1)
                self.altitude_state = 1
            else:
                self.initialize_integrator = 0

        elif self.altitude_state == 3:  # descend zone
            delta_t = 0.0
            theta_c = self.airspeed_with_pitch_hold(Va_c, Va, self.initialize_integrator)
            if h <= h_c + P.altitude_hold_zone:
                self._prepare_bumpless_transfer(theta_c, next_state=4)
                self.altitude_state = 4
            else:
                self.initialize_integrator = 0

        elif self.altitude_state == 4:  # altitude hold zone
            delta_t = self.airspeed_with_throttle_hold(Va_c, Va, self.initialize_integrator)
            theta_c = self.altitude_hold(h_c, h, self.initialize_integrator)
            if h <= h_c - P.altitude_hold_zone:
                self._prepare_bumpless_transfer(theta_c, next_state=2)
                self.altitude_state = 2
            elif h >= h_c + P.altitude_hold_zone:
                self._prepare_bumpless_transfer(theta_c, next_state=3)
                self.altitude_state = 3
            else:
                self.initialize_integrator = 0
        else:
            raise RuntimeError(f"Invalid altitude_state: {self.altitude_state}")

        if pitch_command_override_rad is not None and self.altitude_state != 1:
            theta_c = sat(float(pitch_command_override_rad), 15.0 * pi / 180.0, -10.0 * pi / 180.0)

        self.last_theta_c = theta_c
        delta_e = self.pitch_hold(theta_c, theta, q, reset_flag)

        delta = np.array([delta_e, delta_a, delta_r, delta_t], dtype=float)
        x_command = np.array(
            [
                0.0,
                0.0,
                -h_c,
                Va_c,
                0.0,
                0.0,
                phi_c,
                theta_c * P.K_theta_DC,
                chi_c,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        xhat = np.array([pn, pe, -h, u, v, w, phi, theta, psi, p, q, r], dtype=float)
        return np.concatenate([delta, x_command, xhat])

    def heading_hold(self, chi_c: float, psi: float, r: float, flag: int) -> float:
        P, st = self.P, self.heading_state
        if flag == 1:
            st.reset()
        # For ordinary heading commands, wrap to [-π, π] so ψ = -179° and
        # χ_c = +179° produces a small -2° error rather than a 358° jump.  For
        # compact circle missions the backend deliberately sends an unwrapped
        # multi-turn heading reference (|χ_c| > π) so the aircraft keeps turning
        # through 360° instead of taking the shortest way back after every wrap.
        error = chi_c - psi if abs(chi_c) > pi else wrap_to_pi(chi_c - psi)
        delta_i = (P.Ts / 2.0) * (error + st.error_d1)
        st.integrator += delta_i
        up = P.heading_kp * error
        ui = P.heading_ki * st.integrator
        ud = -P.heading_kd * r
        unsat = up + ui + ud
        up_lim = 45.0 * pi / 180.0
        low_lim = -45.0 * pi / 180.0
        out = sat(unsat, up_lim, low_lim)
        if _should_undo_integrator(unsat, up_lim, low_lim, P.heading_ki * delta_i):
            st.integrator -= delta_i
        st.error_d1 = error
        return out

    def roll_hold(self, phi_c: float, phi: float, p: float, flag: int) -> float:
        P, st = self.P, self.roll_state
        if flag == 1:
            st.reset()
        error = phi_c - phi
        delta_i = (P.Ts / 2.0) * (error + st.error_d1)
        st.integrator += delta_i
        up = P.roll_kp * error
        ui = P.roll_ki * st.integrator
        ud = -P.roll_kd * p
        unsat = up + ui + ud
        up_lim = 45.0 * pi / 180.0
        low_lim = -45.0 * pi / 180.0
        out = sat(unsat, up_lim, low_lim)
        if _should_undo_integrator(unsat, up_lim, low_lim, P.roll_ki * delta_i):
            st.integrator -= delta_i
        st.error_d1 = error
        return out

    def pitch_hold(self, theta_c: float, theta: float, q: float, flag: int) -> float:
        P, st = self.P, self.pitch_state
        if flag == 1:
            st.reset()
        error = theta_c - theta
        delta_i = (P.Ts / 2.0) * (error + st.error_d1)
        st.integrator += delta_i
        up = P.pitch_kp * error
        ui = P.pitch_ki * st.integrator
        ud = -P.pitch_kd * q
        unsat = P.u_trim[0] + up + ui + ud
        up_lim = 45.0 * pi / 180.0
        low_lim = -45.0 * pi / 180.0
        out = sat(unsat, up_lim, low_lim)
        if _should_undo_integrator(unsat, up_lim, low_lim, P.pitch_ki * delta_i):
            st.integrator -= delta_i
        st.error_d1 = error
        return out

    def _pid_common(self, st: PIDState, error: float, kp: float, ki: float, kd: float, up_lim: float, low_lim: float, flag: int) -> float:
        P = self.P
        if flag == 1:
            st.reset()
        delta_i = (P.Ts / 2.0) * (error + st.error_d1)
        st.integrator += delta_i
        st.differentiator = ((2.0 * P.tau - P.Ts) / (2.0 * P.tau + P.Ts)) * st.differentiator_d1 + (
            2.0 / (2.0 * P.tau + P.Ts)
        ) * (error - st.error_d1)
        up = kp * error
        ui = ki * st.integrator
        ud = kd * st.differentiator
        unsat = up + ui + ud
        out = sat(unsat, up_lim, low_lim)
        if _should_undo_integrator(unsat, up_lim, low_lim, ki * delta_i):
            st.integrator -= delta_i
        st.error_d1 = error
        st.differentiator_d1 = st.differentiator
        return out

    def airspeed_with_pitch_hold(self, Va_c: float, Va: float, flag: int) -> float:
        P = self.P
        error = Va_c - Va
        return self._pid_common(
            self.airspeed_pitch_state,
            error,
            P.airspeed_pitch_kp,
            P.airspeed_pitch_ki,
            P.airspeed_pitch_kd,
            30.0 * pi / 180.0,
            -30.0 * pi / 180.0,
            flag,
        )

    def airspeed_with_throttle_hold(self, Va_c: float, Va: float, flag: int) -> float:
        P, st = self.P, self.airspeed_throttle_state
        if flag == 1:
            st.reset()
        error = Va_c - Va
        delta_i = (P.Ts / 2.0) * (error + st.error_d1)
        st.integrator += delta_i
        st.differentiator = ((2.0 * P.tau - P.Ts) / (2.0 * P.tau + P.Ts)) * st.differentiator_d1 + (
            2.0 / (2.0 * P.tau + P.Ts)
        ) * (error - st.error_d1)
        up = P.airspeed_throttle_kp * error
        ui = P.airspeed_throttle_ki * st.integrator
        ud = P.airspeed_throttle_kd * st.differentiator
        unsat = P.u_trim[3] + up + ui + ud
        out = sat(unsat, 1.0, 0.0)
        if _should_undo_integrator(unsat, 1.0, 0.0, P.airspeed_throttle_ki * delta_i):
            st.integrator -= delta_i
        st.error_d1 = error
        st.differentiator_d1 = st.differentiator
        return out

    def altitude_hold(self, h_c: float, h: float, flag: int) -> float:
        P = self.P
        error = h_c - h
        correction = self._pid_common(
            self.altitude_hold_state,
            error,
            P.altitude_kp,
            P.altitude_ki,
            P.altitude_kd,
            20.0 * pi / 180.0 - P.theta0,
            -10.0 * pi / 180.0 - P.theta0,
            flag,
        )
        return P.theta0 + correction

    def coordinated_turn_hold(self, v: float, flag: int) -> float:
        P, st = self.P, self.sideslip_state
        if flag == 1:
            st.reset()
        error = -v
        delta_i = (P.Ts / 2.0) * (error + st.error_d1)
        st.integrator += delta_i
        up = P.sideslip_kp * error
        ui = P.sideslip_ki * st.integrator
        ud = 0.0
        unsat = up + ui + ud
        up_lim = 30.0 * pi / 180.0
        low_lim = -30.0 * pi / 180.0
        out = sat(unsat, up_lim, low_lim)
        if _should_undo_integrator(unsat, up_lim, low_lim, P.sideslip_ki * delta_i):
            st.integrator -= delta_i
        st.error_d1 = error
        return out

    # ------------------------------------------------------------------
    # Bumpless transfer between longitudinal state-machine branches
    # ------------------------------------------------------------------
    def _prepare_bumpless_transfer(self, theta_c_current: float, *, next_state: int) -> None:
        """Pre-load the integrator of the PID that owns theta_c after the
        imminent state transition so the next call produces ``theta_c_current``.

        Without this preload, switching from *climb* (theta_c from
        airspeed-with-pitch) to *altitude hold* (theta_c from altitude hold)
        with both PIDs flagged for integrator reset would produce
        theta_c = kp * (h_c - h) on the very first sample of the new branch -- a
        step that propagates through the inner pitch loop as a visible jump.
        We solve  u = kp*e + ki*I + kd*d = theta_c_current for I assuming
        d = 0 and e ~= 0, then write I back into the destination PID. For
        altitude-hold, ``u`` is the pitch correction around trim, so the trim
        pitch offset is removed before preloading.
        """
        P = self.P
        # State 1 uses a constant theta_c, so no preload is required there.
        # States 2 and 3 (climb / descend) drive theta_c via the
        # airspeed-with-pitch PID. State 4 (altitude hold) drives theta_c via
        # the altitude-hold PID.
        preload_output = float(theta_c_current)
        if next_state in (2, 3):
            state, ki = self.airspeed_pitch_state, P.airspeed_pitch_ki
        elif next_state == 4:
            state, ki = self.altitude_hold_state, P.altitude_ki
            preload_output -= P.theta0
        else:
            self.initialize_integrator = 1
            return
        state.reset()
        if abs(ki) > 1e-12:
            state.integrator = preload_output / ki
        state.error_d1 = 0.0
        state.differentiator = 0.0
        state.differentiator_d1 = 0.0
        # Subsequent steps must accumulate normally, not re-reset.
        self.initialize_integrator = 0
