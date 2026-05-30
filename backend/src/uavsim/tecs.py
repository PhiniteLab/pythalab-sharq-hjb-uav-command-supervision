"""TECS-style energy management for speed/altitude command allocation."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
import numpy as np


@dataclass
class TECSState:
    total_energy_error: float = 0.0
    balance_energy_error: float = 0.0
    throttle_command: float = 0.0
    pitch_command_rad: float = 0.0
    elevator_bias_rad: float = 0.0


@dataclass
class TECSController:
    kp_total: float = 0.0012
    kp_balance: float = 0.00018
    kp_elevator_bias: float = 0.00008
    min_throttle: float = 0.0
    max_throttle: float = 1.0
    min_pitch_rad: float = -10.0 * pi / 180.0
    max_pitch_rad: float = 15.0 * pi / 180.0
    max_elevator_bias_rad: float = 2.0 * pi / 180.0
    state: TECSState = field(default_factory=TECSState)

    def reset(self) -> None:
        self.state = TECSState()

    def update(
        self,
        *,
        altitude_m: float,
        airspeed_mps: float,
        target_altitude_m: float,
        target_airspeed_mps: float,
        trim_throttle: float,
        trim_pitch_rad: float = 0.0,
    ) -> TECSState:
        va = max(float(airspeed_mps), 0.0)
        va_c = max(float(target_airspeed_mps), 0.0)
        h = float(altitude_m)
        h_c = float(target_altitude_m)
        e_total = 9.80665 * (h_c - h) + 0.5 * (va_c * va_c - va * va)
        e_balance = 9.80665 * (h_c - h) - 0.5 * (va_c * va_c - va * va)
        throttle = np.clip(float(trim_throttle) + self.kp_total * e_total, self.min_throttle, self.max_throttle)
        pitch = np.clip(float(trim_pitch_rad) + self.kp_balance * e_balance, self.min_pitch_rad, self.max_pitch_rad)
        # A very small elevator feed-forward makes the TECS balance allocation
        # observable in dynamics while preserving the existing pitch loop.
        elevator_bias = np.clip(-self.kp_elevator_bias * e_balance, -self.max_elevator_bias_rad, self.max_elevator_bias_rad)
        self.state = TECSState(float(e_total), float(e_balance), float(throttle), float(pitch), float(elevator_bias))
        return self.state
