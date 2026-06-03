"""Actuator lag/rate-limit model for runtime simulations."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
import numpy as np


@dataclass(frozen=True)
class ActuatorLimits:
    lower: np.ndarray = field(default_factory=lambda: np.array([-45.0*pi/180.0, -45.0*pi/180.0, -30.0*pi/180.0, 0.0], dtype=float))
    upper: np.ndarray = field(default_factory=lambda: np.array([45.0*pi/180.0, 45.0*pi/180.0, 30.0*pi/180.0, 1.0], dtype=float))
    rate: np.ndarray = field(default_factory=lambda: np.array([120.0*pi/180.0, 160.0*pi/180.0, 120.0*pi/180.0, 1.8], dtype=float))
    tau_s: float = 0.08


@dataclass
class ActuatorState:
    limits: ActuatorLimits = field(default_factory=ActuatorLimits)
    actual: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0], dtype=float))
    command: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0], dtype=float))

    def reset(self, initial: np.ndarray | None = None) -> None:
        if initial is None:
            self.actual = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            self.actual = np.clip(np.asarray(initial, dtype=float).reshape(4), self.limits.lower, self.limits.upper)
        self.command = self.actual.copy()

    def update(self, command: np.ndarray, dt: float) -> np.ndarray:
        cmd = np.clip(np.asarray(command, dtype=float).reshape(4), self.limits.lower, self.limits.upper)
        self.command = cmd
        dt = max(float(dt), 0.0)
        if dt <= 0.0:
            return self.actual.copy()
        desired_rate = (cmd - self.actual) / max(self.limits.tau_s, 1e-9)
        limited_rate = np.clip(desired_rate, -self.limits.rate, self.limits.rate)
        self.actual = np.clip(self.actual + limited_rate * dt, self.limits.lower, self.limits.upper)
        return self.actual.copy()

    def saturation_ratio(self) -> float:
        span = np.maximum(np.abs(self.limits.upper), np.abs(self.limits.lower))
        span[span < 1e-9] = 1.0
        return float(np.clip(np.max(np.abs(self.actual) / span), 0.0, 1.0))
