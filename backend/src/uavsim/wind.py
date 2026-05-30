"""Wind models corresponding to the Simulink wind subsystem."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, cast
import numpy as np
from scipy import signal

from .parameters import UAVParameters


@dataclass
class DrydenWind:
    """Approximate discrete Dryden gust generator.

    Simulink used continuous transfer functions driven by Band-Limited White
    Noise and then a zero-order hold at ``P.Ts``. This class discretizes the
    same transfer functions with a ZOH and propagates them at ``P.Ts``. Exact
    bit-for-bit stochastic equivalence with Simulink noise is not guaranteed.

    ``P.Va0`` is set to the current trim/cruise speed by
    :func:`uavsim.parameters.build_default_parameters`, so turbulence bandwidth
    is scheduled to the active operating point instead of the legacy low-speed
    default.
    """
    P: UAVParameters
    seed: int = 23341
    enabled: bool = True

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._states: list[np.ndarray] = []
        self._systems: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        P = self.P
        # H_wx = sigma_wx*sqrt(2*Va0/L_wx)/(s+Va0/L_wx)
        self._add_tf([P.sigma_wx * np.sqrt(2.0 * P.Va0 / P.L_wx)], [1.0, P.Va0 / P.L_wx])
        # H_wy/H_wz = sigma*sqrt(3*Va0/L)*(s + Va0/(sqrt(3)L))/(s+Va0/L)^2
        self._add_tf(
            P.sigma_wy * np.sqrt(3.0 * P.Va0 / P.L_wy) * np.array([1.0, P.Va0 / np.sqrt(3.0) / P.L_wy]),
            np.poly([-P.Va0 / P.L_wy, -P.Va0 / P.L_wy]),
        )
        self._add_tf(
            P.sigma_wz * np.sqrt(3.0 * P.Va0 / P.L_wz) * np.array([1.0, P.Va0 / np.sqrt(3.0) / P.L_wz]),
            np.poly([-P.Va0 / P.L_wz, -P.Va0 / P.L_wz]),
        )

    def _add_tf(self, num: Any, den: Any) -> None:
        P = self.P
        A, B, C, D = signal.tf2ss(num, den)
        Ad, Bd, Cd, Dd, _ = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
            signal.cont2discrete((A, B, C, D), P.Ts, method="zoh"),
        )
        self._systems.append((Ad, Bd, Cd, Dd))
        self._states.append(np.zeros((Ad.shape[0], 1)))

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        for i, x in enumerate(self._states):
            self._states[i] = np.zeros_like(x)

    def gust(self) -> np.ndarray:
        if not self.enabled:
            return np.zeros(3)
        outs = []
        for i, (Ad, Bd, Cd, Dd) in enumerate(self._systems):
            # Unit variance white noise input, as in the mdl block Cov [1].
            u = np.array([[self.rng.standard_normal()]])
            y = Cd @ self._states[i] + Dd @ u
            self._states[i] = Ad @ self._states[i] + Bd @ u
            outs.append(float(y.ravel()[0]))
        return np.asarray(outs, dtype=float)

    def wind_vector(self) -> np.ndarray:
        P = self.P
        steady = np.array([P.wind_n, P.wind_e, P.wind_d], dtype=float)
        return np.concatenate([steady, self.gust()])


def zero_wind(P: UAVParameters) -> np.ndarray:
    return np.array([P.wind_n, P.wind_e, P.wind_d, 0.0, 0.0, 0.0], dtype=float)
