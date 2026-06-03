"""Command profiles extracted from ``mavsim_auto.mdl`` Stateflow charts."""
from __future__ import annotations
import numpy as np


def h_destination(t: float) -> float:
    """Stateflow MATLAB Function chart ``h_destination(t)`` from mavsim_auto.mdl."""
    r_t = 1.0 * t
    r_t_50 = 1.0 * (t - 50.0)
    if t < 50.0:
        return r_t
    if t >= 50.0 and t < 100.0:
        return r_t - r_t_50
    if t >= 100.0 and t < 150.0:
        return r_t - r_t_50 + 150.0
    return r_t - r_t_50 + 150.0 - 100.0


def phi_destination(t: float) -> float:
    """Stateflow MATLAB Function chart ``phi_destination(t)`` from mavsim_auto.mdl.

    Despite the chart name, this signal is connected to the heading command and
    is interpreted by ``autopilot.m`` as degrees.
    """
    r_t = 1.0 * t
    return 60.0 - 0.1 * r_t


def mavsim_auto_commands(t: float, Va_cmd: float = 140.0, h_cmd: float = 100.0) -> np.ndarray:
    return np.array([Va_cmd, h_cmd, phi_destination(t)], dtype=float)
