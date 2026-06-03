"""Numerical analogue of ``compute_ss_model.m``."""
from __future__ import annotations
import numpy as np

from .parameters import UAVParameters
from .dynamics import mav_derivatives


def numerical_jacobians(
    x_trim: np.ndarray,
    u_trim: np.ndarray,
    P: UAVParameters,
    *,
    wind: np.ndarray | None = None,
    eps: float = 1e-5,
    exact_source: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    x_trim = np.asarray(x_trim, dtype=float).reshape(12)
    u_trim = np.asarray(u_trim, dtype=float).reshape(4)
    if wind is None:
        wind = np.zeros(6)
    f0 = mav_derivatives(x_trim, u_trim, wind, P, exact_source=exact_source)
    A = np.zeros((12, 12), dtype=float)
    B = np.zeros((12, 4), dtype=float)
    for i in range(12):
        xp = x_trim.copy()
        xp[i] += eps
        A[:, i] = (mav_derivatives(xp, u_trim, wind, P, exact_source=exact_source) - f0) / eps
    for i in range(4):
        up = u_trim.copy()
        up[i] += eps
        B[:, i] = (mav_derivatives(x_trim, up, wind, P, exact_source=exact_source) - f0) / eps
    return A, B


def compute_ss_model(
    x_trim: np.ndarray,
    u_trim: np.ndarray,
    P: UAVParameters,
    *,
    eps: float = 1e-5,
    exact_source: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``A_lon, B_lon, A_lat, B_lat, A, B``.

    The extraction matrices are exactly those in ``compute_ss_model.m``.
    """
    A, B = numerical_jacobians(x_trim, u_trim, P, eps=eps, exact_source=exact_source)
    E1_lat = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=float,
    )
    E2_lat = np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    A_lat = E1_lat @ A @ E1_lat.T
    B_lat = E1_lat @ B @ E2_lat.T

    E1_lon = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    E2_lon = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    A_lon = E1_lon @ A @ E1_lon.T
    B_lon = E1_lon @ B @ E2_lon.T
    return A_lon, B_lon, A_lat, B_lat, A, B
