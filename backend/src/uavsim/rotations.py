"""Rotation matrices used by the UAV model."""
from __future__ import annotations
import numpy as np
from math import sin, cos


def R_v_to_b(phi: float, theta: float, psi: float) -> np.ndarray:
    """Rotation from vehicle/inertial-aligned frame to body frame.

    This is the matrix used in the book and in ``forces_moments.m``:
    R_b_v = R_x(phi) R_y(theta) R_z(psi).
    """
    return np.array(
        [
            [cos(theta) * cos(psi), cos(theta) * sin(psi), -sin(theta)],
            [
                sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
                sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
                sin(phi) * cos(theta),
            ],
            [
                cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
                cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),
                cos(phi) * cos(theta),
            ],
        ],
        dtype=float,
    )


def R_b_to_v(phi: float, theta: float, psi: float) -> np.ndarray:
    return R_v_to_b(phi, theta, psi).T
