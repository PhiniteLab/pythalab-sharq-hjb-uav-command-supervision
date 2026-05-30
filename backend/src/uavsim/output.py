"""Python analogue of ``output.m``."""
from __future__ import annotations
import numpy as np
from math import sqrt, atan2


def output(in_vec: np.ndarray) -> np.ndarray:
    x = np.asarray(in_vec, dtype=float).reshape(-1)
    u, v, w = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    ur, vr, wr = u, v, w
    Va = sqrt(ur * ur + vr * vr + wr * wr)
    alpha = atan2(wr, ur)
    beta = atan2(vr, sqrt(ur * ur + wr * wr))
    return np.array([Va, alpha, beta, phi, theta, psi, p, q, r], dtype=float)
