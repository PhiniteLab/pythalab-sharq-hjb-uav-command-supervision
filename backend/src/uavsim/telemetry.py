"""Telemetry extraction, CSV export, and summary statistics."""
from __future__ import annotations

from pathlib import Path
import csv
from math import atan2, pi
from typing import Any

import numpy as np

from .simulation import SimulationResult, SimulationStep

STATE_NAMES = ["pn", "pe", "pd", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r"]
DELTA_NAMES = ["delta_e", "delta_a", "delta_r", "delta_t"]
COMMAND_NAMES = ["Va_c", "h_c", "chi_c_deg"]
XCOMMAND_NAMES = ["pn_c", "pe_c", "pd_c", "u_c", "v_c", "w_c", "phi_c", "theta_c", "psi_c", "p_c", "q_c", "r_c"]
XHAT_NAMES = [f"xhat_{name}" for name in STATE_NAMES]
WIND_NAMES = ["wind_n", "wind_e", "wind_d", "gust_u", "gust_v", "gust_w"]
FORCE_NAMES = ["Fx", "Fy", "Fz", "ell", "m", "n"]
DX_NAMES = [f"d_{name}" for name in STATE_NAMES]
AIR_NAMES = ["Va", "alpha", "beta", "wind_body_u", "wind_body_v", "wind_body_w"]
DERIVED_NAMES = [
    "altitude_h",
    "h",
    "Vg",
    "chi",
    "chi_deg",
    "altitude_state",
    "autopilot_altitude_state",
    "mode_takeoff_climb_descend_hold",
    "phi_deg",
    "theta_deg",
    "psi_deg",
    "p_deg",
    "q_deg",
    "r_deg",
    "p_deg_s",
    "q_deg_s",
    "r_deg_s",
    "delta_e_deg",
    "delta_a_deg",
    "delta_r_deg",
    "phi_c_deg",
    "theta_c_deg",
    "psi_c_deg",
    "alpha_deg",
    "beta_deg",
]

TELEMETRY_COLUMNS = (
    ["t"]
    + STATE_NAMES
    + DERIVED_NAMES
    + DELTA_NAMES
    + COMMAND_NAMES
    + XCOMMAND_NAMES
    + XHAT_NAMES
    + WIND_NAMES
    + FORCE_NAMES
    + DX_NAMES
    + AIR_NAMES
)


def rad2deg(x: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(x) * 180.0 / pi


def _groundspeed_and_course(x: np.ndarray) -> tuple[float, float]:
    u, v, w = x[3:6]
    phi, theta, psi = x[6:9]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    R_b_to_v = np.array(
        [
            [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
            [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
            [-sth, sphi * cth, cphi * cth],
        ],
        dtype=float,
    )
    vel_ned = R_b_to_v @ np.array([u, v, w], dtype=float)
    Vg = float(np.linalg.norm(vel_ned[:2]))
    chi = float(atan2(vel_ned[1], vel_ned[0]))
    return Vg, chi


def telemetry_row(
    t: float,
    x: np.ndarray,
    delta: np.ndarray,
    commands: np.ndarray,
    x_command: np.ndarray,
    xhat: np.ndarray,
    wind: np.ndarray,
    fm: np.ndarray,
    dx: np.ndarray,
    air: np.ndarray,
    altitude_state: int,
) -> dict[str, float]:
    """Build one dense telemetry row from raw simulator signals."""
    x = np.asarray(x, dtype=float).reshape(12)
    delta = np.asarray(delta, dtype=float).reshape(4)
    commands = np.asarray(commands, dtype=float).reshape(3)
    x_command = np.asarray(x_command, dtype=float).reshape(12)
    xhat = np.asarray(xhat, dtype=float).reshape(12)
    wind = np.asarray(wind, dtype=float).reshape(6)
    fm = np.asarray(fm, dtype=float).reshape(6)
    dx = np.asarray(dx, dtype=float).reshape(12)
    air = np.asarray(air, dtype=float).reshape(6)
    Vg, chi = _groundspeed_and_course(x)

    row: dict[str, float] = {"t": float(t)}
    for name, value in zip(STATE_NAMES, x):
        row[name] = float(value)

    row.update(
        {
            "altitude_h": float(-x[2]),
            "h": float(-x[2]),
            "Vg": Vg,
            "chi": chi,
            "chi_deg": float(rad2deg(chi)),
            "altitude_state": float(altitude_state),
            "autopilot_altitude_state": float(altitude_state),
            "mode_takeoff_climb_descend_hold": float(altitude_state),
            "phi_deg": float(rad2deg(x[6])),
            "theta_deg": float(rad2deg(x[7])),
            "psi_deg": float(rad2deg(x[8])),
            "p_deg": float(rad2deg(x[9])),
            "q_deg": float(rad2deg(x[10])),
            "r_deg": float(rad2deg(x[11])),
            "p_deg_s": float(rad2deg(x[9])),
            "q_deg_s": float(rad2deg(x[10])),
            "r_deg_s": float(rad2deg(x[11])),
            "delta_e_deg": float(rad2deg(delta[0])),
            "delta_a_deg": float(rad2deg(delta[1])),
            "delta_r_deg": float(rad2deg(delta[2])),
            "phi_c_deg": float(rad2deg(x_command[6])),
            "theta_c_deg": float(rad2deg(x_command[7])),
            "psi_c_deg": float(rad2deg(x_command[8])),
            "alpha_deg": float(rad2deg(air[1])),
            "beta_deg": float(rad2deg(air[2])),
        }
    )
    for name, value in zip(DELTA_NAMES, delta):
        row[name] = float(value)
    for name, value in zip(COMMAND_NAMES, commands):
        row[name] = float(value)
    for name, value in zip(XCOMMAND_NAMES, x_command):
        row[name] = float(value)
    for name, value in zip(XHAT_NAMES, xhat):
        row[name] = float(value)
    for name, value in zip(WIND_NAMES, wind):
        row[name] = float(value)
    for name, value in zip(FORCE_NAMES, fm):
        row[name] = float(value)
    for name, value in zip(DX_NAMES, dx):
        row[name] = float(value)
    for name, value in zip(AIR_NAMES, air):
        row[name] = float(value)
    return row


def step_to_row(step: SimulationStep) -> dict[str, float]:
    return telemetry_row(
        step.t,
        step.x,
        step.delta,
        step.commands,
        step.x_command,
        step.xhat,
        step.wind,
        step.forces_moments,
        step.derivatives,
        step.air_data,
        step.altitude_state,
    )


def telemetry_rows_from_result(result: SimulationResult, P: Any | None = None, *, exact_source: bool = True) -> list[dict[str, float]]:
    """Convert a :class:`SimulationResult` into list-of-dict telemetry rows.

    ``P`` and ``exact_source`` are accepted for compatibility with earlier helper
    scripts; extended simulation results already carry all logged signals.
    """
    n = len(result.t)
    wind = result.wind if result.wind is not None else np.full((n, 6), np.nan)
    fm = result.forces_moments if result.forces_moments is not None else np.full((n, 6), np.nan)
    dx = result.derivatives if result.derivatives is not None else np.full((n, 12), np.nan)
    if result.air_data is not None:
        air = result.air_data
    elif result.airdata is not None:
        air = np.column_stack([result.airdata, np.full((n, 3), np.nan)])
    else:
        air = np.full((n, 6), np.nan)
    altitude_state = result.altitude_state if result.altitude_state is not None else np.zeros(n, dtype=int)
    return [
        telemetry_row(
            float(result.t[k]),
            result.x[k],
            result.delta[k],
            result.commands[k],
            result.x_command[k],
            result.xhat[k],
            wind[k],
            fm[k],
            dx[k],
            air[k],
            int(altitude_state[k]),
        )
        for k in range(n)
    ]


def result_to_matrix(result: SimulationResult) -> tuple[list[str], np.ndarray]:
    """Return telemetry columns and dense numeric matrix."""
    rows = telemetry_rows_from_result(result)
    if not rows:
        return list(TELEMETRY_COLUMNS), np.empty((0, len(TELEMETRY_COLUMNS)))
    cols = [c for c in TELEMETRY_COLUMNS if c in rows[0]]
    cols += [c for c in rows[0].keys() if c not in cols]
    data = np.asarray([[row.get(c, np.nan) for c in cols] for row in rows], dtype=float)
    return cols, data


def write_rows_csv(path: str | Path, rows: list[dict[str, float]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No telemetry rows to write")
    fieldnames = [c for c in TELEMETRY_COLUMNS if c in rows[0]]
    fieldnames += [c for c in rows[0].keys() if c not in fieldnames]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_result_csv(result: SimulationResult, path: str | Path) -> Path:
    return write_rows_csv(path, telemetry_rows_from_result(result))


def write_telemetry_csv(arg1, arg2=None) -> Path:
    """Compatibility CSV writer.

    Supported forms:

    ``write_telemetry_csv(path, rows)``
        Write precomputed rows.

    ``write_telemetry_csv(result, path)``
        Convert a ``SimulationResult`` and write it.
    """
    if isinstance(arg1, SimulationResult):
        if arg2 is None:
            raise TypeError("write_telemetry_csv(result, path) requires a path")
        return write_result_csv(arg1, arg2)
    if arg2 is None:
        raise TypeError("write_telemetry_csv(path, rows) requires rows")
    return write_rows_csv(arg1, arg2)


class StreamingCSVLogger:
    """CSV logger for live ``SimulationStep`` samples."""

    def __init__(self, filename: str | Path):
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=TELEMETRY_COLUMNS)
        self._writer.writeheader()

    def write(self, step: SimulationStep) -> None:
        row = step_to_row(step)
        self._writer.writerow({key: row.get(key, np.nan) for key in TELEMETRY_COLUMNS})

    def close(self) -> None:
        self._file.flush()
        self._file.close()

    def __enter__(self) -> "StreamingCSVLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def result_summary(result: SimulationResult) -> dict[str, float]:
    x = np.asarray(result.x, dtype=float)
    t = np.asarray(result.t, dtype=float)
    delta = np.asarray(result.delta, dtype=float)
    if result.air_data is not None:
        air = np.asarray(result.air_data, dtype=float)
        Va = air[:, 0]
        alpha = air[:, 1]
        beta = air[:, 2]
    elif result.airdata is not None:
        air3 = np.asarray(result.airdata, dtype=float)
        Va = air3[:, 0]
        alpha = air3[:, 1]
        beta = air3[:, 2]
    else:
        Va = np.linalg.norm(x[:, 3:6], axis=1)
        alpha = np.zeros_like(Va)
        beta = np.zeros_like(Va)
    h = -x[:, 2]
    return {
        "t_final": float(t[-1]),
        "samples": float(len(t)),
        "altitude_final_m": float(h[-1]),
        "altitude_min_m": float(np.min(h)),
        "altitude_max_m": float(np.max(h)),
        "h_final": float(h[-1]),
        "h_min": float(np.min(h)),
        "h_max": float(np.max(h)),
        "north_final_m": float(x[-1, 0]),
        "east_final_m": float(x[-1, 1]),
        "roll_abs_max_deg": float(np.max(np.abs(rad2deg(x[:, 6])))),
        "pitch_abs_max_deg": float(np.max(np.abs(rad2deg(x[:, 7])))),
        "yaw_final_deg": float(rad2deg(x[-1, 8])),
        "psi_final_deg": float(rad2deg(x[-1, 8])),
        "Va_final_mps": float(Va[-1]),
        "Va_min_mps": float(np.min(Va)),
        "Va_max_mps": float(np.max(Va)),
        "Va_final": float(Va[-1]),
        "Va_min": float(np.min(Va)),
        "Va_max": float(np.max(Va)),
        "alpha_abs_max_deg": float(np.max(np.abs(rad2deg(alpha)))),
        "beta_abs_max_deg": float(np.max(np.abs(rad2deg(beta)))),
        "aileron_abs_max_deg": float(np.max(np.abs(rad2deg(delta[:, 1])))),
        "elevator_abs_max_deg": float(np.max(np.abs(rad2deg(delta[:, 0])))),
        "rudder_abs_max_deg": float(np.max(np.abs(rad2deg(delta[:, 2])))),
        "throttle_min": float(np.min(delta[:, 3])),
        "throttle_max": float(np.max(delta[:, 3])),
    }


def print_live_line(t: float, x: np.ndarray, delta: np.ndarray, commands: np.ndarray, air: np.ndarray, altitude_state: int) -> str:
    x = np.asarray(x, dtype=float)
    delta = np.asarray(delta, dtype=float)
    commands = np.asarray(commands, dtype=float)
    air = np.asarray(air, dtype=float)
    return (
        f"t={t:7.2f}s | mode={altitude_state:d} | "
        f"N={x[0]:8.1f} E={x[1]:8.1f} h={-x[2]:7.2f}m | "
        f"Va={air[0]:6.2f}m/s alpha={float(rad2deg(air[1])):7.2f}deg beta={float(rad2deg(air[2])):7.2f}deg | "
        f"phi={float(rad2deg(x[6])):7.2f} theta={float(rad2deg(x[7])):7.2f} psi={float(rad2deg(x[8])):7.2f}deg | "
        f"de={float(rad2deg(delta[0])):7.2f} da={float(rad2deg(delta[1])):7.2f} dr={float(rad2deg(delta[2])):7.2f}deg dt={delta[3]:5.2f} | "
        f"cmd: Va={commands[0]:5.1f} h={commands[1]:6.1f} chi={commands[2]:7.2f}deg"
    )
