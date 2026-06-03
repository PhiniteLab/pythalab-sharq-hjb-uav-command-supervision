"""Closed-loop simulation for the converted ``mavsim_auto`` model.

The top-level Simulink model is represented as

    commands -> Autopilot -> MAV(Forces & Moments + Dynamics) -> states

with the Wind subsystem feeding the MAV block and the draw/plot blocks replaced
by Python-side telemetry/visualization utilities.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterator
import numpy as np

from .parameters import UAVParameters
from .autopilot import Autopilot
from .dynamics import mav_derivatives, mav_derivatives_from_forces, rk4_step
from .forces_moments import forces_moments, air_data
from .wind import zero_wind, DrydenWind
from .commands import mavsim_auto_commands

CommandFn = Callable[[float], np.ndarray]
WindFn = Callable[[float], np.ndarray]


@dataclass
class SimulationResult:
    """Complete sampled output of the converted model.

    The core arrays match the Simulink draw/plot signal structure:
    ``x`` is the true 12-state vector, ``delta`` is the four actuator command
    vector, ``commands`` is the three-command input vector, ``x_command`` is the
    commanded state vector from ``autopilot.m``, and ``xhat`` is the estimated
    state vector.  The additional arrays are Python telemetry extras computed at
    the same sample instants.
    """

    t: np.ndarray
    x: np.ndarray
    delta: np.ndarray
    commands: np.ndarray
    x_command: np.ndarray
    xhat: np.ndarray
    wind: np.ndarray | None = None
    forces_moments: np.ndarray | None = None
    derivatives: np.ndarray | None = None
    air_data: np.ndarray | None = None  # columns: Va, alpha, beta, wind_body_x, wind_body_y, wind_body_z
    airdata: np.ndarray | None = None  # compatibility alias: columns Va, alpha, beta
    altitude_state: np.ndarray | None = None


@dataclass
class SimulationStep:
    """One live sample from ``simulate_mavsim_auto_stream``."""

    t: float
    x: np.ndarray
    delta: np.ndarray
    commands: np.ndarray
    x_command: np.ndarray
    xhat: np.ndarray
    wind: np.ndarray
    forces_moments: np.ndarray
    derivatives: np.ndarray
    air_data: np.ndarray  # [Va, alpha, beta, wind_body_x, wind_body_y, wind_body_z]
    airdata: np.ndarray  # [Va, alpha, beta]
    altitude_state: int


def _default_wind_fn(P: UAVParameters, use_dryden_gusts: bool, wind_fn: WindFn | None) -> WindFn:
    if use_dryden_gusts and wind_fn is None:
        wind_model = DrydenWind(P, enabled=True)
        return lambda _t: wind_model.wind_vector()
    if wind_fn is None:
        return lambda _t: zero_wind(P)
    return wind_fn


def _default_command_fn(P: UAVParameters, command_fn: CommandFn | None) -> CommandFn:
    if command_fn is not None:
        return command_fn
    trim_speed = float(P.Va)
    return lambda t: mavsim_auto_commands(t, Va_cmd=trim_speed)


def _sample_step(
    t: float,
    x: np.ndarray,
    ap: Autopilot,
    P: UAVParameters,
    command_fn: CommandFn,
    wind_fn: WindFn,
    *,
    exact_source: bool,
) -> SimulationStep:
    commands = np.asarray(command_fn(t), dtype=float).reshape(3)
    wind = np.asarray(wind_fn(t), dtype=float).reshape(6)
    Va, alpha, beta, wind_body = air_data(x, wind, P, exact_source=exact_source)
    y_ap = ap.update(x, commands, t, measured_airspeed=Va)
    delta = y_ap[:4]
    x_command = y_ap[4:16]
    xhat = y_ap[16:28]
    fm = forces_moments(x, delta, wind, P, exact_source=exact_source)
    dx = mav_derivatives_from_forces(x, fm[:3], fm[3:], P, exact_source=exact_source)
    air3 = np.array([Va, alpha, beta], dtype=float)
    air6 = np.concatenate([air3, wind_body.astype(float)])
    return SimulationStep(
        float(t),
        x.copy(),
        delta.copy(),
        commands.copy(),
        x_command.copy(),
        xhat.copy(),
        wind.copy(),
        fm.copy(),
        dx.copy(),
        air6.copy(),
        air3.copy(),
        int(ap.altitude_state if ap.altitude_state is not None else -1),
    )


def simulate_mavsim_auto_stream(
    P: UAVParameters,
    *,
    t_final: float = 200.0,
    command_fn: CommandFn | None = None,
    wind_fn: WindFn | None = None,
    use_dryden_gusts: bool = False,
    exact_source: bool = True,
    rk4_substeps: int = 5,
    yield_stride: int = 10,
) -> Iterator[SimulationStep]:
    """Yield live samples while simulating the converted top-level model.

    Parameters mirror :func:`simulate_mavsim_auto`.  ``yield_stride`` controls
    how often samples are yielded.  ``yield_stride=10`` with the default
    ``P.Ts=0.01`` produces visual updates at 0.1 s, the same sample time used by
    the original Simulink ``drawAircraft`` and ``plotStateVariables`` blocks.
    """
    command_fn = _default_command_fn(P, command_fn)
    wind_fn = _default_wind_fn(P, use_dryden_gusts, wind_fn)

    ap = Autopilot(P)
    x = P.initial_state().copy()
    Ts = P.Ts
    steps = int(np.floor(t_final / Ts)) + 1
    substeps = max(1, int(rk4_substeps))
    yield_stride = max(1, int(yield_stride))

    for k in range(steps):
        t = k * Ts
        step = _sample_step(t, x, ap, P, command_fn, wind_fn, exact_source=exact_source)
        if k % yield_stride == 0 or k == steps - 1:
            yield step
        if k == steps - 1:
            break

        dt = Ts / substeps
        # Zero-order hold of the current actuator and wind command over the
        # integration substeps, matching the sampled MATLAB Fcn + continuous MAV.
        for j in range(substeps):
            tj = t + j * dt
            f = lambda tt, xx: mav_derivatives(xx, step.delta, step.wind, P, exact_source=exact_source)
            x = rk4_step(f, x, tj, dt)
            if not np.all(np.isfinite(x)):
                raise FloatingPointError(f"Non-finite state at t={tj:.3f}: {x}")


def simulate_mavsim_auto(
    P: UAVParameters,
    *,
    t_final: float = 200.0,
    command_fn: CommandFn | None = None,
    wind_fn: WindFn | None = None,
    use_dryden_gusts: bool = False,
    exact_source: bool = True,
    rk4_substeps: int = 5,
) -> SimulationResult:
    """Simulate the top-level ``mavsim_auto`` architecture.

    The autopilot is updated at ``P.Ts`` and held constant over the RK4
    integration substeps, matching the zero-order-hold nature of the sampled
    MATLAB Function block in the Simulink model.
    """
    command_fn = _default_command_fn(P, command_fn)
    wind_fn = _default_wind_fn(P, use_dryden_gusts, wind_fn)

    ap = Autopilot(P)
    x = P.initial_state().copy()
    Ts = P.Ts
    steps = int(np.floor(t_final / Ts)) + 1
    substeps = max(1, int(rk4_substeps))

    t_hist = np.zeros(steps)
    x_hist = np.zeros((steps, 12))
    delta_hist = np.zeros((steps, 4))
    commands_hist = np.zeros((steps, 3))
    x_command_hist = np.zeros((steps, 12))
    xhat_hist = np.zeros((steps, 12))
    wind_hist = np.zeros((steps, 6))
    fm_hist = np.zeros((steps, 6))
    dx_hist = np.zeros((steps, 12))
    air_data_hist = np.zeros((steps, 6))
    air_hist = np.zeros((steps, 3))
    altitude_state_hist = np.zeros(steps, dtype=int)

    for k in range(steps):
        t = k * Ts
        step = _sample_step(t, x, ap, P, command_fn, wind_fn, exact_source=exact_source)

        t_hist[k] = step.t
        x_hist[k] = step.x
        delta_hist[k] = step.delta
        commands_hist[k] = step.commands
        x_command_hist[k] = step.x_command
        xhat_hist[k] = step.xhat
        wind_hist[k] = step.wind
        fm_hist[k] = step.forces_moments
        dx_hist[k] = step.derivatives
        air_data_hist[k] = step.air_data
        air_hist[k] = step.airdata
        altitude_state_hist[k] = step.altitude_state

        if k == steps - 1:
            break
        dt = Ts / substeps
        for j in range(substeps):
            tj = t + j * dt
            f = lambda tt, xx: mav_derivatives(xx, step.delta, step.wind, P, exact_source=exact_source)
            x = rk4_step(f, x, tj, dt)
            if not np.all(np.isfinite(x)):
                raise FloatingPointError(f"Non-finite state at t={tj:.3f}: {x}")

    return SimulationResult(
        t_hist,
        x_hist,
        delta_hist,
        commands_hist,
        x_command_hist,
        xhat_hist,
        wind_hist,
        fm_hist,
        dx_hist,
        air_data_hist,
        air_hist,
        altitude_state_hist,
    )
