"""Offline experiment runner for baseline/Q/SHARQ-HJB studies.

This module reuses :class:`uavsim.server.Runtime` so CSV/JSONL experiments
exercise the same plant, wind, actuator, TECS, guidance, Q-table, and SHARQ-HJB
code path as the live WebSocket backend.  It supports:

* paired fixed/Q/SHARQ-HJB comparison summaries over profiles and seeds;
* Q-learning or SHARQ-HJB training with persistent checkpoints;
* frozen-policy evaluation from JSON or NPZ checkpoints;
* a 20-scenario challenging-reference benchmark written under
  ``experiments/results/raw/full-duration`` by default.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Literal, Mapping, Sequence, cast

import numpy as np

from .q_learning import TabularQLearningSupervisor
from .server import (
    FIXED_CONTROLLER_MODE,
    Q_LEARNING_CONTROLLER_MODE,
    SHARQ_HJB_CONTROLLER_MODE,
    Runtime,
    apply_effects,
)
from .sharq_hjb import SHARQHJBResidualSupervisor


ControllerMode = Literal["fixed_matlab_autopilot", "online_q_learning", "sharq_hjb"]
CheckpointFormat = Literal["json", "npz"]
Learner = TabularQLearningSupervisor | SHARQHJBResidualSupervisor

CONTROLLER_MODES: tuple[ControllerMode, ...] = (
    FIXED_CONTROLLER_MODE,
    Q_LEARNING_CONTROLLER_MODE,
    SHARQ_HJB_CONTROLLER_MODE,
)
METHOD_FOLDER: dict[ControllerMode, str] = {
    FIXED_CONTROLLER_MODE: "baseline",
    Q_LEARNING_CONTROLLER_MODE: "baseline-q",
    SHARQ_HJB_CONTROLLER_MODE: "sharq-hjb",
}
METHOD_LABEL: dict[ControllerMode, str] = {
    FIXED_CONTROLLER_MODE: "Baseline autopilot",
    Q_LEARNING_CONTROLLER_MODE: "Baseline + tabular Q residual",
    SHARQ_HJB_CONTROLLER_MODE: "Baseline + SHARQ-HJB residual",
}


@dataclass(frozen=True)
class EpisodeSummary:
    profile: str
    controller_mode: ControllerMode
    seed: int
    duration_s: float
    steps: int
    rms_altitude_error_m: float
    rms_airspeed_error_mps: float
    rms_reference_error_m: float
    mean_reference_error_m: float
    max_reference_error_m: float
    control_energy_integral: float
    saturation_time_fraction: float
    min_airspeed_after_10s_mps: float
    max_abs_roll_rad: float
    max_abs_pitch_rad: float
    max_abs_load_factor_nz: float
    mean_abs_load_factor_nz: float
    safety_time_fraction: float
    safety_violations: int
    q_episode_return: float
    q_table_size: int
    q_updates: int
    finite: bool
    scenario_name: str = "default"
    scenario_description: str = ""
    target_altitude_m: float = 0.0
    circle_diameter_m: float = 0.0
    circle_airspeed_mps: float = 0.0
    circle_direction: int = 1
    steady_wind_n: float = 0.0
    steady_wind_e: float = 0.0
    steady_wind_d: float = 0.0
    gust_body_u: float = 0.0
    gust_body_v: float = 0.0
    gust_body_w: float = 0.0
    turbulence_std: float = 0.0
    residual_active_fraction: float = 0.0
    mean_hard_condition_score: float = 0.0
    mean_hjb_value: float = 0.0
    mean_hjb_advantage: float = 0.0
    mean_hjb_stage_cost: float = 0.0
    shield_active_fraction: float = 0.0
    mean_candidate_count: float = 0.0

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class StepRecord:
    profile: str
    controller_mode: ControllerMode
    seed: int
    t_s: float
    altitude_error_m: float
    airspeed_error_mps: float
    reference_error_m: float
    control_energy: float
    saturation_ratio: float
    roll_rad: float
    pitch_rad: float
    reward: float
    episode_return: float
    td_error: float
    action_index: int
    epsilon: float
    explored: bool
    q_state: str
    q_value: float
    max_next_q: float
    safety_violation: int
    scenario_name: str = "default"
    method: str = ""
    residual_active: bool = False
    hard_condition_score: float = 0.0
    hjb_value: float = 0.0
    hjb_advantage: float = 0.0
    hjb_stage_cost: float = 0.0
    shield_active: bool = False
    candidate_count: int = 0
    load_factor_nz: float = 1.0
    safety_risk_score: float = 0.0

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkScenario:
    """One challenging reference/wind condition for controller comparisons."""

    name: str
    profile: str
    description: str
    effects: Mapping[str, float]
    mission_config: tuple[float, float, float, int] | None = None
    duration_s: float = 45.0
    seed: int = 4101
    straight_length_m: float = 200.0
    lookahead_m: float = 80.0

    def to_dict(self) -> dict[str, float | int | str | None | dict[str, float]]:
        target_altitude_m = None
        circle_diameter_m = None
        circle_airspeed_mps = None
        circle_direction = None
        if self.mission_config is not None:
            target_altitude_m, circle_diameter_m, circle_airspeed_mps, circle_direction = self.mission_config
        return {
            "name": self.name,
            "profile": self.profile,
            "description": self.description,
            "target_altitude_m": target_altitude_m,
            "circle_diameter_m": circle_diameter_m,
            "circle_airspeed_mps": circle_airspeed_mps,
            "circle_direction": circle_direction,
            "duration_s": self.duration_s,
            "seed": self.seed,
            "straight_length_m": self.straight_length_m,
            "lookahead_m": self.lookahead_m,
            "effects": dict(self.effects),
        }


REFERENCE_BENCHMARK_SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario(
        name="orbit_crosswind_turbulence_low_alt",
        profile="loiter_orbit",
        description="Low-altitude 200 m orbit with maximum crosswind and turbulence.",
        mission_config=(180.0, 200.0, 30.0, 1),
        effects={"steady_wind_e": 10.0, "gust_body_v": 3.0, "turbulence_std": 2.0},
        duration_s=45.0,
        seed=4101,
        straight_length_m=160.0,
    ),
    BenchmarkScenario(
        name="tight_orbit_tailwind_gust",
        profile="high_speed_climb_s_turn_200",
        description="Tighter orbit and tailwind/gust stress after climb-out.",
        mission_config=(210.0, 160.0, 34.0, 1),
        effects={"steady_wind_n": 10.0, "gust_body_v": 4.0, "turbulence_std": 1.4},
        duration_s=45.0,
        seed=4102,
        straight_length_m=150.0,
    ),
    BenchmarkScenario(
        name="wide_orbit_headwind_energy",
        profile="racetrack",
        description="Wide faster orbit under strong headwind to stress total-energy recovery.",
        mission_config=(240.0, 320.0, 45.0, 1),
        effects={"steady_wind_n": -10.0, "gust_body_u": -3.5, "turbulence_std": 1.5},
        duration_s=50.0,
        seed=4103,
        straight_length_m=240.0,
    ),
    BenchmarkScenario(
        name="reverse_orbit_shear",
        profile="figure_eight",
        description="Reverse-turn orbit with vertical wind component and lateral shear proxy.",
        mission_config=(200.0, 220.0, 32.0, -1),
        effects={"steady_wind_e": -10.0, "steady_wind_d": 3.0, "gust_body_v": -4.0, "turbulence_std": 1.2},
        duration_s=45.0,
        seed=4104,
        straight_length_m=180.0,
    ),
    BenchmarkScenario(
        name="high_altitude_headwind_climb",
        profile="straight_climb_altitude_hold",
        description="Higher target altitude and headwind during climb/straight/orbit transition.",
        mission_config=(300.0, 260.0, 40.0, 1),
        effects={"steady_wind_n": -8.0, "gust_body_u": -4.0, "gust_body_w": 2.5, "turbulence_std": 1.8},
        duration_s=55.0,
        seed=4105,
        straight_length_m=220.0,
    ),
    BenchmarkScenario(
        name="low_alt_vertical_gust_recovery",
        profile="takeoff_climbout_200",
        description="Lower-altitude reference with maximum vertical gust and turbulence.",
        mission_config=(140.0, 180.0, 28.0, 1),
        effects={"gust_body_w": 4.0, "gust_body_u": -2.0, "turbulence_std": 2.0},
        duration_s=45.0,
        seed=4106,
        straight_length_m=130.0,
    ),
    BenchmarkScenario(
        name="fast_circle_mixed_crosswind",
        profile="runway_takeoff_accel_200",
        description="Fast 300 m circle with diagonal wind and gust coupling.",
        mission_config=(250.0, 300.0, 55.0, 1),
        effects={"steady_wind_n": 8.0, "steady_wind_e": -8.0, "gust_body_u": 3.0, "gust_body_v": -3.0, "turbulence_std": 1.6},
        duration_s=50.0,
        seed=4107,
        straight_length_m=230.0,
    ),
    BenchmarkScenario(
        name="small_radius_crosswind_orbit",
        profile="loiter_orbit",
        description="Small-radius orbit under lateral gusts; intentionally lateral-error dominated.",
        mission_config=(200.0, 140.0, 32.0, -1),
        effects={"steady_wind_e": 10.0, "gust_body_v": -4.0, "turbulence_std": 1.9},
        duration_s=45.0,
        seed=4108,
        straight_length_m=140.0,
    ),
    BenchmarkScenario(
        name="fight_vertical_gust",
        profile="fight_mode",
        description="Cinematic fight-mode S-turn with vertical gusts and headwind.",
        mission_config=(240.0, 480.0, 112.0, 1),
        effects={"steady_wind_n": -9.0, "gust_body_w": 4.0, "turbulence_std": 1.6},
        duration_s=55.0,
        seed=4109,
    ),
    BenchmarkScenario(
        name="fight_crosswind_turbulence",
        profile="fight_mode",
        description="Fight-mode S-turn with maximum crosswind/turbulence and mixed gusts.",
        mission_config=(240.0, 480.0, 112.0, 1),
        effects={"steady_wind_e": -10.0, "gust_body_u": -3.0, "gust_body_v": 4.0, "turbulence_std": 2.0},
        duration_s=55.0,
        seed=4110,
    ),
    BenchmarkScenario(
        name="oblique_takeoff_crosswind",
        profile="runway_takeoff_accel_200",
        description="Runway takeoff with diagonal crosswind and mild vertical gust before orbit capture.",
        mission_config=(170.0, 240.0, 34.0, 1),
        effects={"steady_wind_n": 6.0, "steady_wind_e": 9.0, "gust_body_w": 2.5, "turbulence_std": 1.4},
        duration_s=45.0,
        seed=4111,
        straight_length_m=170.0,
    ),
    BenchmarkScenario(
        name="steep_climb_headwind_turbulence",
        profile="takeoff_climbout_200",
        description="Higher climb target under headwind/turbulence to stress energy allocation.",
        mission_config=(340.0, 280.0, 42.0, 1),
        effects={"steady_wind_n": -10.0, "gust_body_u": -4.0, "gust_body_w": 3.0, "turbulence_std": 2.0},
        duration_s=60.0,
        seed=4112,
        straight_length_m=260.0,
    ),
    BenchmarkScenario(
        name="long_straight_crosswind_rejoin",
        profile="straight_climb_altitude_hold",
        description="Long straight leg before orbit entry with sustained crosswind rejoin demand.",
        mission_config=(220.0, 260.0, 38.0, 1),
        effects={"steady_wind_e": 10.0, "gust_body_v": 3.5, "turbulence_std": 1.3},
        duration_s=55.0,
        seed=4113,
        straight_length_m=360.0,
        lookahead_m=120.0,
    ),
    BenchmarkScenario(
        name="reverse_fast_orbit_tailwind",
        profile="racetrack",
        description="Reverse fast orbit with tailwind and lateral gust forcing.",
        mission_config=(230.0, 260.0, 52.0, -1),
        effects={"steady_wind_n": 9.0, "gust_body_v": -4.0, "turbulence_std": 1.5},
        duration_s=50.0,
        seed=4114,
        straight_length_m=220.0,
    ),
    BenchmarkScenario(
        name="low_speed_vertical_shear_hold",
        profile="loiter_orbit",
        description="Lower-speed orbit with vertical shear proxy and high turbulence.",
        mission_config=(190.0, 220.0, 24.0, 1),
        effects={"steady_wind_d": -4.0, "gust_body_w": 4.0, "turbulence_std": 2.0},
        duration_s=45.0,
        seed=4115,
        straight_length_m=180.0,
    ),
    BenchmarkScenario(
        name="compact_circle_windshift",
        profile="figure_eight",
        description="Compact circle under mixed north/east wind and opposing body gusts.",
        mission_config=(210.0, 150.0, 30.0, 1),
        effects={"steady_wind_n": -7.0, "steady_wind_e": 7.0, "gust_body_u": 4.0, "gust_body_v": -4.0, "turbulence_std": 1.7},
        duration_s=45.0,
        seed=4116,
        straight_length_m=150.0,
    ),
    BenchmarkScenario(
        name="high_speed_climb_crosswind",
        profile="high_speed_climb_s_turn_200",
        description="Fast climb/orbit entry with strong crosswind and positive vertical gust.",
        mission_config=(280.0, 300.0, 58.0, 1),
        effects={"steady_wind_e": -10.0, "gust_body_w": -3.5, "gust_body_v": 3.0, "turbulence_std": 1.6},
        duration_s=55.0,
        seed=4117,
        straight_length_m=250.0,
    ),
    BenchmarkScenario(
        name="fight_tailwind_lateral_gust",
        profile="fight_mode",
        description="Fight-mode S-turn with tailwind and alternating lateral gust stress.",
        mission_config=(240.0, 480.0, 112.0, 1),
        effects={"steady_wind_n": 10.0, "gust_body_v": -4.0, "gust_body_w": 2.5, "turbulence_std": 1.5},
        duration_s=55.0,
        seed=4118,
    ),
    BenchmarkScenario(
        name="fight_diagonal_max_turbulence",
        profile="fight_mode",
        description="Fight-mode S-turn with diagonal wind, mixed gusts, and maximum turbulence.",
        mission_config=(240.0, 480.0, 112.0, 1),
        effects={"steady_wind_n": -8.0, "steady_wind_e": 8.0, "gust_body_u": -4.0, "gust_body_v": 4.0, "gust_body_w": 4.0, "turbulence_std": 2.0},
        duration_s=55.0,
        seed=4119,
    ),
    BenchmarkScenario(
        name="racetrack_reverse_vertical_mixed",
        profile="racetrack",
        description="Reverse-turn racetrack-like orbit with vertical wind and mixed gusts.",
        mission_config=(260.0, 240.0, 44.0, -1),
        effects={"steady_wind_n": -6.0, "steady_wind_e": -6.0, "steady_wind_d": 4.0, "gust_body_u": 3.0, "gust_body_w": -3.0, "turbulence_std": 1.8},
        duration_s=50.0,
        seed=4120,
        straight_length_m=240.0,
    ),
)


def _as_controller_mode(value: str) -> ControllerMode:
    aliases = {
        "fixed": FIXED_CONTROLLER_MODE,
        "baseline": FIXED_CONTROLLER_MODE,
        "fixed_matlab_baseline": FIXED_CONTROLLER_MODE,
        FIXED_CONTROLLER_MODE: FIXED_CONTROLLER_MODE,
        "q": Q_LEARNING_CONTROLLER_MODE,
        "baseline_q": Q_LEARNING_CONTROLLER_MODE,
        "baseline+q": Q_LEARNING_CONTROLLER_MODE,
        Q_LEARNING_CONTROLLER_MODE: Q_LEARNING_CONTROLLER_MODE,
        "sharq": SHARQ_HJB_CONTROLLER_MODE,
        "sharq-hjb": SHARQ_HJB_CONTROLLER_MODE,
        "baseline_sharq_hjb": SHARQ_HJB_CONTROLLER_MODE,
        "baseline+sharq-hjb": SHARQ_HJB_CONTROLLER_MODE,
        SHARQ_HJB_CONTROLLER_MODE: SHARQ_HJB_CONTROLLER_MODE,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported controller mode: {value}")
    return cast(ControllerMode, aliases[value])


def _load_checkpoint(path: str | Path, controller_mode: ControllerMode = Q_LEARNING_CONTROLLER_MODE) -> Learner:
    checkpoint = Path(path)
    if controller_mode == SHARQ_HJB_CONTROLLER_MODE:
        if checkpoint.suffix.lower() == ".npz":
            return SHARQHJBResidualSupervisor.load_npz(checkpoint)
        return SHARQHJBResidualSupervisor.load_json(checkpoint)
    if checkpoint.suffix.lower() == ".npz":
        return TabularQLearningSupervisor.load_npz(checkpoint)
    return TabularQLearningSupervisor.load_json(checkpoint)


def _save_checkpoint(learner: Learner, path: str | Path, fmt: CheckpointFormat | None = None) -> None:
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    chosen = fmt or ("npz" if checkpoint.suffix.lower() == ".npz" else "json")
    if chosen == "npz":
        learner.save_npz(checkpoint)
    else:
        learner.save_json(checkpoint)


def _prepare_runtime(
    *,
    profile: str,
    controller_mode: ControllerMode,
    seed: int,
    checkpoint_in: str | Path | None,
    training_enabled: bool,
    effects: Mapping[str, float] | None = None,
    mission_config: tuple[float, float, float, int] | None = None,
    straight_length_m: float | None = None,
    lookahead_m: float | None = None,
    substeps: int | None = None,
    sample_time_s: float | None = None,
) -> Runtime:
    rt = Runtime.create(profile=profile)
    rt.controller_mode = controller_mode
    if sample_time_s is not None:
        sample_time = float(sample_time_s)
        if not 0.001 <= sample_time <= 0.02:
            raise ValueError("sample_time_s must be in [0.001, 0.02] for bounded benchmark stability")
        rt.P.Ts = sample_time
    if substeps is not None:
        rt.substeps = max(1, int(substeps))
    if effects:
        apply_effects(rt.wind, dict(effects))
    if mission_config is not None:
        rt.mission_profile_config_override = mission_config
    if straight_length_m is not None:
        rt.mission_straight_length_m = max(float(straight_length_m), 1.0)
    if lookahead_m is not None:
        rt.mission_waypoint_lookahead_m = max(float(lookahead_m), 1.0)
    if rt.wind.dryden is not None:
        rt.wind.dryden.seed = int(seed)
        rt.wind.dryden.reset()
    if controller_mode == Q_LEARNING_CONTROLLER_MODE:
        if checkpoint_in is not None:
            learner = _load_checkpoint(checkpoint_in, controller_mode)
            if not isinstance(learner, TabularQLearningSupervisor):  # pragma: no cover - defensive narrowing
                raise TypeError("Expected a tabular Q-learning checkpoint")
            rt.q_learner = learner
        rt.q_learner.seed = int(seed)
        rt.q_learner.rng = np.random.default_rng(int(seed))
        rt.q_learner.training_enabled = bool(training_enabled)
        if not training_enabled:
            rt.q_learner.freeze_for_evaluation()
        rt.q_learner.reset()
    elif controller_mode == SHARQ_HJB_CONTROLLER_MODE:
        if checkpoint_in is not None:
            learner = _load_checkpoint(checkpoint_in, controller_mode)
            if not isinstance(learner, SHARQHJBResidualSupervisor):  # pragma: no cover - defensive narrowing
                raise TypeError("Expected a SHARQ-HJB checkpoint")
            rt.sharq_hjb_learner = learner
        rt.sharq_hjb_learner.seed = int(seed)
        rt.sharq_hjb_learner.rng = np.random.default_rng(int(seed))
        rt.sharq_hjb_learner.training_enabled = bool(training_enabled)
        if not training_enabled:
            rt.sharq_hjb_learner.freeze_for_evaluation()
        rt.sharq_hjb_learner.reset()
    return rt


def run_episode_records(
    *,
    profile: str = "loiter_orbit",
    controller_mode: ControllerMode = FIXED_CONTROLLER_MODE,
    duration_s: float = 60.0,
    seed: int = 23341,
    checkpoint_in: str | Path | None = None,
    checkpoint_out: str | Path | None = None,
    training_enabled: bool | None = None,
    scenario_name: str = "default",
    scenario_description: str = "",
    effects: Mapping[str, float] | None = None,
    mission_config: tuple[float, float, float, int] | None = None,
    straight_length_m: float | None = None,
    lookahead_m: float | None = None,
    step_log_stride: int = 1,
    substeps: int | None = None,
    sample_time_s: float | None = None,
) -> tuple[EpisodeSummary, list[StepRecord]]:
    """Run one offline episode and return scalar metrics plus sampled rows."""

    train = bool(training_enabled) if training_enabled is not None else controller_mode in {
        Q_LEARNING_CONTROLLER_MODE,
        SHARQ_HJB_CONTROLLER_MODE,
    }
    rt = _prepare_runtime(
        profile=profile,
        controller_mode=controller_mode,
        seed=seed,
        checkpoint_in=checkpoint_in,
        training_enabled=train,
        effects=effects,
        mission_config=mission_config,
        straight_length_m=straight_length_m,
        lookahead_m=lookahead_m,
        substeps=substeps,
        sample_time_s=sample_time_s,
    )
    steps = max(1, int(float(duration_s) / rt.P.Ts))
    altitude_errors: list[float] = []
    airspeed_errors: list[float] = []
    reference_errors: list[float] = []
    step_rows: list[StepRecord] = []
    control_energy = 0.0
    saturation_steps = 0
    safety_violations = 0
    min_va_after_10s = float("inf")
    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    max_abs_load = 0.0
    abs_load_samples: list[float] = []
    finite = True
    residual_active_steps = 0
    hard_condition_scores: list[float] = []
    hjb_values: list[float] = []
    hjb_advantages: list[float] = []
    hjb_stage_costs: list[float] = []
    shield_active_steps = 0
    candidate_counts: list[float] = []

    for idx in range(steps):
        info = rt.step()
        if not np.all(np.isfinite(rt.x)):
            finite = False
            break
        commands = np.asarray(info["commands"], dtype=float)
        delta = np.asarray(info["delta"], dtype=float)
        altitude_m = -float(rt.x[2])
        va = float(info["Va"])
        ref_error = sqrt((float(rt.x[0]) - rt.reference_n) ** 2 + (float(rt.x[1]) - rt.reference_e) ** 2)
        altitude_error = float(commands[1]) - altitude_m
        airspeed_error = float(commands[0]) - va
        altitude_errors.append(altitude_error)
        airspeed_errors.append(airspeed_error)
        reference_errors.append(ref_error)
        instant_control_energy = float(np.dot(delta, delta))
        control_energy += instant_control_energy * rt.P.Ts
        saturation_ratio = rt.actuators.saturation_ratio()
        load_factor_nz = float(info.get("load_factor_nz", 1.0))
        abs_load = abs(load_factor_nz)
        max_abs_load = max(max_abs_load, abs_load)
        abs_load_samples.append(abs_load)
        saturation_steps += int(saturation_ratio > 0.98)
        q_metrics = info["q_learning"]
        airborne_envelope = float(rt.t) > 2.0 and altitude_m > 5.0
        episode_safety = int(airborne_envelope and (abs_load > 6.0 or saturation_ratio > 0.98))
        safety_violations += episode_safety
        residual_active_steps += int(q_metrics.residual_active)
        hard_condition_scores.append(float(q_metrics.hard_condition_score))
        hjb_values.append(float(q_metrics.hjb_value))
        hjb_advantages.append(float(q_metrics.hjb_advantage))
        hjb_stage_costs.append(float(q_metrics.hjb_stage_cost))
        shield_active_steps += int(q_metrics.shield_active)
        candidate_counts.append(float(q_metrics.candidate_count))
        if idx * rt.P.Ts > 10.0:
            min_va_after_10s = min(min_va_after_10s, va)
        max_abs_roll = max(max_abs_roll, abs(float(rt.x[6])))
        max_abs_pitch = max(max_abs_pitch, abs(float(rt.x[7])))
        if step_log_stride > 0 and idx % step_log_stride == 0:
            step_rows.append(
                StepRecord(
                    profile=rt.profile,
                    controller_mode=controller_mode,
                    seed=int(seed),
                    t_s=float(rt.t),
                    altitude_error_m=altitude_error,
                    airspeed_error_mps=airspeed_error,
                    reference_error_m=ref_error,
                    control_energy=instant_control_energy,
                    saturation_ratio=float(saturation_ratio),
                    roll_rad=float(rt.x[6]),
                    pitch_rad=float(rt.x[7]),
                    reward=float(q_metrics.reward),
                    episode_return=float(q_metrics.episode_return),
                    td_error=float(q_metrics.td_error),
                    action_index=int(q_metrics.action_index),
                    epsilon=float(q_metrics.epsilon),
                    explored=bool(q_metrics.explored),
                    q_state=",".join(map(str, q_metrics.q_state)) if q_metrics.q_state is not None else "",
                    q_value=float(q_metrics.q_value),
                    max_next_q=float(q_metrics.max_next_q),
                    safety_violation=int(episode_safety),
                    scenario_name=scenario_name,
                    method=str(q_metrics.method),
                    residual_active=bool(q_metrics.residual_active),
                    hard_condition_score=float(q_metrics.hard_condition_score),
                    hjb_value=float(q_metrics.hjb_value),
                    hjb_advantage=float(q_metrics.hjb_advantage),
                    hjb_stage_cost=float(q_metrics.hjb_stage_cost),
                    shield_active=bool(q_metrics.shield_active),
                    candidate_count=int(q_metrics.candidate_count),
                    load_factor_nz=load_factor_nz,
                    safety_risk_score=float(q_metrics.safety_risk_score),
                )
            )

    if checkpoint_out is not None and controller_mode == Q_LEARNING_CONTROLLER_MODE:
        _save_checkpoint(rt.q_learner, checkpoint_out)
    elif checkpoint_out is not None and controller_mode == SHARQ_HJB_CONTROLLER_MODE:
        _save_checkpoint(rt.sharq_hjb_learner, checkpoint_out)

    def _rms(values: list[float]) -> float:
        if not values:
            return float("nan")
        arr = np.asarray(values, dtype=float)
        return float(np.sqrt(np.mean(arr * arr)))

    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(np.mean(np.asarray(values, dtype=float)))

    ref_arr = np.asarray(reference_errors or [float("nan")], dtype=float)
    target_altitude_m, circle_diameter_m, circle_airspeed_mps, circle_direction = rt.active_circle_profile_config()
    if min_va_after_10s == float("inf"):
        min_va_after_10s = float("nan")
    summary = EpisodeSummary(
        profile=rt.profile,
        controller_mode=controller_mode,
        seed=int(seed),
        duration_s=float(duration_s),
        steps=len(reference_errors),
        rms_altitude_error_m=_rms(altitude_errors),
        rms_airspeed_error_mps=_rms(airspeed_errors),
        rms_reference_error_m=_rms(reference_errors),
        mean_reference_error_m=float(np.mean(ref_arr)),
        max_reference_error_m=float(np.max(ref_arr)),
        control_energy_integral=float(control_energy),
        saturation_time_fraction=float(saturation_steps / max(len(reference_errors), 1)),
        min_airspeed_after_10s_mps=float(min_va_after_10s),
        max_abs_roll_rad=float(max_abs_roll),
        max_abs_pitch_rad=float(max_abs_pitch),
        max_abs_load_factor_nz=float(max_abs_load),
        mean_abs_load_factor_nz=_mean(abs_load_samples),
        safety_time_fraction=float(safety_violations / max(len(reference_errors), 1)),
        safety_violations=int(safety_violations),
        q_episode_return=float(
            rt.q_learner.episode_return
            if controller_mode == Q_LEARNING_CONTROLLER_MODE
            else rt.sharq_hjb_learner.episode_return
            if controller_mode == SHARQ_HJB_CONTROLLER_MODE
            else 0.0
        ),
        q_table_size=int(
            len(rt.q_learner.q)
            if controller_mode == Q_LEARNING_CONTROLLER_MODE
            else len(rt.sharq_hjb_learner.q)
            if controller_mode == SHARQ_HJB_CONTROLLER_MODE
            else 0
        ),
        q_updates=int(
            rt.q_learner.updates
            if controller_mode == Q_LEARNING_CONTROLLER_MODE
            else rt.sharq_hjb_learner.updates
            if controller_mode == SHARQ_HJB_CONTROLLER_MODE
            else 0
        ),
        finite=finite,
        scenario_name=scenario_name,
        scenario_description=scenario_description,
        target_altitude_m=float(target_altitude_m),
        circle_diameter_m=float(circle_diameter_m),
        circle_airspeed_mps=float(circle_airspeed_mps),
        circle_direction=int(circle_direction),
        steady_wind_n=float(rt.wind.steady_n),
        steady_wind_e=float(rt.wind.steady_e),
        steady_wind_d=float(rt.wind.steady_d),
        gust_body_u=float(rt.wind.gust_body_u),
        gust_body_v=float(rt.wind.gust_body_v),
        gust_body_w=float(rt.wind.gust_body_w),
        turbulence_std=float(rt.wind.turbulence_std),
        residual_active_fraction=float(residual_active_steps / max(len(reference_errors), 1)),
        mean_hard_condition_score=_mean(hard_condition_scores),
        mean_hjb_value=_mean(hjb_values),
        mean_hjb_advantage=_mean(hjb_advantages),
        mean_hjb_stage_cost=_mean(hjb_stage_costs),
        shield_active_fraction=float(shield_active_steps / max(len(reference_errors), 1)),
        mean_candidate_count=_mean(candidate_counts),
    )
    return summary, step_rows


def run_episode_summary(
    *,
    profile: str = "loiter_orbit",
    controller_mode: ControllerMode = FIXED_CONTROLLER_MODE,
    duration_s: float = 60.0,
    seed: int = 23341,
    checkpoint_in: str | Path | None = None,
    checkpoint_out: str | Path | None = None,
    training_enabled: bool | None = None,
    step_log_path: str | Path | None = None,
    scenario_name: str = "default",
    scenario_description: str = "",
    effects: Mapping[str, float] | None = None,
    mission_config: tuple[float, float, float, int] | None = None,
    straight_length_m: float | None = None,
    lookahead_m: float | None = None,
    step_log_stride: int = 1,
    substeps: int | None = None,
    sample_time_s: float | None = None,
) -> EpisodeSummary:
    """Run one offline episode and return scalar metrics."""

    summary, rows = run_episode_records(
        profile=profile,
        controller_mode=controller_mode,
        duration_s=duration_s,
        seed=seed,
        checkpoint_in=checkpoint_in,
        checkpoint_out=checkpoint_out,
        training_enabled=training_enabled,
        scenario_name=scenario_name,
        scenario_description=scenario_description,
        effects=effects,
        mission_config=mission_config,
        straight_length_m=straight_length_m,
        lookahead_m=lookahead_m,
        step_log_stride=step_log_stride,
        substeps=substeps,
        sample_time_s=sample_time_s,
    )
    if step_log_path is not None:
        write_step_jsonl(step_log_path, rows)
    return summary


def compare_fixed_vs_q_learning(*, profile: str = "loiter_orbit", duration_s: float = 60.0, seed: int = 23341) -> list[EpisodeSummary]:
    """Run paired fixed/Q-learning summaries on the same runtime profile and seed."""

    return [
        run_episode_summary(profile=profile, controller_mode=FIXED_CONTROLLER_MODE, duration_s=duration_s, seed=seed),
        run_episode_summary(profile=profile, controller_mode=Q_LEARNING_CONTROLLER_MODE, duration_s=duration_s, seed=seed),
    ]


def compare_all_methods(*, profile: str = "loiter_orbit", duration_s: float = 60.0, seed: int = 23341) -> list[EpisodeSummary]:
    """Run baseline, baseline+Q, and baseline+SHARQ-HJB on one profile."""

    return [run_episode_summary(profile=profile, controller_mode=mode, duration_s=duration_s, seed=seed) for mode in CONTROLLER_MODES]


def run_batch(
    *,
    profiles: list[str],
    seeds: list[int],
    duration_s: float,
    controller_modes: list[ControllerMode],
    checkpoint_in: str | Path | None = None,
    training_enabled: bool = False,
) -> list[EpisodeSummary]:
    summaries: list[EpisodeSummary] = []
    for profile in profiles:
        for seed in seeds:
            for mode in controller_modes:
                summaries.append(
                    run_episode_summary(
                        profile=profile,
                        controller_mode=mode,
                        duration_s=duration_s,
                        seed=seed,
                        checkpoint_in=checkpoint_in if mode != FIXED_CONTROLLER_MODE else None,
                        training_enabled=training_enabled if mode != FIXED_CONTROLLER_MODE else False,
                    )
                )
    return summaries


def write_summary_csv(path: str | Path, summaries: Sequence[EpisodeSummary]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary.to_dict() for summary in summaries]
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_step_jsonl(path: str | Path, rows: Sequence[StepRecord]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def _aggregate_summaries(summaries: Sequence[EpisodeSummary]) -> list[dict[str, float | int | str]]:
    metrics = [
        "rms_altitude_error_m",
        "rms_airspeed_error_mps",
        "rms_reference_error_m",
        "mean_reference_error_m",
        "max_reference_error_m",
        "control_energy_integral",
        "saturation_time_fraction",
        "min_airspeed_after_10s_mps",
        "max_abs_roll_rad",
        "max_abs_pitch_rad",
        "max_abs_load_factor_nz",
        "mean_abs_load_factor_nz",
        "safety_time_fraction",
        "safety_violations",
        "q_episode_return",
        "q_table_size",
        "q_updates",
        "residual_active_fraction",
        "mean_hard_condition_score",
        "mean_hjb_value",
        "mean_hjb_advantage",
        "mean_hjb_stage_cost",
        "shield_active_fraction",
        "mean_candidate_count",
    ]
    rows: list[dict[str, float | int | str]] = []
    for mode in CONTROLLER_MODES:
        selected = [summary for summary in summaries if summary.controller_mode == mode]
        if not selected:
            continue
        row: dict[str, float | int | str] = {
            "controller_mode": mode,
            "method_label": METHOD_LABEL[mode],
            "episodes": len(selected),
        }
        for metric in metrics:
            values = np.asarray([float(getattr(summary, metric)) for summary in selected], dtype=float)
            finite_values = values[np.isfinite(values)]
            row[f"mean_{metric}"] = float(np.mean(finite_values)) if finite_values.size else float("nan")
            row[f"std_{metric}"] = float(np.std(finite_values)) if finite_values.size else float("nan")
            row[f"ci95_{metric}"] = (
                float(1.96 * np.std(finite_values, ddof=1) / np.sqrt(finite_values.size))
                if finite_values.size > 1
                else 0.0
                if finite_values.size == 1
                else float("nan")
            )
        rows.append(row)
    return rows


def write_dicts_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_method_readme(path: Path, mode: ControllerMode, summaries: Sequence[EpisodeSummary]) -> None:
    aggregate = _aggregate_summaries(summaries)
    aggregate_row = aggregate[0] if aggregate else {}
    path.write_text(
        "\n".join(
            [
                f"# {METHOD_LABEL[mode]} Results",
                "",
                "This directory contains method-specific scalar metrics and step-level time series.",
                "",
                "## Files",
                "",
                "- `episode_summary.csv`: scalar metrics per scenario/seed.",
                "- `steps.jsonl`: step-level time series when non-empty.",
                "- `aggregate_metrics.csv`: method-level mean/std summaries.",
                "- `scenario_catalog.json`: scenario definitions copied for local context.",
                "",
                "## Quick metrics",
                "",
                f"- Episodes: {len(summaries)}",
                f"- Mean RMS reference error [m]: {aggregate_row.get('mean_rms_reference_error_m', 'n/a')}",
                f"- Mean RMS altitude error [m]: {aggregate_row.get('mean_rms_altitude_error_m', 'n/a')}",
                f"- Mean RMS airspeed error [m/s]: {aggregate_row.get('mean_rms_airspeed_error_mps', 'n/a')}",
                f"- Mean actuator-command activity index: {aggregate_row.get('mean_control_energy_integral', 'n/a')}",
                "",
                "These are simulator result files, not certified flight evidence.",
            ]
        ),
        encoding="utf-8",
    )


def run_reference_benchmark(
    *,
    output_dir: str | Path = Path("experiments/results/raw/full-duration"),
    scenarios: Sequence[BenchmarkScenario] = REFERENCE_BENCHMARK_SCENARIOS,
    controller_modes: Sequence[ControllerMode] = CONTROLLER_MODES,
    seeds: Sequence[int] | None = None,
    duration_override_s: float | None = None,
    step_log_stride: int = 10,
    substeps: int | None = None,
    sample_time_s: float | None = None,
) -> list[EpisodeSummary]:
    """Run the reference benchmark and write experiment outputs."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    scenario_catalog = [scenario.to_dict() for scenario in scenarios]
    (root / "scenario_catalog.json").write_text(json.dumps(scenario_catalog, indent=2, sort_keys=True), encoding="utf-8")

    summaries: list[EpisodeSummary] = []
    rows_by_mode: dict[ControllerMode, list[StepRecord]] = {mode: [] for mode in controller_modes}
    summaries_by_mode: dict[ControllerMode, list[EpisodeSummary]] = {mode: [] for mode in controller_modes}

    for scenario in scenarios:
        run_seeds = list(seeds) if seeds is not None else [scenario.seed]
        duration_s = float(duration_override_s) if duration_override_s is not None else float(scenario.duration_s)
        for seed in run_seeds:
            for mode in controller_modes:
                summary, rows = run_episode_records(
                    profile=scenario.profile,
                    controller_mode=mode,
                    duration_s=duration_s,
                    seed=int(seed),
                    training_enabled=mode != FIXED_CONTROLLER_MODE,
                    scenario_name=scenario.name,
                    scenario_description=scenario.description,
                    effects=scenario.effects,
                    mission_config=scenario.mission_config,
                    straight_length_m=scenario.straight_length_m,
                    lookahead_m=scenario.lookahead_m,
                    step_log_stride=step_log_stride,
                    substeps=substeps,
                    sample_time_s=sample_time_s,
                )
                summaries.append(summary)
                summaries_by_mode[mode].append(summary)
                rows_by_mode[mode].extend(rows)

    write_summary_csv(root / "comparative" / "all_episode_summary.csv", summaries)
    write_dicts_csv(root / "comparative" / "aggregate_by_method.csv", _aggregate_summaries(summaries))
    (root / "comparative" / "scenario_catalog.json").write_text(
        json.dumps(scenario_catalog, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for mode in controller_modes:
        method_dir = root / METHOD_FOLDER[mode]
        method_dir.mkdir(parents=True, exist_ok=True)
        write_summary_csv(method_dir / "episode_summary.csv", summaries_by_mode[mode])
        write_step_jsonl(method_dir / "steps.jsonl", rows_by_mode[mode])
        write_dicts_csv(method_dir / "aggregate_metrics.csv", _aggregate_summaries(summaries_by_mode[mode]))
        (method_dir / "scenario_catalog.json").write_text(
            json.dumps(scenario_catalog, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_method_readme(method_dir / "README.md", mode, summaries_by_mode[mode])

    return summaries


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_seed_list(value: str) -> list[int]:
    return [int(part) for part in _parse_csv_list(value)]


def _parse_controller_modes(value: str) -> list[ControllerMode]:
    return [_as_controller_mode(part) for part in _parse_csv_list(value)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run UAV baseline/Q/SHARQ-HJB experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--profile", default="loiter_orbit")
        p.add_argument("--duration", type=float, default=60.0)
        p.add_argument("--seed", type=int, default=23341)
        p.add_argument("--summary-csv", type=Path, default=Path("experiments/tmp/uavsim/summary.csv"))

    compare = sub.add_parser("compare", help="Run paired fixed/Q-learning summaries.")
    add_common(compare)
    compare.add_argument("--all-methods", action="store_true", help="Compare baseline, Q, and SHARQ-HJB.")

    train = sub.add_parser("train", help="Train online Q-learning or SHARQ-HJB and write a checkpoint.")
    add_common(train)
    train.add_argument("--controller", default=Q_LEARNING_CONTROLLER_MODE)
    train.add_argument("--checkpoint-in", type=Path)
    train.add_argument("--checkpoint-out", type=Path, default=Path("experiments/tmp/uavsim/q_learning_policy.json"))
    train.add_argument("--step-jsonl", type=Path)
    train.add_argument("--step-log-stride", type=int, default=1)

    evaluate = sub.add_parser("eval", help="Evaluate a frozen Q-learning/SHARQ-HJB checkpoint.")
    add_common(evaluate)
    evaluate.add_argument("--controller", default=Q_LEARNING_CONTROLLER_MODE)
    evaluate.add_argument("--checkpoint-in", type=Path, required=True)
    evaluate.add_argument("--step-jsonl", type=Path)
    evaluate.add_argument("--step-log-stride", type=int, default=1)

    batch = sub.add_parser("batch", help="Run CSV batch over comma-separated profiles/seeds/modes.")
    batch.add_argument("--profiles", default="loiter_orbit,racetrack,fight_mode")
    batch.add_argument("--seeds", default="23341,23342,23343")
    batch.add_argument("--duration", type=float, default=60.0)
    batch.add_argument("--controller-modes", default=",".join(CONTROLLER_MODES))
    batch.add_argument("--checkpoint-in", type=Path)
    batch.add_argument("--summary-csv", type=Path, default=Path("experiments/tmp/uavsim/batch_summary.csv"))
    batch.add_argument("--train", action="store_true", help="Allow Q/SHARQ-HJB updates during batch runs.")

    benchmark = sub.add_parser("benchmark", help="Run the reference benchmark scenarios.")
    benchmark.add_argument("--output-dir", type=Path, default=Path("experiments/results/raw/full-duration"))
    benchmark.add_argument("--duration", type=float, help="Override every scenario duration.")
    benchmark.add_argument("--seeds", default="", help="Comma-separated seed override. Empty uses each scenario seed.")
    benchmark.add_argument("--seed-count", type=int, default=0, help="Use this many sequential seeds when --seeds is empty.")
    benchmark.add_argument("--seed-start", type=int, default=5001, help="First seed for --seed-count.")
    benchmark.add_argument("--max-scenarios", type=int, default=0, help="Limit scenario count for smoke runs; 0 uses all scenarios.")
    benchmark.add_argument("--controller-modes", default=",".join(CONTROLLER_MODES))
    benchmark.add_argument("--step-log-stride", type=int, default=10, help="Use 0 to skip time-series JSONL and write scalar summaries only.")
    benchmark.add_argument("--substeps", type=int)
    benchmark.add_argument("--sample-time", type=float, help="Optional coarse autopilot sample time for large statistical sweeps.")

    args = parser.parse_args(argv)
    if args.command == "compare":
        summaries = compare_all_methods(profile=args.profile, duration_s=args.duration, seed=args.seed) if args.all_methods else compare_fixed_vs_q_learning(profile=args.profile, duration_s=args.duration, seed=args.seed)
        write_summary_csv(args.summary_csv, summaries)
    elif args.command == "train":
        mode = _as_controller_mode(args.controller)
        if mode == FIXED_CONTROLLER_MODE:
            raise ValueError("Training requires a residual controller mode")
        summaries = [
            run_episode_summary(
                profile=args.profile,
                controller_mode=mode,
                duration_s=args.duration,
                seed=args.seed,
                checkpoint_in=args.checkpoint_in,
                checkpoint_out=args.checkpoint_out,
                training_enabled=True,
                step_log_path=args.step_jsonl,
                step_log_stride=args.step_log_stride,
            )
        ]
        write_summary_csv(args.summary_csv, summaries)
    elif args.command == "eval":
        mode = _as_controller_mode(args.controller)
        summaries = [
            run_episode_summary(
                profile=args.profile,
                controller_mode=mode,
                duration_s=args.duration,
                seed=args.seed,
                checkpoint_in=args.checkpoint_in,
                training_enabled=False,
                step_log_path=args.step_jsonl,
                step_log_stride=args.step_log_stride,
            )
        ]
        write_summary_csv(args.summary_csv, summaries)
    elif args.command == "batch":
        summaries = run_batch(
            profiles=_parse_csv_list(args.profiles),
            seeds=_parse_seed_list(args.seeds),
            duration_s=args.duration,
            controller_modes=_parse_controller_modes(args.controller_modes),
            checkpoint_in=args.checkpoint_in,
            training_enabled=bool(args.train),
        )
        write_summary_csv(args.summary_csv, summaries)
    elif args.command == "benchmark":
        if str(args.seeds).strip():
            seed_override = _parse_seed_list(args.seeds)
        elif int(args.seed_count) > 0:
            seed_override = list(range(int(args.seed_start), int(args.seed_start) + min(int(args.seed_count), 50)))
        else:
            seed_override = None
        scenarios = REFERENCE_BENCHMARK_SCENARIOS
        if int(args.max_scenarios) > 0:
            scenarios = scenarios[: int(args.max_scenarios)]
        summaries = run_reference_benchmark(
            output_dir=args.output_dir,
            scenarios=scenarios,
            controller_modes=_parse_controller_modes(args.controller_modes),
            seeds=seed_override,
            duration_override_s=args.duration,
            step_log_stride=max(0, int(args.step_log_stride)),
            substeps=args.substeps,
            sample_time_s=args.sample_time,
        )
        print(json.dumps({"episodes": len(summaries), "output_dir": str(args.output_dir)}, sort_keys=True))
    else:  # pragma: no cover - argparse enforces command choices
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
