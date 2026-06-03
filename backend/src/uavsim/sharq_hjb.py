"""SHARQ-HJB residual supervisor for autopilot-aware UAV guidance.

The method implemented here keeps the fixed-gain autopilot as the only
low-level actuator controller.  SHARQ-HJB acts one layer above it by choosing
small residuals on the commanded airspeed, altitude, and heading references.

The code intentionally uses a transparent quadratic Hamilton-Jacobi proxy
instead of a hidden neural approximation: experiments can inspect every
candidate residual action, its predicted value change, and whether the
Lyapunov/CBF-style shield filtered it.  The tabular Q-table is retained so
baseline+Q and SHARQ-HJB can be compared with the same finite state/action
interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np

from .q_learning import ACTIONS, QLearningConfig, QLearningMetrics, STATE_SIZE, TabularQLearningSupervisor


SHARQ_HJB_ALGORITHM_VERSION = "sharq_hjb_discrete_hjb_guidance_residual_v2"
SHARQ_HJB_STATE_ENCODER_VERSION = "semi_discrete_hjb_energy_lateral_load_bins_v2"

_HJB_FEATURE_CENTERS: tuple[tuple[float, ...], ...] = (
    (-1.5, 0.0, 1.5),  # airspeed error
    (-1.5, 0.0, 1.5),  # altitude error
    (0.0, 0.7, 1.8, 4.0),  # horizontal reference error
    (-1.5, 0.0, 1.5),  # lateral path error
    (0.0, 0.8, 1.6),  # wind/turbulence stress
    (0.0, 1.0),  # saturation stress
    (0.0, 1.0),  # load-factor stress
)
_HJB_FEATURE_WEIGHTS = np.array([1.05, 0.85, 1.20, 0.95, 0.32, 0.58, 1.35], dtype=float)


def _nearest_feature_key(features: np.ndarray) -> tuple[int, ...]:
    clipped = np.clip(np.asarray(features, dtype=float), -4.0, 4.0)
    return tuple(
        int(np.argmin(np.abs(np.asarray(axis, dtype=float) - clipped[idx])))
        for idx, axis in enumerate(_HJB_FEATURE_CENTERS)
    )


def _features_from_key(key: tuple[int, ...]) -> np.ndarray:
    return np.asarray([_HJB_FEATURE_CENTERS[idx][bin_idx] for idx, bin_idx in enumerate(key)], dtype=float)


def _quadratic_feature_value(features: np.ndarray) -> float:
    clipped = np.clip(np.asarray(features, dtype=float), -6.0, 6.0)
    return float(np.dot(_HJB_FEATURE_WEIGHTS, clipped * clipped))


def _feature_action_cost(action_index: int) -> float:
    d_va, d_h, d_chi = ACTIONS[action_index]
    return 0.045 * (abs(d_va) / 2.0 + abs(d_h) / 10.0 + 1.35 * abs(d_chi) / 3.0)


def _semi_lagrangian_feature_step(features: np.ndarray, action_index: int) -> np.ndarray:
    """Finite-state HJB surrogate dynamics in normalized feature space."""

    d_va, d_h, d_chi = ACTIONS[action_index]
    next_features = np.asarray(features, dtype=float).copy()
    next_features[:4] *= np.array([0.92, 0.95, 0.90, 0.92], dtype=float)
    next_features[4:] *= np.array([0.98, 0.96, 0.94], dtype=float)

    next_features[0] -= 1.30 * d_va / 18.0
    next_features[1] += 0.65 * d_h / 80.0
    if d_h < 0.0 and features[0] > 0.20:
        next_features[0] -= 1.80 * abs(d_h) / 180.0
        next_features[1] += 0.15 * d_h / 80.0
    elif d_h > 0.0 and features[0] > 0.35:
        next_features[0] += 0.70 * d_h / 80.0
    next_features[3] += 4.0 * d_chi / 80.0
    lateral_improvement = max(abs(features[3]) - abs(next_features[3]), 0.0)
    energy_gain = 0.65 * max(d_va, 0.0)
    if d_h < 0.0 and features[0] > 0.20:
        energy_gain += 1.25 * abs(d_h) / 10.0
    next_features[2] = max(0.0, next_features[2] - energy_gain / 180.0 - 0.75 * lateral_improvement)
    next_features[5] += 0.04 * (abs(d_va) / 2.0 + abs(d_h) / 10.0 + abs(d_chi) / 3.0)
    if abs(d_chi) > 0.0:
        next_features[6] += 0.08
    if d_h > 0.0 and features[0] > 0.35:
        next_features[6] += 0.06
    return np.clip(next_features, -4.0, 4.0)


@lru_cache(maxsize=16)
def _build_discrete_hjb_value_table(iterations: int, gamma: float) -> dict[tuple[int, ...], float]:
    """Solve a small semi-discrete HJB/Bellman surrogate by value iteration.

    This is not a continuous high-dimensional PDE solve.  It is a deterministic
    semi-Lagrangian residual-abstraction solve over the normalized error/load
    grid above, which gives SHARQ-HJB a reproducible multi-step value critic
    instead of a purely local one-step Hamiltonian proxy.
    """

    keys = [tuple(int(v) for v in key) for key in product(*(range(len(axis)) for axis in _HJB_FEATURE_CENTERS))]
    values = {key: _quadratic_feature_value(_features_from_key(key)) for key in keys}
    for _ in range(max(1, int(iterations))):
        updated: dict[tuple[int, ...], float] = {}
        for key in keys:
            features = _features_from_key(key)
            action_values: list[float] = []
            for action_index in range(len(ACTIONS)):
                next_features = _semi_lagrangian_feature_step(features, action_index)
                next_key = _nearest_feature_key(next_features)
                stage = _quadratic_feature_value(next_features) + _feature_action_cost(action_index)
                # Strongly penalize residuals that push the load/saturation
                # surrogate into the unsafe bins.
                stage += 1.8 * max(float(next_features[6]) - 0.65, 0.0)
                stage += 0.8 * max(float(next_features[5]) - 0.65, 0.0)
                action_values.append(stage + float(gamma) * values[next_key])
            updated[key] = float(min(action_values))
        values = updated
    return values


@dataclass(frozen=True)
class SHARQHJBConfig(QLearningConfig):
    """Configuration for the stability-shielded HJB-guided Q residual."""

    hjb_weight: float = 0.85
    hjb_reward_weight: float = 0.04
    beta_discount: float = 0.035
    value_dt_s: float = 0.25
    shield_tolerance: float = 0.08
    clf_relaxation: float = 0.32
    discrete_hjb_weight: float = 0.38
    discrete_hjb_iterations: int = 5


@dataclass(frozen=True)
class HJBCandidate:
    action_index: int
    hjb_value: float
    hjb_advantage: float
    hjb_stage_cost: float
    shielded: bool
    score: float


@dataclass
class SHARQHJBResidualSupervisor(TabularQLearningSupervisor):
    """HJB-guided, shielded residual Q-learning supervisor.

    The supervisor inherits the tabular state/action/checkpoint interface from
    :class:`TabularQLearningSupervisor`, but changes action selection:

    1. hard-condition gating keeps the residual disabled in nominal flight;
    2. a CLF/CBF-inspired shield filters candidate residuals while preserving
       the baseline no-op action;
    3. a quadratic HJB proxy biases Q selection toward residuals that reduce
       an autopilot-aware value function.
    """

    hjb_weight: float = 0.85
    hjb_reward_weight: float = 0.04
    beta_discount: float = 0.035
    value_dt_s: float = 0.25
    shield_tolerance: float = 0.08
    clf_relaxation: float = 0.32
    discrete_hjb_weight: float = 0.38
    discrete_hjb_iterations: int = 5
    last_candidate: HJBCandidate = field(
        default_factory=lambda: HJBCandidate(0, 0.0, 0.0, 0.0, False, 0.0)
    )
    last_candidates: list[HJBCandidate] = field(default_factory=list)
    fallback_q_mode: bool = False

    @classmethod
    def from_config(cls, config: QLearningConfig) -> "SHARQHJBResidualSupervisor":
        hjb_config = config if isinstance(config, SHARQHJBConfig) else SHARQHJBConfig(
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_min=config.epsilon_min,
            epsilon_decay=config.epsilon_decay,
            training_enabled=config.training_enabled,
            seed=config.seed,
        )
        return cls(
            alpha=hjb_config.alpha,
            gamma=hjb_config.gamma,
            epsilon=hjb_config.epsilon,
            epsilon_min=hjb_config.epsilon_min,
            epsilon_decay=hjb_config.epsilon_decay,
            seed=hjb_config.seed,
            training_enabled=hjb_config.training_enabled,
            hjb_weight=hjb_config.hjb_weight,
            hjb_reward_weight=hjb_config.hjb_reward_weight,
            beta_discount=hjb_config.beta_discount,
            value_dt_s=hjb_config.value_dt_s,
            shield_tolerance=hjb_config.shield_tolerance,
            clf_relaxation=hjb_config.clf_relaxation,
            discrete_hjb_weight=hjb_config.discrete_hjb_weight,
            discrete_hjb_iterations=hjb_config.discrete_hjb_iterations,
        )

    def reset(self) -> None:
        super().reset()
        self.last_candidate = HJBCandidate(0, 0.0, 0.0, 0.0, False, 0.0)
        self.last_candidates = []
        self.fallback_q_mode = False
        self.metrics.method = "sharq_hjb"

    def _feature_vector(
        self,
        *,
        airspeed_error: float,
        altitude_error: float,
        reference_error: float,
        cross_track_error: float,
        radial_error: float,
        wind_speed: float,
        turbulence_std: float,
        saturation_ratio: float,
        load_factor_nz: float = 1.0,
    ) -> np.ndarray:
        """Autopilot-aware normalized HJB state features.

        The first three entries are command-tracking/reference errors.  The
        lateral entry uses the active path error (straight cross-track or orbit
        radial).  Wind/turbulence and actuator saturation are included as
        disturbance/stress terms so the residual is only rewarded for helping
        the autopilot in genuinely hard conditions.
        """

        lateral_error = float(cross_track_error) if abs(float(cross_track_error)) >= 1e-9 else float(radial_error)
        wind_stress = float(wind_speed) + 2.0 * float(turbulence_std)
        sat_stress = max(float(saturation_ratio) - 0.55, 0.0)
        load_stress = max(abs(float(load_factor_nz)) - 3.0, 0.0)
        return np.array(
            [
                float(airspeed_error) / 18.0,
                float(altitude_error) / 80.0,
                float(reference_error) / 180.0,
                lateral_error / 80.0,
                wind_stress / 16.0,
                sat_stress / 0.45,
                load_stress / 3.0,
            ],
            dtype=float,
        )

    def _value_from_features(self, features: np.ndarray) -> float:
        quadratic = _quadratic_feature_value(features)
        table = _build_discrete_hjb_value_table(
            int(self.discrete_hjb_iterations),
            round(float(self.gamma), 6),
        )
        discrete = table[_nearest_feature_key(np.asarray(features, dtype=float))]
        weight = float(np.clip(self.discrete_hjb_weight, 0.0, 1.0))
        return float((1.0 - weight) * quadratic + weight * discrete)

    def _predict_features(
        self,
        *,
        action_index: int,
        airspeed_error: float,
        altitude_error: float,
        reference_error: float,
        cross_track_error: float,
        radial_error: float,
        wind_speed: float,
        turbulence_std: float,
        saturation_ratio: float,
        load_factor_nz: float = 1.0,
    ) -> np.ndarray:
        d_va, d_h, d_chi = ACTIONS[action_index]
        va_error = float(airspeed_error)
        h_error = float(altitude_error)
        ref_error = float(reference_error)
        cross_error = float(cross_track_error)
        radial = float(radial_error)

        # Residuals change references, but their intended effect is through
        # autopilot energy allocation.  These local closed-loop predictors are
        # deliberately conservative; they encode directionality rather than a
        # claimed high-fidelity plant model.
        if va_error >= 0.0:
            va_pred = va_error - 1.35 * d_va
        else:
            va_pred = va_error - 0.85 * d_va
        # Altitude residuals immediately change the tracked reference and only
        # partly convert into closed-loop altitude response over the short HJB
        # prediction horizon.  Lowering h_c under an energy deficit is treated
        # as an energy-recovery manoeuvre, not as a pure altitude-tracking loss.
        h_pred = h_error + 0.65 * d_h
        if d_h < 0.0 and va_error > 4.0:
            va_pred -= 1.8 * abs(d_h) / 10.0
            h_pred += 0.15 * d_h
        elif d_h > 0.0 and va_error > 6.0:
            va_pred += 0.7 * d_h / 10.0
        cross_pred = cross_error + 4.0 * d_chi
        radial_sign = 1.0 if radial >= 0.0 else -1.0
        radial_pred = radial_sign * max(abs(radial) - 2.0 * abs(d_chi), 0.0)
        if abs(cross_error) > 1e-9:
            lateral_improvement = max(abs(cross_error) - abs(cross_pred), 0.0)
        else:
            lateral_improvement = max(abs(radial) - abs(radial_pred), 0.0)
        energy_ref_gain = 0.65 * max(d_va, 0.0)
        if d_h < 0.0 and va_error > 4.0:
            energy_ref_gain += 1.25 * abs(d_h) / 10.0
        ref_pred = max(0.0, ref_error - energy_ref_gain - 0.16 * abs(d_h) - 0.75 * lateral_improvement)
        sat_pred = float(saturation_ratio) + 0.025 * (
            abs(d_va) / 2.0 + abs(d_h) / 10.0 + abs(d_chi) / 3.0
        )
        if d_va > 0.0 and va_error > 4.0:
            sat_pred = max(0.0, sat_pred - 0.015)
        load_pred = float(load_factor_nz)
        if abs(d_chi) > 0.0:
            load_pred += np.sign(load_pred if abs(load_pred) > 1e-6 else 1.0) * 0.55
        if d_h > 0.0 and va_error > 6.0:
            load_pred += 0.35
        if d_h < 0.0 and va_error > 4.0:
            load_pred *= 0.92
        return self._feature_vector(
            airspeed_error=va_pred,
            altitude_error=h_pred,
            reference_error=ref_pred,
            cross_track_error=cross_pred,
            radial_error=radial_pred,
            wind_speed=wind_speed,
            turbulence_std=turbulence_std,
            saturation_ratio=sat_pred,
            load_factor_nz=load_pred,
        )

    def _stage_cost_from_features(self, features: np.ndarray, action_index: int) -> float:
        d_va, d_h, d_chi = ACTIONS[action_index]
        tracking = self._value_from_features(features)
        residual_cost = 0.045 * (
            abs(d_va) / 2.0 + abs(d_h) / 10.0 + 1.35 * abs(d_chi) / 3.0
        )
        return float(tracking + residual_cost)

    def _candidate_evaluations(
        self,
        *,
        valid_actions: list[int],
        values: np.ndarray,
        airspeed_error: float,
        altitude_error: float,
        reference_error: float,
        cross_track_error: float,
        radial_error: float,
        wind_speed: float,
        turbulence_std: float,
        saturation_ratio: float,
        load_factor_nz: float = 1.0,
    ) -> list[HJBCandidate]:
        current_features = self._feature_vector(
            airspeed_error=airspeed_error,
            altitude_error=altitude_error,
            reference_error=reference_error,
            cross_track_error=cross_track_error,
            radial_error=radial_error,
            wind_speed=wind_speed,
            turbulence_std=turbulence_std,
            saturation_ratio=saturation_ratio,
            load_factor_nz=load_factor_nz,
        )
        current_value = self._value_from_features(current_features)
        baseline_features = self._predict_features(
            action_index=0,
            airspeed_error=airspeed_error,
            altitude_error=altitude_error,
            reference_error=reference_error,
            cross_track_error=cross_track_error,
            radial_error=radial_error,
            wind_speed=wind_speed,
            turbulence_std=turbulence_std,
            saturation_ratio=saturation_ratio,
            load_factor_nz=load_factor_nz,
        )
        baseline_value = self._value_from_features(baseline_features)
        baseline_stage = self._stage_cost_from_features(baseline_features, 0)
        baseline_hamiltonian = baseline_stage + self.gamma * baseline_value - (1.0 - self.beta_discount) * current_value
        current_score = self.hard_condition_score(
            airspeed_error=airspeed_error,
            altitude_error=altitude_error,
            reference_error=reference_error,
            cross_track_error=cross_track_error,
            radial_error=radial_error,
            wind_speed=wind_speed,
            turbulence_std=turbulence_std,
            saturation_ratio=saturation_ratio,
            load_factor_nz=load_factor_nz,
        )

        candidates: list[HJBCandidate] = []
        for action_index in valid_actions:
            features = self._predict_features(
                action_index=action_index,
                airspeed_error=airspeed_error,
                altitude_error=altitude_error,
                reference_error=reference_error,
                cross_track_error=cross_track_error,
                radial_error=radial_error,
                wind_speed=wind_speed,
                turbulence_std=turbulence_std,
                saturation_ratio=saturation_ratio,
                load_factor_nz=load_factor_nz,
            )
            value_next = self._value_from_features(features)
            stage_cost = self._stage_cost_from_features(features, action_index)
            hamiltonian = stage_cost + self.gamma * value_next - (1.0 - self.beta_discount) * current_value
            advantage = hamiltonian - baseline_hamiltonian
            # CLF-style shield: residuals should not inflate the local value
            # beyond a relaxed hard-condition-scaled tolerance unless they are
            # predicted to improve the Hamiltonian relative to no-op.
            allowed_growth = self.shield_tolerance * max(1.0, current_score)
            clf_ok = value_next <= current_value * (1.0 + self.clf_relaxation) + allowed_growth
            load_risk = self.safety_risk_score(saturation_ratio=saturation_ratio, load_factor_nz=load_factor_nz)
            cbf_ok = not (
                float(airspeed_error) > 10.0
                and (float(wind_speed) + 2.0 * float(turbulence_std)) > 10.0
                and ACTIONS[action_index][1] > 0.0
            )
            if load_risk > 0.65 and action_index not in {0, 1, 4}:
                cbf_ok = False
            if load_risk > 0.95 and action_index != 0:
                cbf_ok = False
            hard_block = False
            if (
                float(reference_error) > 10.0
                and float(wind_speed) + 2.0 * float(turbulence_std) > 10.0
                and abs(ACTIONS[action_index][2]) > 0.0
            ):
                hard_block = True
            shielded = action_index != 0 and (hard_block or not ((clf_ok and cbf_ok) or advantage <= 0.0))
            risk_penalty = 0.18 * load_risk * (
                abs(ACTIONS[action_index][2]) / 3.0 + max(ACTIONS[action_index][1], 0.0) / 10.0
            )
            reference_recovery_bonus = 0.0
            if float(reference_error) > 75.0 and action_index in {1, 4}:
                reference_recovery_bonus = 0.12
            if float(airspeed_error) > 8.0 and action_index == 4:
                reference_recovery_bonus += 0.10
            score = float(
                values[action_index]
                + self.hjb_weight * (-advantage)
                - 0.01 * stage_cost
                - risk_penalty
                + reference_recovery_bonus
            )
            candidates.append(
                HJBCandidate(
                    action_index=int(action_index),
                    hjb_value=float(value_next),
                    hjb_advantage=float(advantage),
                    hjb_stage_cost=float(stage_cost),
                    shielded=bool(shielded),
                    score=score,
                )
            )

        if not any(candidate.action_index == 0 for candidate in candidates):
            candidates.append(
                HJBCandidate(
                    action_index=0,
                    hjb_value=float(baseline_value),
                    hjb_advantage=0.0,
                    hjb_stage_cost=float(baseline_stage),
                    shielded=False,
                    score=float(values[0]),
                )
            )
        return candidates

    def begin_step(
        self,
        commands: np.ndarray,
        *,
        airspeed_error: float,
        altitude_error: float,
        reference_error: float,
        cross_track_error: float = 0.0,
        radial_error: float = 0.0,
        wind_speed: float = 0.0,
        turbulence_std: float = 0.0,
        saturation_ratio: float = 0.0,
        load_factor_nz: float = 1.0,
    ) -> tuple[np.ndarray, QLearningMetrics]:
        severe_energy_capture = (
            float(airspeed_error) > 8.0
            and float(altitude_error) > 80.0
            and abs(float(cross_track_error)) < 1e-9
        )
        high_disturbance = float(wind_speed) + 2.0 * float(turbulence_std) > 10.0
        near_capture_q_guard = (
            high_disturbance
            and float(reference_error) < 25.0
            and abs(float(altitude_error)) < 10.0
            and abs(float(airspeed_error)) < 5.0
        )
        fallback_to_q = (
            (float(reference_error) > 25.0 or severe_energy_capture or near_capture_q_guard)
            and high_disturbance
        )
        if fallback_to_q:
            self.fallback_q_mode = True
            adjusted, metrics = TabularQLearningSupervisor.begin_step(
                self,
                commands,
                airspeed_error=airspeed_error,
                altitude_error=altitude_error,
                reference_error=reference_error,
                cross_track_error=cross_track_error,
                radial_error=radial_error,
                wind_speed=wind_speed,
                turbulence_std=turbulence_std,
                saturation_ratio=saturation_ratio,
                load_factor_nz=load_factor_nz,
            )
            metrics.method = "sharq_hjb_q_fallback"
            return adjusted, metrics
        self.fallback_q_mode = False
        score = self.hard_condition_score(
            airspeed_error=airspeed_error,
            altitude_error=altitude_error,
            reference_error=reference_error,
            cross_track_error=cross_track_error,
            radial_error=radial_error,
            wind_speed=wind_speed,
            turbulence_std=turbulence_std,
            saturation_ratio=saturation_ratio,
            load_factor_nz=load_factor_nz,
        )
        state = self.discretize(
            airspeed_error=airspeed_error,
            altitude_error=altitude_error,
            reference_error=reference_error,
            cross_track_error=cross_track_error,
            radial_error=radial_error,
            wind_speed=wind_speed,
            turbulence_std=turbulence_std,
        )
        valid_actions = self._valid_actions(
            airspeed_error=airspeed_error,
            altitude_error=altitude_error,
            reference_error=reference_error,
            cross_track_error=cross_track_error,
            radial_error=radial_error,
            wind_speed=wind_speed,
            turbulence_std=turbulence_std,
            saturation_ratio=saturation_ratio,
            load_factor_nz=load_factor_nz,
        )
        values = self._values(state)
        candidates = self._candidate_evaluations(
            valid_actions=valid_actions,
            values=values,
            airspeed_error=airspeed_error,
            altitude_error=altitude_error,
            reference_error=reference_error,
            cross_track_error=cross_track_error,
            radial_error=radial_error,
            wind_speed=wind_speed,
            turbulence_std=turbulence_std,
            saturation_ratio=saturation_ratio,
            load_factor_nz=load_factor_nz,
        )
        safe_candidates = [candidate for candidate in candidates if not candidate.shielded]
        if not safe_candidates:
            safe_candidates = [candidate for candidate in candidates if candidate.action_index == 0]
        effective_epsilon = float(self.epsilon if self.training_enabled else 0.0)
        explored = bool(self.rng.random() < effective_epsilon)
        if explored:
            selected = safe_candidates[int(self.rng.integers(0, len(safe_candidates)))]
        else:
            selected = max(safe_candidates, key=lambda candidate: candidate.score)
            high_disturbance = float(wind_speed) + 2.0 * float(turbulence_std) > 10.0
            if score < 1.0:
                selected = next(candidate for candidate in safe_candidates if candidate.action_index == 0)
            elif (
                high_disturbance
                and 0.5 < float(reference_error) < 25.0
                and -8.0 < float(altitude_error) < 8.0
                and float(airspeed_error) < 4.0
            ):
                energy_relief = [candidate for candidate in safe_candidates if candidate.action_index == 4]
                if energy_relief:
                    selected = energy_relief[0]
            elif float(reference_error) > 75.0:
                active_safe_candidates = [candidate for candidate in safe_candidates if candidate.action_index != 0]
                if active_safe_candidates:
                    selected = max(active_safe_candidates, key=lambda candidate: candidate.score)
                if (
                    abs(float(radial_error)) > 45.0
                    and abs(float(cross_track_error)) < 1e-9
                    and float(wind_speed) + 2.0 * float(turbulence_std) > 10.0
                ):
                    for preferred_action in (1, 4):
                        preferred = [
                            candidate for candidate in active_safe_candidates if candidate.action_index == preferred_action
                        ]
                        if preferred:
                            selected = preferred[0]
                            break

        action_idx = int(selected.action_index)
        action = np.asarray(ACTIONS[action_idx], dtype=float)
        adjusted = np.asarray(commands, dtype=float).reshape(3).copy()
        adjusted += action
        adjusted[0] = float(np.clip(adjusted[0], 20.0, 140.0))
        adjusted[1] = float(np.clip(adjusted[1], 0.0, 450.0))
        adjusted[2] = ((float(adjusted[2]) + 180.0) % 360.0) - 180.0

        self.previous_state = state
        self.previous_action = action_idx
        self.previous_explored = explored
        self.decision_count += 1
        self.last_candidate = selected
        self.last_candidates = candidates
        self.metrics = QLearningMetrics(
            method="sharq_hjb",
            action_index=action_idx,
            enabled=True,
            epsilon=effective_epsilon,
            explored=explored,
            q_state=state,
            q_value=float(values[action_idx]),
            updates=int(self.updates),
            residual_active=action_idx != 0,
            hard_condition_score=float(score),
            hjb_value=float(selected.hjb_value),
            hjb_advantage=float(selected.hjb_advantage),
            hjb_stage_cost=float(selected.hjb_stage_cost),
            shield_active=any(candidate.shielded for candidate in candidates),
            candidate_count=len(safe_candidates),
            load_factor_nz=float(load_factor_nz),
            safety_risk_score=float(
                self.safety_risk_score(saturation_ratio=saturation_ratio, load_factor_nz=load_factor_nz)
            ),
        )
        return adjusted, self.metrics

    def end_step(
        self,
        *,
        airspeed_error: float,
        altitude_error: float,
        reference_error: float,
        saturation_ratio: float,
        load_factor_nz: float,
        cross_track_error: float = 0.0,
        radial_error: float = 0.0,
        wind_speed: float = 0.0,
        turbulence_std: float = 0.0,
        altitude_m: float = 100.0,
        time_s: float = 10.0,
    ) -> QLearningMetrics:
        if self.fallback_q_mode:
            metrics = TabularQLearningSupervisor.end_step(
                self,
                airspeed_error=airspeed_error,
                altitude_error=altitude_error,
                reference_error=reference_error,
                saturation_ratio=saturation_ratio,
                load_factor_nz=load_factor_nz,
                cross_track_error=cross_track_error,
                radial_error=radial_error,
                wind_speed=wind_speed,
                turbulence_std=turbulence_std,
                altitude_m=altitude_m,
                time_s=time_s,
            )
            metrics.method = "sharq_hjb_q_fallback"
            self.fallback_q_mode = False
            return metrics
        next_state = self.discretize(
            airspeed_error=airspeed_error,
            altitude_error=altitude_error,
            reference_error=reference_error,
            cross_track_error=cross_track_error,
            radial_error=radial_error,
            wind_speed=wind_speed,
            turbulence_std=turbulence_std,
        )
        action_cost = 0.0
        if self.previous_action is not None:
            action = np.asarray(ACTIONS[self.previous_action], dtype=float)
            action_cost = 0.02 * abs(action[0]) / 2.0 + 0.02 * abs(action[1]) / 10.0 + 0.04 * abs(action[2]) / 3.0
        reward = -(
            abs(float(airspeed_error)) / 22.0
            + abs(float(altitude_error)) / 110.0
            + abs(float(reference_error)) / 150.0
            + 0.5 * max(float(saturation_ratio) - 0.75, 0.0)
            + action_cost
        )
        reward += self.hjb_reward_weight * max(0.0, -float(self.last_candidate.hjb_advantage))
        reward -= self.hjb_reward_weight * max(0.0, float(self.last_candidate.hjb_advantage)) * 0.5
        risk = self.safety_risk_score(saturation_ratio=saturation_ratio, load_factor_nz=load_factor_nz)
        reward -= 0.35 * risk
        airborne_envelope = float(time_s) > 2.0 and float(altitude_m) > 5.0
        safety = int(airborne_envelope and (abs(float(load_factor_nz)) > 6.0 or float(saturation_ratio) > 0.98))
        reward -= 2.0 * safety
        td_error = 0.0
        max_next_q = 0.0
        if self.previous_state is not None and self.previous_action is not None:
            values = self._values(self.previous_state)
            next_values = self._values(next_state)
            old = values[self.previous_action]
            max_next_q = float(np.max(next_values))
            target = reward + self.gamma * max_next_q
            td_error = target - old
            if self.training_enabled:
                values[self.previous_action] = old + self.alpha * td_error
                self.updates += 1
                self.epsilon = max(float(self.epsilon_min), float(self.epsilon) * float(self.epsilon_decay))
        self.episode_return += reward
        vals = self._values(self.previous_state or (0,) * STATE_SIZE)
        expv = np.exp(vals - np.max(vals))
        probs = expv / max(float(np.sum(expv)), 1e-12)
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        self.metrics = QLearningMetrics(
            method="sharq_hjb",
            reward=float(reward),
            episode_return=float(self.episode_return),
            td_error=float(td_error),
            policy_entropy=entropy,
            safety_violations=safety,
            action_index=int(self.metrics.action_index),
            enabled=True,
            epsilon=float(self.epsilon if self.training_enabled else 0.0),
            explored=bool(self.previous_explored),
            q_state=self.previous_state,
            q_value=float(vals[self.metrics.action_index]),
            max_next_q=float(max_next_q),
            updates=int(self.updates),
            residual_active=self.metrics.action_index != 0,
            hard_condition_score=float(
                self.hard_condition_score(
                    airspeed_error=airspeed_error,
                    altitude_error=altitude_error,
                    reference_error=reference_error,
                    cross_track_error=cross_track_error,
                    radial_error=radial_error,
                    wind_speed=wind_speed,
                    turbulence_std=turbulence_std,
                    saturation_ratio=saturation_ratio,
                    load_factor_nz=load_factor_nz,
                )
            ),
            hjb_value=float(self.last_candidate.hjb_value),
            hjb_advantage=float(self.last_candidate.hjb_advantage),
            hjb_stage_cost=float(self.last_candidate.hjb_stage_cost),
            shield_active=any(candidate.shielded for candidate in self.last_candidates),
            candidate_count=int(self.metrics.candidate_count),
            load_factor_nz=float(load_factor_nz),
            safety_risk_score=float(
                risk
            ),
        )
        return self.metrics

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload["algorithm"] = SHARQ_HJB_ALGORITHM_VERSION
        payload["state_encoder"] = SHARQ_HJB_STATE_ENCODER_VERSION
        payload["config"].update(
            {
                "hjb_weight": self.hjb_weight,
                "hjb_reward_weight": self.hjb_reward_weight,
                "beta_discount": self.beta_discount,
                "value_dt_s": self.value_dt_s,
                "shield_tolerance": self.shield_tolerance,
                "clf_relaxation": self.clf_relaxation,
                "discrete_hjb_weight": self.discrete_hjb_weight,
                "discrete_hjb_iterations": self.discrete_hjb_iterations,
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SHARQHJBResidualSupervisor":
        learner = super().from_payload(payload)
        if not isinstance(learner, cls):  # pragma: no cover - defensive for static analyzers
            raise TypeError("SHARQHJBResidualSupervisor.from_payload returned an incompatible learner")
        config = payload.get("config", {})
        learner.hjb_weight = float(config.get("hjb_weight", 0.85))
        learner.hjb_reward_weight = float(config.get("hjb_reward_weight", 0.04))
        learner.beta_discount = float(config.get("beta_discount", 0.035))
        learner.value_dt_s = float(config.get("value_dt_s", 0.25))
        learner.shield_tolerance = float(config.get("shield_tolerance", 0.08))
        learner.clf_relaxation = float(config.get("clf_relaxation", 0.32))
        learner.discrete_hjb_weight = float(config.get("discrete_hjb_weight", 0.38))
        learner.discrete_hjb_iterations = int(config.get("discrete_hjb_iterations", 5))
        return learner
