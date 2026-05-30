"""Modular tabular Q-learning supervisor for academic baseline comparisons.

The learner is intentionally outside the low-level fixed-gain autopilot: it
adjusts guidance-level commands by small bounded increments so experiments can
compare fixed guidance vs. Q-assisted guidance without replacing the plant or
PID autopilot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


ACTIONS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (-2.0, 0.0, 0.0),
    (0.0, 10.0, 0.0),
    (0.0, -10.0, 0.0),
    (0.0, 0.0, 3.0),
    (0.0, 0.0, -3.0),
)
ACTIONS_VERSION = "shielded_guidance_residual_v2"
STATE_ENCODER_VERSION = "wind_lateral_energy_bins_v2"
STATE_SIZE = 7


@dataclass(frozen=True)
class QLearningConfig:
    """Configuration for the tabular guidance-residual learner.

    ``training_enabled=False`` makes the policy frozen/evaluative: actions are
    selected greedily, rewards/TD targets are still reported, but Q values are
    not changed.  This keeps publication evaluation separate from training.
    """

    alpha: float = 0.12
    gamma: float = 0.95
    epsilon: float = 0.05
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    training_enabled: bool = True
    seed: int = 17


@dataclass
class QLearningMetrics:
    method: str = "none"
    reward: float = 0.0
    episode_return: float = 0.0
    td_error: float = 0.0
    policy_entropy: float = 0.0
    safety_violations: int = 0
    action_index: int = 0
    enabled: bool = False
    epsilon: float = 0.0
    explored: bool = False
    q_state: tuple[int, ...] | None = None
    q_value: float = 0.0
    max_next_q: float = 0.0
    updates: int = 0
    residual_active: bool = False
    hard_condition_score: float = 0.0
    hjb_value: float = 0.0
    hjb_advantage: float = 0.0
    hjb_stage_cost: float = 0.0
    shield_active: bool = False
    candidate_count: int = 0
    load_factor_nz: float = 1.0
    safety_risk_score: float = 0.0


@dataclass
class TabularQLearningSupervisor:
    alpha: float = 0.12
    gamma: float = 0.95
    epsilon: float = 0.05
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    seed: int = 17
    training_enabled: bool = True
    q: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)
    episode_return: float = 0.0
    previous_state: tuple[int, ...] | None = None
    previous_action: int | None = None
    previous_explored: bool = False
    updates: int = 0
    decision_count: int = 0
    rng: np.random.Generator = field(init=False)
    metrics: QLearningMetrics = field(default_factory=QLearningMetrics)

    @classmethod
    def from_config(cls, config: QLearningConfig) -> "TabularQLearningSupervisor":
        return cls(**asdict(config))

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.episode_return = 0.0
        self.previous_state = None
        self.previous_action = None
        self.previous_explored = False
        self.decision_count = 0
        self.metrics = QLearningMetrics(
            method="baseline_q",
            enabled=True,
            epsilon=float(self.epsilon),
            updates=self.updates,
        )

    def hard_condition_score(
        self,
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
    ) -> float:
        """Return a normalized disturbance/tracking score used to gate residuals."""

        return max(
            abs(float(airspeed_error)) / 10.0,
            abs(float(altitude_error)) / 35.0,
            abs(float(reference_error)) / 75.0,
            abs(float(cross_track_error)) / 20.0,
            abs(float(radial_error)) / 35.0,
            abs(float(wind_speed)) / 8.0,
            abs(float(turbulence_std)) / 1.0,
            max(float(saturation_ratio) - 0.65, 0.0) / 0.2,
            max(abs(float(load_factor_nz)) - 3.5, 0.0) / 1.5,
        )

    def safety_risk_score(self, *, saturation_ratio: float = 0.0, load_factor_nz: float = 1.0) -> float:
        """Return a normalized pre-violation risk score for residual shielding."""

        return max(
            max(abs(float(load_factor_nz)) - 3.5, 0.0) / 2.5,
            max(float(saturation_ratio) - 0.70, 0.0) / 0.28,
        )

    def discretize(
        self,
        *,
        airspeed_error: float,
        altitude_error: float,
        reference_error: float,
        cross_track_error: float = 0.0,
        radial_error: float = 0.0,
        wind_speed: float = 0.0,
        turbulence_std: float = 0.0,
    ) -> tuple[int, ...]:
        va_bin = int(np.digitize([airspeed_error], [-12.0, -4.0, 4.0, 12.0])[0])
        h_bin = int(np.digitize([altitude_error], [-40.0, -10.0, 10.0, 40.0])[0])
        ref_bin = int(np.digitize([reference_error], [25.0, 75.0, 150.0, 300.0])[0])
        cross_bin = int(np.digitize([cross_track_error], [-80.0, -25.0, -8.0, 8.0, 25.0, 80.0])[0])
        radial_bin = int(np.digitize([radial_error], [-80.0, -25.0, -8.0, 8.0, 25.0, 80.0])[0])
        wind_bin = int(np.digitize([wind_speed + 3.0 * turbulence_std], [4.0, 8.0, 12.0, 16.0])[0])
        low_energy_bin = int(float(airspeed_error) > 8.0 or (wind_speed + 2.0 * turbulence_std) > 10.0)
        return va_bin, h_bin, ref_bin, cross_bin, radial_bin, wind_bin, low_energy_bin

    def _values(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q:
            self.q[state] = np.zeros(len(ACTIONS), dtype=float)
        return self.q[state]

    def _valid_actions(
        self,
        *,
        airspeed_error: float,
        altitude_error: float,
        reference_error: float,
        cross_track_error: float,
        radial_error: float,
        wind_speed: float,
        turbulence_std: float,
        saturation_ratio: float = 0.0,
        load_factor_nz: float = 1.0,
    ) -> list[int]:
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
        if score < 1.0:
            return [0]
        if (
            float(reference_error) > 100.0
            and float(wind_speed) + 2.0 * float(turbulence_std) < 4.0
        ):
            # In calm-air cinematic/high-curvature references, a large
            # horizontal reference error is often a mission geometry mismatch
            # rather than a disturbance-rejection opportunity.  Preserve the
            # baseline instead of adding residuals that can increase load.
            return [0]

        valid = [0, 1]
        low_energy = float(airspeed_error) > 8.0 or (float(wind_speed) + 2.0 * float(turbulence_std)) > 10.0
        safety_risk = self.safety_risk_score(saturation_ratio=saturation_ratio, load_factor_nz=load_factor_nz)
        if safety_risk > 0.65:
            # Near the load/saturation envelope, prevent additional lateral or
            # climb residuals and keep only energy-relief actions.
            return [0, 1, 4]
        if not low_energy and float(airspeed_error) < -4.0:
            valid.append(2)
        if not low_energy and float(altitude_error) > 12.0:
            valid.append(3)
        if float(altitude_error) < -12.0 or low_energy:
            valid.append(4)
        if abs(float(cross_track_error)) > 8.0:
            # For the straight-path sign convention, positive cross-track
            # means the aircraft is to the right/east of the path and a
            # negative heading residual points it back toward the path.
            valid.append(6 if cross_track_error > 0.0 else 5)
        elif abs(float(radial_error)) > 45.0 and not low_energy:
            valid.extend([5, 6])
        return sorted(set(valid))

    def _prior_action(
        self,
        valid: list[int],
        *,
        airspeed_error: float,
        altitude_error: float,
        cross_track_error: float,
    ) -> int:
        if 1 in valid and float(airspeed_error) > 8.0:
            return 1
        if 4 in valid and float(airspeed_error) > 5.0:
            return 4
        if 3 in valid and float(altitude_error) > 20.0:
            return 3
        if 4 in valid and float(altitude_error) < -20.0:
            return 4
        if float(cross_track_error) > 10.0 and 6 in valid:
            return 6
        if float(cross_track_error) < -10.0 and 5 in valid:
            return 5
        return 0

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
        effective_epsilon = float(self.epsilon if self.training_enabled else 0.0)
        explored = bool(self.rng.random() < effective_epsilon)
        if explored:
            action_idx = int(valid_actions[int(self.rng.integers(0, len(valid_actions)))])
        else:
            masked = np.full(len(ACTIONS), -np.inf, dtype=float)
            masked[valid_actions] = values[valid_actions]
            if float(np.max(np.abs(values[valid_actions]))) < 1e-12:
                action_idx = self._prior_action(
                    valid_actions,
                    airspeed_error=airspeed_error,
                    altitude_error=altitude_error,
                    cross_track_error=cross_track_error,
                )
            else:
                action_idx = int(np.argmax(masked))
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
        self.metrics.action_index = int(action_idx)
        self.metrics.method = "baseline_q"
        self.metrics.enabled = True
        self.metrics.epsilon = effective_epsilon
        self.metrics.explored = explored
        self.metrics.q_state = state
        self.metrics.q_value = float(values[action_idx])
        self.metrics.updates = int(self.updates)
        self.metrics.residual_active = action_idx != 0
        self.metrics.hard_condition_score = float(score)
        self.metrics.candidate_count = len(valid_actions)
        self.metrics.load_factor_nz = float(load_factor_nz)
        self.metrics.safety_risk_score = float(
            self.safety_risk_score(saturation_ratio=saturation_ratio, load_factor_nz=load_factor_nz)
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
        risk = self.safety_risk_score(saturation_ratio=saturation_ratio, load_factor_nz=load_factor_nz)
        reward -= 0.25 * risk
        airborne_envelope = float(time_s) > 2.0 and float(altitude_m) > 5.0
        safety = int(airborne_envelope and (abs(float(load_factor_nz)) > 6.0 or float(saturation_ratio) > 0.98))
        reward -= 2.0 * safety
        td_error = 0.0
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
        else:
            max_next_q = 0.0
        self.episode_return += reward
        vals = self._values(self.previous_state or (0, 0, 0))
        expv = np.exp(vals - np.max(vals))
        probs = expv / max(float(np.sum(expv)), 1e-12)
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        self.metrics = QLearningMetrics(
            method="baseline_q",
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
            candidate_count=int(self.metrics.candidate_count),
            load_factor_nz=float(load_factor_nz),
            safety_risk_score=float(
                risk
            ),
        )
        return self.metrics

    def freeze_for_evaluation(self) -> None:
        self.training_enabled = False
        self.epsilon = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "algorithm": "tabular_q_learning",
            "state_encoder": STATE_ENCODER_VERSION,
            "action_set": ACTIONS_VERSION,
            "config": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "training_enabled": self.training_enabled,
                "seed": self.seed,
            },
            "episode_return": self.episode_return,
            "updates": self.updates,
            "decision_count": self.decision_count,
            "q_table": {",".join(map(str, state)): values.astype(float).tolist() for state, values in sorted(self.q.items())},
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TabularQLearningSupervisor":
        config = payload.get("config", {})
        learner = cls(
            alpha=float(config.get("alpha", 0.12)),
            gamma=float(config.get("gamma", 0.95)),
            epsilon=float(config.get("epsilon", 0.05)),
            epsilon_min=float(config.get("epsilon_min", 0.01)),
            epsilon_decay=float(config.get("epsilon_decay", 0.995)),
            seed=int(config.get("seed", 17)),
            training_enabled=bool(config.get("training_enabled", True)),
        )
        for raw_state, raw_values in dict(payload.get("q_table", {})).items():
            parts = tuple(int(part) for part in str(raw_state).split(","))
            if len(parts) == 3:
                state = (*parts, 3, 3, 0, 0)
            else:
                state = parts
            if len(state) != STATE_SIZE:
                raise ValueError(f"Invalid Q-table state key: {raw_state!r}")
            values = np.asarray(raw_values, dtype=float)
            if values.shape != (len(ACTIONS),):
                raise ValueError(f"Invalid Q-table values for state {raw_state!r}: {values.shape}")
            learner.q[state] = values
        learner.episode_return = float(payload.get("episode_return", 0.0))
        learner.updates = int(payload.get("updates", 0))
        learner.decision_count = int(payload.get("decision_count", 0))
        return learner

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "TabularQLearningSupervisor":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_npz(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        states = np.asarray(list(self.q.keys()), dtype=np.int64).reshape((-1, STATE_SIZE))
        values = np.asarray(list(self.q.values()), dtype=float).reshape((-1, len(ACTIONS)))
        meta = json.dumps({key: value for key, value in self.to_payload().items() if key != "q_table"}, sort_keys=True)
        np.savez_compressed(out, states=states, values=values, meta=np.asarray(meta))

    @classmethod
    def load_npz(cls, path: str | Path) -> "TabularQLearningSupervisor":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            learner = cls.from_payload({**meta, "q_table": {}})
            states = np.asarray(data["states"], dtype=np.int64).reshape((-1, STATE_SIZE))
            values = np.asarray(data["values"], dtype=float).reshape((-1, len(ACTIONS)))
            for state, row in zip(states, values, strict=True):
                key = tuple(int(v) for v in state)
                learner.q[key] = row.astype(float)
            return learner
