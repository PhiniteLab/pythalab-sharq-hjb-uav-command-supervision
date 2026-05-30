"""Smoke tests for the converted uavsim package."""
from __future__ import annotations

import numpy as np
import pytest

from uavsim import (
    build_default_parameters,
    forces_moments,
    mav_derivatives,
    simulate_mavsim_auto,
    simulate_mavsim_auto_stream,
)
from uavsim.autopilot import Autopilot
from uavsim.commands import h_destination, phi_destination
from uavsim.telemetry import (
    TELEMETRY_COLUMNS,
    result_summary,
    result_to_matrix,
    telemetry_rows_from_result,
    write_result_csv,
)
from uavsim.actuators import ActuatorState
from uavsim.aircraft_config import load_aircraft_config, parameters_from_config
from uavsim.atmosphere import density_at_altitude, sample_atmosphere
from uavsim.experiment_runner import (
    REFERENCE_BENCHMARK_SCENARIOS,
    compare_all_methods,
    compare_fixed_vs_q_learning,
    run_batch,
    run_episode_summary,
    run_reference_benchmark,
    write_summary_csv,
)
from uavsim.guidance import orbit_guidance, straight_path_guidance
from uavsim.q_learning import TabularQLearningSupervisor
from uavsim.sharq_hjb import SHARQHJBResidualSupervisor
from uavsim.tecs import TECSController


def test_default_parameters_trim_and_gains() -> None:
    P = build_default_parameters(compute_trim_and_gains=True, exact_source=True)
    assert P.x_trim.shape == (12,)
    assert P.u_trim.shape == (4,)
    assert 139.0 <= P.Va <= 141.0
    assert 0.55 <= P.u_trim[3] <= 0.75
    assert np.isfinite(P.roll_kp)
    assert np.isfinite(P.pitch_kp)


def test_default_parameters_schedule_dryden_to_trim_speed() -> None:
    from uavsim.wind import DrydenWind

    P = build_default_parameters(compute_trim_and_gains=False, Va=140.0)
    assert P.Va == 140.0
    assert P.Va0 == 140.0

    wind = DrydenWind(P)
    gust_a = wind.gust()
    assert gust_a.shape == (3,)
    assert np.all(np.isfinite(gust_a))
    wind.reset()
    gust_b = wind.gust()
    assert np.allclose(gust_a, gust_b)


def test_high_speed_trim_has_propulsion_margin() -> None:
    P = build_default_parameters(compute_trim_and_gains=True, exact_source=True)
    fm_trim = forces_moments(P.x_trim, P.u_trim, np.zeros(6), P, exact_source=True)
    max_throttle = P.u_trim.copy()
    max_throttle[3] = 1.0
    fm_max = forces_moments(P.x_trim, max_throttle, np.zeros(6), P, exact_source=True)

    # Level 140 m/s trim should be force-balanced, while the resized motor has
    # substantial positive-x margin for high-speed path/gust transients.
    assert abs(fm_trim[0]) < 1e-6
    assert abs(fm_trim[2]) < 1e-6
    assert abs(fm_trim[4]) < 1e-6
    assert fm_max[0] > 500.0


def test_forces_and_dynamics_are_finite() -> None:
    P = build_default_parameters(compute_trim_and_gains=True, exact_source=True)
    fm = forces_moments(P.x_trim, P.u_trim, np.zeros(6), P, exact_source=True)
    dx = mav_derivatives(P.x_trim, P.u_trim, np.zeros(6), P, exact_source=True)
    assert fm.shape == (6,)
    assert dx.shape == (12,)
    assert np.all(np.isfinite(fm))
    assert np.all(np.isfinite(dx))


def test_autopilot_can_use_wind_relative_airspeed_measurement() -> None:
    P = build_default_parameters(compute_trim_and_gains=True, exact_source=True)
    x = P.initial_state().copy()
    x[2] = -100.0  # altitude-hold zone for the command below
    commands = np.array([140.0, 100.0, 0.0], dtype=float)

    baseline = Autopilot(P).update(x, commands, 1.0)
    headwind_measured = Autopilot(P).update(x, commands, 1.0, measured_airspeed=144.0)

    # In altitude hold, a measured airspeed above the 140 m/s target must
    # reduce throttle relative to the legacy body-speed fallback.
    assert headwind_measured[3] < baseline[3]


def test_command_profiles_match_stateflow_charts() -> None:
    assert h_destination(10) == 10
    assert h_destination(60) == 50
    assert h_destination(120) == 200
    assert h_destination(160) == 100
    assert phi_destination(10) == 59


def test_short_simulation_runs_and_logs_extended_telemetry() -> None:
    P = build_default_parameters(compute_trim_and_gains=True, exact_source=True)
    res = simulate_mavsim_auto(P, t_final=1.0, exact_source=True, rk4_substeps=2)
    assert res.x.shape[1] == 12
    assert res.delta.shape[1] == 4
    assert res.wind is not None and res.wind.shape[1] == 6
    assert res.forces_moments is not None and res.forces_moments.shape[1] == 6
    assert res.derivatives is not None and res.derivatives.shape[1] == 12
    assert res.air_data is not None and res.air_data.shape[1] == 6
    assert np.all(np.isfinite(res.x))
    assert np.all(np.isfinite(res.forces_moments))
    rows = telemetry_rows_from_result(res, P)
    assert len(rows) == len(res.t)
    assert "alpha_deg" in rows[0]
    summary = result_summary(res)
    assert summary["t_final"] == res.t[-1]


def test_default_simulation_command_follows_parameter_trim_speed() -> None:
    P = build_default_parameters(compute_trim_and_gains=True, exact_source=True, Va=200.0)
    res = simulate_mavsim_auto(P, t_final=0.1, exact_source=True, rk4_substeps=1)
    assert np.allclose(res.commands[:, 0], 200.0)


def test_streaming_simulation_yields_finite_steps() -> None:
    P = build_default_parameters(compute_trim_and_gains=True, exact_source=True)
    steps = list(
        simulate_mavsim_auto_stream(
            P, t_final=0.1, exact_source=True, rk4_substeps=1, yield_stride=5
        )
    )
    assert len(steps) >= 2
    for step in steps:
        assert np.all(np.isfinite(step.x))
        assert np.all(np.isfinite(step.delta))
        assert step.forces_moments.shape == (6,)
        assert step.airdata.shape == (3,)


def test_telemetry_matrix_and_csv(tmp_path) -> None:
    P = build_default_parameters(compute_trim_and_gains=True, exact_source=True)
    result = simulate_mavsim_auto(P, t_final=0.1, exact_source=True, rk4_substeps=1)
    cols, data = result_to_matrix(result)
    assert cols == list(TELEMETRY_COLUMNS)
    assert data.shape[0] == result.x.shape[0]
    assert "Fx" in cols and "Va" in cols and "altitude_state" in cols
    path = write_result_csv(result, tmp_path / "telemetry.csv")
    assert path.exists()
    assert path.read_text().startswith("t,pn,pe,pd")


def test_atmosphere_dynamic_pressure_and_density_are_finite() -> None:
    rho0 = density_at_altitude(1.2682, 0.0)
    rho2k = density_at_altitude(1.2682, 2_000.0)
    sample = sample_atmosphere(1.2682, 250.0, 40.0)

    assert rho0 == 1.2682
    assert 0.0 < rho2k < rho0
    assert abs(sample.dynamic_pressure_pa - 0.5 * sample.density_kg_m3 * 40.0 * 40.0) < 1e-9
    assert 0.0 < sample.mach < 1.0


def test_actuator_state_rate_limits_and_clamps() -> None:
    actuators = ActuatorState()
    command = np.array([10.0, -10.0, 10.0, 2.0])
    actual = actuators.update(command, 0.01)

    assert actual.shape == (4,)
    assert actual[3] <= 0.018 + 1e-9
    assert np.all(actual <= actuators.limits.upper)
    assert np.all(actual >= actuators.limits.lower)


def test_guidance_helpers_emit_finite_path_commands() -> None:
    straight = straight_path_guidance(
        pn=20.0,
        pe=5.0,
        start_n=0.0,
        start_e=0.0,
        unit_n=1.0,
        unit_e=0.0,
        lookahead_m=80.0,
    )
    orbit = orbit_guidance(pn=120.0, pe=0.0, center_n=0.0, center_e=0.0, radius_m=100.0, direction=1)

    assert -180.0 <= straight.heading_deg <= 180.0
    assert straight.cross_track_error_m == 5.0
    assert np.isfinite(straight.lateral_accel_mps2)
    assert -np.pi / 4.0 <= straight.roll_command_rad <= np.pi / 4.0
    assert -180.0 <= orbit.heading_deg <= 180.0
    assert orbit.radial_error_m == 20.0
    assert orbit.lookahead_m == 50.0


def test_tecs_and_q_learning_supervisor_are_deterministic_and_finite() -> None:
    tecs = TECSController()
    tecs_state = tecs.update(altitude_m=100.0, airspeed_mps=30.0, target_altitude_m=120.0, target_airspeed_mps=35.0, trim_throttle=0.2)
    assert 0.0 <= tecs_state.throttle_command <= 1.0
    assert -10.0 * np.pi / 180.0 <= tecs_state.pitch_command_rad <= 15.0 * np.pi / 180.0
    assert abs(tecs_state.elevator_bias_rad) <= 2.0 * np.pi / 180.0
    assert np.isfinite(tecs_state.total_energy_error)

    learner = TabularQLearningSupervisor(seed=3, epsilon=0.0)
    commands, metrics = learner.begin_step(
        np.array([30.0, 100.0, 0.0]),
        airspeed_error=4.0,
        altitude_error=5.0,
        reference_error=20.0,
    )
    metrics = learner.end_step(airspeed_error=3.0, altitude_error=4.0, reference_error=18.0, saturation_ratio=0.2, load_factor_nz=1.0)
    assert commands.shape == (3,)
    assert metrics.enabled
    assert metrics.updates == 1
    assert np.isfinite(metrics.reward)


def test_q_learning_checkpoint_round_trip_json_and_npz(tmp_path) -> None:
    learner = TabularQLearningSupervisor(alpha=0.3, gamma=0.7, epsilon=0.2, seed=9)
    learner.begin_step(np.array([30.0, 100.0, 0.0]), airspeed_error=2.0, altitude_error=3.0, reference_error=10.0)
    learner.end_step(airspeed_error=1.0, altitude_error=2.0, reference_error=8.0, saturation_ratio=0.1, load_factor_nz=1.0)

    json_path = tmp_path / "policy.json"
    npz_path = tmp_path / "policy.npz"
    learner.save_json(json_path)
    learner.save_npz(npz_path)

    from_json = TabularQLearningSupervisor.load_json(json_path)
    from_npz = TabularQLearningSupervisor.load_npz(npz_path)
    assert from_json.updates == from_npz.updates == learner.updates
    assert from_json.to_payload()["q_table"] == from_npz.to_payload()["q_table"] == learner.to_payload()["q_table"]

    sharq = SHARQHJBResidualSupervisor(alpha=0.2, gamma=0.8, epsilon=0.0, seed=19)
    sharq.begin_step(
        np.array([30.0, 100.0, 0.0]),
        airspeed_error=12.0,
        altitude_error=5.0,
        reference_error=90.0,
        wind_speed=10.0,
        turbulence_std=1.0,
    )
    sharq.end_step(
        airspeed_error=10.0,
        altitude_error=4.0,
        reference_error=75.0,
        saturation_ratio=0.2,
        load_factor_nz=1.0,
    )
    sharq_json_path = tmp_path / "sharq_policy.json"
    sharq_npz_path = tmp_path / "sharq_policy.npz"
    sharq.save_json(sharq_json_path)
    sharq.save_npz(sharq_npz_path)

    sharq_from_json = SHARQHJBResidualSupervisor.load_json(sharq_json_path)
    sharq_from_npz = SHARQHJBResidualSupervisor.load_npz(sharq_npz_path)
    assert sharq_from_json.to_payload()["algorithm"] == "sharq_hjb_discrete_hjb_guidance_residual_v2"
    assert sharq_from_json.updates == sharq_from_npz.updates == sharq.updates
    assert sharq_from_json.to_payload()["q_table"] == sharq_from_npz.to_payload()["q_table"] == sharq.to_payload()["q_table"]


def test_q_learning_update_uses_next_state_bootstrap() -> None:
    learner = TabularQLearningSupervisor(alpha=0.5, gamma=0.5, epsilon=0.0, seed=5)
    start_state = learner.discretize(airspeed_error=0.0, altitude_error=0.0, reference_error=20.0)
    next_state = learner.discretize(airspeed_error=20.0, altitude_error=50.0, reference_error=400.0)
    learner.q[next_state] = np.array([1.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    learner.begin_step(
        np.array([30.0, 100.0, 0.0]),
        airspeed_error=0.0,
        altitude_error=0.0,
        reference_error=20.0,
    )
    metrics = learner.end_step(
        airspeed_error=20.0,
        altitude_error=50.0,
        reference_error=400.0,
        saturation_ratio=0.2,
        load_factor_nz=1.0,
    )

    expected_reward = -(20.0 / 22.0 + 50.0 / 110.0 + 400.0 / 150.0)
    expected_td = expected_reward + 0.5 * 6.0
    assert metrics.td_error == pytest.approx(expected_td)
    assert learner.q[start_state][0] == pytest.approx(0.5 * expected_td)


def test_q_learning_residual_gate_and_shield_prevent_unneeded_heading_actions() -> None:
    learner = TabularQLearningSupervisor(epsilon=0.0, seed=11)
    commands, metrics = learner.begin_step(
        np.array([30.0, 100.0, 0.0]),
        airspeed_error=1.0,
        altitude_error=2.0,
        reference_error=5.0,
        cross_track_error=0.0,
        wind_speed=0.0,
        turbulence_std=0.0,
    )
    assert metrics.action_index == 0
    assert metrics.residual_active is False
    assert np.allclose(commands, np.array([30.0, 100.0, 0.0]))

    learner = TabularQLearningSupervisor(epsilon=0.0, seed=12)
    commands, metrics = learner.begin_step(
        np.array([30.0, 100.0, 0.0]),
        airspeed_error=14.0,
        altitude_error=5.0,
        reference_error=40.0,
        cross_track_error=0.0,
        wind_speed=10.0,
        turbulence_std=1.0,
    )
    assert metrics.action_index == 1
    assert metrics.residual_active is True
    assert commands[0] == 32.0

    learner = TabularQLearningSupervisor(epsilon=0.0, seed=13)
    commands, metrics = learner.begin_step(
        np.array([30.0, 100.0, 0.0]),
        airspeed_error=0.0,
        altitude_error=0.0,
        reference_error=50.0,
        cross_track_error=20.0,
        wind_speed=8.0,
        turbulence_std=1.0,
    )
    assert metrics.action_index == 6
    assert commands[2] == -3.0


def test_sharq_hjb_residual_uses_hjb_metrics_and_preserves_nominal_baseline() -> None:
    learner = SHARQHJBResidualSupervisor(epsilon=0.0, seed=21)
    nominal_commands, nominal_metrics = learner.begin_step(
        np.array([30.0, 100.0, 0.0]),
        airspeed_error=1.0,
        altitude_error=2.0,
        reference_error=5.0,
        cross_track_error=0.0,
        wind_speed=0.0,
        turbulence_std=0.0,
    )
    assert nominal_metrics.method == "sharq_hjb"
    assert nominal_metrics.action_index == 0
    assert nominal_metrics.residual_active is False
    assert np.allclose(nominal_commands, np.array([30.0, 100.0, 0.0]))

    hard_commands, hard_metrics = learner.begin_step(
        np.array([30.0, 100.0, 0.0]),
        airspeed_error=14.0,
        altitude_error=5.0,
        reference_error=80.0,
        cross_track_error=18.0,
        wind_speed=7.0,
        turbulence_std=0.5,
    )
    assert hard_metrics.method == "sharq_hjb"
    assert hard_metrics.hard_condition_score >= 1.0
    assert hard_metrics.candidate_count >= 1
    assert np.isfinite(hard_metrics.hjb_value)
    assert np.isfinite(hard_metrics.hjb_advantage)
    assert hard_commands.shape == (3,)


def test_aircraft_yaml_configs_load_current_parameters() -> None:
    small = load_aircraft_config("small_mav")
    generic = load_aircraft_config("generic_jet")
    P = parameters_from_config("small_mav")

    assert small["mass"] == 1.56
    assert generic["name"] == "generic_fast_uav_v1"
    assert generic["mass"] > small["mass"]
    assert generic["S_wing"] > small["S_wing"]
    assert P.mass == 1.56
    assert P.S_wing == 0.2589


def test_experiment_runner_reports_and_persists_fixed_and_q_learning_metrics(tmp_path) -> None:
    checkpoint = tmp_path / "q_policy.json"
    summary_csv = tmp_path / "summary.csv"
    step_jsonl = tmp_path / "steps.jsonl"
    fixed = run_episode_summary(profile="loiter_orbit", controller_mode="fixed_matlab_autopilot", duration_s=1.0)
    trained = run_episode_summary(
        profile="loiter_orbit",
        controller_mode="online_q_learning",
        duration_s=1.0,
        seed=101,
        checkpoint_out=checkpoint,
        training_enabled=True,
        step_log_path=step_jsonl,
    )
    evaluated = run_episode_summary(
        profile="loiter_orbit",
        controller_mode="online_q_learning",
        duration_s=1.0,
        seed=101,
        checkpoint_in=checkpoint,
        training_enabled=False,
    )
    paired = compare_fixed_vs_q_learning(profile="loiter_orbit", duration_s=1.0)
    all_methods = compare_all_methods(profile="loiter_orbit", duration_s=1.0)
    batch = run_batch(profiles=["loiter_orbit"], seeds=[1], duration_s=1.0, controller_modes=["fixed_matlab_autopilot", "online_q_learning"])
    write_summary_csv(summary_csv, batch)

    assert fixed.finite
    assert fixed.q_table_size == 0
    assert fixed.to_dict()["controller_mode"] == "fixed_matlab_autopilot"
    assert trained.q_updates > 0
    assert checkpoint.exists()
    assert step_jsonl.exists() and step_jsonl.read_text(encoding="utf-8").strip()
    assert evaluated.q_updates == trained.q_updates
    assert [summary.controller_mode for summary in paired] == ["fixed_matlab_autopilot", "online_q_learning"]
    assert [summary.controller_mode for summary in all_methods] == ["fixed_matlab_autopilot", "online_q_learning", "sharq_hjb"]
    assert paired[1].finite
    assert paired[1].q_table_size > 0
    assert np.isfinite(paired[1].rms_altitude_error_m)
    assert summary_csv.exists() and "controller_mode" in summary_csv.read_text(encoding="utf-8")


def test_reference_benchmark_writes_method_result_folders(tmp_path) -> None:
    assert len(REFERENCE_BENCHMARK_SCENARIOS) == 20
    summaries = run_reference_benchmark(
        output_dir=tmp_path / "results",
        scenarios=REFERENCE_BENCHMARK_SCENARIOS[:1],
        controller_modes=["fixed_matlab_autopilot", "online_q_learning", "sharq_hjb"],
        duration_override_s=0.2,
        step_log_stride=10,
        substeps=1,
    )
    root = tmp_path / "results"
    assert len(summaries) == 3
    assert (root / "baseline" / "episode_summary.csv").exists()
    assert (root / "baseline-q" / "steps.jsonl").exists()
    assert (root / "sharq-hjb" / "aggregate_metrics.csv").exists()
    assert (root / "comparative" / "all_episode_summary.csv").exists()
