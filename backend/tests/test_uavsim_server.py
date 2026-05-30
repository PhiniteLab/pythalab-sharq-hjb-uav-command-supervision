"""Integration tests for the FastAPI WebSocket bridge."""
from __future__ import annotations

from fastapi.testclient import TestClient
import numpy as np
import pytest

from uavsim.server import (
    CRUISE_AIRSPEED_MPS,
    FIGHT_MODE_PROFILE,
    Runtime,
    SIMULATION_FRAME_SCHEMA_VERSION,
    TAKEOFF_PROFILES,
    TRAJECTORY_PROFILES,
    app,
    apply_effects,
    build_simulation_frame,
    reference_commands,
    _handle_command,
)


def _schema_snapshot(value: object) -> object:
    if isinstance(value, dict):
        return {key: _schema_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_schema_snapshot(item) for item in value]
    return type(value).__name__


def _wrapped_delta_deg(next_heading_deg: float, prev_heading_deg: float) -> float:
    return ((next_heading_deg - prev_heading_deg + 180.0) % 360.0) - 180.0


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_reference_commands_cover_all_profiles() -> None:
    for profile in TRAJECTORY_PROFILES:
        cmd = reference_commands(profile, 12.0)
        assert cmd.shape == (3,)
        assert 20.0 <= cmd[0] <= CRUISE_AIRSPEED_MPS
        assert cmd[1] >= 0.0


def test_apply_effects_clamps_extreme_values() -> None:
    rt = Runtime.create()
    apply_effects(
        rt.wind,
        {
            "steady_wind_n": "1000.0",
            "steady_wind_e": -1000.0,
            "steady_wind_d": 10.0,
            "gust_body_u": -50.0,
            "gust_body_v": 50.0,
            "turbulence_std": 50.0,
            "gust_body_w": -50.0,
        },
    )
    assert rt.wind.steady_n == 10.0
    assert rt.wind.steady_e == -10.0
    assert rt.wind.steady_d == 4.0
    assert rt.wind.gust_body_u == -4.0
    assert rt.wind.gust_body_v == 4.0
    assert rt.wind.turbulence_std == 2.0
    assert rt.wind.gust_body_w == -4.0


def test_apply_effects_ignores_non_finite_values() -> None:
    rt = Runtime.create()
    original_steady_n = rt.wind.steady_n
    original_turbulence = rt.wind.turbulence_std
    original_gust_u = rt.wind.gust_body_u

    apply_effects(
        rt.wind,
        {
            "steady_wind_n": float("nan"),
            "turbulence_std": float("inf"),
            "gust_body_u": "not-a-number",
        },
    )

    assert rt.wind.steady_n == original_steady_n
    assert rt.wind.turbulence_std == original_turbulence
    assert rt.wind.gust_body_u == original_gust_u


def test_runtime_reset_clears_profile_specific_guidance_state_and_preserves_wind_settings() -> None:
    rt = Runtime.create(profile=FIGHT_MODE_PROFILE)
    rt.t = 37.5
    rt.x[0] = 123.0
    rt.x[1] = -45.0
    rt.x[2] = -260.0
    rt.x[3] = 80.0
    rt.reference_n = 1200.0
    rt.reference_e = -800.0
    rt.reference_circle_started = True
    rt.mission_straight_start_n = 10.0
    rt.mission_straight_start_e = -20.0
    rt.mission_circle_elapsed_s = 12.0
    rt.mission_circle_center_n = 220.0
    rt.mission_circle_center_e = -130.0
    rt.mission_heading_command_deg = 45.0
    rt.mission_altitude_command_m = 275.0
    rt.fight_mode_start_time_s = 8.0
    rt.fight_mode_anchor_n = 50.0
    rt.fight_mode_anchor_e = -60.0
    rt.fight_mode_heading_rad = np.deg2rad(25.0)
    rt.wind.steady_n = -3.5
    rt.wind.turbulence_std = 1.25

    rt.reset()

    assert rt.profile == FIGHT_MODE_PROFILE
    assert rt.t == 0.0
    assert rt.x[2] == 0.0
    assert rt.x[3] == 0.0
    assert rt.reference_n == 0.0
    assert rt.reference_e == 0.0
    assert rt.reference_circle_started is False
    assert rt.mission_straight_start_n is None
    assert rt.mission_straight_start_e is None
    assert rt.mission_circle_elapsed_s == 0.0
    assert rt.mission_circle_center_n is None
    assert rt.mission_circle_center_e is None
    assert rt.mission_heading_command_deg is None
    assert rt.mission_altitude_command_m is None
    assert rt.fight_mode_start_time_s is None
    assert rt.fight_mode_anchor_n == 0.0
    assert rt.fight_mode_anchor_e == 0.0
    assert rt.fight_mode_heading_rad == 0.0
    assert rt.wind.steady_n == -3.5
    assert rt.wind.turbulence_std == 1.25


def test_handle_command_rejects_non_object_payload_and_preserves_pause_on_configure() -> None:
    rt = Runtime.create()

    for payload in ([], "pause", 1):
        try:
            _handle_command(rt, payload)
        except ValueError as exc:
            assert "JSON object" in str(exc)
        else:  # pragma: no cover - defensive assertion clarity
            raise AssertionError(f"accepted invalid payload {payload!r}")

    original_profile = rt.profile
    original_steady_n = rt.wind.steady_n
    try:
        _handle_command(
            rt,
            {"command": "bogus", "profile": "racetrack", "effects": {"steady_wind_n": -10.0}},
            True,
        )
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:  # pragma: no cover - defensive assertion clarity
        raise AssertionError("accepted unsupported command")
    assert rt.profile == original_profile
    assert rt.wind.steady_n == original_steady_n

    running, *_ = _handle_command(rt, {"command": "pause"}, True)
    assert running is False
    invalid_commands = [
        {"command": "configure", "controller": "unknown_controller", "effects": {"steady_wind_n": -5.0}},
        {"command": "configure", "profile": "not_a_profile", "effects": {"steady_wind_n": -5.0}},
        {"command": "configure", "profile": "racetrack", "scenario": "loiter_orbit", "effects": {"steady_wind_n": -5.0}},
    ]
    for invalid in invalid_commands:
        try:
            _handle_command(rt, invalid, running)
        except ValueError:
            pass
        else:  # pragma: no cover - defensive assertion clarity
            raise AssertionError(f"accepted invalid command {invalid!r}")
        assert rt.profile == original_profile
        assert rt.wind.steady_n == original_steady_n

    running, episode, scenario, controller = _handle_command(
        rt,
        {
            "command": "configure",
            "controller": "fixed_matlab_baseline",
            "scenario": rt.profile,
            "effects": {"steady_wind_n": -2.0, "turbulence_std": "nan"},
        },
        running,
    )
    assert running is False
    assert episode == 1
    assert scenario == rt.profile
    assert controller == "fixed_matlab_autopilot"
    assert rt.wind.steady_n == -2.0
    assert rt.wind.turbulence_std == 0.0


def test_handle_command_reset_clamps_effects_and_preserves_profile_schema() -> None:
    rt = Runtime.create(profile=FIGHT_MODE_PROFILE)
    rt.t = 15.0
    rt.x[0] = 90.0
    rt.x[1] = -20.0
    rt.x[2] = -210.0
    rt.reference_circle_started = True
    rt.mission_circle_elapsed_s = 4.0
    rt.mission_circle_center_n = 100.0
    rt.mission_circle_center_e = -120.0
    rt.fight_mode_start_time_s = 7.0

    running, episode, scenario, controller = _handle_command(
        rt,
        {
            "command": "reset",
            "controller": "online_q_learning",
            "effects": {
                "steady_wind_n": 99.0,
                "steady_wind_d": -99.0,
                "gust_body_u": -99.0,
                "turbulence_std": 99.0,
            },
        },
        False,
    )

    assert running is True
    assert episode == 1
    assert scenario == FIGHT_MODE_PROFILE
    assert controller == "online_q_learning"
    assert rt.controller_mode == "online_q_learning"
    assert rt.profile == FIGHT_MODE_PROFILE
    assert rt.t == 0.0
    assert rt.x[2] == 0.0
    assert rt.x[3] == 0.0
    assert rt.reference_circle_started is False
    assert rt.mission_circle_elapsed_s == 0.0
    assert rt.mission_circle_center_n is None
    assert rt.mission_circle_center_e is None
    assert rt.fight_mode_start_time_s is None
    assert rt.wind.steady_n == 10.0
    assert rt.wind.steady_d == -4.0
    assert rt.wind.gust_body_u == -4.0
    assert rt.wind.turbulence_std == 2.0


def test_handle_command_resets_initial_conditions_when_profile_changes() -> None:
    rt = Runtime.create(profile="loiter_orbit")
    rt.x = rt.P.initial_state().copy()
    rt.x[2] = -100.0
    rt.t = 220.0
    rt.wind.steady_n = -3.0
    info = rt.step()
    assert info["Va"] > 100.0

    running, _, scenario, _ = _handle_command(
        rt,
        {
            "command": "configure",
            "profile": "runway_takeoff_accel_200",
            "scenario": "runway_takeoff_accel_200",
        },
        True,
    )

    assert running is True
    assert scenario == "runway_takeoff_accel_200"
    assert rt.profile == "runway_takeoff_accel_200"
    assert rt.t == 0.0
    assert rt.x[2] == 0.0
    assert rt.x[3] == 0.0
    assert rt.wind.steady_n == -3.0


def test_runtime_mission_commands_transition_from_climb_to_straight_to_circle() -> None:
    rt = Runtime.create(profile="racetrack")

    climb = rt.mission_commands()
    assert climb.shape == (3,)
    assert climb[1] == 200.0
    assert climb[2] == 0.0
    assert rt.reference_circle_started is False

    rt.x[0] = 40.0
    rt.x[1] = -15.0
    rt.x[2] = -200.0
    rt.x[8] = np.deg2rad(35.0)
    rt.t = 25.0
    rt.mission_speed_command_mps = 70.0

    straight = rt.mission_commands()
    assert straight[0] == pytest.approx(70.0)
    assert straight[1] == 200.0
    assert -180.0 <= straight[2] <= 180.0
    assert rt.mission_straight_start_n == pytest.approx(40.0)
    assert rt.mission_straight_start_e == pytest.approx(-15.0)
    assert rt.reference_n == pytest.approx(40.0)
    assert rt.reference_e == pytest.approx(-15.0)
    straight_start_n = rt.mission_straight_start_n
    straight_start_e = rt.mission_straight_start_e
    assert straight_start_n is not None
    assert straight_start_e is not None

    rt.mission_speed_command_mps = 25.0
    rt.mission_straight_start_time_s = rt.t - 9.0
    rt.x[0] = straight_start_n + 240.0 * rt.mission_straight_unit_n
    rt.x[1] = straight_start_e + 240.0 * rt.mission_straight_unit_e

    circle = rt.mission_commands()
    assert np.all(np.isfinite(circle))
    assert circle[0] == pytest.approx(25.0)
    assert circle[1] == 200.0
    assert rt.reference_circle_started is True
    assert rt.mission_circle_center_n is not None
    assert rt.mission_circle_center_e is not None
    reference_radius = np.hypot(
        rt.reference_n - rt.mission_circle_center_n,
        rt.reference_e - rt.mission_circle_center_e,
    )
    assert reference_radius == pytest.approx(100.0, abs=1e-6)


def test_runtime_fight_mode_commands_activate_with_bounded_rate_limited_updates() -> None:
    rt = Runtime.create(profile=FIGHT_MODE_PROFILE)

    climb = rt.fight_mode_commands()
    assert climb.shape == (3,)
    assert climb[1] == 240.0
    assert rt.fight_mode_start_time_s is None
    assert rt.reference_circle_started is False

    rt.x[0] = 120.0
    rt.x[1] = -35.0
    rt.x[2] = -238.0
    rt.x[8] = np.deg2rad(25.0)
    rt.t = 48.0
    rt.mission_speed_command_mps = 112.0

    cmd_1 = rt.fight_mode_commands()
    assert np.all(np.isfinite(cmd_1))
    assert rt.fight_mode_start_time_s == pytest.approx(48.0)
    assert rt.reference_circle_started is True
    assert rt.mission_circle_center_n == pytest.approx(120.0)
    assert rt.mission_circle_center_e == pytest.approx(-35.0)
    assert 112.0 <= cmd_1[0] <= 128.0
    assert 120.0 <= cmd_1[1] <= 390.0
    assert -180.0 <= cmd_1[2] <= 180.0

    heading_rad = rt.fight_mode_heading_rad
    rt.x[0] = rt.fight_mode_anchor_n + 900.0 * np.cos(heading_rad)
    rt.x[1] = rt.fight_mode_anchor_e + 900.0 * np.sin(heading_rad)
    rt.t += rt.P.Ts

    cmd_2 = rt.fight_mode_commands()
    assert np.all(np.isfinite(cmd_2))
    assert abs(cmd_2[0] - cmd_1[0]) <= 12.0 * rt.P.Ts + 1e-9
    assert abs(cmd_2[1] - cmd_1[1]) <= 9.0 * rt.P.Ts + 1e-9
    assert abs(_wrapped_delta_deg(cmd_2[2], cmd_1[2])) <= 58.0 * rt.P.Ts + 1e-9
    assert np.isfinite(rt.reference_n)
    assert np.isfinite(rt.reference_e)


def test_runtime_frame_air_data_matches_post_step_state() -> None:
    from uavsim.forces_moments import air_data

    rt = Runtime.create(profile="loiter_orbit")
    rt.wind.steady_n = -10.0
    rt.wind.gust_body_w = 4.0

    info = rt.step()
    frame = build_simulation_frame(
        rt=rt,
        info=info,
        episode=1,
        scenario=rt.profile,
        controller="fixed_matlab_autopilot",
    )
    va_now, alpha_now, beta_now, _ = air_data(rt.x, info["wind"], rt.P, exact_source=True)

    assert frame["uav_state"]["airspeed"] == va_now
    assert frame["aero_state"]["angle_of_attack_deg"] == np.degrees(alpha_now)
    assert frame["aero_state"]["sideslip_deg"] == np.degrees(beta_now)
    assert np.isfinite(frame["aero_state"]["load_factor_nz"])


def test_websocket_ignores_non_object_json_payload() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/uav-digital-twin") as ws:
            ws.send_json([])
            frame = ws.receive_json()
            assert frame["type"] == "simulation_frame"


def test_build_simulation_frame_schema_snapshot() -> None:
    rt = Runtime.create(profile="straight_climb_altitude_hold")
    rt.wind.steady_n = -4.0
    rt.wind.gust_body_w = 1.5
    rt.wind.turbulence_std = 0.0

    info = rt.step()
    frame = build_simulation_frame(
        rt=rt,
        info=info,
        episode=3,
        scenario="straight_climb_altitude_hold",
        controller="fixed_matlab_autopilot",
    )

    assert frame["schema_version"] == SIMULATION_FRAME_SCHEMA_VERSION
    assert _schema_snapshot(frame) == {
        "type": "str",
        "schema_version": "int",
        "timestamp": "float",
        "episode": "int",
        "controller": "str",
        "scenario": "str",
        "profile": "str",
        "uav_state": {
            "position": ["float", "float", "float"],
            "attitude": ["float", "float", "float"],
            "pitch_rate_deg_s": "float",
            "airspeed": "float",
            "mach": "float",
            "flight_mode": "str",
        },
        "reference_state": {
            "autopilot": "str",
            "target_altitude": "float",
            "altitude_error": "float",
            "target_airspeed": "float",
            "airspeed_error": "float",
            "target_pitch_deg": "float",
            "target_roll_deg": "float",
            "target_heading_deg": "float",
            "target_lateral_offset": "float",
            "target_throttle": "float",
            "reference_position_n": "float",
            "reference_position_e": "float",
            "reference_altitude": "float",
            "horizontal_reference_error": "float",
            "distance_to_reference": "float",
            "distance_error": "float",
            "mission_area_size_m": "float",
            "circle_diameter_m": "float",
            "circle_radius_m": "float",
            "circle_airspeed_mps": "float",
            "circle_direction": "int",
            "circle_start_time_s": "float",
            "trajectory_profile": "str",
        },
        "wing_state": {
            "tip_deflection": "float",
            "max_deflection": "float",
            "average_strain": "float",
            "twist_angle": "float",
            "node_displacements": ["float", "float", "float", "float", "float"],
        },
        "aero_state": {
            "wind_speed": "float",
            "wind_direction": ["float", "float", "float"],
            "wind_body": ["float", "float", "float"],
            "turbulence_intensity": "float",
            "gust_level": "float",
            "angle_of_attack_deg": "float",
            "sideslip_deg": "float",
            "flight_path_angle_deg": "float",
            "load_factor_nz": "float",
            "density_kg_m3": "float",
            "dynamic_pressure_pa": "float",
            "speed_of_sound_mps": "float",
        },
        "control_state": {
            "control_energy": "float",
            "actuator_commands": ["float", "float", "float", "float", "float"],
            "actuator_commanded": ["float", "float", "float", "float", "float"],
            "saturation_ratio": "float",
            "roll_command_saturation_ratio": "float",
            "guidance_saturation_ratio": "float",
            "aileron_deflection": {"left": "float", "right": "float"},
            "elevator_deflection": "float",
            "rudder_deflection": "float",
            "flap_deflection": {"left": "float", "right": "float"},
            "spoiler_deployment": {"left": "float", "right": "float"},
            "throttle": "float",
            "flight_phase": "str",
            "tecs_total_energy_error": "float",
            "tecs_balance_energy_error": "float",
            "tecs_throttle_command": "float",
            "tecs_pitch_command_rad": "float",
            "tecs_elevator_bias_rad": "float",
        },
        "rl_metrics": {
            "method": "str",
            "reward": "float",
            "episode_return": "float",
            "td_error": "float",
            "policy_entropy": "float",
            "safety_violations": "int",
            "action_index": "int",
            "enabled": "bool",
            "epsilon": "float",
            "explored": "bool",
            "q_state": "NoneType",
            "q_value": "float",
            "max_next_q": "float",
            "updates": "int",
            "residual_active": "bool",
            "hard_condition_score": "float",
            "hjb_value": "float",
            "hjb_advantage": "float",
            "hjb_stage_cost": "float",
            "shield_active": "bool",
            "candidate_count": "int",
            "load_factor_nz": "float",
            "safety_risk_score": "float",
        },
        "guidance_state": {
            "cross_track_error_m": "float",
            "radial_error_m": "float",
            "lookahead_m": "float",
            "bearing_error_rad": "float",
            "lateral_accel_mps2": "float",
            "roll_command_rad": "float",
        },
    }
    assert frame["type"] == "simulation_frame"
    assert frame["episode"] == 3
    assert frame["controller"] == "fixed_matlab_autopilot"
    assert frame["scenario"] == "straight_climb_altitude_hold"
    assert frame["profile"] == "straight_climb_altitude_hold"
    assert frame["reference_state"]["trajectory_profile"] == frame["profile"]
    assert frame["reference_state"]["distance_error"] == frame["reference_state"]["distance_to_reference"]


def test_runtime_step_produces_finite_frame() -> None:
    rt = Runtime.create(profile="straight_climb_altitude_hold")
    info = rt.step()
    frame = build_simulation_frame(
        rt=rt,
        info=info,
        episode=1,
        scenario="straight_climb_altitude_hold",
        controller="fixed_matlab_autopilot",
    )
    assert frame["type"] == "simulation_frame"
    pos = frame["uav_state"]["position"]
    att = frame["uav_state"]["attitude"]
    assert len(pos) == 3 and all(isinstance(v, float) for v in pos)
    assert len(att) == 3
    assert len(frame["control_state"]["actuator_commands"]) == 5
    assert frame["reference_state"]["target_altitude"] >= 0.0
    assert np.isfinite(frame["reference_state"]["reference_position_n"])
    assert np.isfinite(frame["reference_state"]["reference_position_e"])
    assert np.isfinite(frame["reference_state"]["reference_altitude"])
    assert frame["reference_state"]["circle_diameter_m"] == 200.0
    assert frame["reference_state"]["mission_area_size_m"] == 10_000.0
    assert 0.0 <= frame["aero_state"]["load_factor_nz"] <= 2.0
    assert frame["profile"] == "straight_climb_altitude_hold"


def test_runtime_all_profiles_takeoff_climb_to_200_then_enter_compact_circle_smoke() -> None:
    assert TAKEOFF_PROFILES == TRAJECTORY_PROFILES
    sample_20_step = int(20.0 / Runtime.create().P.Ts) - 1
    final_step = int(120.0 / Runtime.create().P.Ts)
    for profile in TRAJECTORY_PROFILES - {FIGHT_MODE_PROFILE}:
        rt = Runtime.create(profile=profile)
        assert rt.x[3] == 0.0
        assert rt.x[2] == 0.0

        frame_20 = None
        max_airspeed = 0.0
        info = rt.step()
        for step in range(final_step - 1):
            info = rt.step()
            max_airspeed = max(max_airspeed, float(info["Va"]))
            if step == sample_20_step:
                frame_20 = build_simulation_frame(
                    rt=rt,
                    info=info,
                    episode=1,
                    scenario=profile,
                    controller="fixed_matlab_autopilot",
                )

        frame = build_simulation_frame(
            rt=rt,
            info=info,
            episode=1,
            scenario=profile,
            controller="fixed_matlab_autopilot",
        )
        assert frame_20 is not None
        assert np.all(np.isfinite(rt.x))

        assert frame_20["profile"] == profile
        assert 65.0 <= frame_20["reference_state"]["target_airspeed"] <= 91.0
        assert 60.0 <= frame_20["uav_state"]["airspeed"] <= 95.0
        assert frame_20["uav_state"]["position"][2] >= 195.0
        assert max_airspeed >= 65.0

        assert frame["profile"] == profile
        assert 24.0 <= frame["reference_state"]["target_airspeed"] <= 26.0
        assert abs(frame["uav_state"]["airspeed"] - frame["reference_state"]["target_airspeed"]) < 5.0
        assert 195.0 <= frame["uav_state"]["position"][2] <= 205.0
        assert rt.reference_circle_started
        assert rt.mission_circle_center_n is not None
        assert rt.mission_circle_center_e is not None
        assert 0.0 < frame["reference_state"]["circle_start_time_s"] < 40.0
        assert -180.0 <= frame["reference_state"]["target_heading_deg"] <= 180.0
        reference_radius = np.hypot(
            frame["reference_state"]["reference_position_n"] - rt.mission_circle_center_n,
            frame["reference_state"]["reference_position_e"] - rt.mission_circle_center_e,
        )
        assert abs(reference_radius - 100.0) < 1.0
        aircraft_radius = np.hypot(
            frame["uav_state"]["position"][0] - rt.mission_circle_center_n,
            frame["uav_state"]["position"][1] - rt.mission_circle_center_e,
        )
        assert abs(aircraft_radius - 100.0) < 20.0
        assert frame["reference_state"]["circle_diameter_m"] == 200.0
        assert frame["reference_state"]["horizontal_reference_error"] < 300.0
        assert frame["reference_state"]["distance_to_reference"] < 300.0
        assert abs(frame["aero_state"]["angle_of_attack_deg"]) < 10.0
        assert frame["control_state"]["saturation_ratio"] < 0.98
        assert frame["control_state"]["guidance_saturation_ratio"] < 0.90


def test_runtime_fight_mode_generates_finite_maneuver_reference() -> None:
    rt = Runtime.create(profile=FIGHT_MODE_PROFILE)
    final_step = int(85.0 / rt.P.Ts)
    refs: list[tuple[float, float, float]] = []
    info = rt.step()
    sample_stride = int(1.0 / rt.P.Ts)
    for step in range(final_step - 1):
        info = rt.step()
        if rt.fight_mode_start_time_s is not None and step % sample_stride == 0:
            refs.append((rt.reference_n, rt.reference_e, float(info["commands"][1])))

    frame = build_simulation_frame(
        rt=rt,
        info=info,
        episode=1,
        scenario=FIGHT_MODE_PROFILE,
        controller="fixed_matlab_autopilot",
    )
    assert frame["profile"] == FIGHT_MODE_PROFILE
    assert rt.fight_mode_start_time_s is not None
    assert np.all(np.isfinite(rt.x))
    assert np.all(np.isfinite(info["commands"]))
    assert 105.0 <= frame["reference_state"]["target_airspeed"] <= 135.0
    assert 115.0 <= frame["reference_state"]["reference_altitude"] <= 395.0
    assert -180.0 <= frame["reference_state"]["target_heading_deg"] <= 180.0
    assert frame["reference_state"]["circle_diameter_m"] == 480.0
    assert len(refs) > 20
    lateral_span = max(e for _, e, _ in refs) - min(e for _, e, _ in refs)
    altitude_span = max(h for _, _, h in refs) - min(h for _, _, h in refs)
    assert lateral_span > 15.0
    assert altitude_span > 20.0
    assert frame["control_state"]["throttle"] >= 0.34
    assert frame["reference_state"]["distance_to_reference"] < 700.0
    assert 0.0 <= frame["control_state"]["guidance_saturation_ratio"] <= 1.0


def test_runtime_autopilot_receives_wind_relative_airspeed() -> None:
    calm = Runtime.create(profile="straight_climb_altitude_hold")
    relative_wind = Runtime.create(profile="straight_climb_altitude_hold")
    for rt in (calm, relative_wind):
        rt.x[2] = -200.0  # force altitude-hold zone at the first command
        rt.x[3] = 70.0
        rt.t = 20.0
        rt.mission_straight_start_n = 0.0
        rt.mission_straight_start_e = 0.0
        rt.mission_speed_command_mps = 70.0
        rt.wind.turbulence_std = 0.0
    relative_wind.wind.steady_n = -4.0

    calm_info = calm.step()
    relative_wind_info = relative_wind.step()

    # The post-step airspeed can move either way because the different
    # throttle command has already affected the integrated state.  The
    # controller-facing effect we need to preserve is that wind-relative
    # airspeed is used before the actuator command is chosen.
    assert relative_wind_info["delta"][3] < calm_info["delta"][3]

    frame = build_simulation_frame(
        rt=relative_wind,
        info=relative_wind_info,
        episode=1,
        scenario="straight_climb_altitude_hold",
        controller="fixed_matlab_autopilot",
    )
    assert frame["control_state"]["throttle"] < calm_info["delta"][3]
    assert np.isfinite(frame["reference_state"]["airspeed_error"])


def test_fixed_and_q_learning_controller_modes_are_distinct_in_runtime_frames() -> None:
    fixed = Runtime.create(profile="loiter_orbit")
    fixed_info = fixed.step()
    fixed_frame = build_simulation_frame(
        rt=fixed,
        info=fixed_info,
        episode=1,
        scenario="loiter_orbit",
        controller="fixed_matlab_autopilot",
    )
    assert fixed.controller_mode == "fixed_matlab_autopilot"
    assert fixed_frame["rl_metrics"]["enabled"] is False
    assert fixed_frame["control_state"]["tecs_throttle_command"] == 0.0

    q_rt = Runtime.create(profile="loiter_orbit")
    running, episode, scenario, controller = _handle_command(
        q_rt,
        {
            "command": "start",
            "profile": "loiter_orbit",
            "scenario": "loiter_orbit",
            "controller": "online_q_learning",
        },
        True,
    )
    q_info = q_rt.step()
    q_frame = build_simulation_frame(rt=q_rt, info=q_info, episode=episode, scenario=scenario, controller=controller)

    assert running is True
    assert q_rt.controller_mode == "online_q_learning"
    assert q_frame["controller"] == "online_q_learning"
    assert q_frame["rl_metrics"]["enabled"] is True
    assert np.isfinite(q_frame["rl_metrics"]["reward"])
    assert q_frame["control_state"]["tecs_throttle_command"] >= 0.0

    sharq_rt = Runtime.create(profile="loiter_orbit")
    _, sharq_episode, sharq_scenario, sharq_controller = _handle_command(
        sharq_rt,
        {
            "command": "start",
            "profile": "loiter_orbit",
            "scenario": "loiter_orbit",
            "controller": "sharq_hjb",
            "effects": {"steady_wind_e": 10.0, "turbulence_std": 2.0},
        },
        True,
    )
    sharq_info = sharq_rt.step()
    sharq_frame = build_simulation_frame(
        rt=sharq_rt,
        info=sharq_info,
        episode=sharq_episode,
        scenario=sharq_scenario,
        controller=sharq_controller,
    )

    assert sharq_rt.controller_mode == "sharq_hjb"
    assert sharq_frame["controller"] == "sharq_hjb"
    assert sharq_frame["rl_metrics"]["method"] in {"sharq_hjb", "sharq_hjb_q_fallback"}
    assert sharq_frame["rl_metrics"]["enabled"] is True
    assert np.isfinite(sharq_frame["rl_metrics"]["hjb_value"])
    assert sharq_frame["control_state"]["tecs_throttle_command"] >= 0.0


def test_websocket_emits_simulation_frame() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/uav-digital-twin") as ws:
            ws.send_json({
                "command": "start",
                "scenario": "loiter_orbit",
                "profile": "loiter_orbit",
                "controller": "fixed_matlab_autopilot",
                "effects": {"steady_wind_n": 0.0},
            })
            frame = ws.receive_json()
            assert frame["type"] == "simulation_frame"
            assert frame["profile"] == "loiter_orbit"
            assert "uav_state" in frame and "control_state" in frame


def test_websocket_ignores_malformed_payload() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/uav-digital-twin") as ws:
            ws.send_text("this-is-not-json")
            # Server should still emit a frame on the next loop iteration.
            frame = ws.receive_json()
            assert frame["type"] == "simulation_frame"
