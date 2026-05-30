export interface SimulationFrameMessage {
  type: 'simulation_frame';
  /** Increment when the top-level frame contract intentionally changes. */
  schema_version: 1;
  timestamp: number;
  episode: number;
  controller: string;
  scenario: string;
  profile: 'runway_takeoff_accel_200' | 'takeoff_climbout_200' | 'high_speed_climb_s_turn_200' | 'straight_climb_altitude_hold' | 'figure_eight' | 'racetrack' | 'loiter_orbit' | 'fight_mode' | string;
  uav_state: {
    /** [pn, pe, altitude] in metres. Backend state stores pd down-positive; frame exports altitude = -pd. */
    position: [number, number, number];
    /** [roll, pitch, yaw] in degrees. */
    attitude: [number, number, number];
    pitch_rate_deg_s: number;
    airspeed: number;
    mach?: number;
    flight_mode: 'takeoff' | 'climb' | 'descend' | 'altitude_hold' | 'airborne' | string;
  };
  reference_state: {
    /** Controller label echoed from the latest command: baseline, baseline+Q, or SHARQ-HJB. */
    autopilot: string;
    target_altitude: number;
    altitude_error: number;
    target_airspeed: number;
    airspeed_error: number;
    target_pitch_deg: number;
    target_roll_deg: number;
    target_heading_deg: number;
    target_lateral_offset: number;
    target_throttle: number;
    reference_position_n: number;
    reference_position_e?: number;
    reference_altitude?: number;
    horizontal_reference_error?: number;
    distance_to_reference: number;
    distance_error: number;
    mission_area_size_m?: number;
    circle_diameter_m?: number;
    circle_radius_m?: number;
    circle_airspeed_mps?: number;
    circle_direction?: number;
    circle_start_time_s?: number;
    trajectory_profile: string;
  };
  /** Compatibility fields only: the active backend is rigid-body and currently emits zeros here. */
  wing_state: {
    tip_deflection: number;
    max_deflection: number;
    average_strain: number;
    twist_angle: number;
    node_displacements: number[];
  };
  aero_state: {
    wind_speed: number;
    wind_direction: [number, number, number];
    wind_body?: [number, number, number];
    turbulence_intensity: number;
    gust_level: number;
    angle_of_attack_deg?: number;
    sideslip_deg?: number;
    flight_path_angle_deg?: number;
    /** Acceleration-derived normal load factor, clipped by the backend for display safety. */
    load_factor_nz?: number;
    /** ISA-scaled density used by the force model at the current altitude. */
    density_kg_m3?: number;
    /** Dynamic pressure qbar = 0.5 rho Va². */
    dynamic_pressure_pa?: number;
    /** Local speed of sound from the atmosphere model. */
    speed_of_sound_mps?: number;
  };
  control_state: {
    control_energy: number;
    actuator_commands: number[];
    /** Raw controller command before actuator lag/rate limiting. */
    actuator_commanded?: number[];
    saturation_ratio: number;
    roll_command_saturation_ratio?: number;
    guidance_saturation_ratio?: number;
    aileron_deflection?: {
      left: number;
      right: number;
    };
    elevator_deflection?: number;
    rudder_deflection?: number;
    flap_deflection?: {
      left: number;
      right: number;
    };
    spoiler_deployment?: {
      left: number;
      right: number;
    };
    throttle?: number;
    flight_phase?: 'ground_roll' | 'airborne' | string;
    tecs_total_energy_error?: number;
    tecs_balance_energy_error?: number;
    tecs_throttle_command?: number;
    tecs_pitch_command_rad?: number;
    tecs_elevator_bias_rad?: number;
  };
  /** Fixed modes emit zeros; residual modes emit bounded tabular/SHARQ-HJB diagnostics. */
  rl_metrics: {
    method?: string;
    reward: number;
    episode_return: number;
    td_error: number;
    policy_entropy: number;
    safety_violations: number;
    action_index?: number;
    enabled?: boolean;
    epsilon?: number;
    explored?: boolean;
    q_state?: number[] | null;
    q_value?: number;
    max_next_q?: number;
    updates?: number;
    residual_active?: boolean;
    hard_condition_score?: number;
    hjb_value?: number;
    hjb_advantage?: number;
    hjb_stage_cost?: number;
    shield_active?: boolean;
    candidate_count?: number;
    load_factor_nz?: number;
    safety_risk_score?: number;
  };
  guidance_state?: {
    cross_track_error_m: number;
    radial_error_m: number;
    lookahead_m?: number;
    bearing_error_rad?: number;
    lateral_accel_mps2?: number;
    roll_command_rad?: number;
  };
}

export interface ExperimentCommandMessage {
  command: 'start' | 'stop' | 'pause' | 'reset' | 'configure';
  episode?: number;
  scenario?: string;
  profile?: 'runway_takeoff_accel_200' | 'takeoff_climbout_200' | 'high_speed_climb_s_turn_200' | 'straight_climb_altitude_hold' | 'figure_eight' | 'racetrack' | 'loiter_orbit' | 'fight_mode' | string;
  /** Controller label. Supported backend modes include baseline, baseline+Q, and SHARQ-HJB. */
  controller?: string;
  seed?: number;
  effects?: Partial<{
    steady_wind_n: number;
    steady_wind_e: number;
    steady_wind_d: number;
    gust_body_u: number;
    gust_body_v: number;
    gust_body_w: number;
    turbulence_std: number;
  }>;
}
