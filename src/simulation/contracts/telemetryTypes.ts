export interface TelemetryPoint {
  time: number;
  tipDeflection: number; // mm
  pitchRate: number; // deg/s
  controlEffort: number; // general unit
  reward: number; // RL reward proxy
  strain: number; // %
  forwardPosition: number; // m (inertial north, pn)
  eastPosition: number; // m (inertial east, pe)
  altitude: number; // m
  airspeed: number; // m/s
  mach: number; // Mach number
  pitchAngle: number; // deg
  rollAngle: number; // deg
  yawAngle: number; // deg
  angleOfAttack: number; // deg
  sideslipAngle: number; // deg
  flightPathAngle: number; // deg
  loadFactorNz: number; // g
  leftAileron: number; // deg
  rightAileron: number; // deg
  elevator: number; // deg
  rudder: number; // deg
  flapLeft: number; // deg
  flapRight: number; // deg
  leftSpoiler: number; // 0-1 normalized deployment
  rightSpoiler: number; // 0-1 normalized deployment
  throttle: number; // 0-1
  flightMode?: string;
  autopilot?: string;
  trajectoryProfile?: string;
  targetAltitude?: number; // m
  altitudeError?: number; // m
  targetAirspeed?: number; // m/s
  airspeedError?: number; // m/s
  distanceError?: number; // m
  distanceToReference?: number; // m
  targetPitch?: number; // deg
  targetRoll?: number; // deg
  targetHeading?: number; // deg
  targetLateralOffset?: number; // m
  referenceForwardPosition?: number; // m (reference pn)
  referenceEastPosition?: number; // m (reference pe)
  referenceAltitude?: number; // m
  horizontalReferenceError?: number; // m
  missionAreaSize?: number; // m
  circleDiameter?: number; // m
  circleRadius?: number; // m
  circleAirspeed?: number; // m/s
  circleDirection?: number; // +/-1
  circleStartTime?: number; // s
  windSpeed?: number; // m/s
  windDirectionX?: number; // body/inertial normalized x
  windDirectionY?: number; // body/inertial normalized y
  windDirectionZ?: number; // body/inertial normalized z
  windBodyX?: number; // m/s
  windBodyY?: number; // m/s
  windBodyZ?: number; // m/s
  turbulenceIntensity?: number; // 0-1
  gustLevel?: number; // m/s proxy
}

export interface TelemetryStream {
  /** UI history is bounded by the Zustand store; do not append unbounded raw backend frames here. */
  history: TelemetryPoint[];
  /** Maximum retained UI telemetry samples for charts/traces. */
  maxPoints: number;
}
