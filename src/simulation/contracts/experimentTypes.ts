export type Scenario =
  | 'calm'
  | 'steady'
  | 'gust'
  | 'crosswind'
  | 'turbulence_burst'
  | 'severe'
  | 'actuator_saturation'
  | 'sensor_noise'
  | 'stiffness_variation'
  | 'payload_shift';

export interface ExperimentConfig {
  scenario: Scenario;
  windSpeed: number; // m/s
  crosswind: number; // m/s, lateral
  turbulenceIntensity: number; // 0-1
  angleOfAttack: number; // degrees
  wingStiffness: number; // scaling factor
  payloadMass: number; // kg
  sensorNoiseLevel: number; // 0-1
}

export interface ExperimentState {
  isRunning: boolean;
  time: number;
  currentConfig: ExperimentConfig;
}
