import { create } from 'zustand';
import { TelemetryPoint, TelemetryStream } from '../simulation/contracts/telemetryTypes';
import { ControllerMode, ControllerMetrics } from '../simulation/contracts/controllerTypes';
import { Scenario, ExperimentConfig, ExperimentState } from '../simulation/contracts/experimentTypes';

export type BackendEffectMode = 'calm' | 'headwind' | 'tailwind' | 'crosswind' | 'turbulence' | 'gust' | 'q_learning' | 'sharq_hjb';
export type BackendTrajectoryProfile = 'runway_takeoff_accel_200' | 'takeoff_climbout_200' | 'high_speed_climb_s_turn_200' | 'straight_climb_altitude_hold' | 'figure_eight' | 'racetrack' | 'loiter_orbit' | 'fight_mode';
export type FlightUiMode = 'cinematic' | 'operator';

interface SimulationState {
  // Experiment / Physics configs
  experiment: ExperimentState;

  // Controller
  controllerMode: ControllerMode;
  controllerMetrics: ControllerMetrics;

  // Data
  telemetry: TelemetryStream;
  latestTelemetry?: TelemetryPoint;
  backendActive: boolean;
  backendEffectMode: BackendEffectMode;
  backendTrajectoryProfile: BackendTrajectoryProfile;
  backendResetNonce: number;
  uiMode: FlightUiMode;

  // Actions
  setControllerMode: (mode: ControllerMode) => void;
  setExperimentConfig: (partial: Partial<ExperimentConfig>) => void;
  toggleSimulation: () => void;
  addTelemetry: (point: Omit<TelemetryPoint, 'time'>) => void;
  resetTelemetry: () => void;
  incrementTime: (dt: number) => void;
  setTime: (t: number) => void;
  setBackendActive: (active: boolean) => void;
  setBackendEffectMode: (mode: BackendEffectMode) => void;
  setBackendTrajectoryProfile: (profile: BackendTrajectoryProfile) => void;
  requestBackendReset: () => void;
  setUiMode: (mode: FlightUiMode) => void;
  toggleUiMode: () => void;
}

const MAX_TELEMETRY_POINTS = 480;

export const useSimulationStore = create<SimulationState>((set) => ({
  experiment: {
    isRunning: true,
    time: 0,
    currentConfig: {
      scenario: 'steady',
      windSpeed: 25,
      crosswind: 0,
      turbulenceIntensity: 0.2,
      angleOfAttack: 2.5,
      wingStiffness: 1.0,
      payloadMass: 5.0,
      sensorNoiseLevel: 0,
    }
  },

  controllerMode: 'passive',
  controllerMetrics: {
    controlEffort: 0,
    saturationLimit: 100,
    active: false,
  },

  telemetry: {
    history: [],
    maxPoints: MAX_TELEMETRY_POINTS,
  },
  latestTelemetry: undefined,
  backendActive: false,
  backendEffectMode: 'calm',
  backendTrajectoryProfile: 'runway_takeoff_accel_200',
  backendResetNonce: 0,
  uiMode: 'cinematic',

  setControllerMode: (mode) => set((state) => ({
    controllerMode: mode,
    telemetry: { ...state.telemetry, history: [] },
    latestTelemetry: undefined,
    experiment: { ...state.experiment, time: 0 }
  })),

  setExperimentConfig: (partial) => set((state) => {
    const isNewScenario = partial.scenario && partial.scenario !== state.experiment.currentConfig.scenario;
    return {
      experiment: {
        ...state.experiment,
        time: isNewScenario ? 0 : state.experiment.time,
        currentConfig: { ...state.experiment.currentConfig, ...partial }
      },
      telemetry: {
        ...state.telemetry,
        history: isNewScenario ? [] : state.telemetry.history
      },
      latestTelemetry: isNewScenario ? undefined : state.latestTelemetry
    };
  }),

  toggleSimulation: () => set((state) => ({
    experiment: { ...state.experiment, isRunning: !state.experiment.isRunning }
  })),

  addTelemetry: (point) => set((state) => {
    const newPoint: TelemetryPoint = { ...point, time: state.experiment.time };
    const newHistory = [...state.telemetry.history, newPoint].slice(-state.telemetry.maxPoints);
    return {
      telemetry: { ...state.telemetry, history: newHistory },
      latestTelemetry: newPoint,
      controllerMetrics: { ...state.controllerMetrics, controlEffort: point.controlEffort }
    };
  }),

  resetTelemetry: () => set((state) => ({
    telemetry: { ...state.telemetry, history: [] },
    latestTelemetry: undefined,
    experiment: { ...state.experiment, time: 0 }
  })),

  incrementTime: (dt) => set((state) => ({
    experiment: { ...state.experiment, time: state.experiment.time + dt }
  })),

  setTime: (t) => set((state) => ({
    experiment: { ...state.experiment, time: t }
  })),

  setBackendActive: (active) => set({ backendActive: active }),
  setBackendEffectMode: (mode) => set({ backendEffectMode: mode }),
  // Changing the trajectory profile already triggers the bridge's
  // ``[trajectoryProfile]`` effect, which resets telemetry and sends a reset
  // command. We must not also bump ``backendResetNonce`` here or the bridge
  // fires two reset commands back-to-back on every profile switch.
  setBackendTrajectoryProfile: (profile) => set({ backendTrajectoryProfile: profile }),
  requestBackendReset: () => set((state) => ({ backendResetNonce: state.backendResetNonce + 1 })),
  setUiMode: (mode) => set({ uiMode: mode }),
  toggleUiMode: () => set((state) => ({ uiMode: state.uiMode === 'cinematic' ? 'operator' : 'cinematic' }))
}));
