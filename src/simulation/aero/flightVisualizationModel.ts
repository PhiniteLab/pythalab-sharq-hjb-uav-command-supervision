import { TelemetryPoint } from '../contracts/telemetryTypes';

export type ControlSurfaceName =
  | 'leftAileron'
  | 'rightAileron'
  | 'elevator'
  | 'rudder'
  | 'flapLeft'
  | 'flapRight'
  | 'leftSpoiler'
  | 'rightSpoiler';

export interface SurfaceLimit {
  label: string;
  min: number;
  max: number;
  unit: 'deg' | '%';
}

export const CONTROL_SURFACE_LIMITS: Record<ControlSurfaceName, SurfaceLimit> = {
  leftAileron: { label: 'L Aileron', min: -22, max: 22, unit: 'deg' },
  rightAileron: { label: 'R Aileron', min: -22, max: 22, unit: 'deg' },
  elevator: { label: 'Elevator', min: -24, max: 24, unit: 'deg' },
  rudder: { label: 'Rudder', min: -28, max: 28, unit: 'deg' },
  flapLeft: { label: 'L Flap', min: 0, max: 25, unit: 'deg' },
  flapRight: { label: 'R Flap', min: 0, max: 25, unit: 'deg' },
  leftSpoiler: { label: 'L Spoiler', min: 0, max: 1, unit: '%' },
  rightSpoiler: { label: 'R Spoiler', min: 0, max: 1, unit: '%' },
};

export interface EnvelopeStatus {
  severity: 'nominal' | 'caution' | 'critical';
  messages: string[];
  alphaLimit: number;
  betaLimit: number;
  nzLimit: number;
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function isSurfaceSaturated(name: ControlSurfaceName, value: number, threshold = 0.96) {
  const limit = CONTROL_SURFACE_LIMITS[name];
  return value >= limit.max * threshold || value <= limit.min * threshold;
}

function getSurfaceValue(frame: TelemetryPoint | undefined, name: ControlSurfaceName) {
  if (!frame) return 0;
  return frame[name] ?? 0;
}

export function envelopeStatus(frame: TelemetryPoint | undefined): EnvelopeStatus {
  const alphaLimit = 18;
  const betaLimit = 14;
  const nzLimit = 4.5;
  if (!frame) return { severity: 'nominal', messages: [], alphaLimit, betaLimit, nzLimit };

  const messages: string[] = [];
  if (Math.abs(frame.angleOfAttack) > alphaLimit) messages.push('AoA limit');
  if (Math.abs(frame.sideslipAngle) > betaLimit) messages.push('Sideslip limit');
  if (Math.abs(frame.loadFactorNz) > nzLimit) messages.push('Nz structural limit');
  const saturated = (Object.keys(CONTROL_SURFACE_LIMITS) as ControlSurfaceName[])
    .filter((name) => isSurfaceSaturated(name, getSurfaceValue(frame, name)));
  if (saturated.length > 0) messages.push(`${saturated.length} surface saturation`);

  const maxIncidence = Math.max(Math.abs(frame.angleOfAttack) / alphaLimit, Math.abs(frame.sideslipAngle) / betaLimit);
  const maxLoad = Math.abs(frame.loadFactorNz) / nzLimit;
  const severity = messages.length === 0
    ? 'nominal'
    : maxIncidence > 1.25 || maxLoad > 1.1 || saturated.length > 2
      ? 'critical'
      : 'caution';

  return { severity, messages, alphaLimit, betaLimit, nzLimit };
}
