export type ControllerMode = 'passive' | 'pid' | 'mpc' | 'lqr' | 'hjb' | 'rl' | 'external' | 'all';

export interface ControllerMetrics {
  controlEffort: number;
  saturationLimit: number;
  active: boolean;
}
