import { expect, test } from '@playwright/test';

test.describe('frontend browser smoke', () => {
  test('loads the homepage and reflects backend status without requiring Start', async ({ page, request }) => {
    const statusResponse = await request.get('/api/backend/status');
    expect(statusResponse.ok()).toBeTruthy();

    const status = await statusResponse.json() as {
      state: string;
      message?: string;
      pid?: number | null;
    };

    expect(status.state).toMatch(/^(stopped|starting|running|stopping|error)$/);

    await page.goto('/', { waitUntil: 'domcontentloaded' });

    await expect(page.getByText('UAV Flight Console')).toBeVisible();
    await expect(page.getByText('SAAB Gripen Digital Twin')).toBeVisible();

    const systemTab = page.getByRole('button', { name: 'System' });
    if (!(await systemTab.isVisible())) {
      await page.getByRole('button', { name: /Cinematic|Operator/i }).click();
    }

    await expect(page.getByText(/Awaiting backend telemetry/i)).toBeVisible();
    await expect(systemTab).toBeVisible();
    await systemTab.click({ force: true });

    const diagnosticsPanel = page.locator('section').filter({ hasText: 'Backend Diagnostics' });
    await expect(diagnosticsPanel).toBeVisible();
    await expect(diagnosticsPanel).toContainText('Process');
    await expect(diagnosticsPanel).toContainText(/stopped|starting|running|stopping|error|unavailable/i);
    if (status.message) {
      await expect(diagnosticsPanel).toContainText(/Backend|port|started|stopped|available|Not checked|API/i);
    }
  });

  test('keeps telemetry history bounded during burst updates', async ({ page }) => {
    await page.goto('/', { waitUntil: 'domcontentloaded' });

    const result = await page.evaluate(async () => {
      const modulePath = new URL('/src/state/simulationStore.ts', window.location.origin).toString();
      const { useSimulationStore } = await import(/* @vite-ignore */ modulePath);
      const store = useSimulationStore;

      store.getState().resetTelemetry();
      const { maxPoints } = store.getState().telemetry;
      const overflow = 125;

      for (let index = 0; index < maxPoints + overflow; index += 1) {
        store.getState().setTime(index * 0.1);
        store.getState().addTelemetry({
          tipDeflection: index % 7,
          pitchRate: 0.15 * index,
          controlEffort: index % 100,
          reward: 1,
          strain: 0.05,
          forwardPosition: index,
          eastPosition: index * 0.25,
          altitude: 120 + (index % 9),
          airspeed: 72 + (index % 5),
          mach: 0.21,
          pitchAngle: 1.5,
          rollAngle: 0.5,
          yawAngle: 3,
          angleOfAttack: 2.2,
          sideslipAngle: 0.1,
          flightPathAngle: 1.1,
          loadFactorNz: 1.02,
          leftAileron: 0.3,
          rightAileron: -0.3,
          elevator: -0.2,
          rudder: 0.1,
          flapLeft: 0,
          flapRight: 0,
          leftSpoiler: 0,
          rightSpoiler: 0,
          throttle: 0.68,
          flightMode: 'smoke',
          autopilot: 'fixed_matlab',
          trajectoryProfile: 'runway_takeoff_accel_200',
          targetAltitude: 200,
          altitudeError: 80,
          targetAirspeed: 90,
          airspeedError: 12,
          distanceError: 0,
          distanceToReference: 0,
          targetPitch: 2,
          targetRoll: 0,
          targetHeading: 3,
          targetLateralOffset: 0,
          referenceForwardPosition: index,
          referenceEastPosition: index * 0.25,
          referenceAltitude: 200,
          horizontalReferenceError: 0,
          missionAreaSize: 200,
          circleDiameter: 200,
          circleRadius: 100,
          circleAirspeed: 90,
          circleDirection: 1,
          circleStartTime: 0,
          windSpeed: 0,
          windDirectionX: 1,
          windDirectionY: 0,
          windDirectionZ: 0,
          windBodyX: 0,
          windBodyY: 0,
          windBodyZ: 0,
          turbulenceIntensity: 0,
          gustLevel: 0,
        });
      }

      const current = store.getState();
      const history = current.telemetry.history;

      return {
        maxPoints,
        overflow,
        historyLength: history.length,
        oldestForwardPosition: history[0]?.forwardPosition ?? null,
        newestForwardPosition: history[history.length - 1]?.forwardPosition ?? null,
        latestForwardPosition: current.latestTelemetry?.forwardPosition ?? null,
      };
    });

    expect(result.historyLength).toBe(result.maxPoints);
    expect(result.oldestForwardPosition).toBe(result.overflow);
    expect(result.newestForwardPosition).toBe(result.maxPoints + result.overflow - 1);
    expect(result.latestForwardPosition).toBe(result.newestForwardPosition);
  });
});
