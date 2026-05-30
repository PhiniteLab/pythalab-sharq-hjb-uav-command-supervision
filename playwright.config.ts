import { defineConfig } from '@playwright/test';

const port = Number(process.env.PLAYWRIGHT_BASE_PORT ?? 3000);
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : 'list',
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  use: {
    browserName: 'chromium',
    baseURL,
    viewport: { width: 1440, height: 900 },
    headless: true,
    trace: 'on-first-retry',
  },
  webServer: {
    command: `npm run dev:test -- --host 127.0.0.1 --port ${port} --strictPort`,
    env: {
      VITE_BACKEND_WS_URL: 'ws://127.0.0.1:9/ws/uav-digital-twin',
    },
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
