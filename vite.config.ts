import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import fs from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import path from 'node:path';
import { defineConfig, loadEnv, type Plugin } from 'vite';

type BackendState = 'stopped' | 'starting' | 'running' | 'stopping' | 'error';

function sendJson(res: ServerResponse, status: number, payload: Record<string, unknown>) {
  res.statusCode = status;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(payload));
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

function uavBackendLifecyclePlugin(): Plugin {
  let child: ChildProcessWithoutNullStreams | undefined;
  let state: BackendState = 'stopped';
  let message = 'Backend not started';
  let lastLog = '';
  let stopping = false;

  const backendDir = path.resolve(__dirname, 'backend');
  const venvPython = path.resolve(__dirname, '.venv/bin/python');
  const python = fs.existsSync(venvPython) ? venvPython : (process.env.PYTHON ?? 'python3');

  const appendLog = (prefix: string, data: Buffer) => {
    lastLog = `${lastLog}${prefix}${data.toString()}`.slice(-2500);
  };

  const isAlive = () => Boolean(child && child.exitCode === null && !child.killed);

  const stopBackend = (reason = 'stop requested') => {
    if (!child) {
      state = 'stopped';
      message = 'Backend stopped';
      return;
    }
    if (stopping) return;
    stopping = true;
    state = 'stopping';
    message = `Stopping backend: ${reason}`;
    const proc = child;
    proc.kill('SIGTERM');
    setTimeout(() => {
      if (proc.exitCode === null && !proc.killed) proc.kill('SIGKILL');
    }, 1800).unref();
  };

  const probeBackendHealth = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/health', { signal: AbortSignal.timeout(450) });
      return response.ok;
    } catch {
      return false;
    }
  };

  const startBackend = async () => {
    if (isAlive()) {
      state = 'running';
      message = 'Backend already running';
      return;
    }
    if (await probeBackendHealth()) {
      state = 'running';
      message = 'Backend already available on port 8000 (external process)';
      return;
    }
    stopping = false;
    state = 'starting';
    message = `Starting backend with ${python}`;
    lastLog = '';
    child = spawn(python, ['-m', 'uavsim.server'], {
      cwd: backendDir,
      env: {
        ...process.env,
        PYTHONPATH: path.resolve(backendDir, 'src'),
        PYTHONUNBUFFERED: '1',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    child.stdout.on('data', data => appendLog('[backend] ', data));
    child.stderr.on('data', data => appendLog('[backend] ', data));
    child.on('spawn', () => {
      state = 'running';
      message = 'Backend process started';
    });
    child.on('error', error => {
      state = 'error';
      message = error.message;
      child = undefined;
      stopping = false;
    });
    child.on('exit', (code, signal) => {
      state = code === 0 || stopping ? 'stopped' : 'error';
      message = `Backend exited (${signal ?? code ?? 'unknown'})`;
      child = undefined;
      stopping = false;
    });
  };

  const statusPayload = async () => {
    if (!isAlive() && await probeBackendHealth()) {
      state = 'running';
      message = 'Backend available on port 8000 (external process)';
    } else if (!isAlive() && state === 'running') {
      state = 'stopped';
      message = 'Backend stopped';
    }
    return {
      state: isAlive() ? (state === 'starting' ? 'starting' : 'running') : state,
      pid: isAlive() ? child?.pid : null,
      message,
      lastLog,
    };
  };

  const cleanup = (reason: string) => stopBackend(reason);
  // Synchronous fallback: when the Node process is exiting, async setTimeout
  // SIGKILL will never fire because the event loop is being torn down. So we
  // also issue an immediate SIGKILL here as a best-effort to avoid leaving
  // the spawned backend running as a zombie when Vite dies unexpectedly.
  const killChildSync = () => {
    if (child && child.exitCode === null && !child.killed) {
      try { child.kill('SIGKILL'); } catch { /* ignore */ }
    }
  };
  process.once('exit', () => { cleanup('vite exit'); killChildSync(); });
  process.once('SIGINT', () => { cleanup('SIGINT'); killChildSync(); process.exit(130); });
  process.once('SIGTERM', () => { cleanup('SIGTERM'); killChildSync(); process.exit(143); });

  return {
    name: 'uav-backend-lifecycle',
    configureServer(server) {
      server.httpServer?.once('close', () => cleanup('vite dev server closed'));
      server.middlewares.use(async (req, res, next) => {
        const url = req.url?.split('?')[0];
        if (!url?.startsWith('/api/backend')) return next();
        if (url === '/api/backend/status' && req.method === 'GET') {
          return sendJson(res, 200, await statusPayload());
        }
        if (url === '/api/backend/start' && req.method === 'POST') {
          await readBody(req).catch(() => '');
          await startBackend();
          return sendJson(res, 200, await statusPayload());
        }
        if (url === '/api/backend/stop' && req.method === 'POST') {
          await readBody(req).catch(() => '');
          stopBackend('ui stop');
          return sendJson(res, 200, { state: isAlive() ? 'stopping' : 'stopped', pid: isAlive() ? child?.pid : null, message, lastLog });
        }
        return sendJson(res, 404, { state, message: 'Unknown backend endpoint' });
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  loadEnv(mode, '.', '');
  return {
    plugins: [uavBackendLifecyclePlugin(), react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            react: ['react', 'react-dom', 'zustand'],
            three: ['three', '@react-three/fiber', '@react-three/drei', '@react-three/postprocessing'],
            charts: ['recharts'],
          },
        },
      },
    },
    server: {
      open: true,
      // Set DISABLE_HMR=true when file watching is unstable in the local environment.
      hmr: process.env.DISABLE_HMR !== 'true',
    },
  };
});
