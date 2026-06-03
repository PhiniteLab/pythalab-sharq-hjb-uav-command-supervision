# Setup

This document lists the local dependencies and first validation commands.

## Requirements

- Node.js and npm for the Vite/React frontend.
- Python 3.10+ for the backend package.
- A modern WebGL-capable browser for the 3D scene.

The examples assume a virtual environment at repository root:

```bash
python -m venv .venv
. .venv/bin/activate
```

## Frontend setup

```bash
npm install
```

Package scripts:

```bash
npm run dev      # Vite development server on port 3000
npm run lint     # TypeScript noEmit type-check
npm run build    # production build
npm run preview  # preview built dist output
npm run clean    # remove dist
```

## Backend setup

```bash
cd backend
../.venv/bin/python -m pip install -e '.[dev]'
```

The backend package name is `pythalab-uavsim-backend`, the import package is `uavsim`, and the console script is `uavsim-backend`.

## Environment variables

Frontend WebSocket URL selection order:

1. `VITE_BACKEND_WS_URL`
2. `VITE_UAV_BACKEND_WS`
3. default: `ws://localhost:8000/ws/uav-digital-twin`

Python selection for the Vite backend lifecycle plugin:

1. repository-root `.venv/bin/python` if it exists,
2. `PYTHON` environment variable,
3. `python3` fallback.

Optional HMR control:

```bash
DISABLE_HMR=true npm run dev
```

Use this only when local file watching is unstable.

## First validation

```bash
npm run lint
npm run build
cd backend && ../.venv/bin/python -m pytest -q
```

Expected backend test result: `41 passed`.
