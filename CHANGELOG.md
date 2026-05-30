# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-05-31

First stable, citable baseline of the simulation-only fixed-wing UAV
command-supervision system.

### Added
- 12-state rigid-body fixed-wing UAV simulator with a fixed / gain-scheduled
  autopilot that remains the actuator-facing controller.
- Bounded tabular Q-learning residual supervisor and an HJB residual supervisor,
  both restricted to small bounded residuals on the commanded airspeed,
  altitude, and heading references.
- React / Three.js live visualization frontend and a WebSocket backend bridge.
- Reproducible benchmark package under `experiments/` (raw runs, aggregated
  data, tables, and figures) with a SHA-256 manifest under `reproducibility/`.
- Citation metadata (`CITATION.cff`) and project documentation under `docs/`.

[0.0.1]: https://github.com/PhiniteLab/pythalab-sharq-hjb-uav-command-supervision/releases/tag/v0.0.1
