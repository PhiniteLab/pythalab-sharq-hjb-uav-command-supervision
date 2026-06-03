# Full-Duration Benchmark Results

This directory contains full scenario duration benchmark outputs.

## Scope

- 20 challenging scenarios.
- 3 controller modes.
- 60 total episodes.
- Scenario durations are 45-60 seconds.

## Aggregate result snapshot

| Method | Episodes | RMS ref [m] | CI95 | RMS alt [m] | CI95 | RMS Va [m/s] | Control idx. | Safety frac | Max abs(nz) | Residual active | HJB adv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline autopilot | 20 | 338.617 | +/-129.248 | 114.137 | +/-11.312 | 10.546 | 5.665 | 0.0048 | 7.414 | 0.000 | 0.0000 |
| Baseline + tabular Q residual | 20 | 88.809 | +/-76.550 | 65.819 | +/-6.059 | 15.086 | 6.524 | 0.0031 | 6.844 | 0.913 | 0.0000 |
| Baseline + SHARQ-HJB residual | 20 | 44.809 | +/-32.781 | 64.710 | +/-5.808 | 15.191 | 6.346 | 0.0033 | 6.800 | 0.888 | -0.3263 |

## Files

| File | Use |
| --- | --- |
| `scenario_catalog.json` | Scenario definitions. |
| `comparative/aggregate_by_method.csv` | Method-level aggregate table. |
| `comparative/all_episode_summary.csv` | Per-scenario/per-seed scalar metrics. |
| `comparative/fight_mode_60s_smoke.csv` | Fight-mode finite-runtime smoke metrics. |
| `<method>/episode_summary.csv` | Method-specific scalar metrics. |
| `<method>/aggregate_metrics.csv` | Method-specific aggregate metrics. |
| `<method>/steps.jsonl` | Step-level time series. |
