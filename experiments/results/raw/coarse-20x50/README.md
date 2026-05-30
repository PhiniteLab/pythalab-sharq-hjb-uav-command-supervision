# 20x50 Coarse Sweep Results

This directory contains short-horizon early-phase sweep outputs.

## Scope

- 20 scenarios.
- 50 seeds.
- 3 controller modes.
- 3000 total episodes.
- `duration=8 s`, `sample_time=0.02 s`, `substeps=1`, `step_log_stride=0`.

## Aggregate result snapshot

| Method | Episodes | RMS ref [m] | CI95 | RMS alt [m] | CI95 | RMS Va [m/s] | Control idx. | Safety frac | Max abs(nz) | Residual active | HJB adv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline autopilot | 1000 | 0.866 | +/-0.007 | 221.417 | +/-2.751 | 18.944 | 0.732 | 0.0022 | 4.259 | 0.000 | 0.0000 |
| Baseline + tabular Q residual | 1000 | 1.183 | +/-0.007 | 156.781 | +/-2.188 | 35.169 | 0.833 | 0.0004 | 7.145 | 0.924 | 0.0000 |
| Baseline + SHARQ-HJB residual | 1000 | 1.174 | +/-0.007 | 155.149 | +/-2.213 | 35.303 | 0.822 | 0.0004 | 7.147 | 0.951 | -0.3008 |

## Use

This result set is useful for seed sensitivity and early-phase confidence checks. It is not a full mission-length comparison.

## Files

| File | Use |
| --- | --- |
| `scenario_catalog.json` | Scenario definitions. |
| `comparative/aggregate_by_method.csv` | Method-level aggregate table. |
| `comparative/all_episode_summary.csv` | Per-scenario/per-seed scalar metrics. |
| `<method>/episode_summary.csv` | Method-specific scalar metrics. |
| `<method>/aggregate_metrics.csv` | Method-specific aggregate metrics. |
| `<method>/steps.jsonl` | Step-level time series when non-empty. |
