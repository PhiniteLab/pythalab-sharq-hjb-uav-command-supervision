# Baseline autopilot Results

This directory contains full-duration benchmark scalar metrics and step-level time series for `fixed_matlab_autopilot`.

## Aggregate snapshot

| Metric | Value |
| --- | ---: |
| Episodes | 20 |
| Mean RMS reference error [m] | 338.617 |
| Mean RMS altitude error [m] | 114.137 |
| Mean RMS airspeed error [m/s] | 10.546 |
| Mean actuator-command activity index | 5.665 |
| Mean safety-threshold fraction | 0.0048 |
| Mean max abs load factor | 7.414 |
| Mean residual active fraction | 0.0000 |
| Mean HJB advantage | 0.0000 |

## Role

- Baseline comparator; residual metrics should remain zero.
- Shows the behavior of the fixed/gain-scheduled autopilot on the same scenario set.

## Files

- `episode_summary.csv`: scalar metrics per scenario/seed.
- `aggregate_metrics.csv`: method-level summary metrics.
- `scenario_catalog.json`: scenario definitions copied for local context.
- `steps.jsonl`: step-level time series when non-empty.
