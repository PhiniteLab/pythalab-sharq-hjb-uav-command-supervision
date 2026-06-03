# Baseline autopilot Results

This directory contains short-horizon coarse sweep scalar metrics and step-level time series for `fixed_matlab_autopilot`.

## Aggregate snapshot

| Metric | Value |
| --- | ---: |
| Episodes | 1000 |
| Mean RMS reference error [m] | 0.866 |
| Mean RMS altitude error [m] | 221.417 |
| Mean RMS airspeed error [m/s] | 18.944 |
| Mean actuator-command activity index | 0.732 |
| Mean safety-threshold fraction | 0.0022 |
| Mean max abs load factor | 4.259 |
| Mean residual active fraction | 0.0000 |
| Mean HJB advantage | 0.0000 |

## Role

- Baseline comparator; residual metrics should remain zero.
- Shows the behavior of the fixed/gain-scheduled autopilot on the same scenario set.
- This is an 8 second early-phase sweep; do not read it as a full mission-length comparison.

## Files

- `episode_summary.csv`: scalar metrics per scenario/seed.
- `aggregate_metrics.csv`: method-level summary metrics.
- `scenario_catalog.json`: scenario definitions copied for local context.
- `steps.jsonl`: step-level time series when non-empty.
