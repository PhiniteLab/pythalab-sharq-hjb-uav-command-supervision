# Baseline + tabular Q residual Results

This directory contains short-horizon coarse sweep scalar metrics and step-level time series for `online_q_learning`.

## Aggregate snapshot

| Metric | Value |
| --- | ---: |
| Episodes | 1000 |
| Mean RMS reference error [m] | 1.183 |
| Mean RMS altitude error [m] | 156.781 |
| Mean RMS airspeed error [m/s] | 35.169 |
| Mean actuator-command activity index | 0.833 |
| Mean safety-threshold fraction | 0.0004 |
| Mean max abs load factor | 7.145 |
| Mean residual active fraction | 0.9239 |
| Mean HJB advantage | 0.0000 |

## Role

- Tabular Q residual comparator using the bounded command-action set.
- Use this mode to separate Q-learning effects from the added HJB residual value and risk-filter layer.
- This is an 8 second early-phase sweep; do not read it as a full mission-length comparison.

## Files

- `episode_summary.csv`: scalar metrics per scenario/seed.
- `aggregate_metrics.csv`: method-level summary metrics.
- `scenario_catalog.json`: scenario definitions copied for local context.
- `steps.jsonl`: step-level time series when non-empty.
