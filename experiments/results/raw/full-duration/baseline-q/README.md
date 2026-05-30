# Baseline + tabular Q residual Results

This directory contains full-duration benchmark scalar metrics and step-level time series for `online_q_learning`.

## Aggregate snapshot

| Metric | Value |
| --- | ---: |
| Episodes | 20 |
| Mean RMS reference error [m] | 88.809 |
| Mean RMS altitude error [m] | 65.819 |
| Mean RMS airspeed error [m/s] | 15.086 |
| Mean actuator-command activity index | 6.524 |
| Mean safety-threshold fraction | 0.0031 |
| Mean max abs load factor | 6.844 |
| Mean residual active fraction | 0.9129 |
| Mean HJB advantage | 0.0000 |

## Role

- Tabular Q residual comparator using the bounded command-action set.
- Use this mode to separate Q-learning effects from the added HJB residual value and risk-filter layer.

## Files

- `episode_summary.csv`: scalar metrics per scenario/seed.
- `aggregate_metrics.csv`: method-level summary metrics.
- `scenario_catalog.json`: scenario definitions copied for local context.
- `steps.jsonl`: step-level time series when non-empty.
