# HJB Residual Results

This directory contains full-duration benchmark scalar metrics and step-level time series for `sharq_hjb`.

## Aggregate snapshot

| Metric | Value |
| --- | ---: |
| Episodes | 20 |
| Mean RMS reference error [m] | 44.809 |
| Mean RMS altitude error [m] | 64.710 |
| Mean RMS airspeed error [m/s] | 15.191 |
| Mean actuator-command activity index | 6.346 |
| Mean safety-threshold fraction | 0.0033 |
| Mean max abs load factor | 6.800 |
| Mean residual active fraction | 0.8882 |
| Mean HJB advantage | -0.3263 |

## Role

- HJB residual mode with value, advantage, candidate, and risk-filter diagnostics.
- Use residual activity, HJB advantage, shield-active fraction, and candidate count for method inspection.

## Files

- `episode_summary.csv`: scalar metrics per scenario/seed.
- `aggregate_metrics.csv`: method-level summary metrics.
- `scenario_catalog.json`: scenario definitions copied for local context.
- `steps.jsonl`: step-level time series when non-empty.
