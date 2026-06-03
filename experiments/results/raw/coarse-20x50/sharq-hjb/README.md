# HJB Residual Results

This directory contains short-horizon coarse sweep scalar metrics and step-level time series for `sharq_hjb`.

## Aggregate snapshot

| Metric | Value |
| --- | ---: |
| Episodes | 1000 |
| Mean RMS reference error [m] | 1.174 |
| Mean RMS altitude error [m] | 155.149 |
| Mean RMS airspeed error [m/s] | 35.303 |
| Mean actuator-command activity index | 0.822 |
| Mean safety-threshold fraction | 0.0004 |
| Mean max abs load factor | 7.147 |
| Mean residual active fraction | 0.9515 |
| Mean HJB advantage | -0.3008 |

## Role

- HJB residual mode with value, advantage, candidate, and risk-filter diagnostics.
- Use residual activity, HJB advantage, shield-active fraction, and candidate count for method inspection.
- This is an 8 second early-phase sweep; do not read it as a full mission-length comparison.

## Files

- `episode_summary.csv`: scalar metrics per scenario/seed.
- `aggregate_metrics.csv`: method-level summary metrics.
- `scenario_catalog.json`: scenario definitions copied for local context.
- `steps.jsonl`: step-level time series when non-empty.
