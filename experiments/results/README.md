# Results

This directory contains the stored benchmark results and the compact result surfaces used by the table and figure scripts.

## Layout

| Path | Contents |
| --- | --- |
| `raw/full-duration/` | Full-duration benchmark outputs: 20 scenarios x 3 controller modes = 60 episodes. |
| `raw/coarse-20x50/` | Short-horizon coarse sweep: 20 scenarios x 50 seeds x 3 controller modes = 3000 episodes. |
| `data/` | Compact CSV/JSON/JSONL copies used by scripts. |
| `tables/` | CSV table outputs. |
| `figures/` | PNG figure outputs only. |

## Controller modes

- `fixed_matlab_autopilot` — fixed/gain-scheduled autopilot only.
- `online_q_learning` — tabular Q residual command supervisor.
- `sharq_hjb` — HJB residual command supervisor.

## Result boundaries

- Full-duration results are the main mission-scale comparison in this repository.
- Coarse 20x50 results are short-horizon seed-sensitivity outputs.
- Fight-mode smoke output is a finite-runtime sanity check.
- Metrics are simulator diagnostics, not real-flight guarantees.
