#!/usr/bin/env python3
"""Sync raw benchmark outputs into the compact experiment result data package.

Raw runner outputs live under ``experiments/results/raw``.  The compact CSV/JSONL
copy under ``experiments/results/data`` is consumed by the table and figure scripts.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "experiments" / "results" / "raw" / "full-duration"
COARSE = ROOT / "experiments" / "results" / "raw" / "coarse-20x50"
DATA = ROOT / "experiments" / "results" / "data"

COPIES = [
    (FULL / "comparative" / "aggregate_by_method.csv", DATA / "aggregate_by_method.csv"),
    (FULL / "comparative" / "all_episode_summary.csv", DATA / "all_episode_summary.csv"),
    (FULL / "comparative" / "aggregate_by_method.csv", DATA / "full_duration_aggregate_by_method.csv"),
    (FULL / "comparative" / "all_episode_summary.csv", DATA / "full_duration_all_episode_summary.csv"),
    (FULL / "comparative" / "scenario_catalog.json", DATA / "scenario_catalog.json"),
    (FULL / "comparative" / "scenario_catalog.json", DATA / "full_duration_scenario_catalog.json"),
    (FULL / "comparative" / "fight_mode_60s_smoke.csv", DATA / "fight_mode_60s_smoke.csv"),
    (COARSE / "comparative" / "aggregate_by_method.csv", DATA / "coarse_20x50_aggregate_by_method.csv"),
    (COARSE / "comparative" / "all_episode_summary.csv", DATA / "coarse_20x50_all_episode_summary.csv"),
]

TIMESERIES = [
    (FULL / "baseline" / "steps.jsonl", DATA / "timeseries" / "baseline" / "steps.jsonl"),
    (FULL / "baseline-q" / "steps.jsonl", DATA / "timeseries" / "baseline-q" / "steps.jsonl"),
    (FULL / "sharq-hjb" / "steps.jsonl", DATA / "timeseries" / "sharq-hjb" / "steps.jsonl"),
]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required experiment: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def main() -> None:
    for src, dst in COPIES + TIMESERIES:
        copy_file(src, dst)


if __name__ == "__main__":
    main()
