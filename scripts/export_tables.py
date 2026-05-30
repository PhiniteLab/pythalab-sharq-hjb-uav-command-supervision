#!/usr/bin/env python3
"""Export compact CSV tables from the stored experiment result data.

The script reads ``experiments/results/data`` and writes method-comparison tables
under ``experiments/results/tables``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "results" / "data"
OUT = ROOT / "experiments" / "results" / "tables"

METHOD_LABEL = {
    "fixed_matlab_autopilot": "Baseline",
    "online_q_learning": "Q residual",
    "sharq_hjb": "SHARQ-HJB",
}
METHOD_ORDER = ["fixed_matlab_autopilot", "online_q_learning", "sharq_hjb"]


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    if "controller_mode" not in df.columns:
        return df
    out = df.copy()
    out["method_label"] = out["controller_mode"].map(METHOD_LABEL).fillna(out["controller_mode"])
    out["_order"] = out["controller_mode"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    return out.sort_values("_order").drop(columns=["_order"])


def write_csv(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"wrote {OUT / name}")


def winner_counts(ep: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rms_reference_error_m",
        "rms_altitude_error_m",
        "rms_airspeed_error_mps",
        "control_energy_integral",
        "safety_time_fraction",
        "max_abs_load_factor_nz",
    ]
    rows = []
    for metric in metrics:
        for _, group in ep.groupby("scenario_name"):
            best = group.loc[group[metric].astype(float).idxmin(), "controller_mode"]
            rows.append({"metric": metric, "controller_mode": best})
    out = pd.DataFrame(rows).value_counts(["metric", "controller_mode"]).reset_index(name="wins")
    out["method_label"] = out["controller_mode"].map(METHOD_LABEL).fillna(out["controller_mode"])
    return out.sort_values(["metric", "controller_mode"])


def profile_summary(ep: pd.DataFrame) -> pd.DataFrame:
    out = ep.groupby(["profile", "controller_mode"], as_index=False)["rms_reference_error_m"].mean()
    out = ordered(out)
    return out.rename(columns={"rms_reference_error_m": "mean_rms_reference_error_m"})


def main() -> None:
    full = ordered(pd.read_csv(DATA / "aggregate_by_method.csv"))
    coarse = ordered(pd.read_csv(DATA / "coarse_20x50_aggregate_by_method.csv"))
    fight = ordered(pd.read_csv(DATA / "fight_mode_60s_smoke.csv"))
    ep = pd.read_csv(DATA / "all_episode_summary.csv")

    write_csv(full, "full_duration_aggregate.csv")
    write_csv(coarse, "coarse_20x50_aggregate.csv")
    write_csv(fight, "fight_mode_60s_smoke.csv")
    write_csv(winner_counts(ep), "full_duration_winner_counts.csv")
    write_csv(profile_summary(ep), "full_duration_profile_reference_summary.csv")


if __name__ == "__main__":
    main()
