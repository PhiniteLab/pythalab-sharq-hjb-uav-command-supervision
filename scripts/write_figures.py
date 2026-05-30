#!/usr/bin/env python3
"""Write PNG figures from packaged CSV/JSON experiment result data.

The script does not run simulations or modify raw result files. It reads
``experiments/results/data`` and writes PNG files under ``experiments/results/figures``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "results" / "data"
FIG = ROOT / "experiments" / "results" / "figures"

METHOD_ORDER = ["fixed_matlab_autopilot", "online_q_learning", "sharq_hjb"]
METHOD_LABEL = {
    "fixed_matlab_autopilot": "Baseline",
    "online_q_learning": "Q residual",
    "sharq_hjb": "SHARQ-HJB",
}
METHOD_COLOR = {
    "fixed_matlab_autopilot": "#4C78A8",
    "online_q_learning": "#F58518",
    "sharq_hjb": "#54A24B",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 360,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.6,
    "lines.linewidth": 2.6,
    "patch.linewidth": 1.4,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.32,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save(fig: plt.Figure, name: str) -> None:
    """Save one PNG file for each figure."""
    FIG.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def polish(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.4, length=5)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.1f}", dy: float = 0.02, color: str = "#222222") -> None:
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy * span,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=color,
        )


def box(ax: plt.Axes, xy, w, h, text, fc="#F7F9FC", ec="#2F3B52", fontsize=13):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")
    return p


def arrow(ax: plt.Axes, start, end, color="#2F3B52"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17, linewidth=1.9, color=color))


def fig01_framework():
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(r"Autopilot-preserving residual command supervision", pad=14)

    y = 0.54
    w, h = 0.135, 0.20
    xs = [0.03, 0.21, 0.39, 0.57, 0.75]
    labels = [
        "Mission\nreference\n$r_k=[V_{a,c},h_c,\\psi_c]^\\top$",
        "Finite residual\nsupervisor\n$\\Delta r(a_k)$",
        "Projection\n$\\tilde r_k=\\Pi_{\\mathcal{C}}(r_k+\\Delta r)$",
        "Classical\nautopilot\n$\\kappa_{\\rm AP}$",
        "Actuators +\n12-state plant\n$F_{\\Delta t}$",
    ]
    colors = ["#EAF2FF", "#FFF2E3", "#EAF7EA", "#F2ECFF", "#F7F7F7"]
    for x, lab, c in zip(xs, labels, colors):
        box(ax, (x, y), w, h, lab, fc=c, fontsize=12)
    for x in xs[:-1]:
        arrow(ax, (x + w, y + h/2), (x + 0.18, y + h/2))

    box(ax, (0.54, 0.20), 0.18, 0.15, "Energy-allocation\nhelper\n$\\mathcal{E}_{\\mathrm{eng}}$", fc="#FFF9DB", fontsize=12)
    arrow(ax, (0.63, y), (0.63, 0.35), color="#8A6D00")
    arrow(ax, (0.72, 0.275), (0.75, y+0.02), color="#8A6D00")
    ax.text(0.50, 0.12, r"Learned authority is limited to command residuals; actuator-facing changes outside the autopilot are deterministic and disclosed.",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#333333")
    ax.text(0.88, 0.83, r"Telemetry: reward, $Q$, HJB value, shield diagnostics", ha="center", fontsize=12, fontweight="bold")
    arrow(ax, (0.88, y+h), (0.29, y+h+0.04), color="#5B6C8A")
    save(fig, "fig01_framework_architecture.png")


def fig02_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(r"SHARQ-HJB finite-candidate evaluation pipeline", pad=14)
    xs = [0.025, 0.19, 0.355, 0.52, 0.685, 0.85]
    w, h, y = 0.125, 0.21, 0.54
    labels = [
        "Encode\n$s_k,\\ z_k$",
        "Generate\n$\\mathcal{A}_{\\mathrm{valid}}$",
        "Predict\n$\\hat z_{k+1}^a$",
        "Score\n$A_H(z_k,a)$",
        "Filter\nCLF/CBF-style",
        "Select\n$a_k^H$ + no-op",
    ]
    cols = ["#EAF2FF", "#FFF2E3", "#EAF7EA", "#F2ECFF", "#FFE8E8", "#E8FAF7"]
    for x, lab, c in zip(xs, labels, cols):
        box(ax, (x, y), w, h, lab, fc=c, fontsize=12)
    for i in range(len(xs)-1):
        arrow(ax, (xs[i]+w, y+h/2), (xs[i+1], y+h/2))
    box(ax, (0.32, 0.18), 0.36, 0.15, "Finite shield: keep $a_0$; reject hard-blocked candidates;\nrank survivors by $Q+[-A_H]-\\rho_{\\rm risk}$", fc="#F7F7F7", fontsize=12)
    arrow(ax, (0.745, y), (0.62, 0.33), color="#555555")
    ax.text(0.50, 0.09, r"The pipeline ranks seven bounded command residuals; it is not a continuous CBF-QP or full HJB PDE solver.",
            ha="center", fontsize=13, fontweight="bold")
    save(fig, "fig02_candidate_pipeline.png")


def fig03_reference_rms(agg: pd.DataFrame):
    df = agg.set_index("controller_mode").loc[METHOD_ORDER]
    vals = df["mean_rms_reference_error_m"].to_numpy()
    ci = df["ci95_rms_reference_error_m"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, yerr=ci, capsize=8, color=[METHOD_COLOR[m] for m in METHOD_ORDER], edgecolor="#222222")
    ax.set_xticks(x, [METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.set_ylabel(r"RMS spatial reference/path error [m]")
    ax.set_title(r"Full-duration spatial reference/path tracking error")
    polish(ax)
    upper = max(vals + ci) * 1.34
    ax.set_ylim(0, upper)
    span = upper
    for b, val, err in zip(bars, vals, ci):
        ax.text(
            b.get_x() + b.get_width() / 2,
            val + err + 0.035 * span,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#DDDDDD", alpha=0.92),
        )
    save(fig, "fig03_full_reference_rms.png")


def fig04_normalized(agg: pd.DataFrame):
    metrics = [
        ("Ref. RMS", "mean_rms_reference_error_m", False),
        ("Alt. RMS", "mean_rms_altitude_error_m", False),
        ("Airspeed RMS", "mean_rms_airspeed_error_mps", False),
        ("Ctrl. idx.", "mean_control_energy_integral", False),
        ("Viol. frac.", "mean_safety_time_fraction", False),
        ("Max |$n_z$|", "mean_max_abs_load_factor_nz", False),
    ]
    df = agg.set_index("controller_mode").loc[METHOD_ORDER]
    base = df.loc["fixed_matlab_autopilot"]
    data = []
    for _, col, _ in metrics:
        b = base[col]
        data.append([(df.loc[m, col] / b) if b != 0 else np.nan for m in METHOD_ORDER])
    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(9.8, 6.6), constrained_layout=True)
    im = ax.imshow(arr, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(1.65, np.nanmax(arr))))
    ax.set_xticks(range(len(METHOD_ORDER)), [METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.set_yticks(range(len(metrics)), [m[0] for m in metrics])
    ax.set_title(r"Metrics normalized to baseline ($<1$ is lower/better)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = "--" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12, fontweight="bold", color="#111111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(r"Ratio to baseline", fontweight="bold")
    ax.tick_params(width=1.4, length=0)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", rotation_mode="anchor")
    ax.set_xlabel(r"Controller package", labelpad=12)
    ax.set_ylabel(r"Metric", labelpad=12)
    save(fig, "fig04_normalized_metrics.png")


def fig05_tradeoff(agg: pd.DataFrame):
    df = agg.set_index("controller_mode").loc[METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    energy = df["mean_control_energy_integral"]
    sizes = 650 * (energy / energy.max()) ** 2
    for m in METHOD_ORDER:
        x = df.loc[m, "mean_rms_airspeed_error_mps"]
        y = df.loc[m, "mean_rms_reference_error_m"]
        ax.scatter(x, y, s=float(sizes.loc[m]), color=METHOD_COLOR[m], edgecolor="#222222", linewidth=1.5, alpha=0.90, label=METHOD_LABEL[m])
        ax.annotate(METHOD_LABEL[m], (x, y), xytext=(8, 8), textcoords="offset points", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel(r"RMS airspeed error [m s$^{-1}$]")
    ax.set_ylabel(r"RMS reference error [m], log scale")
    ax.set_title(r"Tracking--airspeed--energy trade-off")
    polish(ax, "both")
    ax.legend(title=r"Bubble area $\propto$ energy", frameon=True)
    save(fig, "fig05_reference_airspeed_tradeoff.png")


def winner_counts(ep: pd.DataFrame):
    metrics = [
        ("Ref.", "rms_reference_error_m", "min"),
        ("Alt.", "rms_altitude_error_m", "min"),
        ("Airspeed", "rms_airspeed_error_mps", "min"),
        ("Energy", "control_energy_integral", "min"),
        ("Viol.", "safety_time_fraction", "min"),
        ("Load", "max_abs_load_factor_nz", "min"),
    ]
    scenarios = sorted(ep["scenario_name"].unique())
    counts = pd.DataFrame(0, index=[m[0] for m in metrics], columns=[METHOD_LABEL[m] for m in METHOD_ORDER] + ["Tie"])
    for name, col, direction in metrics:
        for sc in scenarios:
            g = ep[ep["scenario_name"] == sc].set_index("controller_mode").loc[METHOD_ORDER]
            vals = g[col]
            best = vals.min() if direction == "min" else vals.max()
            tol = max(1e-9, abs(best) * 1e-6)
            winners = [m for m in METHOD_ORDER if abs(vals.loc[m] - best) <= tol]
            if len(winners) == 1:
                counts.loc[name, METHOD_LABEL[winners[0]]] += 1
            else:
                counts.loc[name, "Tie"] += 1
    return counts


def fig06_winners(ep: pd.DataFrame):
    counts = winner_counts(ep)
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    x = np.arange(len(counts))
    width = 0.20
    series = [METHOD_LABEL[m] for m in METHOD_ORDER] + ["Tie"]
    colors = [METHOD_COLOR[m] for m in METHOD_ORDER] + ["#B8B8B8"]
    for k, (s, c) in enumerate(zip(series, colors)):
        ax.bar(x + (k - 1.5) * width, counts[s].to_numpy(), width, label=s, color=c, edgecolor="#222222")
    ax.set_xticks(x, counts.index)
    ax.set_ylabel("Scenario count")
    ax.set_ylim(0, 21)
    ax.set_title("Per-scenario winners by metric")
    polish(ax)
    ax.legend(ncols=4, frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save(fig, "fig06_winner_counts.png")


def fig07_diagnostics(agg: pd.DataFrame):
    df = agg.set_index("controller_mode").loc[METHOD_ORDER[1:]]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    labels = [METHOD_LABEL[m] for m in METHOD_ORDER[1:]]
    cols = [METHOD_COLOR[m] for m in METHOD_ORDER[1:]]

    panels = [
        (axes[0], ["mean_residual_active_fraction", "mean_shield_active_fraction"], ["Residual active", "Shield active"], r"Fraction of time"),
        (axes[1], ["mean_mean_hard_condition_score", "mean_mean_candidate_count"], [r"Hard score $\chi$", "Candidates"], r"Mean value"),
        (axes[2], ["mean_mean_hjb_value", "mean_mean_hjb_advantage", "mean_mean_hjb_stage_cost"], [r"$\widehat V_H$", r"$A_H$", r"$\ell_H$"], r"HJB diagnostic"),
    ]
    for ax, metrics, names, ylabel in panels:
        x = np.arange(len(metrics))
        width = 0.34
        for i, m in enumerate(METHOD_ORDER[1:]):
            vals = [df.loc[m, col] for col in metrics]
            bars = ax.bar(x + (i - 0.5)*width, vals, width, color=cols[i], edgecolor="#222222", label=labels[i])
            for b in bars:
                h = b.get_height()
                va = "bottom" if h >= 0 else "top"
                ax.text(b.get_x()+b.get_width()/2, h + (0.03 if h>=0 else -0.03)*(ax.get_ylim()[1]-ax.get_ylim()[0] if ax.get_ylim()[1]!=ax.get_ylim()[0] else 1), f"{h:.2f}", ha="center", va=va, fontsize=10, fontweight="bold")
        ax.set_xticks(x, names)
        ax.set_ylabel(ylabel)
        polish(ax)
    axes[1].legend(frameon=True, loc="upper right")
    fig.suptitle(r"Residual-supervisor diagnostics", fontsize=18, fontweight="bold", y=1.02)
    save(fig, "fig07_residual_diagnostics.png")


def fig08_coarse(coarse: pd.DataFrame):
    df = coarse.set_index("controller_mode").loc[METHOD_ORDER]
    metrics = [("Ref.", "mean_rms_reference_error_m"), ("Alt.", "mean_rms_altitude_error_m"), ("Airspeed", "mean_rms_airspeed_error_mps"), ("Energy", "mean_control_energy_integral"), ("Viol.", "mean_safety_time_fraction")]
    base = df.loc["fixed_matlab_autopilot"]
    arr = np.array([[df.loc[m, col] / base[col] if base[col] else np.nan for colname, col in metrics] for m in METHOD_ORDER])
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    x = np.arange(len(metrics)); width = 0.24
    for i, m in enumerate(METHOD_ORDER):
        vals = arr[i]
        bars = ax.bar(x + (i-1)*width, vals, width, color=METHOD_COLOR[m], label=METHOD_LABEL[m], edgecolor="#222222")
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x()+b.get_width()/2, v+0.04, f"{v:.2f}", ha="center", fontsize=9.5, fontweight="bold", rotation=90 if v>3 else 0)
    ax.axhline(1, color="#111111", linewidth=1.4, linestyle="--")
    ax.set_xticks(x, [m[0] for m in metrics])
    ax.set_ylabel(r"Ratio to baseline")
    ax.set_title(r"Coarse $20\times50$ sweep: normalized metrics")
    ax.set_ylim(0, np.nanmax(arr)*1.22)
    polish(ax)
    ax.legend(ncols=3, frameon=True)
    save(fig, "fig08_coarse_sweep.png")


def fig09_fight(fight: pd.DataFrame):
    df = fight.set_index("controller_mode").loc[METHOD_ORDER]
    metrics = [("Ref.", "rms_reference_error_m"), ("Alt.", "rms_altitude_error_m"), ("Airspeed", "rms_airspeed_error_mps"), ("Energy", "control_energy_integral"), ("Viol.", "safety_time_fraction")]
    base = df.loc["fixed_matlab_autopilot"]
    arr = np.array([[df.loc[m, col] / base[col] if base[col] else np.nan for _, col in metrics] for m in METHOD_ORDER])
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x=np.arange(len(metrics)); width=.24
    for i,m in enumerate(METHOD_ORDER):
        bars=ax.bar(x+(i-1)*width, arr[i], width, color=METHOD_COLOR[m], label=METHOD_LABEL[m], edgecolor="#222222")
        for b,v in zip(bars,arr[i]):
            if np.isfinite(v):
                ax.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1,color="#111111",linewidth=1.4,linestyle="--")
    ax.set_xticks(x,[m[0] for m in metrics])
    ax.set_ylabel(r"Ratio to baseline")
    ax.set_title(r"Fight-mode 60 s smoke: normalized metrics")
    ax.set_ylim(0, np.nanmax(arr)*1.22)
    polish(ax)
    ax.legend(ncols=3, frameon=True)
    save(fig, "fig09_fight_smoke.png")


def short_scenario(name: str, i: int) -> str:
    words = name.replace("_", " ").replace("crosswind", "xwind").replace("turbulence", "turb.").replace("altitude", "alt.")
    parts = words.split()
    label = " ".join(parts[:5])
    return f"S{i:02d}  {label}"


def fig10_heatmap(ep: pd.DataFrame):
    scenarios = list(ep.drop_duplicates("scenario_name").sort_values(["profile", "scenario_name"])["scenario_name"])
    arr = np.zeros((len(scenarios), len(METHOD_ORDER)))
    for i, sc in enumerate(scenarios):
        g = ep[ep["scenario_name"] == sc].set_index("controller_mode")
        arr[i] = [g.loc[m, "rms_reference_error_m"] for m in METHOD_ORDER]
    vmin = max(np.nanmin(arr[arr>0])*0.8, 0.1)
    vmax = np.nanmax(arr)*1.05
    fig, ax = plt.subplots(figsize=(8.9, 12.4), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xticks(range(len(METHOD_ORDER)), [METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.set_yticks(range(len(scenarios)), [short_scenario(sc, i+1) for i, sc in enumerate(scenarios)])
    ax.set_title(r"Per-scenario RMS spatial reference/path error [m]", pad=14)
    for i in range(arr.shape[0]):
        row_min = np.nanmin(arr[i])
        for j in range(arr.shape[1]):
            val = arr[i, j]
            color = "white" if val > math.sqrt(vmin*vmax) else "#111111"
            txt = f"{val:.1f}" if val >= 10 else f"{val:.2f}"
            weight = "black" if abs(val-row_min) <= max(1e-9, row_min*1e-6) else "bold"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=11.0, fontweight=weight,
                    bbox=dict(boxstyle="round,pad=0.20", facecolor=(1,1,1,0.26), edgecolor="none"))
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(r"RMS spatial reference/path error [m]", fontweight="bold")
    ax.tick_params(width=1.4, length=0)
    for lab in ax.get_xticklabels()+ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("Controller package")
    save(fig, "fig10_per_scenario_heatmap.png")


def load_timeseries() -> dict[str, pd.DataFrame]:
    out = {}
    for sub, method in [("baseline", "fixed_matlab_autopilot"), ("baseline-q", "online_q_learning"), ("sharq-hjb", "sharq_hjb")]:
        rows = []
        path = DATA / "timeseries" / sub / "steps.jsonl"
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        out[method] = pd.DataFrame(rows)
    return out


def fig11_timeseries(ts: dict[str, pd.DataFrame]):
    scenario = "orbit_crosswind_turbulence_low_alt"
    seed = 4101
    panels = [
        ("reference_error_m", r"Reference error [m]", "symlog"),
        ("altitude_error_m", r"Altitude error [m]", "linear"),
        ("airspeed_error_mps", r"Airspeed error [m s$^{-1}$]", "linear"),
        ("load_factor_nz", r"Load factor $n_z$", "linear"),
        ("residual_active", r"Residual / shield", "binary"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.8, 10.8), sharex=True, gridspec_kw={"height_ratios": [1.1,1,1,1,0.72]})
    for ax, (col, ylabel, scale) in zip(axes, panels):
        for m in METHOD_ORDER:
            df = ts[m]
            g = df[(df["scenario_name"] == scenario) & (df["seed"] == seed)].copy().reset_index(drop=True)
            if g.empty: continue
            t = g["t_s"].astype(float).to_numpy()
            if scale == "binary":
                if m == "fixed_matlab_autopilot":
                    continue
                y = g["residual_active"].astype(float).rolling(50, min_periods=1).mean()
                ax.plot(t, y, color=METHOD_COLOR[m], label=METHOD_LABEL[m] + " activity", linewidth=2.0)
                if "shield_active" in g:
                    ys = g["shield_active"].astype(float).rolling(50, min_periods=1).mean()
                    ax.plot(t, ys, color="#6F4ACB", linestyle="--", label=METHOD_LABEL[m] + " shield", linewidth=2.0)
            else:
                y = g[col].astype(float)
                # Reference error spans orders of magnitude; plot absolute value on symlog for readability.
                if col in {"reference_error_m", "altitude_error_m", "airspeed_error_mps"}:
                    y = y.abs()
                ax.plot(t, y, color=METHOD_COLOR[m], label=METHOD_LABEL[m])
        ax.set_ylabel(ylabel)
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=1.0)
        if col == "load_factor_nz":
            ax.axhline(0, color="#333333", linewidth=1.0)
            ax.axhline(8, color="#B00020", linestyle=":", linewidth=1.5)
        polish(ax, "y")
    axes[0].legend(ncols=3, frameon=True, loc="upper right")
    axes[-1].legend(ncols=2, frameon=True, loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Representative low-altitude crosswind/turbulence orbit", fontsize=18, fontweight="bold", y=0.995)
    save(fig, "fig11_representative_timeseries.png")


def fig12_profile(ep: pd.DataFrame):
    g = ep.groupby(["profile", "controller_mode"], as_index=False)["rms_reference_error_m"].mean()
    profiles = list(g.drop_duplicates("profile").sort_values("profile")["profile"])
    y = np.arange(len(profiles))
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    height = 0.22
    for i, m in enumerate(METHOD_ORDER):
        vals = []
        for p in profiles:
            val = g[(g.profile == p) & (g.controller_mode == m)]["rms_reference_error_m"]
            vals.append(float(val.iloc[0]) if not val.empty else np.nan)
        bars = ax.barh(y + (i-1)*height, vals, height, color=METHOD_COLOR[m], edgecolor="#222222", label=METHOD_LABEL[m])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(v*1.05, b.get_y()+b.get_height()/2, f"{v:.1f}", va="center", fontsize=10.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yticks(y, [p.replace("_", " ") for p in profiles])
    ax.set_xlabel(r"Mean RMS reference error [m], log scale")
    ax.set_title(r"Reference tracking by mission profile")
    polish(ax, "x")
    ax.legend(ncols=3, frameon=True, loc="lower right")
    save(fig, "fig12_profile_reference_summary.png")


def main() -> None:
    agg = pd.read_csv(DATA / "aggregate_by_method.csv")
    ep = pd.read_csv(DATA / "all_episode_summary.csv")
    coarse = pd.read_csv(DATA / "coarse_20x50_aggregate_by_method.csv")
    fight = pd.read_csv(DATA / "fight_mode_60s_smoke.csv")

    fig01_framework()
    fig02_pipeline()
    fig03_reference_rms(agg)
    fig04_normalized(agg)
    fig05_tradeoff(agg)
    fig06_winners(ep)
    fig07_diagnostics(agg)
    fig08_coarse(coarse)
    fig09_fight(fight)
    fig10_heatmap(ep)
    fig11_timeseries(load_timeseries())
    fig12_profile(ep)
    print("Wrote 12 PNG experiment figures in", FIG)


if __name__ == "__main__":
    main()
