from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
FIGS = BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.25
})


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

def _save(fig, name: str):
    fig.tight_layout()
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")

def _mean_std_bar(ax, labels, means, stds, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


# Load data
runs = _safe_read_csv(DATA / "runs.csv")
# Normalize column names (lowercase)
runs.columns = [c.strip().lower() for c in runs.columns]

has_variant = "variant" in runs.columns
has_constraints = "n_constraints" in runs.columns

# Optional timeline
timeline = None
tl_path = DATA / "timeline.csv"
if tl_path.exists():
    timeline = pd.read_csv(tl_path)
    timeline.columns = [c.strip().lower() for c in timeline.columns]

# Basic cleaning
for col in ["feasible"]:
    if col in runs.columns:
        runs[col] = runs[col].astype(str).str.lower().isin(["1", "true", "yes"])

for col in ["t_first", "t_total", "penalty_total"]:
    if col in runs.columns:
        runs[col] = pd.to_numeric(runs[col], errors="coerce")

if timeline is not None:
    timeline["t"] = pd.to_numeric(timeline["t"], errors="coerce")
    if "best_penalty" in timeline.columns:
        timeline["best_penalty"] = pd.to_numeric(timeline["best_penalty"], errors="coerce")

# Figure 1: Runtime comparison by solver
def fig_runtime_by_solver():
    if not {"solver", "t_total"}.issubset(runs.columns):
        print("[SKIP] fig_runtime_by_solver: missing columns")
        return
    agg = runs.groupby("solver")["t_total"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    _mean_std_bar(
        ax,
        labels=agg["solver"].tolist(),
        means=agg["mean"].tolist(),
        stds=agg["std"].fillna(0).tolist(),
        title="Runtime Comparison (mean ± std)",
        ylabel="Runtime (seconds)" if agg["mean"].max() < 10 else "Runtime (seconds)"
    )
    _save(fig, "runtime_comparison_real.jpg")

# Figure 2: Penalty comparison by solver
def fig_penalty_by_solver():
    if not {"solver", "penalty_total"}.issubset(runs.columns):
        print("[SKIP] fig_penalty_by_solver: missing columns")
        return
    # Only compare on feasible runs to avoid skew (optional; comment out if you want all)
    df = runs.copy()
    if "feasible" in df.columns:
        df = df[df["feasible"]]
        if df.empty:
            df = runs  # fall back to all
    agg = df.groupby("solver")["penalty_total"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    _mean_std_bar(
        ax,
        labels=agg["solver"].tolist(),
        means=agg["mean"].tolist(),
        stds=agg["std"].fillna(0).tolist(),
        title="Penalty Comparison (mean ± std)",
        ylabel="Penalty (soft constraints)"
    )
    _save(fig, "penalty_comparison_real.jpg")

# Figure 3: Convergence curve for a chosen instance (aggregate over seeds)
def fig_convergence_curve(instance_hint: str | None = None):
    if timeline is None or not {"instance", "solver", "t", "best_penalty"}.issubset(timeline.columns):
        print("[SKIP] fig_convergence_curve: missing timeline.csv or columns")
        return
    # Choose an instance to display; default to the most frequent one
    if instance_hint is None:
        instance_hint = timeline["instance"].value_counts().index[0]
    df = timeline[timeline["instance"] == instance_hint].copy()
    if df.empty:
        print(f"[SKIP] convergence: no rows for instance '{instance_hint}'")
        return

    # Aggregate across seeds: at each time, take median best_penalty per solver
    # To align times, we bin time into 1s or 2s grid
    bin_sz = max(1, int((df["t"].max() - df["t"].min()) / 200))  # ~200 points
    df["t_bin"] = (df["t"] / bin_sz).round().astype(int) * bin_sz

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for solver, g in df.groupby("solver"):
        gg = g.groupby("t_bin")["best_penalty"].median().reset_index()
        ax.plot(gg["t_bin"], gg["best_penalty"], label=str(solver))
    ax.set_title(f"Convergence Curve (instance: {instance_hint})")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Best Penalty (median across seeds)")
    ax.legend()
    _save(fig, "convergence_curve_real.jpg")

# Figure 4: Heuristic comparison (SAT runtime, ILP penalty)
# Requires 'variant' column with values like 'baseline' vs 'heuristic'
def fig_heuristic_effects():
    if not has_variant or not {"solver", "variant"}.issubset(runs.columns):
        print("[SKIP] fig_heuristic_effects: no 'variant' column")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: SAT runtime by variant
    sat = runs[(runs["solver"].str.upper() == "SAT")]
    if not sat.empty and "t_total" in sat.columns:
        sat_agg = sat.groupby("variant")["t_total"].mean().reindex(["baseline", "heuristic"])
        ax1.bar(sat_agg.index.tolist(), sat_agg.fillna(0).tolist())
        ax1.set_title("SAT Runtime by Variant")
        ax1.set_ylabel("Runtime (seconds)")
    else:
        ax1.text(0.5, 0.5, "No SAT data", ha="center", va="center")
        ax1.set_axis_off()

    # Right: ILP penalty by variant
    ilp = runs[(runs["solver"].str.upper() == "ILP")]
    if not ilp.empty and "penalty_total" in ilp.columns:
        ilp_agg = ilp.groupby("variant")["penalty_total"].mean().reindex(["baseline", "heuristic"])
        ax2.bar(ilp_agg.index.tolist(), ilp_agg.fillna(0).tolist())
        ax2.set_title("ILP Penalty by Variant")
        ax2.set_ylabel("Penalty (soft constraints)")
    else:
        ax2.text(0.5, 0.5, "No ILP data", ha="center", va="center")
        ax2.set_axis_off()

    fig.suptitle("Effect of Heuristic Improvements", y=1.02)
    _save(fig, "heuristic_comparison_real.jpg")

# Figure 5: Scalability curve (t_total vs n_constraints; log-log)
def fig_scalability():
    if not has_constraints or not {"solver", "n_constraints", "t_total"}.issubset(runs.columns):
        print("[SKIP] fig_scalability: missing 'n_constraints'")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for solver, g in runs.groupby("solver"):
        gg = g.groupby("n_constraints")["t_total"].median().reset_index()
        gg = gg.sort_values("n_constraints")
        ax.plot(gg["n_constraints"], gg["t_total"], marker="o", label=str(solver))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Scalability: Runtime vs Number of Constraints (median)")
    ax.set_xlabel("Number of Constraints (log)")
    ax.set_ylabel("Runtime (seconds, log)")
    ax.legend()
    _save(fig, "scalability_curve_real.jpg")

# Run all figures

if __name__ == "__main__":
    fig_runtime_by_solver()
    fig_penalty_by_solver()
    fig_convergence_curve(instance_hint=None)  # or pass a specific instance id/name
    fig_heuristic_effects()
    fig_scalability()
    print(f"[DONE] Figures saved under: {FIGS.resolve()}")
