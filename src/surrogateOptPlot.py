"""
Surrogate Optimization Convergence Analysis
Pulls all runs from W&B project and plots convergence of mean_contact_num
and mean_seconds to their 365-day values, to justify 7-day surrogate window.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
ENTITY      = "loc_gsopt"          # set to your W&B username/team, or None to auto-detect
PROJECT     = "SurrogateTestsurrogate_opt"
CACHE_FILE  = "wandb_runs_cache.pkl"   # avoids re-downloading every time
MAX_DAYS    = 100            # x-axis cutoff for plots (no need to show all 365)
SURROGATE_DAY = 7           # the vertical line we want to justify
EPSILON     = 0.05          # 2% tolerance for convergence threshold plot

# ─────────────────────────────────────────────
# 1. PULL DATA FROM W&B (with caching)
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 1. PULL DATA FROM W&B (with caching)
# ─────────────────────────────────────────────
 
def pull_runs(entity, project, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file} ...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
 
    print("Fetching runs from W&B ...")
    api = wandb.Api(timeout=60)
    proj_path = f"{entity}/{project}" if entity else project
    runs = api.runs(proj_path)
 
    all_data = {}
    for run in runs:
        try:
            history = run.history(
                keys=["num_days", "total_contact_num", "total_seconds"],
                pandas=True
            )
            if history.empty:
                continue
            history = history.sort_values("num_days").dropna(subset=["num_days"])
            all_data[run.id] = {
                "num_days":          history["num_days"].values,
                "total_contact_num": history["total_contact_num"].values if "total_contact_num" in history else None,
                "total_seconds":     history["total_seconds"].values     if "total_seconds"     in history else None,
            }
        except Exception as e:
            print(f"  Skipping run {run.id}: {e}")
            continue
 
    print(f"Downloaded {len(all_data)} valid runs.")
    with open(cache_file, "wb") as f:
        pickle.dump(all_data, f)
    return all_data
 
 
# ─────────────────────────────────────────────
# 2. BUILD CUMULATIVE MEAN CURVES
# ─────────────────────────────────────────────
 
def build_cumulative_mean_curves(all_data, total_key, max_days=30):
    """
    For each run:
      1. Interpolate cumulative total onto integer days 1..365
      2. Compute per-day values via finite difference
      3. Compute running mean up to each day d: mean(daily[0:d])
      4. Normalize by the run's 365-day mean (ground truth reference)
 
    Returns:
      curves   : (n_runs, max_days) — normalized running mean at each day
      diff_days: (max_days,) — day indices 1..max_days
    """
    curves    = []
    # We need all 365 days to compute the reference mean
    full_days_axis = np.arange(0, 366)
    diff_days      = np.arange(1, max_days + 1)
 
    for run_id, d in all_data.items():
        totals = d.get(total_key)
        days   = d.get("num_days")
 
        if totals is None or days is None or len(days) < 30:
            continue
 
        # Need runs that go close to 365 days for a reliable reference
        if np.max(days) < 100:
            continue
 
        # Interpolate cumulative total onto full integer day grid
        interp_totals = np.interp(full_days_axis, days, totals)
 
        # Per-day values via finite difference
        daily = np.diff(interp_totals)   # shape: (365,), daily[0] = day 1, etc.
 
        # 365-day mean = ground truth reference
        ref_mean = np.mean(daily)
        if ref_mean <= 0 or np.isnan(ref_mean):
            continue
 
        # Running mean up to each day d, normalized by 365-day mean
        running_mean = np.cumsum(daily[:max_days]) / np.arange(1, max_days + 1)
        normalized   = running_mean / ref_mean
 
        curves.append(normalized)
 
    return np.array(curves), diff_days
 
 
# ─────────────────────────────────────────────
# 3. PLOT
# ─────────────────────────────────────────────
 
def plot_convergence(all_data, max_days=MAX_DAYS, surrogate_day=SURROGATE_DAY, epsilon=EPSILON):
 
    curves_c, days = build_cumulative_mean_curves(all_data, "total_contact_num", max_days)
    curves_s, _    = build_cumulative_mean_curves(all_data, "total_seconds",     max_days)
 
    print(f"Runs used for contacts plot : {len(curves_c)}")
    print(f"Runs used for seconds plot  : {len(curves_s)}")
 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # fig.suptitle(
    #     f"Surrogate Optimization Convergence Analysis\n"
    #     f"(n = {len(curves_c)} configurations, varying altitude, inclination, and ground station)",
    #     fontsize=13, y=1.02
    # )
 
    def shade_percentiles(ax, days, curves, color):
        med = np.median(curves, axis=0)
        p25 = np.percentile(curves, 25, axis=0)
        p75 = np.percentile(curves, 75, axis=0)
        p5  = np.percentile(curves,  5, axis=0)
        p95 = np.percentile(curves, 95, axis=0)
        ax.fill_between(days, p5,  p95, alpha=0.30, color=color, label="5–95th pct")
        # ax.fill_between(days, p25, p75, alpha=0.30, color=color, label="IQR (25–75th)")
        # ax.plot(days, med, color=color, lw=2, label="Median")
 
    def style_ax(ax, title, surrogate_day, ylim=None):
        ax.axvline(surrogate_day, color="red", ls="--", lw=1.5, label=f"Day {surrogate_day}")
        ax.axhline(1.0, color="black", ls=":", lw=1, alpha=0.6, label="365-day mean (reference)")
        ax.axhspan(1 - epsilon, 1 + epsilon, color="gray", alpha=0.3, label=f"±{int(epsilon*100)}% band")
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Simulation Duration (days)", fontsize=14)
        ax.set_ylabel("Running mean / 365-day mean", fontsize=14)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend(fontsize=13)
        ax.set_xlim(1, max_days)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
 
    # --- Panel 1: Cumulative mean contacts, normalized to 365-day mean ---
    ax = axes[0]
    if len(curves_c) > 0:
        shade_percentiles(ax, days, curves_c, color="#1f77b4")
        style_ax(ax,
                 title="Running Mean Contacts/Day\n(normalized to 365-day mean)",
                 surrogate_day=surrogate_day,
                 ylim=(0.5, 1.8))
        # ax.text(0.97, 0.05, f"n = {len(curves_c)} runs",
        #         transform=ax.transAxes, ha="right", va="bottom", fontsize=16, color="gray")
 
    # --- Panel 2: Cumulative mean seconds, normalized to 365-day mean ---
    ax = axes[1]
    if len(curves_s) > 0:
        shade_percentiles(ax, days, curves_s, color="#2ca02c")
        style_ax(ax,
                 title="Running Mean Contact Duration/Day\n(normalized to 365-day mean)",
                 surrogate_day=surrogate_day,
                 ylim=(0.5, 1.8))
 
    # --- Panel 3: % of runs whose running mean is within epsilon of 365-day mean ---
    ax = axes[2]
    for label, color, crv in [("Contacts", "#1f77b4", curves_c), ("Duration", "#2ca02c", curves_s)]:
        if len(crv) == 0:
            continue
        # At each day d, what % of runs have their running mean within epsilon of 1.0?
        pct = np.mean(np.abs(crv - 1.0) <= epsilon, axis=0) * 100
        ax.plot(days, pct, color=color, lw=2, label=label)
 
    ax.axvline(surrogate_day, color="red", ls="--", lw=1.5, label=f"Day {surrogate_day}")
    ax.axhline(95, color="black", ls=":", lw=1, alpha=0.5, label="95% threshold")
    ax.set_title(
        f"% of Configurations Whose Running Mean\nIs Within {int(epsilon*100)}% of 365-day Mean",
        fontsize=16
    )
    ax.set_xlabel("Simulation Duration (days)", fontsize=14)
    ax.set_ylabel("% of configurations converged", fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=13)
    ax.set_xlim(1, max_days)
    ax.set_ylim(0, 105)
    # ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    out_path = "figures_final/surrogate_convergence.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    all_data = pull_runs(ENTITY, PROJECT, CACHE_FILE)
    print(f"Total valid runs: {len(all_data)}")
    plot_convergence(all_data)