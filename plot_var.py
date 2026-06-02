"""
Visualizations for the VaR project.

Produces three plots:
    1. var_comparison.png    — VaR and ES across three methods (bar chart)
    2. exception_clusters.png — cumulative P&L with VaR exceptions overlaid
    3. stress_waterfall.png  — stress test losses vs daily VaR (waterfall)

Run with: python3 plot_var.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec


# ----- Load everything -----
asset_returns = pd.read_parquet("data/asset_returns.parquet")
portfolio_returns = pd.read_parquet("data/portfolio_returns.parquet")["portfolio_return_pct"]
var_summary = pd.read_parquet("data/var_summary.parquet")
exception_dates = pd.read_parquet("data/exception_dates.parquet")
stress_results = pd.read_parquet("data/stress_results.parquet")

# Colors used consistently across plots
COLOR_HIST = "#C44E52"       # red — historical
COLOR_PARM = "#4C72B0"       # blue — variance-covariance
COLOR_MC   = "#8172B2"       # purple — monte carlo
COLOR_STRESS = "#DD8452"     # orange — stress

# =============================================================================
# Plot 1: VaR and ES comparison across three methods
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

methods = ["Historical\nSimulation", "Variance-\nCovariance", "Monte Carlo\n(Gaussian)"]
var_values = var_summary["var_dollars"].values
es_values = var_summary["es_dollars"].values

x = np.arange(len(methods))
width = 0.38

bars_var = ax.bar(x - width/2, var_values, width, label="VaR (99%)",
                  color=[COLOR_HIST, COLOR_PARM, COLOR_MC], alpha=0.85,
                  edgecolor="black", linewidth=0.8)
bars_es = ax.bar(x + width/2, es_values, width, label="ES (99%)",
                 color=[COLOR_HIST, COLOR_PARM, COLOR_MC], alpha=0.45,
                 edgecolor="black", linewidth=0.8, hatch="///")

# Annotate values on each bar
for bar, val in zip(bars_var, var_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 30,
            f"${val:,.0f}", ha="center", fontsize=10, fontweight="bold")
for bar, val in zip(bars_es, es_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 30,
            f"${val:,.0f}", ha="center", fontsize=10, fontweight="bold")

# Annotate ES/VaR ratios above each pair
for i, ratio in enumerate(var_summary["es_var_ratio"].values):
    ax.text(i, max(var_values[i], es_values[i]) + 250,
            f"ES/VaR = {ratio:.2f}", ha="center", fontsize=9,
            style="italic", color="dimgray")

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel("Loss ($)", fontsize=12)
ax.set_title("1-Day 99% VaR & Expected Shortfall — Three Methods\n"
             "$100K portfolio, 500-day estimation window",
             fontsize=13)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.set_ylim(0, max(es_values) * 1.25)

# Annotation: the fat-tail gap
hist_var = var_summary.loc[var_summary["method"] == "Historical Simulation", "var_dollars"].values[0]
parm_var = var_summary.loc[var_summary["method"] == "Variance-Covariance", "var_dollars"].values[0]
gap_pct = (hist_var - parm_var) / parm_var * 100

ax.text(0.5, 0.02,
        f"Historical VaR exceeds Gaussian VaR by {gap_pct:.0f}% — direct evidence of fat tails the Gaussian model misses.   "
        f"Gaussian ES/VaR ratio of 1.15 matches closed-form theory (φ(z)/α / |z| = 1.145 for 99%).",
        transform=ax.transAxes, fontsize=9.5, va="bottom", ha="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="gray"))

plt.tight_layout()
plt.savefig("var_comparison.png", dpi=150, bbox_inches="tight")
print("Saved var_comparison.png")
plt.close()


# =============================================================================
# Plot 2: Cumulative P&L with exception clusters
# =============================================================================

fig = plt.figure(figsize=(13, 8))
gs = GridSpec(2, 1, height_ratios=[2.5, 1.5], hspace=0.18)

# Top: cumulative P&L
ax1 = fig.add_subplot(gs[0])
cumulative = (1 + portfolio_returns).cumprod()
ax1.plot(cumulative.index, cumulative.values, color="black", linewidth=1.2, label="Cumulative portfolio value")
ax1.axhline(1.0, color="gray", linewidth=0.5, alpha=0.5)

# Mark major crisis windows for context
ax1.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31"),
            alpha=0.15, color="red", label="2008 GFC")
ax1.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"),
            alpha=0.15, color="orange", label="2020 COVID")

ax1.set_ylabel("Cumulative portfolio value ($1 invested)", fontsize=11)
ax1.set_title("Portfolio Cumulative P&L (2005–2026) with VaR Exception Clusters",
              fontsize=13)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)

# Bottom: rugplot of exceptions by method
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Convert binary exception columns to dates where exceptions occurred
for i, (method, color, label) in enumerate([
    ("historical", COLOR_HIST, "Historical Simulation"),
    ("parametric", COLOR_PARM, "Variance-Covariance"),
    ("monte_carlo", COLOR_MC, "Monte Carlo (Gaussian)"),
]):
    exceptions = exception_dates[method].dropna()
    exception_days = exceptions[exceptions == 1].index
    # Plot each exception as a vertical line
    y_pos = 3 - i  # stack the methods on different y-levels
    ax2.vlines(exception_days, y_pos - 0.4, y_pos + 0.4,
               colors=color, linewidth=1.2, alpha=0.85, label=label)

# Highlight crisis windows here too
ax2.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31"),
            alpha=0.10, color="red")
ax2.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"),
            alpha=0.10, color="orange")

ax2.set_yticks([1, 2, 3])
ax2.set_yticklabels(["Monte Carlo", "Variance-Cov", "Historical"], fontsize=10)
ax2.set_xlabel("Date", fontsize=11)
ax2.set_ylabel("Method", fontsize=11)
ax2.grid(True, alpha=0.3, axis="x")
ax2.set_ylim(0.3, 3.7)

# Annotation with the headline test result
ax2.text(0.99, -0.40,
         "Visible clustering during 2008 GFC and 2020 COVID drives the Christoffersen rejection (p < 0.001 for all methods).",
         transform=ax2.transAxes, fontsize=9.5, va="top", ha="right",
         style="italic", color="dimgray")

plt.tight_layout()
plt.savefig("exception_clusters.png", dpi=150, bbox_inches="tight")
print("Saved exception_clusters.png")
plt.close()


# =============================================================================
# Plot 3: Stress test waterfall
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6.5))

# Use peak drawdown for historical scenarios, total loss for hypotheticals
plot_df = stress_results.copy()
plot_df["display_loss"] = np.where(
    plot_df["scenario_type"] == "historical",
    plot_df["peak_drawdown"],
    plot_df["total_loss"],
)

# Sort by display_loss descending for visual impact
plot_df = plot_df.sort_values("display_loss", ascending=True)

# Reference: 1-day historical VaR
var_dollars = var_summary.loc[
    var_summary["method"] == "Historical Simulation", "var_dollars"
].values[0]

# Horizontal bar chart
colors_per_bar = [
    COLOR_HIST if t == "historical" else COLOR_STRESS
    for t in plot_df["scenario_type"]
]

bars = ax.barh(plot_df["scenario"], plot_df["display_loss"],
               color=colors_per_bar, alpha=0.85,
               edgecolor="black", linewidth=0.8)

# Annotate each bar with the loss and the /VaR ratio
for bar, loss, scenario_type in zip(bars, plot_df["display_loss"], plot_df["scenario_type"]):
    ratio = loss / var_dollars
    label_text = f"${loss:,.0f}  ({ratio:.1f}× VaR)"
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
            label_text, va="center", fontsize=10, fontweight="bold")

# Reference line: 1-day 99% VaR
ax.axvline(var_dollars, color="black", linestyle="--", linewidth=1.5,
           label=f"1-day 99% VaR = ${var_dollars:,.0f}")

# 5× VaR reference (typical capital allocation)
ax.axvline(var_dollars * 5, color="gray", linestyle=":", linewidth=1.2,
           label=f"5× VaR = ${var_dollars * 5:,.0f}  (typical capital allocation)")

ax.set_xlabel("Loss ($)", fontsize=12)
ax.set_title("Stress Test Results — Peak Drawdown by Scenario\n"
             "(Historical replays use peak drawdown; hypotheticals are instantaneous shocks)",
             fontsize=13)
ax.grid(True, axis="x", alpha=0.3)
ax.set_axisbelow(True)

# Custom legend
legend_elements = [
    Patch(facecolor=COLOR_HIST, alpha=0.85, label="Historical replay"),
    Patch(facecolor=COLOR_STRESS, alpha=0.85, label="Hypothetical scenario"),
    plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
               label=f"1-day 99% VaR (${var_dollars:,.0f})"),
    plt.Line2D([0], [0], color="gray", linestyle=":", linewidth=1.2,
               label=f"5× VaR (typical capital)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

# Extend x-axis to accommodate value labels
ax.set_xlim(0, plot_df["display_loss"].max() * 1.30)

plt.tight_layout()
plt.savefig("stress_waterfall.png", dpi=150, bbox_inches="tight")
print("Saved stress_waterfall.png")
plt.close()


print("\nAll three plots generated.")
