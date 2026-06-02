"""
Day 4: Stress Testing.

Runs five stress scenarios on the current portfolio:
    1. 2008 GFC historical replay (Sep 2008 - Mar 2009)
    2. 2020 COVID historical replay (Feb 2020 - Apr 2020)
    3. Hypothetical: Equity Crash
    4. Hypothetical: Stagflation
    5. Hypothetical: Correlation Breakdown

For each, reports total loss, peak drawdown, worst day (historical only),
and the stress/VaR ratio.


Run with: python3 stress_test.py
Produces: data/stress_results.parquet
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from var import historical_var, TOTAL_NOTIONAL


# ----------------------------- CONFIG -----------------------------
PORTFOLIO = {"SPY": 40_000, "TLT": 30_000, "GLD": 30_000}

# Historical stress windows
GFC_START = "2008-09-01"
GFC_END   = "2009-03-31"
COVID_START = "2020-02-20"
COVID_END   = "2020-04-30"

# Hypothetical scenarios — (SPY_shock, TLT_shock, GLD_shock) as decimals
HYPOTHETICAL_SCENARIOS = {
    "Equity Crash":            {"SPY": -0.30, "TLT": +0.05, "GLD": +0.10},
    "Stagflation":             {"SPY": -0.15, "TLT": -0.20, "GLD": +0.20},
    "Correlation Breakdown":   {"SPY": -0.25, "TLT": -0.15, "GLD": -0.10},
}


# ----------------------------- RESULT CONTAINER -----------------------------
@dataclass(frozen=True)
class StressResult:
    scenario: str
    scenario_type: str          # "historical" or "hypothetical"
    total_loss: float           # cumulative $ loss over the window
    peak_drawdown: float        # worst peak-to-trough $ within the window
    worst_day_loss: float       # largest single-day $ loss
    worst_day_date: str         # date of worst day (or "n/a" for hypotheticals)
    n_loss_days: int            # number of negative-PnL days
    var_at_start: float         # 99% VaR at the scenario start (for ratio)


# ----------------------------- HISTORICAL REPLAY -----------------------------
def historical_replay(
    asset_returns: pd.DataFrame,
    portfolio: dict,
    start_date: str,
    end_date: str,
    scenario_name: str,
    var_at_start: float,
    notional: float = TOTAL_NOTIONAL,
) -> StressResult:
    """Apply historical returns from [start_date, end_date] to today's portfolio."""

    # Pull returns for the stress window
    window = asset_returns.loc[start_date:end_date]
    if len(window) == 0:
        raise ValueError(f"No data in window {start_date} to {end_date}")

    # Dollar weights vector aligned to columns
    weights_dollar = np.array([portfolio[t] for t in window.columns])

    # Compute the daily portfolio dollar P&L during the stress window.
    # Same pattern as Day 1 — dot product of asset returns with dollar weights.
    

    daily_pnl = window.dot(weights_dollar)

    # Cumulative P&L path
    cumulative_pnl = daily_pnl.cumsum()

    # Statistics
    total_loss = -cumulative_pnl.iloc[-1]  # final cumulative loss (positive number)

    # Peak drawdown: largest peak-to-trough on the cumulative path
    running_max = cumulative_pnl.cummax()
    drawdown_series = cumulative_pnl - running_max
    peak_drawdown = -drawdown_series.min()  # largest drop, positive number

    # Worst single day
    worst_day_loss = -daily_pnl.min()
    worst_day_date = str(daily_pnl.idxmin().date())

    # Number of loss days
    n_loss_days = int((daily_pnl < 0).sum())

    return StressResult(
        scenario=scenario_name,
        scenario_type="historical",
        total_loss=total_loss,
        peak_drawdown=peak_drawdown,
        worst_day_loss=worst_day_loss,
        worst_day_date=worst_day_date,
        n_loss_days=n_loss_days,
        var_at_start=var_at_start,
    )


# ----------------------------- HYPOTHETICAL SCENARIO -----------------------------
def hypothetical_scenario(
    portfolio: dict,
    shocks: dict,
    scenario_name: str,
    var_at_start: float,
    notional: float = TOTAL_NOTIONAL,
) -> StressResult:
    """Apply instantaneous shocks to each asset and compute portfolio loss."""

    # Compute the total scenario loss in dollars.
    # For each asset, dollar P&L = dollar_weight × shock.
    # The total loss is the negative of the sum (since shocks are P&L, loss is -P&L).
    

    total_pnl = sum(
        portfolio[ticker] * shocks[ticker]
        for ticker in portfolio
    )

    total_loss = -total_pnl

    return StressResult(
        scenario=scenario_name,
        scenario_type="hypothetical",
        total_loss=total_loss,
        peak_drawdown=total_loss,    # instantaneous: drawdown == total loss
        worst_day_loss=total_loss,   # instantaneous: worst day == total loss
        worst_day_date="n/a",
        n_loss_days=1,
        var_at_start=var_at_start,
    )


# ----------------------------- REPORT -----------------------------
def print_stress_table(results: list[StressResult], var_dollars: float):
    """Pretty-print the loss waterfall."""
    print("\n" + "=" * 100)
    print(f"STRESS TEST RESULTS — ${TOTAL_NOTIONAL:,.0f} portfolio")
    print(f"Reference: 1-day 99% Historical VaR = ${var_dollars:,.0f}")
    print("=" * 100)
    print(f"\n{'Scenario':<35}{'Total Loss':>12}{'Peak DD':>12}{'Worst Day':>15}{'Loss Days':>11}{'/VaR':>8}")
    print("-" * 100)

    for r in results:
        worst_day_str = (f"${r.worst_day_loss:,.0f} ({r.worst_day_date})"
                         if r.scenario_type == "historical"
                         else f"${r.worst_day_loss:,.0f} (instant)")

        loss_days_str = str(r.n_loss_days) if r.scenario_type == "historical" else "n/a"

        # Compute the stress/VaR ratio.
        # This is the headline diagnostic: how many days of normal VaR
        # does this stress scenario equal?
        
        stress_var_ratio = r.total_loss / r.var_at_start

        print(f"{r.scenario:<35}"
              f"${r.total_loss:>10,.0f}"
              f"${r.peak_drawdown:>10,.0f}"
              f"  {worst_day_str:<14}"
              f"{loss_days_str:>10}"
              f"{stress_var_ratio:>7.1f}x")

    print("=" * 100)

    # Diagnostic summary
    print("\nKey diagnostics:")

    worst_hist = max((r for r in results if r.scenario_type == "historical"),
                     key=lambda r: r.total_loss, default=None)
    worst_hyp = max((r for r in results if r.scenario_type == "hypothetical"),
                    key=lambda r: r.total_loss, default=None)

    if worst_hist:
        ratio = worst_hist.total_loss / var_dollars
        print(f"  Worst historical replay: {worst_hist.scenario} — ${worst_hist.total_loss:,.0f} "
              f"({ratio:.1f}× single-day VaR)")
    if worst_hyp:
        ratio = worst_hyp.total_loss / var_dollars
        print(f"  Worst hypothetical:      {worst_hyp.scenario} — ${worst_hyp.total_loss:,.0f} "
              f"({ratio:.1f}× single-day VaR)")

    print("\n  Implication: capital sized to 5× daily VaR would NOT cover the worst stress scenarios.")
    print("  Production risk frameworks size capital to max(VaR-based, stress-based) requirements.")


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("Loading data...")
    asset_returns = pd.read_parquet("data/asset_returns.parquet")
    portfolio_returns = pd.read_parquet("data/portfolio_returns.parquet")["portfolio_return_pct"]

    # Compute reference VaR at the start of each stress window
    # (in practice we'd use the VaR at that exact date; for simplicity use latest)
    var_result = historical_var(portfolio_returns)
    var_dollars = var_result.var_dollars
    print(f"  Reference 1-day 99% VaR: ${var_dollars:,.0f}")

    results = []

    # 1. 2008 GFC replay
    print(f"\nRunning 2008 GFC replay ({GFC_START} to {GFC_END})...")
    r = historical_replay(asset_returns, PORTFOLIO, GFC_START, GFC_END,
                          "2008 GFC Replay", var_dollars)
    results.append(r)
    print(f"  Total loss: ${r.total_loss:,.0f}, worst day: ${r.worst_day_loss:,.0f} on {r.worst_day_date}")

    # 2. 2020 COVID replay
    print(f"\nRunning 2020 COVID replay ({COVID_START} to {COVID_END})...")
    r = historical_replay(asset_returns, PORTFOLIO, COVID_START, COVID_END,
                          "2020 COVID Replay", var_dollars)
    results.append(r)
    print(f"  Total loss: ${r.total_loss:,.0f}, worst day: ${r.worst_day_loss:,.0f} on {r.worst_day_date}")

    # 3-5. Hypothetical scenarios
    for name, shocks in HYPOTHETICAL_SCENARIOS.items():
        shock_str = ", ".join(f"{k}{v*100:+.0f}%" for k, v in shocks.items())
        print(f"\nRunning hypothetical: {name} ({shock_str})...")
        r = hypothetical_scenario(PORTFOLIO, shocks, name, var_dollars)
        results.append(r)
        print(f"  Total loss: ${r.total_loss:,.0f}")

    print_stress_table(results, var_dollars)

    # Save
    summary_df = pd.DataFrame([{
        "scenario": r.scenario,
        "scenario_type": r.scenario_type,
        "total_loss": r.total_loss,
        "peak_drawdown": r.peak_drawdown,
        "worst_day_loss": r.worst_day_loss,
        "worst_day_date": r.worst_day_date,
        "n_loss_days": r.n_loss_days,
        "var_at_start": r.var_at_start,
        "stress_var_ratio": r.total_loss / r.var_at_start,
    } for r in results])

    out_dir = Path("data")
    summary_df.to_parquet(out_dir / "stress_results.parquet")
    print("\nSaved to data/stress_results.parquet")
