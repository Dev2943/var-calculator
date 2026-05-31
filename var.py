"""
Day 2: VaR & Expected Shortfall — Three Methods.

Implements three approaches to estimating 1-day VaR and ES:
    1. Historical Simulation (non-parametric)
    2. Variance-Covariance (parametric Gaussian)
    3. Monte Carlo (Gaussian draws from estimated covariance)

Reports VaR and ES under each method as both percentages and dollar amounts,
plus the diagnostic ES/VaR ratios that quantify fat-tail behavior.

Run with: python3 var.py
Produces: data/var_summary.parquet
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------- CONFIG -----------------------------
CONFIDENCE = 0.99
ALPHA = 1 - CONFIDENCE     # 0.01 — tail probability
TOTAL_NOTIONAL = 100_000
HISTORICAL_WINDOW = 500    # Basel III default is 250; we use 500 for stability
MC_SIMULATIONS = 100_000
MC_SEED = 42


# ----------------------------- VAR RESULT CONTAINER -----------------------------
@dataclass(frozen=True)
class VaRResult:
    method: str
    var_pct: float           # VaR as a fraction (e.g., 0.0167 = 1.67%)
    es_pct: float            # ES as a fraction
    var_dollars: float       # VaR in $
    es_dollars: float        # ES in $
    confidence: float

    @property
    def es_var_ratio(self) -> float:
        return self.es_pct / self.var_pct if self.var_pct > 0 else 0.0

    def __repr__(self):
        return (
            f"{self.method:24s}  VaR={self.var_pct*100:.3f}% "
            f"(${self.var_dollars:,.0f})  ES={self.es_pct*100:.3f}% "
            f"(${self.es_dollars:,.0f})  ES/VaR={self.es_var_ratio:.2f}"
        )


# ----------------------------- METHOD 1: HISTORICAL SIMULATION -----------------------------
def historical_var(
    portfolio_returns: pd.Series,
    confidence: float = CONFIDENCE,
    window: int | None = HISTORICAL_WINDOW,
    notional: float = TOTAL_NOTIONAL,
) -> VaRResult:
    """Empirical quantile of past portfolio returns."""
    alpha = 1 - confidence

    # Use the most recent `window` days (or all of history if window=None)
    if window is not None:
        returns = portfolio_returns.tail(window)
    else:
        returns = portfolio_returns

    # compute VaR as the empirical alpha-quantile of returns,
    # then flip the sign so it's reported as a positive loss.
    
    var_pct = -returns.quantile(alpha)

    # compute ES as the mean of returns WORSE than the VaR threshold,
    # then flip the sign.
    
    threshold = returns.quantile(alpha)
    tail_returns = returns[returns <= threshold]
    es_pct = -tail_returns.mean()
    return VaRResult(
        method="Historical Simulation",
        var_pct=var_pct,
        es_pct=es_pct,
        var_dollars=var_pct * notional,
        es_dollars=es_pct * notional,
        confidence=confidence,
    )


# ----------------------------- METHOD 2: VARIANCE-COVARIANCE -----------------------------
def parametric_var(
    asset_returns: pd.DataFrame,
    portfolio: dict,
    confidence: float = CONFIDENCE,
    window: int | None = HISTORICAL_WINDOW,
    notional: float = TOTAL_NOTIONAL,
) -> VaRResult:
    """Closed-form Gaussian VaR/ES using portfolio variance from the covariance matrix.

    Math:
        sigma_p^2 = w' Σ w           (portfolio variance from $-weighted covariances)
        VaR_alpha = z_alpha × sigma_p
        ES_alpha  = sigma_p × phi(z_alpha) / alpha
    """
    alpha = 1 - confidence

    # Use most recent `window` days
    if window is not None:
        returns = asset_returns.tail(window)
    else:
        returns = asset_returns

    # Dollar-weighted covariance — the covariance of dollar P&L
    # Σ_dollar = diag(w) × Σ_returns × diag(w)
    # Then sigma_p^2 = sum(Σ_dollar), but more cleanly we compute:
    #   sigma_p = sqrt(w' Σ_returns w) where w is the dollar weight vector
    weights_dollar = np.array([portfolio[t] for t in returns.columns])
    cov_matrix = returns.cov().values
    portfolio_variance_dollars = weights_dollar @ cov_matrix @ weights_dollar
    portfolio_vol_dollars = np.sqrt(portfolio_variance_dollars)

    # Convert dollar vol back to fractional vol on the total notional
    portfolio_vol_pct = portfolio_vol_dollars / notional

    # compute the alpha-quantile z-score from the inverse normal CDF.
    # For alpha=0.01 this should give about -2.326.
    # Then take the positive VaR threshold.
   
    z_alpha = stats.norm.ppf(alpha)
    var_pct = -z_alpha * portfolio_vol_pct
    

    # closed-form Gaussian ES = sigma * phi(z_alpha) / alpha
    # where phi is the standard normal PDF.
   
    pdf_z = stats.norm.pdf(z_alpha)
    es_pct = portfolio_vol_pct * pdf_z / alpha

    return VaRResult(
        method="Variance-Covariance",
        var_pct=var_pct,
        es_pct=es_pct,
        var_dollars=var_pct * notional,
        es_dollars=es_pct * notional,
        confidence=confidence,
    )


# ----------------------------- METHOD 3: MONTE CARLO -----------------------------
def monte_carlo_var(
    asset_returns: pd.DataFrame,
    portfolio: dict,
    confidence: float = CONFIDENCE,
    window: int | None = HISTORICAL_WINDOW,
    n_sims: int = MC_SIMULATIONS,
    seed: int = MC_SEED,
    notional: float = TOTAL_NOTIONAL,
) -> VaRResult:
    """Monte Carlo VaR: draw asset return samples from MVN, compute portfolio P&L."""
    alpha = 1 - confidence

    if window is not None:
        returns = asset_returns.tail(window)
    else:
        returns = asset_returns

    # Estimate mean vector and covariance matrix
    mean_vec = returns.mean().values
    cov_matrix = returns.cov().values

    weights_dollar = np.array([portfolio[t] for t in returns.columns])

    # Simulate asset returns
    rng = np.random.default_rng(seed)
    simulated_returns = rng.multivariate_normal(mean_vec, cov_matrix, size=n_sims)
    # shape: (n_sims, n_assets)

    # Portfolio dollar P&L for each simulation
    simulated_pnl = simulated_returns @ weights_dollar  # shape: (n_sims,)
    simulated_pct = simulated_pnl / notional

    # VaR and ES from the simulated distribution.
    # Same idea as historical, just on synthetic data.
   
    var_pct = -np.quantile(simulated_pct, alpha)

    threshold = np.quantile(simulated_pct, alpha)
    tail = simulated_pct[simulated_pct <= threshold]
    es_pct = -tail.mean()

    return VaRResult(
        method="Monte Carlo (Gaussian)",
        var_pct=var_pct,
        es_pct=es_pct,
        var_dollars=var_pct * notional,
        es_dollars=es_pct * notional,
        confidence=confidence,
    )


# ----------------------------- REPORT -----------------------------
def print_var_table(results: list[VaRResult], notional: float):
    """Pretty-print the comparison table."""
    confidence = results[0].confidence
    print("\n" + "=" * 78)
    print(f"VaR & ES at {confidence*100:.0f}% confidence — 1-day horizon, "
          f"${notional:,.0f} portfolio")
    print("=" * 78)
    print(f"{'Method':25s}  {'VaR ($)':>10s}  {'ES ($)':>10s}  "
          f"{'VaR (%)':>9s}  {'ES (%)':>9s}  {'ES/VaR':>7s}")
    print("-" * 78)
    for r in results:
        print(f"{r.method:25s}  "
              f"{r.var_dollars:>10,.0f}  "
              f"{r.es_dollars:>10,.0f}  "
              f"{r.var_pct*100:>8.3f}%  "
              f"{r.es_pct*100:>8.3f}%  "
              f"{r.es_var_ratio:>7.2f}")
    print("=" * 78)

    # Diagnostics
    hist = next(r for r in results if r.method.startswith("Historical"))
    parm = next(r for r in results if r.method.startswith("Variance"))

    gap_var = (hist.var_pct - parm.var_pct) / parm.var_pct * 100
    gap_es = (hist.es_pct - parm.es_pct) / parm.es_pct * 100

    print(f"\nFat-tail diagnostics:")
    print(f"  Historical VaR vs. Gaussian VaR:  {gap_var:+.1f}%  "
          f"(positive = Gaussian underestimates tail threshold)")
    print(f"  Historical ES  vs. Gaussian ES:   {gap_es:+.1f}%  "
          f"(positive = Gaussian underestimates tail magnitude)")
    print(f"  Gaussian ES/VaR ratio:            {parm.es_var_ratio:.3f}  "
          f"(theory: 1.145 for 99%)")
    print(f"  Historical ES/VaR ratio:          {hist.es_var_ratio:.3f}  "
          f"(>1.145 indicates fat tails)")


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("Loading data from Day 1...")
    asset_returns = pd.read_parquet("data/asset_returns.parquet")
    portfolio_returns = pd.read_parquet("data/portfolio_returns.parquet")["portfolio_return_pct"]
    print(f"  {len(asset_returns)} days, {len(asset_returns.columns)} assets")

    # Portfolio composition (must match Day 1)
    portfolio = {"SPY": 40_000, "TLT": 30_000, "GLD": 30_000}

    print(f"\nComputing VaR/ES at {CONFIDENCE*100:.0f}% confidence "
          f"using last {HISTORICAL_WINDOW} days...")

    hist_result = historical_var(portfolio_returns)
    parm_result = parametric_var(asset_returns, portfolio)
    mc_result = monte_carlo_var(asset_returns, portfolio)

    results = [hist_result, parm_result, mc_result]
    print_var_table(results, TOTAL_NOTIONAL)

    # Save
    summary = pd.DataFrame([{
        "method": r.method,
        "var_pct": r.var_pct,
        "es_pct": r.es_pct,
        "var_dollars": r.var_dollars,
        "es_dollars": r.es_dollars,
        "es_var_ratio": r.es_var_ratio,
        "confidence": r.confidence,
    } for r in results])
    out_dir = Path("data")
    summary.to_parquet(out_dir / "var_summary.parquet")
    print("\nSaved to data/var_summary.parquet")
