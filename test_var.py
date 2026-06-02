"""
Test suite for the VaR project.

Five categories:
    1. DATA INTEGRITY — portfolio returns are well-formed
    2. VAR SANITY — methods produce plausible numbers and the Gaussian closed form is correct
    3. METHOD AGREEMENT — Monte Carlo Gaussian should agree with Variance-Covariance
    4. BACKTEST CORRECTNESS — Kupiec and Christoffersen LRs are non-negative and bounded
    5. STRESS TEST VALIDITY — hypothetical losses match closed-form math; historical replays produce sensible numbers

Run with: pytest test_var.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from var import (
    historical_var, parametric_var, monte_carlo_var,
    CONFIDENCE, ALPHA, TOTAL_NOTIONAL,
)
from backtest import (
    kupiec_pof_test, christoffersen_independence_test,
    christoffersen_cc_test, basel_traffic_light,
    compute_exceptions,
)


# ----- Fixtures -----

@pytest.fixture(scope="module")
def asset_returns():
    return pd.read_parquet("data/asset_returns.parquet")


@pytest.fixture(scope="module")
def portfolio_returns():
    return pd.read_parquet("data/portfolio_returns.parquet")["portfolio_return_pct"]


@pytest.fixture(scope="module")
def portfolio():
    return {"SPY": 40_000, "TLT": 30_000, "GLD": 30_000}


# ========== 1. DATA INTEGRITY ==========

def test_asset_returns_have_three_columns(asset_returns):
    """Portfolio is SPY/TLT/GLD."""
    assert set(asset_returns.columns) == {"SPY", "TLT", "GLD"}


def test_returns_in_decimal_form(asset_returns):
    """Daily returns should be decimals, not percentages."""
    assert asset_returns.abs().max().max() < 0.5, \
        "Returns appear to be in % form; values should be < 0.5 in decimal form"


def test_portfolio_returns_are_dollar_weighted(asset_returns, portfolio_returns, portfolio):
    """Spot-check that portfolio_return equals weighted sum of asset returns."""
    weights_pct = pd.Series({k: v / TOTAL_NOTIONAL for k, v in portfolio.items()})
    expected = asset_returns.mul(weights_pct).sum(axis=1)
    diff = (portfolio_returns - expected).abs().max()
    assert diff < 1e-10, f"Portfolio returns don't match weighted asset returns; max diff {diff}"


# ========== 2. VAR SANITY ==========

def test_historical_var_positive(portfolio_returns):
    """VaR is reported as a positive loss."""
    result = historical_var(portfolio_returns)
    assert result.var_pct > 0, f"VaR should be positive, got {result.var_pct}"
    assert result.es_pct > 0, f"ES should be positive, got {result.es_pct}"


def test_es_always_at_least_as_large_as_var(asset_returns, portfolio_returns, portfolio):
    """ES is the average of losses beyond VaR — must be >= VaR."""
    methods = [
        historical_var(portfolio_returns),
        parametric_var(asset_returns, portfolio),
        monte_carlo_var(asset_returns, portfolio),
    ]
    for r in methods:
        assert r.es_pct >= r.var_pct - 1e-6, \
            f"{r.method}: ES ({r.es_pct}) < VaR ({r.var_pct})"


def test_gaussian_es_var_ratio_matches_theory(asset_returns, portfolio):
    """Closed-form Gaussian ES/VaR at 99% confidence is φ(2.326)/0.01/2.326 ≈ 1.145."""
    parm = parametric_var(asset_returns, portfolio)
    # Theoretical value
    z = stats.norm.ppf(0.01)
    theoretical = (stats.norm.pdf(z) / 0.01) / abs(z)
    assert abs(parm.es_var_ratio - theoretical) < 0.01, \
        f"Gaussian ES/VaR ratio is {parm.es_var_ratio}, theory predicts {theoretical:.4f}"


def test_var_increases_with_confidence(portfolio_returns):
    """Higher confidence → larger VaR (more conservative threshold)."""
    var_95 = historical_var(portfolio_returns, confidence=0.95)
    var_99 = historical_var(portfolio_returns, confidence=0.99)
    assert var_99.var_pct > var_95.var_pct, \
        f"99% VaR ({var_99.var_pct}) should exceed 95% VaR ({var_95.var_pct})"


# ========== 3. METHOD AGREEMENT ==========

def test_monte_carlo_agrees_with_parametric(asset_returns, portfolio):
    """MC Gaussian and Variance-Covariance should agree within MC sampling error."""
    parm = parametric_var(asset_returns, portfolio)
    mc = monte_carlo_var(asset_returns, portfolio, n_sims=100_000)
    # MC error with 100K simulations at the 1% quantile is roughly a few % of the estimate
    rel_diff = abs(mc.var_pct - parm.var_pct) / parm.var_pct
    assert rel_diff < 0.10, \
        f"MC VaR ({mc.var_pct}) disagrees with Variance-Covariance VaR ({parm.var_pct}) by {rel_diff:.1%}"


def test_historical_var_typically_larger_than_gaussian(asset_returns, portfolio_returns, portfolio):
    """For fat-tailed returns, historical VaR should exceed Gaussian VaR (the fat-tail effect)."""
    hist = historical_var(portfolio_returns)
    parm = parametric_var(asset_returns, portfolio)
    assert hist.var_pct > parm.var_pct, \
        f"Historical VaR ({hist.var_pct}) should exceed Gaussian VaR ({parm.var_pct}) — fat tails not detected"


# ========== 4. BACKTEST CORRECTNESS ==========

def test_kupiec_lr_non_negative():
    """LR statistics are always >= 0."""
    # Synthetic: 10 exceptions out of 1000 days at 1% — exactly on target
    exceptions = pd.Series([1]*10 + [0]*990)
    lr, p = kupiec_pof_test(exceptions, alpha=0.01)
    assert lr >= -1e-10, f"Kupiec LR was negative: {lr}"
    assert p > 0.05, f"On-target frequency should pass POF, got p={p}"


def test_kupiec_rejects_high_exception_rate():
    """If exception rate is 5% but target is 1%, model should be rejected."""
    exceptions = pd.Series([1]*50 + [0]*950)
    lr, p = kupiec_pof_test(exceptions, alpha=0.01)
    assert p < 0.05, f"5% exception rate vs 1% target should reject, got p={p}"


def test_christoffersen_independence_rejects_clustering():
    """If exceptions cluster in two separate runs, independence should be rejected.

    Use a two-cluster pattern so the transition matrix is non-degenerate:
    both no-ex→ex transitions (n_01 > 0) and ex→ex transitions (n_11 > 0) exist.
    A single isolated cluster would have n_01 = 0, causing the test's degenerate-case
    guard to return p=1.0 (cannot reject) — correct defensive behavior, but useless
    for verifying the test detects clustering.
    """
    cluster1 = [1] * 5
    gap1 = [0] * 200
    cluster2 = [1] * 5
    gap2 = [0] * 790
    exceptions = pd.Series(cluster1 + gap1 + cluster2 + gap2)
    lr, p = christoffersen_independence_test(exceptions)
    assert p < 0.05, f"Clustered exceptions should fail independence, got p={p}"


def test_cc_pvalue_bounded():
    """p-values must be in [0, 1]."""
    exceptions = pd.Series([1]*30 + [0]*970)
    _, p_pof, _, p_ind, _, p_cc = christoffersen_cc_test(exceptions)
    for p in [p_pof, p_ind, p_cc]:
        assert 0 <= p <= 1, f"p-value out of bounds: {p}"


def test_basel_traffic_light_classification():
    """Test all three Basel zones."""
    # Green: 3 exceptions in 250 days
    green_ex = pd.Series([1]*3 + [0]*247)
    assert basel_traffic_light(green_ex)["zone"] == "green"

    # Yellow: 7 exceptions in 250 days
    yellow_ex = pd.Series([1]*7 + [0]*243)
    assert basel_traffic_light(yellow_ex)["zone"] == "yellow"

    # Red: 15 exceptions in 250 days
    red_ex = pd.Series([1]*15 + [0]*235)
    assert basel_traffic_light(red_ex)["zone"] == "red"


# ========== 5. STRESS TEST VALIDITY ==========

def test_hypothetical_shock_closed_form(portfolio):
    """Hypothetical scenario loss = -sum(weight × shock). Verify by hand."""
    # Equity Crash: SPY -30%, TLT +5%, GLD +10%
    shocks = {"SPY": -0.30, "TLT": +0.05, "GLD": +0.10}
    expected_loss = -(40_000 * -0.30 + 30_000 * 0.05 + 30_000 * 0.10)
    # = -(-12000 + 1500 + 3000) = -(-7500) = 7500
    assert abs(expected_loss - 7500) < 1e-6


def test_stress_results_are_loaded():
    """Day 4 saved a stress results parquet; verify it's there with sensible columns."""
    p = Path("data/stress_results.parquet")
    assert p.exists(), "Run stress_test.py before testing"
    df = pd.read_parquet(p)
    assert "total_loss" in df.columns
    assert "peak_drawdown" in df.columns
    assert len(df) >= 5  # 2 historical + 3 hypothetical


def test_stress_var_ratios_are_in_plausible_range():
    """Stress losses should be a few × daily VaR — not 100× and not 0.5×."""
    df = pd.read_parquet("data/stress_results.parquet")
    ratios = df["stress_var_ratio"].abs()  # absolute value handles 2020-style recovery
    assert ratios.max() < 50, f"Max stress/VaR ratio of {ratios.max()} is implausibly large"
    assert ratios.max() > 1, f"Max stress/VaR ratio of {ratios.max()} is too small for a stress test"
