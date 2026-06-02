"""
Day 3: VaR Backtesting.

Rolls each VaR method day-by-day over the full sample, counts exceptions
(days the actual loss exceeded predicted VaR), and runs:

    - Kupiec's POF test: is the exception RATE consistent with confidence level?
    - Christoffersen's CC test: are exceptions INDEPENDENT (not clustered)?
    - Basel traffic-light: green/yellow/red zone on a 250-day rolling window

Run with: python3 backtest.py
Produces: data/backtest_results.parquet, data/exception_dates.parquet
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from var import historical_var, parametric_var, monte_carlo_var, TOTAL_NOTIONAL


# ----------------------------- CONFIG -----------------------------
CONFIDENCE = 0.99
ALPHA = 1 - CONFIDENCE
ESTIMATION_WINDOW = 500   # days of history used to estimate VaR each day
BACKTEST_PORTFOLIO = {"SPY": 40_000, "TLT": 30_000, "GLD": 30_000}


# ----------------------------- BACKTEST RESULT CONTAINER -----------------------------
@dataclass(frozen=True)
class BacktestResult:
    method: str
    n_days: int
    n_exceptions: int
    exception_rate: float        # observed
    expected_rate: float          # = alpha (e.g., 0.01)

    # Kupiec
    lr_pof: float
    pof_pvalue: float
    pof_pass: bool

    # Christoffersen independence
    lr_ind: float
    ind_pvalue: float
    ind_pass: bool

    # Combined (Conditional Coverage)
    lr_cc: float
    cc_pvalue: float
    cc_pass: bool


# ----------------------------- ROLLING VAR FORECAST -----------------------------
def rolling_var_forecast(
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    method: str,
    portfolio: dict,
    estimation_window: int = ESTIMATION_WINDOW,
) -> pd.Series:
    """For each date t, compute VaR_t using ONLY data from [t-window, t-1].

    This is the look-ahead-bias-safe rolling backtest pattern.

    Parameters
    ----------
    method : str
        One of "historical", "parametric", "monte_carlo"

    Returns
    -------
    pd.Series of predicted VaR (positive losses) indexed by date.
    """
    var_forecasts = pd.Series(np.nan, index=portfolio_returns.index)
    dates = portfolio_returns.index

    # We need at least `estimation_window` days of history before producing the first forecast
    for i in range(estimation_window, len(dates)):
        # Window: dates[i - window] through dates[i - 1] (exclusive of today)
        window_assets = asset_returns.iloc[i - estimation_window: i]
        window_portfolio = portfolio_returns.iloc[i - estimation_window: i]

        if method == "historical":
            result = historical_var(window_portfolio, window=None)
        elif method == "parametric":
            result = parametric_var(window_assets, portfolio, window=None)
        elif method == "monte_carlo":
            # Reduce MC sims for speed during backtest loop
            result = monte_carlo_var(
                window_assets, portfolio, window=None, n_sims=20_000,
                seed=i,  # vary seed so it's not pathological
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        var_forecasts.iloc[i] = result.var_pct

    return var_forecasts


# ----------------------------- EXCEPTIONS -----------------------------
def compute_exceptions(
    actual_returns: pd.Series,
    var_forecasts: pd.Series,
) -> pd.Series:
    """Binary series: 1 if loss exceeded predicted VaR, 0 otherwise.

    actual loss = -actual_return; exception if loss > var_forecast.
    """
    # Drop dates where we don't have a forecast yet
    aligned = pd.DataFrame({"return": actual_returns, "var": var_forecasts}).dropna()
    exceptions = (-aligned["return"] > aligned["var"]).astype(int)
    return exceptions


# ----------------------------- KUPIEC POF TEST -----------------------------
def kupiec_pof_test(exceptions: pd.Series, alpha: float = ALPHA) -> tuple[float, float]:
    """Kupiec's Proportion-of-Failures likelihood-ratio test.

    H_0: true exception rate equals alpha (model's stated tail probability)
    Under H_0, LR_POF ~ chi-squared(1).

    Returns
    -------
    (LR_POF, p-value)
    """
    T = len(exceptions)
    x = int(exceptions.sum())          # observed exceptions
    pi_hat = x / T if T > 0 else 0.0   # observed exception rate

    if x == 0 or x == T:
        # Degenerate cases — return a very small LR and large p-value
        return 0.0, 1.0

    #  implement the Kupiec LR statistic.
    # The formula is:
    #   LR_POF = -2 * [ x*log(alpha) + (T-x)*log(1-alpha)
    #                 - x*log(pi_hat) - (T-x)*log(1-pi_hat) ]
    #
    # That's the log-likelihood ratio comparing "exceptions ~ Binomial(T, alpha)"
    # against "exceptions ~ Binomial(T, pi_hat)".
   
    log_lik_null = x * np.log(alpha) + (T - x) * np.log(1 - alpha)

    log_lik_alt = x * np.log(pi_hat) + (T - x) * np.log(1 - pi_hat)

    lr_pof = -2 * (log_lik_null - log_lik_alt)

    # p-value from chi-squared(1)
    p_value = 1 - stats.chi2.cdf(lr_pof, df=1)

    return lr_pof, p_value


# ----------------------------- CHRISTOFFERSEN INDEPENDENCE TEST -----------------------------
def christoffersen_independence_test(exceptions: pd.Series) -> tuple[float, float]:
    """Independence component of Christoffersen's Conditional Coverage test.

    Tests whether P(exception today | exception yesterday) equals
    P(exception today | no exception yesterday) — i.e., whether exceptions
    cluster across consecutive days.

    Under H_0 (independence), LR_IND ~ chi-squared(1).

    Returns
    -------
    (LR_IND, p-value)
    """
    e = exceptions.values
    T = len(e)

    if T < 2:
        return 0.0, 1.0

    # Count transitions
    n_00 = int(((e[:-1] == 0) & (e[1:] == 0)).sum())  # no-ex → no-ex
    n_01 = int(((e[:-1] == 0) & (e[1:] == 1)).sum())  # no-ex → ex
    n_10 = int(((e[:-1] == 1) & (e[1:] == 0)).sum())  # ex → no-ex
    n_11 = int(((e[:-1] == 1) & (e[1:] == 1)).sum())  # ex → ex

    # Conditional probabilities
    pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0.0
    pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0.0

    # Unconditional probability
    total_ex = n_01 + n_11
    total_no_ex = n_00 + n_10
    pi_hat = total_ex / (total_ex + total_no_ex) if (total_ex + total_no_ex) > 0 else 0.0

    # Degenerate cases (no transitions of one type)
    if pi_01 == 0 or pi_01 == 1 or pi_11 == 0 or pi_11 == 1 or pi_hat == 0 or pi_hat == 1:
        return 0.0, 1.0

    # Implement the independence LR statistic.
    # Compare the null (independence: same exception prob regardless of yesterday)
    # against the alternative (two different conditional probabilities).
    #
    # log L_null = n_01*log(pi_hat) + n_00*log(1-pi_hat)
    #            + n_11*log(pi_hat) + n_10*log(1-pi_hat)
    # log L_alt  = n_01*log(pi_01)  + n_00*log(1-pi_01)
    #            + n_11*log(pi_11)  + n_10*log(1-pi_11)
    # LR_IND = -2 * (log L_null - log L_alt)
    
    log_lik_null = (
    (n_00 + n_10) * np.log(1 - pi_hat)
    + (n_01 + n_11) * np.log(pi_hat)
    )

    log_lik_alt = (
        n_00 * np.log(1 - pi_01)
        + n_01 * np.log(pi_01)
        + n_10 * np.log(1 - pi_11)
        + n_11 * np.log(pi_11)
    )

    lr_ind = -2 * (log_lik_null - log_lik_alt)

    p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
    return lr_ind, p_value


# ----------------------------- CHRISTOFFERSEN CC TEST -----------------------------
def christoffersen_cc_test(
    exceptions: pd.Series,
    alpha: float = ALPHA,
) -> tuple[float, float, float, float, float, float]:
    """Christoffersen's Conditional Coverage test = POF + Independence.

    LR_CC = LR_POF + LR_IND, distributed chi-squared(2) under H_0.

    Returns
    -------
    (LR_POF, p_POF, LR_IND, p_IND, LR_CC, p_CC)
    """
    lr_pof, p_pof = kupiec_pof_test(exceptions, alpha)
    lr_ind, p_ind = christoffersen_independence_test(exceptions)

    # Combine into the CC statistic and compute its p-value
    # under chi-squared(2).
    

    lr_cc = lr_pof + lr_ind

    p_cc = 1 - stats.chi2.cdf(lr_cc, df=2)

    return lr_pof, p_pof, lr_ind, p_ind, lr_cc, p_cc


# ----------------------------- BASEL TRAFFIC LIGHT -----------------------------
def basel_traffic_light(
    exceptions: pd.Series,
    window: int = 250,
    alpha: float = ALPHA,
) -> dict:
    """Basel III traffic-light zones on a 250-day rolling window.

    For 250 days at 99% confidence:
        green:  0-4 exceptions
        yellow: 5-9 exceptions
        red:    10+ exceptions

    We compute the count in the FINAL 250-day window (the standard regulatory snapshot).
    """
    if len(exceptions) < window:
        return {"window_count": int(exceptions.sum()), "zone": "insufficient_data"}

    recent_count = int(exceptions.iloc[-window:].sum())

    # Classify into the Basel zone.
    # Standard Basel III thresholds at 99% confidence on 250-day windows:
    #   0-4 → "green"
    #   5-9 → "yellow"
    #   10+ → "red"

    
    if recent_count <= 4:
        zone = "green"
    elif recent_count <= 9:
        zone = "yellow"
    else:
        zone = "red"

    return {"window_count": recent_count, "zone": zone}


# ----------------------------- RUN BACKTEST FOR ONE METHOD -----------------------------
def run_full_backtest(
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    method: str,
    portfolio: dict,
) -> tuple[BacktestResult, pd.Series, dict]:
    """Run the full pipeline for one VaR method.

    Returns
    -------
    (BacktestResult, exceptions Series, Basel traffic-light dict)
    """
    print(f"  Running rolling {method} VaR forecast...")
    var_forecasts = rolling_var_forecast(
        asset_returns, portfolio_returns, method, portfolio
    )

    exceptions = compute_exceptions(portfolio_returns, var_forecasts)

    print(f"    Backtest length: {len(exceptions)} days, "
          f"exceptions: {int(exceptions.sum())}")

    # Statistical tests
    lr_pof, p_pof, lr_ind, p_ind, lr_cc, p_cc = christoffersen_cc_test(exceptions)

    # Traffic light
    traffic_light = basel_traffic_light(exceptions)

    result = BacktestResult(
        method=method,
        n_days=len(exceptions),
        n_exceptions=int(exceptions.sum()),
        exception_rate=exceptions.mean(),
        expected_rate=ALPHA,
        lr_pof=lr_pof,
        pof_pvalue=p_pof,
        pof_pass=(p_pof > 0.05),
        lr_ind=lr_ind,
        ind_pvalue=p_ind,
        ind_pass=(p_ind > 0.05),
        lr_cc=lr_cc,
        cc_pvalue=p_cc,
        cc_pass=(p_cc > 0.05),
    )

    return result, exceptions, traffic_light


# ----------------------------- REPORT -----------------------------
def print_backtest_table(
    results: dict[str, BacktestResult],
    traffic_lights: dict[str, dict],
):
    """Pretty-print the full backtest comparison."""
    print("\n" + "=" * 100)
    print(f"VAR BACKTEST RESULTS — {CONFIDENCE*100:.0f}% confidence")
    print("=" * 100)

    # Header
    print(f"\n{'Method':<25}{'Days':>7}{'Exc':>6}{'Rate':>8}{'Expected':>10}"
          f"{'POF(p)':>9}{'IND(p)':>9}{'CC(p)':>9}{'Traffic':>10}")
    print("-" * 100)

    for method_label, key in [
        ("Historical Simulation", "historical"),
        ("Variance-Covariance",  "parametric"),
        ("Monte Carlo (Gaussian)", "monte_carlo"),
    ]:
        r = results[key]
        tl = traffic_lights[key]
        print(f"{method_label:<25}{r.n_days:>7}{r.n_exceptions:>6}"
              f"{r.exception_rate*100:>7.2f}%{r.expected_rate*100:>9.2f}%"
              f"{r.pof_pvalue:>9.3f}{r.ind_pvalue:>9.3f}{r.cc_pvalue:>9.3f}"
              f"{tl['zone'].upper():>10}")

    # Detailed test interpretation
    print("\n" + "-" * 100)
    print("Test interpretation:")
    print("  Kupiec POF: p > 0.05 means exception RATE is consistent with confidence level")
    print("  Christoffersen IND: p > 0.05 means exceptions are INDEPENDENT across time")
    print("  Christoffersen CC: combined test; p > 0.05 means BOTH coverage and independence hold")
    print(f"  Basel Traffic Light: zone classification based on exceptions in last 250 days")
    print(f"    GREEN: 0-4, YELLOW: 5-9, RED: 10+")

    print("\n" + "-" * 100)
    print("Detailed LR statistics:")
    for method_label, key in [
        ("Historical Simulation", "historical"),
        ("Variance-Covariance",   "parametric"),
        ("Monte Carlo (Gaussian)", "monte_carlo"),
    ]:
        r = results[key]
        tl = traffic_lights[key]
        print(f"\n  {method_label}:")
        print(f"    Kupiec POF:        LR={r.lr_pof:.3f}  p={r.pof_pvalue:.4f}  "
              f"-> {'PASS' if r.pof_pass else 'REJECT'}")
        print(f"    Christoffersen IND: LR={r.lr_ind:.3f}  p={r.ind_pvalue:.4f}  "
              f"-> {'PASS' if r.ind_pass else 'REJECT'}")
        print(f"    Christoffersen CC:  LR={r.lr_cc:.3f}  p={r.cc_pvalue:.4f}  "
              f"-> {'PASS' if r.cc_pass else 'REJECT'}")
        print(f"    Recent 250-day exceptions: {tl['window_count']}  "
              f"-> {tl['zone'].upper()} zone")


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("Loading data from Day 1...")
    asset_returns = pd.read_parquet("data/asset_returns.parquet")
    portfolio_returns = pd.read_parquet("data/portfolio_returns.parquet")["portfolio_return_pct"]
    print(f"  {len(asset_returns)} days available")

    print("\nRunning backtests for each VaR method (rolling 500-day window)...")
    print("(This takes ~1-2 minutes for all three methods)")

    results = {}
    exceptions_dict = {}
    traffic_lights = {}

    for method in ["historical", "parametric", "monte_carlo"]:
        result, exceptions, traffic_light = run_full_backtest(
            asset_returns, portfolio_returns, method, BACKTEST_PORTFOLIO,
        )
        results[method] = result
        exceptions_dict[method] = exceptions
        traffic_lights[method] = traffic_light

    print_backtest_table(results, traffic_lights)

    # Save
    out_dir = Path("data")
    summary_df = pd.DataFrame([{
        "method": r.method,
        "n_days": r.n_days,
        "n_exceptions": r.n_exceptions,
        "exception_rate": r.exception_rate,
        "lr_pof": r.lr_pof,
        "pof_pvalue": r.pof_pvalue,
        "pof_pass": r.pof_pass,
        "lr_ind": r.lr_ind,
        "ind_pvalue": r.ind_pvalue,
        "ind_pass": r.ind_pass,
        "lr_cc": r.lr_cc,
        "cc_pvalue": r.cc_pvalue,
        "cc_pass": r.cc_pass,
    } for r in results.values()])
    summary_df.to_parquet(out_dir / "backtest_results.parquet")

    exceptions_df = pd.DataFrame(exceptions_dict)
    exceptions_df.to_parquet(out_dir / "exception_dates.parquet")

    print(f"\nSaved to data/backtest_results.parquet and data/exception_dates.parquet")
