"""
Day 1: Portfolio Data Pipeline.

Pulls daily prices for a 3-asset portfolio (SPY, TLT, GLD), computes returns,
and reports tail statistics that will motivate the VaR methodology choices on Day 2.

Portfolio composition:
    - SPY (US equity): $40,000   (40%)
    - TLT (long bonds): $30,000  (30%)
    - GLD (gold):       $30,000  (30%)
    Total notional:    $100,000

This is a classic institutional "60/40/diversifier" allocation. The three
assets have low pairwise correlations and span the major macro risk drivers.


Run with: python3 data.py
Produces: data/asset_returns.parquet, data/portfolio_returns.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------- CONFIG -----------------------------

# Portfolio composition: tickers and dollar weights
PORTFOLIO = {
    "SPY": 40_000,   # US equity
    "TLT": 30_000,   # Long Treasury bonds
    "GLD": 30_000,   # Gold
}
TOTAL_NOTIONAL = sum(PORTFOLIO.values())

# Date range: long enough to include 2008 GFC and 2020 COVID (for Day 4 stress tests)
START_DATE = "2005-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")


# ----------------------------- PRICE PULL -----------------------------
def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """Download adjusted-close daily prices for our portfolio."""
    print(f"  Downloading prices for {tickers} from yfinance...")
    data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # yfinance returns a MultiIndex when multiple tickers; flatten to just close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    # Drop dates with any missing data — keeps the panel balanced
    prices = prices.dropna()
    return prices


# ----------------------------- RETURN COMPUTATION -----------------------------
def compute_asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns for each asset."""
    # compute daily simple returns from prices using pct_change.
    # The first row will be NaN — drop it.
    
    asset_returns = prices.pct_change().dropna()

    return asset_returns


# ----------------------------- PORTFOLIO P&L -----------------------------
def compute_portfolio_returns(
    asset_returns: pd.DataFrame,
    portfolio: dict,
) -> pd.DataFrame:
    """Compute daily portfolio dollar P&L and percentage return.

    Returns
    -------
    pd.DataFrame with columns:
        portfolio_pnl_dollars: daily $ P&L
        portfolio_return_pct:  daily % return on total notional
    """
    # Asset weights (dollar amounts in same order as columns)
    weights_dollar = pd.Series(
        {ticker: portfolio[ticker] for ticker in asset_returns.columns}
    )
    total_notional = weights_dollar.sum()

    # dollar P&L on day t = sum across assets of (weight_i * return_i)
    # This is the dot product of dollar weights and asset returns.
    
    portfolio_pnl_dollars = asset_returns.dot(weights_dollar)

    # Percentage return on the total notional (so we can quote VaR in %)
    portfolio_return_pct = portfolio_pnl_dollars / total_notional

    return pd.DataFrame({
        "portfolio_pnl_dollars": portfolio_pnl_dollars,
        "portfolio_return_pct": portfolio_return_pct,
    })


# ----------------------------- SUMMARY STATISTICS -----------------------------
def report_summary(asset_returns: pd.DataFrame, portfolio_returns: pd.DataFrame):
    """Print summary stats that will motivate the VaR methodology choices."""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(f"\nDate range: {asset_returns.index.min().date()} to {asset_returns.index.max().date()}")
    print(f"Trading days: {len(asset_returns)}")
    print(f"Total notional: ${TOTAL_NOTIONAL:,}")

    # Annualized stats per asset
    print("\nAsset summary (annualized):")
    summary = pd.DataFrame({
        "Annualized Return": asset_returns.mean() * 252,
        "Annualized Volatility": asset_returns.std() * np.sqrt(252),
        "Skewness": asset_returns.skew(),
        "Excess Kurtosis": asset_returns.kurt(),
        "Worst day (%)": asset_returns.min() * 100,
    })
    print(summary.round(4).to_string())

    # compute the pairwise correlation matrix of asset returns.
    # This is the key driver of diversification benefit.
    
    print("\nAsset return correlation matrix:")
    corr_matrix = asset_returns.corr()
    print(corr_matrix.round(3).to_string())

    # Portfolio-level statistics
    pnl_pct = portfolio_returns["portfolio_return_pct"]
    pnl_dollars = portfolio_returns["portfolio_pnl_dollars"]

    print("\nPortfolio summary (annualized):")
    print(f"  Annualized return:        {pnl_pct.mean() * 252 * 100:.2f}%")
    print(f"  Annualized volatility:    {pnl_pct.std() * np.sqrt(252) * 100:.2f}%")
    print(f"  Sharpe ratio (~):         {(pnl_pct.mean() * 252) / (pnl_pct.std() * np.sqrt(252)):.2f}")
    print(f"  Skewness:                 {pnl_pct.skew():.3f}")
    print(f"  Excess kurtosis:          {pnl_pct.kurt():.2f}")

    # Tail behavior — the part that motivates the project
    print("\nPortfolio worst-day analysis:")
    print(f"  Worst single day:         {pnl_pct.min() * 100:.2f}% = ${pnl_dollars.min():,.0f}")
    print(f"  Best single day:          {pnl_pct.max() * 100:.2f}% = ${pnl_dollars.max():,.0f}")
    print(f"  Worst day date:           {pnl_pct.idxmin().date()}")
    print(f"  1st percentile:           {pnl_pct.quantile(0.01) * 100:.2f}%")
    print(f"  5th percentile:           {pnl_pct.quantile(0.05) * 100:.2f}%")

    # sanity check the Gaussian vs empirical tail.
    # Under a normal distribution, the 1st percentile would be -2.33 sigma.
    # Compute what -2.33 sigma actually is for our portfolio and compare to the
    # empirical 1st percentile. If empirical << Gaussian prediction, fat tails confirmed.
   
    gaussian_1pct = -2.33 * pnl_pct.std()
    empirical_1pct = pnl_pct.quantile(0.01)

    print(f"\nGaussian 1st pct prediction: {gaussian_1pct * 100:.2f}%")
    print(f"Empirical 1st pct:           {empirical_1pct * 100:.2f}%")
    print(f"Ratio:                       {empirical_1pct / gaussian_1pct:.2f}x")


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    print("Fetching prices...")
    tickers = list(PORTFOLIO.keys())
    prices = fetch_prices(tickers)
    print(f"  Got {len(prices)} days of prices for {len(prices.columns)} assets")

    print("\nComputing asset returns...")
    asset_returns = compute_asset_returns(prices)

    print("\nComputing portfolio P&L...")
    portfolio_returns = compute_portfolio_returns(asset_returns, PORTFOLIO)

    report_summary(asset_returns, portfolio_returns)

    # Save
    asset_returns.to_parquet(out_dir / "asset_returns.parquet")
    portfolio_returns.to_parquet(out_dir / "portfolio_returns.parquet")
    print(f"\nSaved to data/asset_returns.parquet and data/portfolio_returns.parquet")
