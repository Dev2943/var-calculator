# Portfolio VaR & Expected Shortfall Calculator with Backtesting and Stress Testing

A market-risk measurement framework for a multi-asset portfolio (SPY, TLT, GLD). Implements three VaR/ES methodologies, runs Kupiec and Christoffersen backtests over a 19-year sample, executes 2008 GFC and 2020 COVID historical stress replays, and runs hypothetical scenarios — culminating in a Model Validation memo of the kind bank MV teams produce.

Built as the fourth project in a quantitative finance portfolio. The end-to-end artifact mirrors the daily workflow of bank Market Risk, Model Validation, and CCAR/DFAST stress-testing teams.

## Highlights

- **Three VaR/ES methods** — Historical Simulation, Variance-Covariance, and Monte Carlo — with full diagnostic comparison
- **Rigorous backtesting** using Kupiec's POF and Christoffersen's CC likelihood-ratio tests over 4,884 trading days plus Basel III traffic-light classification
- **Empirical demonstration** that Gaussian VaR systematically under-captures fat tails: 66-78% excess exception rate vs target
- **Identification of clustering failure mode**: Historical VaR passes Kupiec but fails Christoffersen at p < 0.001, exactly the dangerous failure mode that brought down major shops in 1998 and 2008
- **Historical replay stress tests** through the 2008 GFC and 2020 COVID crisis windows
- **Three hypothetical scenarios** including a correlation-breakdown scenario whose loss matches the 2008 GFC peak drawdown
- **15 passing tests** covering data integrity, closed-form Gaussian math, backtest correctness, and stress test validity
- **Validation memo** (`VALIDATION_MEMO.md`) summarizing findings in the format used by bank Model Risk Management groups

---

## Project structure

```
var-calculator/
├── data.py                 # Day 1: portfolio data pipeline with fat-tail diagnostics
├── var.py                  # Day 2: three VaR/ES methods with comparison table
├── backtest.py             # Day 3: Kupiec POF + Christoffersen CC + Basel traffic-light
├── stress_test.py          # Day 4: 2008/2020 replays + 3 hypothetical scenarios
├── test_var.py             # 15 passing tests
├── VALIDATION_MEMO.md      # Model Validation summary memo
├── requirements.txt
└── README.md
```

## Methodology and findings

### Step 1: Data and fat-tail diagnostics

Portfolio: $40K SPY + $30K TLT + $30K GLD = $100K notional. Daily data from 2005-01-04 to 2026-05-29 (5,384 trading days), including the 2008 GFC and 2020 COVID crash.

Asset summary:

| Asset | Annualized Vol | Skewness | Excess Kurtosis | Worst Day |
|---|---|---|---|---|
| SPY | 19.0% | ~0 | **15.4** | -10.9% |
| TLT | 14.6% | +0.08 | 3.5 | -6.7% |
| GLD | 18.1% | -0.31 | 6.7 | -10.3% |

Asset correlation matrix:

| | SPY | TLT | GLD |
|---|---|---|---|
| SPY | 1.000 | **-0.301** | 0.061 |
| TLT | -0.301 | 1.000 | 0.157 |
| GLD | 0.061 | 0.157 | 1.000 |

The SPY-TLT negative correlation captures the classic stock-bond flight-to-safety relationship; GLD's near-zero correlations make it a true diversifier. Portfolio vol (9.94%) is roughly half the weighted-average of individual vols — that gap is the diversification benefit.

### Step 2: Three VaR/ES methods — disagreement quantified

At 99% confidence, 1-day horizon, using a 500-day estimation window:

| Method | VaR ($) | ES ($) | ES/VaR ratio |
|---|---|---|---|
| Historical Simulation | 2,015 | 2,477 | 1.23 |
| Variance-Covariance | 1,655 | 1,896 | 1.15 |
| Monte Carlo (Gaussian) | 1,560 | 1,796 | 1.15 |

Two key findings:

**(a)** Historical VaR is **22% higher** than Variance-Covariance VaR. The Gaussian model under-captures the empirical tail threshold by roughly that amount.

**(b)** The Gaussian ES/VaR ratio of 1.146 matches theory exactly (`φ(z_α)/α/|z_α| = 1.145`), validating the closed-form math. The empirical ratio of 1.23 (modestly elevated above the Gaussian 1.15) confirms fat-tail behavior even on a calm 500-day window. On the full 5,000-day window including 2008 and 2020, the historical ES/VaR ratio rises to ~1.6.

### Step 3: Backtesting — every method has a failure mode

Rolling each VaR forecast day-by-day over 4,884 trading days produced these exception statistics:

| Method | Days | Exceptions | Rate | Kupiec POF p | Christoffersen IND p | CC p | Basel Zone |
|---|---|---|---|---|---|---|---|
| Historical Simulation | 4884 | 63 | 1.29% | 0.051 (PASS) | **0.000 (FAIL)** | 0.000 (FAIL) | Green |
| Variance-Covariance | 4884 | 81 | 1.66% | 0.000 (FAIL) | 0.000 (FAIL) | 0.000 (FAIL) | Yellow |
| Monte Carlo (Gaussian) | 4884 | 87 | 1.78% | 0.000 (FAIL) | 0.000 (FAIL) | 0.000 (FAIL) | Yellow |

The headline finding: **Historical VaR passes Kupiec's POF test but fails Christoffersen's independence test**. The model has roughly the right total exception count but exceptions cluster severely during 2008 and 2020. This is the textbook "passes the average, fails the moment that matters" failure mode of historical VaR — and is exactly why Christoffersen (1998) introduced the independence test in the first place.

The Gaussian methods fail both Kupiec (66-78% excess exception rate) and Christoffersen (clustering during stress regimes).

### Step 4: Stress testing — quantifying what VaR can't see

Five stress scenarios run on the current portfolio:

| Scenario | Total Loss | Peak Drawdown | Worst Day | Peak DD / VaR |
|---|---|---|---|---|
| 2008 GFC Replay (Sep 2008 - Mar 2009) | $5,744 | $18,697 | $4,135 (2008-12-01) | **9.3×** |
| 2020 COVID Replay (Feb - Apr 2020) | -$2,142 | $15,152 | $4,838 (2020-03-12) | **7.5×** |
| Hypothetical: Equity Crash (-30/+5/+10) | $7,500 | $7,500 | $7,500 | 3.7× |
| Hypothetical: Stagflation (-15/-20/+20) | $6,000 | $6,000 | $6,000 | 3.0× |
| Hypothetical: Correlation Breakdown (-25/-15/-10) | $17,500 | $17,500 | $17,500 | **8.7×** |

Three observations:

**(a)** The 2020 COVID *total loss* is negative because the V-shaped policy response (Fed QE, rate cuts, emergency facilities) drove a complete recovery by April 30. The peak drawdown of $15.2K is the meaningful stress number — the cumulative path is misleading without context.

**(b)** The 2008 GFC and Correlation Breakdown hypothetical produce nearly identical peak losses ($18.7K vs $17.5K), both approximately 9× single-day VaR. The qualitative finding: **the failure of cross-asset correlation produces stress losses on par with the worst empirical crisis on record.**

**(c)** Stress losses are 7-10× single-day VaR for diversified portfolios under realistic crisis scenarios. Capital sized purely off VaR would be insufficient for any of these scenarios. This is the mathematical justification for FRTB's stressed-ES requirement and CCAR's separate stress capital framework.

---

## What I learned

**The fat-tail story shows up in three places, all consistently.** Day 1 measured excess kurtosis of 15.4 on SPY. Day 2 measured a 22% gap between historical and Gaussian VaR. Day 3 measured 66-78% excess exception rates on Gaussian backtests. **All three are the same finding viewed through different lenses** — Gaussian models systematically underestimate tail risk. The consistency is the validation; if just one of the three showed fat tails and the others didn't, I'd suspect a bug.

**Kupiec and Christoffersen catch different failure modes.** Historical VaR passing Kupiec at p=0.051 but failing Christoffersen at p<0.001 was the most diagnostic single result in the project. A model can have the right *average* exception rate while breaking down catastrophically during stress periods — and the distinction is the difference between "passable" and "dangerous." This is the kind of nuance bank Model Validation teams catch routinely.

**The 2020 COVID stress is fundamentally different from 2008.** 2008 was a slow-motion credit crisis with a longer drawdown duration (73 loss days) and persistent losses. 2020 was a sharp liquidity shock followed by the most aggressive central-bank response in modern history, producing a higher single-day loss ($4.8K vs $4.1K) but full recovery by April 30. A risk framework that focuses only on cumulative losses would underestimate 2020; one that focuses on peak drawdown captures both correctly.

**Stress losses are 7-10× single-day VaR for diversified portfolios.** This empirical regularity holds across both historical replays and hypothetical scenarios. It's the mathematical reason capital frameworks layer stress testing on top of statistical VaR — and the reason CCAR exists.

**Correlation breakdown is a worse scenario than any single-asset shock.** The hypothetical scenario where SPY/TLT/GLD all fell together produced a $17.5K loss, comparable to the actual 2008 GFC peak. The takeaway: portfolio risk isn't primarily about individual asset shocks; it's about whether diversification holds in the regime that actually occurs. This is exactly the dynamic that played out in March 2020 (when stocks, bonds, and gold all fell during the liquidity panic before policy intervention).

---

## Running it

```bash
pip install -r requirements.txt

# Pull data (~30 sec first time)
python3 data.py

# Compute VaR & ES under three methods
python3 var.py

# Rolling backtest with Kupiec + Christoffersen + Basel traffic light
python3 backtest.py    # takes ~1-2 minutes

# Stress test: 2008 + 2020 + 3 hypothetical scenarios
python3 stress_test.py

# Full test suite
pytest -v
```

---

## Tests

15 tests across five categories:

| Category | What's verified |
|---|---|
| Data integrity | Three assets, returns in decimal form, portfolio P&L correctly dollar-weighted |
| VaR sanity | All methods produce positive losses; ES ≥ VaR; Gaussian ES/VaR ratio matches `φ(z_α)/α/z_α = 1.145` exactly; VaR monotonic in confidence level |
| Method agreement | MC Gaussian and Variance-Covariance agree within MC sampling error |
| Backtest correctness | Kupiec LR non-negative; rejects high exception rates; Christoffersen detects clustering; p-values bounded; Basel zones correctly classified |
| Stress test validity | Closed-form hypothetical shocks match by-hand math; stress/VaR ratios in plausible 1-50× range |

---

## Limitations and future work

1. **Linear portfolio only.** The portfolio is three ETFs with no derivatives. A real risk system handles non-linear payoffs (options, structured products) via full Monte Carlo revaluation or quadratic approximation (delta-gamma VaR). Adding even one option position to this portfolio would require extending the variance-covariance method.

2. **Gaussian Monte Carlo only.** A natural extension is multivariate-t simulation, which captures fat tails directly. Empirically the t with 5-7 degrees of freedom fits equity returns substantially better than Gaussian, especially in stress periods.

3. **No volatility scaling.** Each VaR uses an estimation window with equal weights. EWMA or GARCH would weight recent observations more heavily and react faster to regime changes — RiskMetrics-style. This would partially address the clustering failure mode identified in backtesting.

4. **No Shanken-style measurement error correction.** Backtests don't account for parameter uncertainty in the underlying VaR estimates themselves.

5. **No multi-day VaR.** Basel III specifies 10-day VaR via `√10` scaling. The scaling rule fails during stress periods (volatility autocorrelates), making true multi-day MC more appropriate.

6. **No reverse stress testing.** Required by EU regulators since CRD IV (2014). The natural extension: search for shocks that produce a 50% portfolio loss and characterize the conditions.

7. **Stress scenarios are subjective.** The three hypotheticals were chosen for illustrative value; a real stress testing program would use CCAR-style supervisor-defined scenarios or sample from extreme-value distributions.

---

## References

- Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models," *Journal of Derivatives*, 3.
- Christoffersen, P. (1998). "Evaluating Interval Forecasts," *International Economic Review*, 39. — the conditional coverage test.
- Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). "Coherent Measures of Risk," *Mathematical Finance*, 9. — coherent risk measures and the case against VaR.
- Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market Risk* (FRTB).
- JP Morgan / Reuters (1996). *RiskMetrics — Technical Document*, 4th ed.
- Federal Reserve Board (2024). *Supervisory Stress Test Methodology*.

---

Part of a broader quantitative finance project portfolio:
- Project 1: [Black-Scholes pricer with Greeks and IV solvers](https://github.com/Dev2943/bsm-pricer)
- Project 2: [Monte Carlo with variance reduction and exotic payoffs](https://github.com/Dev2943/mc-pricer)
- Project 3: [Multi-factor equity model with Fama-MacBeth and momentum backtest](https://github.com/Dev2943/factor-model)
- Project 4 (this): VaR & ES calculator with backtesting and stress testing
