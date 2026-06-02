# MODEL VALIDATION MEMO

**Model:** Portfolio Value-at-Risk & Expected Shortfall Calculator
**Author:** Dev Golakiya
**Date:** May 2026
**Status:** Independent validation — for educational purposes
**Scope:** 1-day 99% VaR / ES on a 3-asset diversified portfolio (SPY/TLT/GLD)

---

## 1. Executive Summary

This validation reviews three competing VaR/ES methodologies — Historical Simulation, Variance-Covariance, and Monte Carlo (Gaussian) — applied to a $100K multi-asset portfolio over 2005-2026. The validation finds that **all three methodologies have material limitations**, but they fail in different and complementary ways:

- **Historical Simulation** is the only method passing Kupiec's POF test (p = 0.051) and Basel III's traffic-light test (4 exceptions in last 250 days = Green zone). However, it fails Christoffersen's independence test at p < 0.001 — exceptions cluster severely during 2008 and 2020 stress periods.

- **Variance-Covariance and Monte Carlo (Gaussian)** both produce 66-78% excess exception rates against the 1% target, failing Kupiec decisively. Both methods are currently in Basel III's Yellow zone, which would trigger a capital multiplier of approximately 3.4× under bank operational use.

**Recommendation:** **Historical Simulation is the only method approved for use as a primary daily risk metric**, conditional on supplementing it with stress testing (per Section 5 below) to address the clustering failure mode identified in backtesting. Variance-Covariance and Monte Carlo should be restricted to internal sensitivity analysis only.

---

## 2. Model Description

The model computes 1-day 99% VaR and Expected Shortfall on a portfolio of three ETFs:

| Asset | Notional | Weight |
|---|---|---|
| SPY (US equity) | $40,000 | 40% |
| TLT (long Treasuries) | $30,000 | 30% |
| GLD (gold) | $30,000 | 30% |
| **Total** | **$100,000** | **100%** |

Three methodologies are implemented:

1. **Historical Simulation**: empirical 1st percentile of past portfolio returns over a 500-day rolling window
2. **Variance-Covariance**: parametric Gaussian formula `2.326 × σ_p`, where `σ_p² = w'Σw`
3. **Monte Carlo (Gaussian)**: 100,000 multivariate normal draws from the historical covariance matrix

All methods report both VaR (the loss threshold) and ES (the conditional expected loss in the tail).

---

## 3. Methodology Comparison — Current Snapshot

At end of sample (May 2026), 500-day estimation window:

| Method | 1-day 99% VaR ($) | 1-day 99% ES ($) | ES/VaR Ratio |
|---|---|---|---|
| Historical Simulation | 2,015 | 2,477 | 1.23 |
| Variance-Covariance | 1,655 | 1,896 | 1.15 |
| Monte Carlo (Gaussian) | 1,560 | 1,796 | 1.15 |

**Key observations:**

1. Historical VaR exceeds Gaussian VaR by 22% — the empirical fat-tail effect even on a relatively calm 500-day window.
2. The Variance-Covariance ES/VaR ratio of 1.146 matches theoretical Gaussian (`φ(2.326)/0.01/2.326 = 1.145`), validating the closed-form implementation.
3. Monte Carlo and Variance-Covariance disagree by ~6%, well within MC sampling error for 100K simulations.

---

## 4. Backtesting Results — Full Sample (4,884 days)

| Method | Exceptions | Rate | Kupiec POF (p) | Christoffersen IND (p) | CC (p) | Basel Zone |
|---|---|---|---|---|---|---|
| Historical Simulation | 63 | 1.29% | 0.051 PASS | **0.000 FAIL** | 0.000 FAIL | Green |
| Variance-Covariance | 81 | 1.66% | 0.000 FAIL | 0.000 FAIL | 0.000 FAIL | Yellow |
| Monte Carlo (Gaussian) | 87 | 1.78% | 0.000 FAIL | 0.000 FAIL | 0.000 FAIL | Yellow |

**Diagnostic interpretation:**

- The Gaussian methods reject under both unconditional coverage AND independence. The 66-78% exception rate excess directly quantifies the Gaussian under-capture of fat tails.

- Historical Simulation has the right *total* count of exceptions but they are not independently distributed. Visual inspection of the exception time series confirms severe clustering in October-November 2008 and February-March 2020. This is the failure mode that brought down sophisticated VaR-based capital systems during both crises.

- The Basel traffic-light snapshot (last 250 days) reflects the current calm regime and is not predictive of stress regime behavior.

---

## 5. Stress Testing — Beyond the Statistical Tail

| Scenario | Total Loss | Peak Drawdown | Peak DD / VaR |
|---|---|---|---|
| 2008 GFC Replay | $5,744 | $18,697 | 9.3× |
| 2020 COVID Replay | -$2,142 | $15,152 | 7.5× |
| Hypothetical: Equity Crash (-30/+5/+10) | $7,500 | $7,500 | 3.7× |
| Hypothetical: Stagflation (-15/-20/+20) | $6,000 | $6,000 | 3.0× |
| Hypothetical: Correlation Breakdown (-25/-15/-10) | $17,500 | $17,500 | 8.7× |

**Notes:**

- The 2020 COVID total loss is negative because the Federal Reserve's V-shaped policy response drove full recovery by April 30. **Peak drawdown is the operationally meaningful number** — at the trough (March 23, 2020), the portfolio had lost $15K (7.5× single-day VaR).

- The Correlation Breakdown hypothetical scenario produces a peak loss matching the 2008 GFC, despite using less extreme individual-asset shocks. This is the empirical signature of the most dangerous regime for diversified portfolios: when cross-asset correlations spike toward 1 during stress, the diversification benefit collapses and a multi-asset portfolio behaves like a single concentrated position.

- **Stress losses run 3-9× single-day VaR.** This empirical regularity is the mathematical foundation for FRTB's stressed-ES requirement and CCAR's separate stress capital framework.

---

## 6. Limitations

1. **Linear portfolio only.** No derivatives; the model would require extension (delta-gamma or full revaluation Monte Carlo) for non-linear payoffs.
2. **Gaussian Monte Carlo only.** A multivariate-t implementation would capture fat tails directly and likely produce backtest results closer to historical simulation while retaining the simulation flexibility.
3. **Equal-weighted estimation window.** EWMA or GARCH would react faster to regime changes and partially address clustering. Not implemented.
4. **No reverse stress testing.** Standard EU practice since CRD IV (2014). The natural extension: search for shocks producing a 50% portfolio loss.
5. **Stress scenarios are subjective.** A production framework would use Fed-prescribed CCAR scenarios or extreme-value sampling.
6. **No multi-day VaR.** The √T scaling rule is implemented implicitly but not validated for the 10-day horizon Basel III requires.

---

## 7. Recommendation

**Approved for primary use:** Historical Simulation (Green zone, passes Kupiec) — with the limitation that the model's clustering failure mode requires supplementary stress testing for capital adequacy.

**Restricted to sensitivity analysis only:** Variance-Covariance and Monte Carlo (Gaussian) — Yellow zone with material POF failures and equivalent clustering issues.

**Required risk-management actions:**

1. Daily VaR reporting via Historical Simulation as primary, with the other two methods as model-risk benchmarks
2. Quarterly stress testing using both historical replays (2008, 2020) and forward-looking hypotheticals
3. Capital allocation sized to `max(VaR-based, stress-based)` — empirically, the stress-based number is 7-10× the VaR-based number for this portfolio
4. Monitor for regime changes via realized vs. predicted exception clustering; investigate any 3+ consecutive exception days

---

## 8. Sign-Off

This memo concludes the validation review. The model is approved for the limited scope specified in Section 7. The validation should be revisited annually or upon any material change in portfolio composition, asset universe, or market regime.

**Validator:** Dev Golakiya
**Date:** May 2026

---

*This is an educational artifact produced as part of a quantitative finance project portfolio. It mirrors the structure and language used by bank Model Risk Management groups but does not constitute investment advice or a regulatory filing.*
