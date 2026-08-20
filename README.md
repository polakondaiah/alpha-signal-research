# Alpha Signal Research Notebook

**For:** Quantitative Researcher, WorldQuant (Mumbai) — extends flagship data pipeline with IC-based alpha evaluation (`IC = rank correlation(signal, forward return)`).

## Description
A research notebook that systematically evaluates candidate alpha signals on NSE data. It builds six signals (momentum 10/20/60, mean-reversion z-score, volatility and volume), computes daily rank Information Coefficients against 1/5/20-day forward returns, and analyzes IC decay and stability across sub-periods. The workflow follows a structured alpha research process (hypothesis → factor construction → IC/IR estimation → robustness checks) and outputs tables and decay plots for comparison. Built to show signal research hygiene rather than a single backtest P&L.

## Data
Reuses `../app-0001-.../project/data/prices.csv` (8 NSE large-caps, 2020-2026, 13,856 rows). No new fetch needed — same universe/pipeline.

## Candidate Signals (6)
| Signal | Definition |
|---|---|
| `mom_10/20/60` | Return over 10/20/60 days |
| `z_20` | Mean-reversion z-score `(Close - MA20)/STD20` |
| `vol_20` | Realized vol (20d std of daily returns) |
| `vol_z_20` | Volume z-score |

## Evaluation: Information Coefficient (IC)
For each signal × horizon (1/5/20d forward), compute **daily rank IC** (Spearman correlation cross-sectionally) then mean/std/IR.

### IC Table (synthetic data)
| signal | IC_1 mean | IC_5 mean | IC_20 mean | IR (1/5/20) |
|---|---|---|---|---|
| mom_10 | -0.003 | -0.005 | -0.007 | -0.01/-0.01/-0.02 |
| mom_20 | -0.006 | -0.015 | **+0.009** | -0.02/-0.04/+0.02 |
| mom_60 | +0.001 | -0.021 | -0.014 | 0.00/-0.05/-0.03 |
| z_20 | -0.011 | -0.015 | -0.004 | -0.03/-0.04/-0.01 |
| vol_20 | +0.001 | **+0.027** | **+0.053** | 0.00/+0.07/+0.13 |

`vol_z_20` NaN on synthetic (volume GBM lacks cross-sectional dispersion) — noted as invalid signal, removed in next iteration.

### IC Decay
Plot `results/ic_decay.png` — `vol_20` strengthens with horizon (vol predicts vol, not return direction), others flat near zero (synthetic has no real alpha — expected).

### IC Stability (5d horizon, 3 subperiods)
| signal | 2020-2022 | 2022-2024 | 2024-2026 |
|---|---|---|---|
| mom_10 | -0.051 | +0.011 | +0.025 |
| vol_20 | +0.014 | +0.031 | +0.035 — most stable |

No signal shows stable IC across all periods on synthetic data — correct conclusion to report, not overfit. With real NSE data, stability check guards overfitting.

## Distinction: IC vs Backtest Sharpe
- **IC** = predictive power of *signal* before portfolio construction (research question: does this rank stocks?).
- **Sharpe** = performance after position sizing, costs, risk — conflates signal + execution. WorldQuant posting asks for "identify high-quality predictive signals (alphas)" via structured research — IC is the standard language.

## Reproduce
```bash
python3 alpha_research.py
# outputs results/ic_table.csv, ic_stability.csv, ic_decay.png
```

## Limitations & Overfitting Guard
- Small universe (8) → high IC variance; widen to 50+ NSE stocks with more time.
- Synthetic data has no true alpha — table correctly shows near-zero ICs; do not claim profitability.
- Next step: out-of-sample split, multiple-testing correction (Bonferroni/DefDR).
