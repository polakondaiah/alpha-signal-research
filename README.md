# Alpha Signal Research Notebook

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
For each signal × horizon (1/5/20d forward), compute **daily rank IC** (Spearman correlation cross-sectionally), mean/std/IR, and a **Newey-West significance test** on the IC series (`significance.py`) — a raw mean IC with no significance test invites over-reading noise as signal.

### IC Table (synthetic data, current run)
| signal | IC_1 (p) | IC_5 (p) | IC_20 (p) |
|---|---|---|---|
| mom_10 | -0.001 (0.94) | -0.018 (0.21) | -0.028 (0.24) |
| mom_20 | -0.001 (0.89) | -0.024 (0.13) | -0.032 (0.28) |
| mom_60 | -0.005 (0.62) | -0.016 (0.33) | -0.031 (0.31) |
| z_20 | +0.006 (0.52) | -0.012 (0.38) | -0.029 (0.24) |
| vol_20 | -0.001 (0.95) | +0.012 (0.44) | +0.049 (0.07) |
| vol_z_20 | -0.013 (0.17) | -0.010 (0.24) | -0.016 (**0.03**) |

Full table with std/IR/n: `results/ic_table.csv`. (`vol_z_20` previously returned all-NaN because the upstream synthetic-data generator produced a constant Volume column per ticker, std=0 → division by zero — traced to `app-0001-.../fetch_data.py`'s volume model and fixed there with an independent RNG stream so it doesn't perturb the price series; volume now has real day-to-day and cross-sectional variation.)

### Significance: 18 tests, multiple-comparison correction
1 of 18 signal×horizon tests is significant at raw p<0.05 (`vol_z_20` at H20, p=0.033) — but **0 of 18 survive Benjamini-Hochberg FDR correction** at 5%. With 18 simultaneous tests, ~1 false positive is expected by chance alone even under the null of no real signal anywhere; this is exactly that expected false-positive rate, not evidence of a real vol_z_20/H20 effect. Correctly reporting "no signal survives correction" here — rather than highlighting the one raw-significant cell — is the point of running BH in the first place. Full test-by-test detail: `results/significance_tests.json`.

### IC Decay
Plot `results/ic_decay.png` — `vol_20` strengthens with horizon (vol predicts vol, not return direction; expected, not alpha), others flat near zero.

### IC Stability (5d horizon, 3 subperiods)
No signal is significant *and* stable across periods and after multiple-testing correction on synthetic data — the honest conclusion, not a data-mined one. Full table: `results/ic_stability.csv`.

### Out-of-sample validation (`out_of_sample_validation`, `results/out_of_sample_validation.json`)
Selecting the best signal by |IR| on the first 2/3 of history and evaluating *only that pre-selected signal* on the untouched final 1/3 — the standard guard against the single most common IC-study mistake (selecting and evaluating on the same data). Current run: selected `vol_20` in-sample (IR=+0.10), which then scored IR=-0.10 out-of-sample — a sign flip, i.e. the in-sample "best" signal does **not** replicate out-of-sample, consistent with the "no real signal" conclusion above and a useful illustration of exactly the failure mode this check exists to catch.

## Distinction: IC vs Backtest Sharpe
- **IC** = predictive power of *signal* before portfolio construction (research question: does this rank stocks?).
- **Sharpe** = performance after position sizing, costs, risk — conflates signal + execution. IC is the standard language for alpha research.

## Tests (`tests/test_alpha_research.py`)
5 pytest cases: Benjamini-Hochberg on pure-noise p-values (should reject almost nothing) and on a mix with two genuinely small p-values (should catch exactly those two); Newey-West IC t-test on pure noise (shouldn't spuriously reject); IC computation validated against a signal engineered to be a near-perfect predictor (should score IC > 0.8) and against pure noise (should score near zero). Run: `python3 -m pytest tests/ -v`.

## Reproduce
```bash
python3 alpha_research.py
# outputs results/{ic_table.csv, ic_stability.csv, ic_decay.png,
#                   significance_tests.json, out_of_sample_validation.json, summary.md}
python3 -m pytest tests/ -v
```

## Limitations & Overfitting Guard
- Small universe (8) → high IC variance; widen to 50+ NSE stocks with more time.
- Synthetic data has no true alpha — table correctly shows near-zero, statistically insignificant ICs after correction; do not claim profitability.
- Out-of-sample validation here selects among 6 *pre-specified* signals, not a full walk-forward re-fit of any signal's internal parameters (e.g. the z-score/momentum windows are fixed, not tuned) — a signal with tunable hyperparameters would need in-fold tuning to get the same protection this check provides for signal *selection*.
