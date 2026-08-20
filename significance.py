"""
Statistical significance for the IC study -- an IC table with no
significance testing invites two classic mistakes: (1) trusting a mean IC
that's indistinguishable from noise, and (2) data-snooping across the
signal x horizon grid without correcting for how many comparisons were
made.

1. Newey-West t-stat per (signal, horizon) IC series -- overlapping-horizon
   forward returns (h=5, h=20) induce autocorrelation in the daily IC
   series even if the underlying signal has a real, stable edge, so a
   naive iid t-test overstates significance. maxlags = horizon-1 accounts
   for the known overlap structure directly (Newey & West, 1987).
2. Benjamini-Hochberg FDR control across all signal x horizon tests (18
   in this project's default grid) -- with that many comparisons, ~1 in 20
   "significant at 5%" results is expected by chance alone; BH control
   reports which survive after adjusting for that.
"""
import numpy as np
import statsmodels.api as sm


def newey_west_ic_tstat(ic_series: np.ndarray, horizon: int) -> dict:
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    n = len(ic)
    if n < 5:
        return {"n": n, "t_stat": np.nan, "p_value": np.nan, "mean_ic": np.nan}
    maxlags = max(horizon - 1, 0)
    X = np.ones((n, 1))
    model = sm.OLS(ic, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}) if maxlags > 0 else sm.OLS(ic, X).fit()
    return {
        "n": n,
        "maxlags": maxlags,
        "mean_ic": float(model.params[0]),
        "nw_std_err": float(model.bse[0]),
        "t_stat": float(model.tvalues[0]),
        "p_value": float(model.pvalues[0]),
    }


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Returns a boolean mask of which p-values are significant after BH
    (1995) false-discovery-rate control at the given alpha."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresholds
    if not below.any():
        return [False] * n
    max_rank = np.max(np.where(below)[0])
    reject = np.zeros(n, dtype=bool)
    reject[order[:max_rank + 1]] = True
    return reject.tolist()
