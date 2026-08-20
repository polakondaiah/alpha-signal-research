"""
Correctness tests for the IC study: significance utilities on synthetic
data with a known answer, and a sanity check that a signal engineered to
be genuinely predictive scores a high positive IC while pure noise scores
approximately zero.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from significance import newey_west_ic_tstat, benjamini_hochberg
from alpha_research import ic_series


def test_benjamini_hochberg_all_null():
    rng = np.random.default_rng(0)
    pvals = rng.uniform(0, 1, 50).tolist()  # under the null, uniform p-values
    flags = benjamini_hochberg(pvals, alpha=0.05)
    # expect very few (ideally 0) false discoveries from pure noise
    assert sum(flags) <= 5


def test_benjamini_hochberg_detects_real_signal():
    pvals = [0.001, 0.002, 0.5, 0.6, 0.7, 0.8, 0.9]
    flags = benjamini_hochberg(pvals, alpha=0.05)
    assert flags[0] and flags[1]
    assert not any(flags[2:])


def test_newey_west_ic_tstat_on_noise():
    rng = np.random.default_rng(1)
    ic = rng.normal(0, 0.05, 300)
    result = newey_west_ic_tstat(ic, horizon=5)
    assert abs(result["mean_ic"]) < 0.02
    assert result["p_value"] > 0.01  # should not spuriously reject on pure noise


def test_ic_series_detects_real_relationship():
    """Construct a signal that's exactly the forward return (plus tiny
    noise) and confirm IC comes out strongly positive -- validates the
    Spearman cross-sectional IC computation itself, not just the stats
    layer around it."""
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows = []
    for d in dates:
        for tk in range(10):
            fwd = rng.normal(0, 0.02)
            signal = fwd + rng.normal(0, 0.001)  # near-perfect predictor
            rows.append({"Date": d, "Ticker": f"T{tk}", "sig": signal, "fwd_1": fwd})
    df = pd.DataFrame(rows)
    ics = ic_series(df, "sig", 1)
    assert np.mean(ics) > 0.8


def test_ic_series_near_zero_for_pure_noise():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows = []
    for d in dates:
        for tk in range(10):
            rows.append({"Date": d, "Ticker": f"T{tk}",
                         "sig": rng.normal(), "fwd_1": rng.normal()})
    df = pd.DataFrame(rows)
    ics = ic_series(df, "sig", 1)
    assert abs(np.mean(ics)) < 0.3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
