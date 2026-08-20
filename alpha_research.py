"""
Alpha Signal Research: IC vs forward returns at 1/5/20 days, IC decay, stability.
Reuses flagship data at ../app-0001/.../project/data/prices.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

_candidates = [
    Path(__file__).parent.parent.parent / "app-0001-nk-securities-quant-researcher" / "backtested-strategy-engine" / "data" / "prices.csv",
    Path(__file__).parent.parent.parent / "app-0001-nk-securities-quant-researcher" / "project" / "data" / "prices.csv",
]
DATA = next((p for p in _candidates if p.exists()), _candidates[0])
if not DATA.exists():
    DATA = Path(__file__).parent / "data" / "prices.csv"

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

SIGNALS = ["mom_10","mom_20","mom_60","z_20","vol_20","vol_z_20"]
HORIZONS = [1,5,20]

def load():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df.sort_values(["Ticker","Date"])
    print(f"[load] {len(df):,} rows {df.Ticker.nunique()} tickers {DATA}")
    return df

def add_forwards(df):
    for h in HORIZONS:
        # forward return h days ahead
        df[f"fwd_{h}"] = df.groupby("Ticker")["Close"].transform(lambda s: s.pct_change(h).shift(-h))
    return df

def ic_series(df, signal, horizon):
    col = f"fwd_{horizon}"
    # rank IC per day: spearman between signal and forward return cross-sectionally
    ics = []
    for date, sub in df.groupby("Date"):
        sub = sub[[signal, col]].dropna()
        if len(sub) < 4:
            continue
        ic, _ = spearmanr(sub[signal], sub[col])
        if not np.isnan(ic):
            ics.append(ic)
    return np.array(ics)


def out_of_sample_validation(df, selection_horizon=5):
    """
    Model-selection discipline: pick the "best" signal using only the
    first two-thirds of history (by |IR|, the same metric used to rank
    signals in the headline table), then report that pre-selected
    signal's IC purely on the untouched final third. Doing selection and
    evaluation on the same data is the single most common way an IC study
    quietly overstates itself; this keeps selection and evaluation
    non-overlapping in time.
    """
    dates = sorted(df["Date"].unique())
    split = dates[: int(len(dates) * 2 / 3)]
    train, test = df[df["Date"].isin(split)], df[~df["Date"].isin(split)]

    train_scores = {}
    for sig in SIGNALS:
        ics = ic_series(train, sig, selection_horizon)
        ir = np.mean(ics) / np.std(ics) if len(ics) and np.std(ics) != 0 else np.nan
        train_scores[sig] = ir
    selected = max(train_scores, key=lambda s: abs(train_scores[s]) if not np.isnan(train_scores[s]) else -1)

    test_ics = ic_series(test, selected, selection_horizon)
    return {
        "selection_horizon": selection_horizon,
        "train_period": [str(split[0]), str(split[-1])],
        "test_period": [str(test["Date"].min()), str(test["Date"].max())],
        "train_ir_by_signal": {k: (float(v) if not np.isnan(v) else None) for k, v in train_scores.items()},
        "selected_signal": selected,
        "selected_ir_in_sample": float(train_scores[selected]),
        "out_of_sample_mean_ic": float(np.mean(test_ics)) if len(test_ics) else None,
        "out_of_sample_ir": float(np.mean(test_ics) / np.std(test_ics)) if len(test_ics) and np.std(test_ics) != 0 else None,
        "out_of_sample_n": int(len(test_ics)),
    }

def main():
    from signals import add_all_signals
    from significance import newey_west_ic_tstat, benjamini_hochberg
    df = load()
    df = add_all_signals(df)
    df = add_forwards(df)
    print(f"[signals] added {SIGNALS}")

    # compute IC stats + Newey-West significance per (signal, horizon)
    table = []
    sig_tests = []  # flat list for BH correction across the whole grid
    for sig in SIGNALS:
        row = {"signal": sig}
        for h in HORIZONS:
            ics = ic_series(df, sig, h)
            row[f"IC_{h}_mean"] = float(np.mean(ics)) if len(ics) else np.nan
            row[f"IC_{h}_std"] = float(np.std(ics)) if len(ics) else np.nan
            row[f"IC_{h}_IR"] = float(np.mean(ics)/np.std(ics)) if len(ics) and np.std(ics)!=0 else np.nan
            row[f"IC_{h}_n"] = int(len(ics))
            nw = newey_west_ic_tstat(ics, horizon=h)
            row[f"IC_{h}_nw_pvalue"] = nw["p_value"]
            sig_tests.append({"signal": sig, "horizon": h, "p_value": nw["p_value"]})
        table.append(row)
        print(f"{sig}: " + ", ".join([f"H{h} IC={row[f'IC_{h}_mean']:.4f} IR={row[f'IC_{h}_IR']:.2f} p={row[f'IC_{h}_nw_pvalue']:.3f}" for h in HORIZONS]))

    # Benjamini-Hochberg FDR control across all signal x horizon tests --
    # with 18 simultaneous tests, ~1 "significant at 5%" result is expected
    # by chance alone even under the null of no real signal anywhere.
    pvals = [t["p_value"] for t in sig_tests]
    valid_mask = [not np.isnan(p) for p in pvals]
    bh_flags = [False] * len(pvals)
    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    if valid_idx:
        bh_result = benjamini_hochberg([pvals[i] for i in valid_idx])
        for j, i in enumerate(valid_idx):
            bh_flags[i] = bh_result[j]
    for t, flag in zip(sig_tests, bh_flags):
        t["significant_after_bh"] = flag
    n_sig_raw = sum(1 for p in pvals if not np.isnan(p) and p < 0.05)
    n_sig_bh = sum(bh_flags)
    print(f"\n[significance] {n_sig_raw}/{len(pvals)} tests significant at raw p<0.05; "
          f"{n_sig_bh}/{len(pvals)} survive Benjamini-Hochberg FDR correction at 5%")

    pd.DataFrame(table).to_csv(OUT / "ic_table.csv", index=False)
    with open(OUT/"ic_table.json","w") as f:
        json.dump(table, f, indent=2)
    with open(OUT/"significance_tests.json", "w") as f:
        json.dump(sig_tests, f, indent=2)

    # Out-of-sample validation: select best signal on first 2/3, test on
    # untouched final 1/3 -- checks whether the headline IC table survives
    # honest, non-overlapping model selection.
    oos = out_of_sample_validation(df)
    with open(OUT / "out_of_sample_validation.json", "w") as f:
        json.dump(oos, f, indent=2)
    print(f"\n[out-of-sample] selected '{oos['selected_signal']}' on train "
          f"(IR={oos['selected_ir_in_sample']:.2f}) -> test IC={oos['out_of_sample_mean_ic']}, "
          f"IR={oos['out_of_sample_ir']}")

    # IC decay plot
    fig, ax = plt.subplots(figsize=(8,5))
    for r in table:
        ax.plot(HORIZONS, [r[f"IC_{h}_mean"] for h in HORIZONS], marker="o", label=r["signal"])
    ax.set_xlabel("Forward horizon (days)")
    ax.set_ylabel("Mean IC (rank corr)")
    ax.set_title("IC Decay across horizons")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"ic_decay.png", dpi=150)
    plt.close()

    # Stability: split into 3 subperiods
    dates = sorted(df["Date"].unique())
    n = len(dates)
    splits = [dates[:n//3], dates[n//3:2*n//3], dates[2*n//3:]]
    split_names = ["2020-2022","2022-2024","2024-2026"]
    stab = []
    for sig in SIGNALS:
        row = {"signal": sig}
        for name, dlist in zip(split_names, splits):
            sub = df[df["Date"].isin(dlist)]
            ics = ic_series(sub, sig, 5)  # 5d horizon for stability
            row[f"IC5_{name}"] = float(np.mean(ics)) if len(ics) else np.nan
        stab.append(row)
        print(f"stability {sig}: " + ", ".join([f"{k}={v:.4f}" for k,v in row.items() if k!="signal"]))

    pd.DataFrame(stab).to_csv(OUT/"ic_stability.csv", index=False)

    # Write markdown summary
    md = OUT / "summary.md"
    md.write_text(
        f"# IC Summary\n\nGenerated from {DATA}\n\n"
        f"## IC Table (mean ± std, with Newey-West p-values)\n" + pd.DataFrame(table).to_string(index=False) +
        f"\n\n## Significance\n{n_sig_raw}/{len(pvals)} tests significant at raw p<0.05; "
        f"{n_sig_bh}/{len(pvals)} survive Benjamini-Hochberg FDR correction at 5%.\n\n"
        f"## Stability (5d horizon)\n" + pd.DataFrame(stab).to_string(index=False) +
        f"\n\n## Out-of-sample validation (5d horizon)\n"
        f"Selected on train (first 2/3): **{oos['selected_signal']}** (in-sample IR={oos['selected_ir_in_sample']:.3f})\n"
        f"Test (final 1/3, untouched during selection): mean IC={oos['out_of_sample_mean_ic']}, IR={oos['out_of_sample_ir']}\n"
    )

    print(f"[done] results in {OUT.resolve()}")

if __name__ == "__main__":
    main()
