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

def main():
    from signals import add_all_signals
    df = load()
    df = add_all_signals(df)
    df = add_forwards(df)
    print(f"[signals] added {SIGNALS}")

    # compute IC stats
    table = []
    for sig in SIGNALS:
        row = {"signal": sig}
        ics_all = {}
        for h in HORIZONS:
            ics = ic_series(df, sig, h)
            ics_all[h] = ics
            row[f"IC_{h}_mean"] = float(np.mean(ics)) if len(ics) else np.nan
            row[f"IC_{h}_std"] = float(np.std(ics)) if len(ics) else np.nan
            row[f"IC_{h}_IR"] = float(np.mean(ics)/np.std(ics)) if len(ics) and np.std(ics)!=0 else np.nan
            row[f"IC_{h}_n"] = int(len(ics))
        table.append(row)
        print(f"{sig}: " + ", ".join([f"H{h} IC={row[f'IC_{h}_mean']:.4f} IR={row[f'IC_{h}_IR']:.2f}" for h in HORIZONS]))

    pd.DataFrame(table).to_csv(OUT / "ic_table.csv", index=False)
    with open(OUT/"ic_table.json","w") as f:
        json.dump(table, f, indent=2)

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
    md.write_text(f"# IC Summary\n\nGenerated from {DATA}\n\n## IC Table (mean ± std)\n" + pd.DataFrame(table).to_string(index=False) + "\n\n## Stability (5d horizon)\n" + pd.DataFrame(stab).to_string(index=False) + "\n")

    print(f"[done] results in {OUT.resolve()}")

if __name__ == "__main__":
    main()
