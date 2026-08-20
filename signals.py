"""Candidate alpha signals for IC evaluation."""
import pandas as pd
import numpy as np

def momentum(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(window)

def mean_rev_z(close: pd.Series, window: int = 20) -> pd.Series:
    ma = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    return (close - ma) / std

def vol_signal(close: pd.Series, window: int = 20) -> pd.Series:
    ret = close.pct_change()
    return ret.rolling(window).std()

def volume_z(volume: pd.Series, window: int = 20) -> pd.Series:
    ma = volume.rolling(window).mean()
    std = volume.rolling(window).std(ddof=0)
    return (volume - ma) / std

def add_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker","Date"]).copy()
    g = df.groupby("Ticker")
    df["mom_10"] = g["Close"].transform(lambda s: momentum(s, 10))
    df["mom_20"] = g["Close"].transform(lambda s: momentum(s, 20))
    df["mom_60"] = g["Close"].transform(lambda s: momentum(s, 60))
    df["z_20"] = g["Close"].transform(lambda s: mean_rev_z(s, 20))
    df["vol_20"] = g["Close"].transform(lambda s: vol_signal(s, 20))
    df["vol_z_20"] = g["Volume"].transform(lambda s: volume_z(s, 20))
    return df
