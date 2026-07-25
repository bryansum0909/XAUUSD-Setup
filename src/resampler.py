"""Resample M1 (or tick-derived M1) OHLCV to higher timeframes.
Uses left-closed, left-labeled bars so a bar timestamped 10:05 covers [10:05, 10:10).
This keeps the bar's CLOSE aligned to information available at the bar's end.
"""
from __future__ import annotations
import pandas as pd

_RULE = {"M1": "1min", "M5": "5min", "M10": "10min", "M15": "15min", "M30": "30min",
         "H1": "1h", "H4": "4h", "D1": "1D", "W": "1W"}


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = _RULE[timeframe]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def resample_all(m1: pd.DataFrame, timeframes=("M5", "M30", "H1")) -> dict:
    return {tf: resample_ohlc(m1, tf) for tf in timeframes}
