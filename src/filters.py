"""Causal, lookahead-safe trade filters for 2-component combinations.

The subtle one is the higher-timeframe (HTF) trend filter: an HTF bar labelled
T (left-closed, covering [T, T+Δ)) only *closes* at T+Δ, so its indicator value
must not be visible to any base bar before T+Δ. We therefore shift the HTF series
forward by one HTF bar and forward-fill onto the base index — a base bar can only
ever see HTF bars that have fully closed at or before its own timestamp.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .resampler import resample_ohlc
from . import indicators as ind


def htf_trend(base_df: pd.DataFrame, htf_tf: str, fast: int = 50, slow: int = 200) -> pd.Series:
    """+1 uptrend / -1 downtrend / 0 none, aligned to base index, lookahead-safe."""
    htf = resample_ohlc(base_df, htf_tf)
    ef, es = ind.ema(htf["close"], fast), ind.ema(htf["close"], slow)
    trend = pd.Series(np.where(ef > es, 1.0, np.where(ef < es, -1.0, 0.0)), index=htf.index)
    trend = trend.shift(1)                       # value known only AFTER the bar closes
    aligned = trend.reindex(base_df.index.union(trend.index)).ffill().reindex(base_df.index)
    return aligned.fillna(0.0)


def session_mask(base_df: pd.DataFrame, a: int, b: int) -> pd.Series:
    """True where bar hour (UTC) in [a, b). Uses the bar's own timestamp only."""
    hr = base_df.index.hour
    return pd.Series((hr >= a) & (hr < b), index=base_df.index)


def adx_mask(base_df: pd.DataFrame, thr: float = 25.0, above: bool = True, n: int = 14) -> pd.Series:
    a, _, _ = ind.adx(base_df, n)
    return (a > thr) if above else (a < thr)


def atr_expansion_mask(base_df: pd.DataFrame, n: int = 14, lookback: int = 100) -> pd.Series:
    a = ind.atr(base_df, n)
    med = a.rolling(lookback, min_periods=lookback).median()
    return (a > med).fillna(False)


def atr_contraction_mask(base_df: pd.DataFrame, n: int = 14, lookback: int = 100) -> pd.Series:
    a = ind.atr(base_df, n)
    med = a.rolling(lookback, min_periods=lookback).median()
    return (a < med).fillna(False)
