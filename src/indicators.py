"""Technical indicators. All are causal: value at bar t uses only data up to and
including bar t (rolling / ewm on close/high/low). No forward-looking windows.

Signals derived from these MUST still be executed at t+1 open (handled by the
backtest engine) to avoid using bar t's close to trade at bar t's close.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1)
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s: pd.Series, n: int) -> pd.Series:
    half = max(1, int(n / 2))
    sqrt_n = max(1, int(np.sqrt(n)))
    return wma(2 * wma(s, half) - wma(s, n), sqrt_n)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(s: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(df: pd.DataFrame, k=14, d=3, smooth=3):
    ll = df["low"].rolling(k, min_periods=k).min()
    hh = df["high"].rolling(k, min_periods=k).max()
    raw_k = 100 * (df["close"] - ll) / (hh - ll)
    k_line = raw_k.rolling(smooth, min_periods=smooth).mean()
    d_line = k_line.rolling(d, min_periods=d).mean()
    return k_line, d_line


def bollinger(s: pd.Series, n=20, k=2.0):
    mid = sma(s, n)
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def keltner(df: pd.DataFrame, n=20, mult=2.0):
    mid = ema(df["close"], n)
    rng = atr(df, n)
    return mid, mid + mult * rng, mid - mult * rng


def donchian(df: pd.DataFrame, n=20):
    # Prior-N (exclude current bar) to make breakout comparisons causal.
    upper = df["high"].rolling(n, min_periods=n).max().shift(1)
    lower = df["low"].rolling(n, min_periods=n).min().shift(1)
    return upper, lower


def adx(df: pd.DataFrame, n=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_n = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_line = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return adx_line, plus_di, minus_di


def cci(df: pd.DataFrame, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(n, min_periods=n).mean()
    mad = tp.rolling(n, min_periods=n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)


def supertrend(df: pd.DataFrame, period=10, mult=3.0):
    hl2 = (df["high"] + df["low"]) / 2
    rng = atr(df, period)
    upper = hl2 + mult * rng
    lower = hl2 - mult * rng
    st = pd.Series(index=df.index, dtype=float)
    dir_ = pd.Series(index=df.index, dtype=float)
    close = df["close"].to_numpy()
    up = upper.to_numpy(); lo = lower.to_numpy()
    st_arr = np.full(len(df), np.nan); dir_arr = np.full(len(df), np.nan)
    prev_st = np.nan; prev_dir = 1.0
    for i in range(len(df)):
        if np.isnan(up[i]):
            continue
        if np.isnan(prev_st):
            prev_st = lo[i]; prev_dir = 1.0
        if prev_dir == 1.0:
            cur = max(lo[i], prev_st) if not np.isnan(prev_st) else lo[i]
            if close[i] < cur:
                cur_dir = -1.0; cur = up[i]
            else:
                cur_dir = 1.0
        else:
            cur = min(up[i], prev_st) if not np.isnan(prev_st) else up[i]
            if close[i] > cur:
                cur_dir = 1.0; cur = lo[i]
            else:
                cur_dir = -1.0
        st_arr[i] = cur; dir_arr[i] = cur_dir
        prev_st = cur; prev_dir = cur_dir
    return pd.Series(st_arr, index=df.index), pd.Series(dir_arr, index=df.index)


def rolling_high(df: pd.DataFrame, n: int) -> pd.Series:
    return df["high"].rolling(n, min_periods=n).max().shift(1)


def rolling_low(df: pd.DataFrame, n: int) -> pd.Series:
    return df["low"].rolling(n, min_periods=n).min().shift(1)


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hh = df["high"].rolling(n, min_periods=n).max()
    ll = df["low"].rolling(n, min_periods=n).min()
    return -100 * (hh - df["close"]) / (hh - ll)


def roc(s: pd.Series, n: int = 10) -> pd.Series:
    return 100 * (s - s.shift(n)) / s.shift(n)


def momentum(s: pd.Series, n: int = 10) -> pd.Series:
    return s - s.shift(n)


def aroon(df: pd.DataFrame, n: int = 14):
    # periods since the highest high / lowest low over the last n bars (causal)
    up = df["high"].rolling(n + 1, min_periods=n + 1).apply(
        lambda x: 100 * (n - (len(x) - 1 - int(np.argmax(x)))) / n, raw=True)
    dn = df["low"].rolling(n + 1, min_periods=n + 1).apply(
        lambda x: 100 * (n - (len(x) - 1 - int(np.argmin(x)))) / n, raw=True)
    return up, dn


def vwap_session(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.normalize()
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def parabolic_sar(df: pd.DataFrame, af0=0.02, af_step=0.02, af_max=0.2) -> pd.Series:
    high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    n = len(df)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=df.index)
    up = True
    af = af0
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, n):
        prev = sar[i - 1]
        sar[i] = prev + af * (ep - prev)
        if up:
            sar[i] = min(sar[i], low[i - 1], low[max(0, i - 2)])
            if high[i] > ep:
                ep = high[i]; af = min(af + af_step, af_max)
            if low[i] < sar[i]:
                up = False; sar[i] = ep; ep = low[i]; af = af0
        else:
            sar[i] = max(sar[i], high[i - 1], high[max(0, i - 2)])
            if low[i] < ep:
                ep = low[i]; af = min(af + af_step, af_max)
            if high[i] > sar[i]:
                up = True; sar[i] = ep; ep = high[i]; af = af0
    return pd.Series(sar, index=df.index)


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    return (c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    return (c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1))
