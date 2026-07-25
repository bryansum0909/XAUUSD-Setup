"""Strategy library. Each strategy returns a dict with numpy arrays aligned to df:
    entries : {+1,-1,0} decided at CLOSE of each bar (engine executes at t+1 open)
    sl_dist : stop distance in price dollars
    tp_dist : target distance in price dollars (np.nan allowed)

Exits are standardised as ATR-based unless a strategy overrides. RR sets tp = rr*sl.
Signals use only causal indicator values; the engine enforces the t+1 fill.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import indicators as ind


def _atr_exits(df, atr_period=14, sl_mult=1.5, rr=1.5):
    a = ind.atr(df, atr_period).to_numpy(float)
    sl = sl_mult * a
    tp = rr * sl
    return sl, tp


def _pack(entries, sl, tp):
    return dict(entries=np.asarray(entries, float),
                sl_dist=np.asarray(sl, float),
                tp_dist=np.asarray(tp, float))


# --------------------------------------------------------------------------
# BASELINES  (validate the engine; not expected to be edges)
# --------------------------------------------------------------------------
def baseline_random(df, seed=0, p=0.02, sl_mult=1.5, rr=1.5, atr_period=14):
    rng = np.random.default_rng(seed)
    u = rng.random(len(df))
    entries = np.zeros(len(df))
    entries[u < p] = 1
    entries[u > 1 - p] = -1
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def baseline_ma_direction(df, n=50, sl_mult=1.5, rr=1.5, atr_period=14):
    ma = ind.sma(df["close"], n)
    up = df["close"] > ma
    prev = up.shift(1)
    entries = np.where(up & ~prev.fillna(False), 1,
                       np.where(~up & prev.fillna(False), -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def baseline_donchian(df, n=20, sl_mult=1.5, rr=1.5, atr_period=14):
    up = ind.rolling_high(df, n); lo = ind.rolling_low(df, n)
    long_sig = df["close"] > up
    short_sig = df["close"] < lo
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def baseline_bb_meanrev(df, n=20, k=2.0, sl_mult=1.5, rr=1.0, atr_period=14):
    mid, up, lo = ind.bollinger(df["close"], n, k)
    long_sig = df["close"] < lo
    short_sig = df["close"] > up
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


# --------------------------------------------------------------------------
# SINGLE-INDICATOR STRATEGIES
# --------------------------------------------------------------------------
def ema_cross(df, fast=20, slow=50, sl_mult=1.5, rr=1.5, atr_period=14, mode="trend"):
    ef, es = ind.ema(df["close"], fast), ind.ema(df["close"], slow)
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))
    if mode == "trend":
        entries = np.where(cross_up, 1, np.where(cross_dn, -1, 0))
    else:  # mean-reversion interpretation (fade the cross)
        entries = np.where(cross_up, -1, np.where(cross_dn, 1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def rsi_meanrev(df, n=14, os=30, ob=70, sl_mult=1.5, rr=1.0, atr_period=14):
    r = ind.rsi(df["close"], n)
    long_sig = (r > os) & (r.shift(1) <= os)      # cross up out of oversold
    short_sig = (r < ob) & (r.shift(1) >= ob)     # cross down out of overbought
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def rsi_momentum(df, n=14, mid=50, sl_mult=1.5, rr=1.5, atr_period=14):
    r = ind.rsi(df["close"], n)
    long_sig = (r > mid) & (r.shift(1) <= mid)
    short_sig = (r < mid) & (r.shift(1) >= mid)
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def macd_cross(df, fast=12, slow=26, signal=9, sl_mult=1.5, rr=1.5, atr_period=14):
    m, s, _ = ind.macd(df["close"], fast, slow, signal)
    cross_up = (m > s) & (m.shift(1) <= s.shift(1))
    cross_dn = (m < s) & (m.shift(1) >= s.shift(1))
    entries = np.where(cross_up, 1, np.where(cross_dn, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def bb_breakout(df, n=20, k=2.0, sl_mult=1.5, rr=1.5, atr_period=14):
    mid, up, lo = ind.bollinger(df["close"], n, k)
    long_sig = (df["close"] > up) & (df["close"].shift(1) <= up.shift(1))
    short_sig = (df["close"] < lo) & (df["close"].shift(1) >= lo.shift(1))
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def donchian_breakout(df, n=20, sl_mult=1.5, rr=2.0, atr_period=14):
    up = ind.rolling_high(df, n); lo = ind.rolling_low(df, n)
    long_sig = (df["close"] > up)
    short_sig = (df["close"] < lo)
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def supertrend_flip(df, period=10, mult=3.0, sl_mult=1.5, rr=2.0, atr_period=14):
    _, d = ind.supertrend(df, period, mult)
    flip_up = (d > 0) & (d.shift(1) <= 0)
    flip_dn = (d < 0) & (d.shift(1) >= 0)
    entries = np.where(flip_up, 1, np.where(flip_dn, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def stochastic_meanrev(df, k=14, d=3, smooth=3, os=20, ob=80, sl_mult=1.5, rr=1.0, atr_period=14):
    kk, dd = ind.stochastic(df, k, d, smooth)
    long_sig = (kk > os) & (kk.shift(1) <= os)
    short_sig = (kk < ob) & (kk.shift(1) >= ob)
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def cci_meanrev(df, n=20, thr=100, sl_mult=1.5, rr=1.0, atr_period=14):
    c = ind.cci(df, n)
    long_sig = (c > -thr) & (c.shift(1) <= -thr)
    short_sig = (c < thr) & (c.shift(1) >= thr)
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def psar_flip(df, af0=0.02, af_max=0.2, sl_mult=1.5, rr=2.0, atr_period=14):
    sar = ind.parabolic_sar(df, af0=af0, af_max=af_max)
    below = df["close"] > sar
    flip_up = below & ~below.shift(1).fillna(False)
    flip_dn = ~below & below.shift(1).fillna(False)
    entries = np.where(flip_up, 1, np.where(flip_dn, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def aroon_cross(df, n=14, sl_mult=1.5, rr=1.5, atr_period=14):
    up, dn = ind.aroon(df, n)
    cu = (up > dn) & (up.shift(1) <= dn.shift(1))
    cd = (up < dn) & (up.shift(1) >= dn.shift(1))
    entries = np.where(cu, 1, np.where(cd, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def williams_meanrev(df, n=14, os=-80, ob=-20, sl_mult=1.5, rr=1.0, atr_period=14):
    w = ind.williams_r(df, n)
    long_sig = (w > os) & (w.shift(1) <= os)
    short_sig = (w < ob) & (w.shift(1) >= ob)
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def roc_momentum(df, n=10, sl_mult=1.5, rr=1.5, atr_period=14):
    r = ind.roc(df["close"], n)
    long_sig = (r > 0) & (r.shift(1) <= 0)
    short_sig = (r < 0) & (r.shift(1) >= 0)
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def keltner_breakout(df, n=20, mult=2.0, sl_mult=1.5, rr=2.0, atr_period=14):
    mid, up, lo = ind.keltner(df, n, mult)
    long_sig = (df["close"] > up) & (df["close"].shift(1) <= up.shift(1))
    short_sig = (df["close"] < lo) & (df["close"].shift(1) >= lo.shift(1))
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def vwap_reclaim(df, sl_mult=1.5, rr=1.5, atr_period=14):
    v = ind.vwap_session(df)
    above = df["close"] > v
    long_sig = above & ~above.shift(1).fillna(False)
    short_sig = ~above & above.shift(1).fillna(False)
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def engulfing_trend(df, sl_mult=1.5, rr=1.5, atr_period=14):
    bull = ind.bullish_engulfing(df)
    bear = ind.bearish_engulfing(df)
    entries = np.where(bull, 1, np.where(bear, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def hma_cross(df, fast=9, slow=21, sl_mult=1.5, rr=1.5, atr_period=14):
    hf, hs = ind.hma(df["close"], fast), ind.hma(df["close"], slow)
    cu = (hf > hs) & (hf.shift(1) <= hs.shift(1))
    cd = (hf < hs) & (hf.shift(1) >= hs.shift(1))
    entries = np.where(cu, 1, np.where(cd, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def prevday_breakout(df, sl_mult=1.5, rr=2.0, atr_period=14):
    day = df.index.normalize()
    daily_h = df["high"].groupby(day).max().shift(1)
    daily_l = df["low"].groupby(day).min().shift(1)
    ph = pd.Series(daily_h.reindex(day).to_numpy(), index=df.index)
    pl = pd.Series(daily_l.reindex(day).to_numpy(), index=df.index)
    long_sig = (df["close"] > ph) & (df["close"].shift(1) <= ph)
    short_sig = (df["close"] < pl) & (df["close"].shift(1) >= pl)
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def opening_range_breakout(df, or_bars=4, sl_mult=1.5, rr=2.0, atr_period=14):
    day = df.index.normalize()
    bar_idx = df.groupby(day).cumcount()
    or_h = df["high"].where(bar_idx < or_bars).groupby(day).transform("max")
    or_l = df["low"].where(bar_idx < or_bars).groupby(day).transform("min")
    after = bar_idx >= or_bars
    long_sig = after & (df["close"] > or_h) & (df["close"].shift(1) <= or_h)
    short_sig = after & (df["close"] < or_l) & (df["close"].shift(1) >= or_l)
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def sma_cross(df, fast=20, slow=50, sl_mult=1.5, rr=1.5, atr_period=14):
    ef, es = ind.sma(df["close"], fast), ind.sma(df["close"], slow)
    cu = (ef > es) & (ef.shift(1) <= es.shift(1))
    cd = (ef < es) & (ef.shift(1) >= es.shift(1))
    entries = np.where(cu, 1, np.where(cd, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def dmi_cross(df, n=14, sl_mult=1.5, rr=1.5, atr_period=14):
    _, pdi, mdi = ind.adx(df, n)
    cu = (pdi > mdi) & (pdi.shift(1) <= mdi.shift(1))
    cd = (pdi < mdi) & (pdi.shift(1) >= mdi.shift(1))
    entries = np.where(cu, 1, np.where(cd, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def ichimoku_cross(df, tenkan=9, kijun=26, sl_mult=1.5, rr=1.5, atr_period=14):
    th = (df["high"].rolling(tenkan).max() + df["low"].rolling(tenkan).min()) / 2
    kj = (df["high"].rolling(kijun).max() + df["low"].rolling(kijun).min()) / 2
    cu = (th > kj) & (th.shift(1) <= kj.shift(1))
    cd = (th < kj) & (th.shift(1) >= kj.shift(1))
    entries = np.where(cu, 1, np.where(cd, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def stochrsi_meanrev(df, n=14, os=20, ob=80, sl_mult=1.5, rr=1.0, atr_period=14):
    r = ind.rsi(df["close"], n)
    ll = r.rolling(n, min_periods=n).min(); hh = r.rolling(n, min_periods=n).max()
    sr = 100 * (r - ll) / (hh - ll)
    long_sig = (sr > os) & (sr.shift(1) <= os)
    short_sig = (sr < ob) & (sr.shift(1) >= ob)
    entries = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sl, tp = _atr_exits(df, atr_period, sl_mult, rr)
    return _pack(entries, sl, tp)


def combo(df, base, base_params, filters):
    """Generic 2+-component strategy: a base entry gated by one or more causal
    filters. Everything is JSON-serialisable so it round-trips through the log.

    filters: list of dicts, each one of:
      {"type":"htf_trend","htf":"H4","fast":50,"slow":200}  -> direction gate
      {"type":"session","a":13,"b":16}                        -> time-of-day gate
      {"type":"adx","thr":25,"above":true}                    -> regime gate
      {"type":"atr_expansion"} / {"type":"atr_contraction"}   -> volatility gate
    """
    from . import filters as flt
    sig = REGISTRY[base](df, **base_params)
    e = sig["entries"].astype(float).copy()
    for f in filters:
        t = f["type"]
        if t == "htf_trend":
            tr = flt.htf_trend(df, f.get("htf", "H4"), f.get("fast", 50), f.get("slow", 200)).to_numpy()
            e = np.where((e > 0) & (tr == 1), 1.0,
                         np.where((e < 0) & (tr == -1), -1.0, 0.0))
        elif t == "session":
            m = flt.session_mask(df, f["a"], f["b"]).to_numpy()
            e = np.where(m, e, 0.0)
        elif t == "adx":
            m = flt.adx_mask(df, f.get("thr", 25), f.get("above", True), f.get("n", 14)).to_numpy()
            e = np.where(m, e, 0.0)
        elif t == "atr_expansion":
            m = flt.atr_expansion_mask(df, f.get("n", 14), f.get("lookback", 100)).to_numpy()
            e = np.where(m, e, 0.0)
        elif t == "atr_contraction":
            m = flt.atr_contraction_mask(df, f.get("n", 14), f.get("lookback", 100)).to_numpy()
            e = np.where(m, e, 0.0)
        elif t == "body":
            m = flt.body_mask(df, f.get("thr", 1.0), f.get("n", 14)).to_numpy()
            e = np.where(m, e, 0.0)
        else:
            raise ValueError(f"unknown filter type: {t}")
    return _pack(e, sig["sl_dist"], sig["tp_dist"])


REGISTRY = {
    "combo": combo,
    "baseline_random": baseline_random,
    "baseline_ma_direction": baseline_ma_direction,
    "baseline_donchian": baseline_donchian,
    "baseline_bb_meanrev": baseline_bb_meanrev,
    "ema_cross": ema_cross,
    "rsi_meanrev": rsi_meanrev,
    "rsi_momentum": rsi_momentum,
    "macd_cross": macd_cross,
    "bb_breakout": bb_breakout,
    "donchian_breakout": donchian_breakout,
    "supertrend_flip": supertrend_flip,
    "stochastic_meanrev": stochastic_meanrev,
    "cci_meanrev": cci_meanrev,
    "psar_flip": psar_flip,
    "aroon_cross": aroon_cross,
    "williams_meanrev": williams_meanrev,
    "roc_momentum": roc_momentum,
    "keltner_breakout": keltner_breakout,
    "vwap_reclaim": vwap_reclaim,
    "engulfing_trend": engulfing_trend,
    "hma_cross": hma_cross,
    "prevday_breakout": prevday_breakout,
    "opening_range_breakout": opening_range_breakout,
    "sma_cross": sma_cross,
    "dmi_cross": dmi_cross,
    "ichimoku_cross": ichimoku_cross,
    "stochrsi_meanrev": stochrsi_meanrev,
}
