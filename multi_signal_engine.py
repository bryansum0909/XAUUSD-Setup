"""Multi-strategy signal engine.

Basket (RR 1.5, fresh crossing on last CLOSED H1 bar, per-setup one-position dedup):
  1. DONCH_H4 : Donchian-40 breakout in H4 trend
  2. MACD_H4  : MACD cross in H4 trend
  3. PSAR     : Parabolic SAR flip

PATENTED Method #1 (RR 1:3, LONG-ONLY, technical + fundamental):
  4. KELT_M1  : Keltner-20 breakout LONG, gated by Daily-trend + Weekly-trend +
               a US-Dollar regime filter (DXY < its 100-day SMA = dollar weak = gold bullish).
               Needs ~1.5y H1 history (Weekly EMA) and a DXY daily series.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import indicators as ind
from src import filters as flt
from signal_engine import lot_for_risk

ATR_P, SL_MULT, RR, N = 14, 1.5, 1.5, 40
SETUP_KEYS = ["DONCH_H4", "MACD_H4", "PSAR", "KELT_M1", "DONCH_M1"]


def _fresh_donchian(df, n=N):
    up = df["high"].rolling(n, min_periods=n).max().shift(1)
    lo = df["low"].rolling(n, min_periods=n).min().shift(1)
    c, cp = df["close"].iloc[-1], df["close"].iloc[-2]
    if c > up.iloc[-1] and cp <= up.iloc[-2]:
        return "BUY"
    if c < lo.iloc[-1] and cp >= lo.iloc[-2]:
        return "SELL"
    return None


def _fresh_macd(df, fast=12, slow=26, signal=9):
    m, s, _ = ind.macd(df["close"], fast, slow, signal)
    if m.iloc[-1] > s.iloc[-1] and m.iloc[-2] <= s.iloc[-2]:
        return "BUY"
    if m.iloc[-1] < s.iloc[-1] and m.iloc[-2] >= s.iloc[-2]:
        return "SELL"
    return None


def _fresh_psar(df):
    sar = ind.parabolic_sar(df)
    an, ap = df["close"].iloc[-1] > sar.iloc[-1], df["close"].iloc[-2] > sar.iloc[-2]
    if an and not ap:
        return "BUY"
    if not an and ap:
        return "SELL"
    return None


def _fresh_keltner_long(df, n=20, mult=2.0):
    _, up, _ = ind.keltner(df, n, mult)
    c, cp = df["close"].iloc[-1], df["close"].iloc[-2]
    return "BUY" if (c > up.iloc[-1] and cp <= up.iloc[-2]) else None


def _h4_trend(df):
    v = flt.htf_trend(df, "H4", 50, 200).iloc[-1]
    return int(v) if np.isfinite(v) else 0


def _trend(df, tf, fast, slow):
    v = flt.htf_trend(df, tf, fast, slow).iloc[-1]
    return int(v) if np.isfinite(v) else 0


def dollar_weak(dxy_close: pd.Series) -> bool:
    """US-Dollar regime filter: DXY below its 100-day SMA = dollar weak = gold-bullish."""
    if dxy_close is None or len(dxy_close) < 100:
        return False
    sma100 = dxy_close.rolling(100, min_periods=100).mean().iloc[-1]
    return bool(np.isfinite(sma100) and dxy_close.iloc[-1] < sma100)


def _make(key, label, direction, df, sl_dist, tp_dist, lot, balance, risk_pct, atr, rr):
    entry = float(df["close"].iloc[-1])
    if direction == "BUY":
        sl, tp = entry - sl_dist, entry + tp_dist
    else:
        sl, tp = entry + sl_dist, entry - tp_dist
    return {"key": key, "setup": label, "signal": direction, "bar_time": str(df.index[-1]),
            "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            "sl_distance": round(sl_dist, 2), "tp_distance": round(tp_dist, 2),
            "atr": round(float(atr), 2), "rr": rr, "lot": round(float(lot), 2),
            "risk_money": round(balance * risk_pct / 100.0, 2),
            "risk_pct": risk_pct, "balance": balance, "max_hold_bars": 60}


def evaluate_all(df: pd.DataFrame, *, dxy_close=None, balance=10000.0, risk_pct=0.5, round_lot_step=0.01):
    df = df[~df.index.duplicated()].sort_index()
    if len(df) < N + ATR_P + 5:
        return []
    atr = ind.atr(df, ATR_P).iloc[-1]
    if not np.isfinite(atr) or atr <= 0:
        return []
    sl_dist = SL_MULT * float(atr)

    def lot_for(sld):
        L = lot_for_risk(balance, risk_pct, sld)
        return max(round_lot_step, np.floor(L / round_lot_step) * round_lot_step)

    tp15 = RR * sl_dist
    lot15 = lot_for(sl_dist)
    h4 = _h4_trend(df)
    out = []

    d = _fresh_donchian(df)
    if d and ((d == "BUY" and h4 == 1) or (d == "SELL" and h4 == -1)):
        out.append(_make("DONCH_H4", "Donchian-40 + tren H4", d, df, sl_dist, tp15, lot15, balance, risk_pct, atr, RR))
    m = _fresh_macd(df)
    if m and ((m == "BUY" and h4 == 1) or (m == "SELL" and h4 == -1)):
        out.append(_make("MACD_H4", "MACD + tren H4", m, df, sl_dist, tp15, lot15, balance, risk_pct, atr, RR))
    p = _fresh_psar(df)
    if p:
        out.append(_make("PSAR", "Parabolic SAR", p, df, sl_dist, tp15, lot15, balance, risk_pct, atr, RR))

    # --- Method #1: Keltner LONG + Daily + Weekly + Dollar regime + ADX<25, RR 1:3 ---
    # ADX<25 filter (validated 6/6 bases: only take breakouts before the trend is overextended)
    k = _fresh_keltner_long(df)
    if k == "BUY":
        d1 = _trend(df, "D1", 50, 200)
        wk = _trend(df, "W", 20, 50)
        adx14 = ind.adx(df, 14)[0].iloc[-1]
        adx_ok = np.isfinite(adx14) and adx14 < 25
        if d1 == 1 and wk == 1 and adx_ok and dollar_weak(dxy_close):
            tp3 = 3.0 * sl_dist
            out.append(_make("KELT_M1", "Method#1+ Keltner LONG (RR1:3, DXY+ADX<25 gate)", "BUY",
                             df, sl_dist, tp3, lot_for(sl_dist), balance, risk_pct, atr, 3.0))

    # --- DONCH_M1: 2nd robust breakout (Donchian-50) — same edge filters, adds frequency ---
    dk = _fresh_donchian(df, 50)
    if dk == "BUY":
        d1 = _trend(df, "D1", 50, 200); wk = _trend(df, "W", 20, 50)
        adx14 = ind.adx(df, 14)[0].iloc[-1]
        if d1 == 1 and wk == 1 and np.isfinite(adx14) and adx14 < 25 and dollar_weak(dxy_close):
            out.append(_make("DONCH_M1", "Donchian-50 LONG (RR1:3, DXY+ADX<25 gate)", "BUY",
                             df, sl_dist, 3.0 * sl_dist, lot_for(sl_dist), balance, risk_pct, atr, 3.0))
    return out


def ma_m30_status(df_m30: pd.DataFrame):
    """M30 EMA50/200 golden-cross TREND-POSITION strategy (validated: +119%, PF 2.18, RR~4.4,
    WR 32%, DD 15.8%, ~0.6 trades/wk over 5y). Long while EMA50>EMA200, exit on death cross —
    NOT an RR-1:3 trade. Returns the current long/flat STATE so the bot alerts on cross flips."""
    df_m30 = df_m30[~df_m30.index.duplicated()].sort_index()
    if len(df_m30) < 210:
        return None
    c = df_m30["close"]
    e50 = ind.ema(c, 50).iloc[-1]
    e200 = ind.ema(c, 200).iloc[-1]
    if not (np.isfinite(e50) and np.isfinite(e200)):
        return None
    atr = ind.atr(df_m30, 14).iloc[-1]
    return {"long": bool(e50 > e200), "close": float(c.iloc[-1]),
            "ema50": round(float(e50), 2), "ema200": round(float(e200), 2),
            "atr": round(float(atr), 2) if np.isfinite(atr) else None,
            "bar_time": str(df_m30.index[-1])}


def trend_m30f_status(df_m30: pd.DataFrame):
    """M30 EMA10/30 gated by an H1-uptrend filter — trend-position (~2.4/wk ≈ 1 per 2 days).
    Validated robust: +84%, PF 1.45, RR 3.3, WR 30%, DD 14%, streak 14, positive EVERY year,
    IS/OOS balanced (PF 1.46/1.44), cost-robust. Long while (M30 EMA10>EMA30) AND (H1 uptrend);
    exit when either flips. H1 trend = resample M30->H1, EMA50/200, shift(1) (lookahead-safe)."""
    df_m30 = df_m30[~df_m30.index.duplicated()].sort_index()
    if len(df_m30) < 420:                      # need ~200 H1 bars (=400 M30) for EMA200
        return None
    c = df_m30["close"]
    e10, e30 = ind.ema(c, 10).iloc[-1], ind.ema(c, 30).iloc[-1]
    # H1 uptrend filter, replicating the backtest exactly (resample->EMA->shift(1)->ffill)
    ch = c.resample("1h").last().dropna()
    ht = pd.Series(np.where(ind.ema(ch, 50) > ind.ema(ch, 200), 1.0, 0.0), index=ch.index).shift(1)
    h1_up = ht.reindex(df_m30.index.union(ht.index)).ffill().reindex(df_m30.index).fillna(0.0).iloc[-1] > 0
    if not (np.isfinite(e10) and np.isfinite(e30)):
        return None
    atr = ind.atr(df_m30, 14).iloc[-1]
    m30_up = bool(e10 > e30)
    return {"long": bool(m30_up and h1_up), "m30_up": m30_up, "h1_up": bool(h1_up),
            "close": float(c.iloc[-1]), "ema10": round(float(e10), 2), "ema30": round(float(e30), 2),
            "atr": round(float(atr), 2) if np.isfinite(atr) else None, "bar_time": str(df_m30.index[-1])}


def regime_status(df_h1: pd.DataFrame, dxy_close=None) -> dict:
    """The gate that BOTH Method #1 family AND ORB_M15 require: uptrend + weak dollar.
    Returns a dict the bot uses to (a) gate signals and (b) alert on regime flips."""
    d1 = _trend(df_h1, "D1", 50, 200)
    wk = _trend(df_h1, "W", 20, 50)
    weak = dollar_weak(dxy_close)
    dxy_val = float(dxy_close.iloc[-1]) if dxy_close is not None and len(dxy_close) else None
    sma100 = None
    if dxy_close is not None and len(dxy_close) >= 100:
        s = dxy_close.rolling(100, min_periods=100).mean().iloc[-1]
        sma100 = float(s) if np.isfinite(s) else None
    # regime_on = the shared NECESSARY core for ANY long setup: daily uptrend + weak dollar.
    # (ORB_M15's validated gate is exactly Daily-trend + DXY-weak. Method #1 additionally
    #  requires weekly_up + ADX<25, reported separately.)
    return {
        "regime_on": bool(d1 == 1 and weak),
        "daily_up": bool(d1 == 1), "weekly_up": bool(wk == 1), "dxy_weak": bool(weak),
        "dxy": dxy_val, "dxy_sma100": sma100,
        "dxy_gap": (round(dxy_val - sma100, 2) if (dxy_val is not None and sma100 is not None) else None),
    }


def evaluate_orb_m15(df_m15: pd.DataFrame, df_h1: pd.DataFrame, *, dxy_close=None,
                     balance=10000.0, risk_pct=2.0, or_bars=4, round_lot_step=0.01):
    """Opening-Range-Breakout LONG on M15, RR 1:3, gated by the SAME regime as Method #1
    (Daily-up + Weekly-up + DXY-weak). Regime is read from the long H1 series (needs the
    warmup); the breakout is read from today's M15 bars. Fires only on the fresh cross
    above today's opening-range high, and only once the opening range is complete."""
    if df_m15 is None or len(df_m15) < 20:
        return []
    df_m15 = df_m15[~df_m15.index.duplicated()].sort_index()
    # regime gate (shared with Method #1) — computed on H1 for correct EMA warmup
    st = regime_status(df_h1, dxy_close)
    if not st["regime_on"]:
        return []
    # today's opening range
    day = df_m15.index.normalize()
    tmask = (day == day[-1])
    tdf = df_m15[tmask]
    n_today = len(tdf)
    if n_today <= or_bars:            # opening range not complete / no post-OR bar yet
        return []
    or_high = float(tdf["high"].iloc[:or_bars].max())
    c = float(df_m15["close"].iloc[-1]); cp = float(df_m15["close"].iloc[-2])
    fresh = (c > or_high) and (cp <= or_high)   # first close above the OR high
    if not fresh:
        return []
    atr = ind.atr(df_m15, 14).iloc[-1]
    if not np.isfinite(atr) or atr <= 0:
        return []
    sl_dist = 1.5 * float(atr)
    L = lot_for_risk(balance, risk_pct, sl_dist)
    lot = max(round_lot_step, np.floor(L / round_lot_step) * round_lot_step)
    return [_make("ORB_M15", "ORB M15 LONG (RR1:3, D1+W+DXY gate)", "BUY",
                  df_m15, sl_dist, 3.0 * sl_dist, lot, balance, risk_pct, atr, 3.0)]
