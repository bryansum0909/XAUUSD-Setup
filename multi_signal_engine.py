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
