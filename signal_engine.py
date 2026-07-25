"""Live signal engine for the H1 Donchian breakout candidate — MANUAL execution.

Matches the backtested rules EXACTLY and lookahead-safe:
  * Signal is read off the LAST CLOSED H1 bar (never the forming bar).
  * Donchian(40): upper = highest high of the 40 bars BEFORE the signal bar,
    lower = lowest low of the same. Long if signal-bar close > upper; short if < lower.
  * SL = 1.5 x ATR(14).  TP = RR x SL (default RR 1.5 -> practical for manual trading;
    this is the validated fixed-TP variant: H1 n=40 rr1.5 holdout PF 1.34, +0.18R, WR ~48%).
  * Entry is the NEXT bar's open -> in practice a MARKET order right after the alert.
  * Lot sized by fixed-fractional risk over the stop distance.

Returns a dict describing the signal (or {'signal': None}).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import indicators as ind

CONTRACT = 100.0   # 1 lot XAUUSD = 100 oz -> $1 move = $100 / lot


def lot_for_risk(balance: float, risk_pct: float, sl_distance: float) -> float:
    if sl_distance <= 0:
        return 0.0
    risk_money = balance * risk_pct / 100.0
    loss_per_lot = sl_distance * CONTRACT
    return risk_money / loss_per_lot if loss_per_lot > 0 else 0.0


def evaluate(df_h1: pd.DataFrame, *, n=40, atr_period=14, sl_mult=1.5, rr=1.5,
             balance=10000.0, risk_pct=0.5, round_lot_step=0.01) -> dict:
    """df_h1: H1 OHLC with the LAST ROW being the most recently CLOSED bar."""
    if len(df_h1) < n + atr_period + 2:
        return {"signal": None, "reason": "not enough bars"}
    upper = df_h1["high"].rolling(n, min_periods=n).max().shift(1)
    lower = df_h1["low"].rolling(n, min_periods=n).min().shift(1)
    atr = ind.atr(df_h1, atr_period)

    i = -1  # last CLOSED bar
    c = df_h1["close"].iloc[i]; c_prev = df_h1["close"].iloc[i - 1]
    up = upper.iloc[i]; lo = lower.iloc[i]; a = atr.iloc[i]
    up_prev = upper.iloc[i - 1]; lo_prev = lower.iloc[i - 1]
    if not np.isfinite(up) or not np.isfinite(lo) or not np.isfinite(a) or a <= 0:
        return {"signal": None, "reason": "warmup"}

    direction = None
    # FRESH breakout only (crossing) -> one alert per breakout event, matching the
    # one-position backtest, not every bar the band stays broken.
    if c > up and (not np.isfinite(up_prev) or c_prev <= up_prev):
        direction = "BUY"
    elif c < lo and (not np.isfinite(lo_prev) or c_prev >= lo_prev):
        direction = "SELL"
    if direction is None:
        return {"signal": None, "bar_time": str(df_h1.index[i]),
                "close": round(float(c), 2), "donchian_up": round(float(up), 2),
                "donchian_lo": round(float(lo), 2), "atr": round(float(a), 2)}

    entry = float(c)                       # ~ next bar open; market order on alert
    sl_dist = sl_mult * float(a)
    tp_dist = rr * sl_dist
    if direction == "BUY":
        sl = entry - sl_dist; tp = entry + tp_dist
    else:
        sl = entry + sl_dist; tp = entry - tp_dist
    lot = lot_for_risk(balance, risk_pct, sl_dist)
    lot = max(round_lot_step, np.floor(lot / round_lot_step) * round_lot_step)
    risk_money = balance * risk_pct / 100.0

    return {
        "signal": direction,
        "bar_time": str(df_h1.index[i]),
        "entry": round(entry, 2),
        "sl": round(sl, 2), "tp": round(tp, 2),
        "sl_distance": round(sl_dist, 2), "tp_distance": round(tp_dist, 2),
        "atr": round(float(a), 2), "rr": rr,
        "lot": round(float(lot), 2),
        "risk_money": round(risk_money, 2), "risk_pct": risk_pct, "balance": balance,
        "max_hold_bars": 60,   # optional discipline: exit if not hit within ~60 H1 bars
        "setup": f"Donchian-{n} breakout {'up' if direction=='BUY' else 'down'}",
    }
