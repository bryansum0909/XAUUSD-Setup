"""Gold (XAUUSD) FUNDAMENTAL REGIME monitor — quantitative, rules-derived from the
research. Fetches the measurable macro drivers daily and outputs a BULLISH / NEUTRAL /
BEARISH regime verdict for gold's trend, plus what would flip it. Optional Telegram digest.

Drivers & rules (each = gold-bullish when true):
  1. DXY below its 100-day SMA          -> dollar weak            (Method #1's gate)
  2. DXY falling over 20 days           -> dollar downtrend
  3. US 10Y real yield falling (20d)    -> opportunity cost down
  4. US 10Y real yield below ~2.0%      -> low real-yield headwind
  5. Gold above its 200-day MA          -> long-term uptrend structure
  6. Gold 50DMA above 200DMA (golden)   -> trend up

Usage: python fundamental_monitor.py            # print report
       python fundamental_monitor.py --telegram  # also send to Telegram (uses bot config/env)
Sources: Yahoo (DXY 'DX-Y.NYB', gold 'GC=F'), FRED CSV (real yield 'DFII10', 10Y 'DGS10').
"""
import os, sys, io, json, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import requests
from src import data_loader as dl

ROOT = os.path.dirname(os.path.abspath(__file__))
UA = dl.UA


def fred(series, retries=3):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=15); r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            d.columns = ["date", "val"]
            d["date"] = pd.to_datetime(d["date"])
            d["val"] = pd.to_numeric(d["val"], errors="coerce")
            return d.dropna().set_index("date")["val"]
        except Exception as e:
            last = e
    raise last


def yh(sym, rng="400d"):
    return dl.fetch_yahoo(sym, interval="1d", range_=rng)["close"].sort_index()


def build():
    out = {"as_of": dt.date(2026, 7, 25).isoformat(), "signals": [], "errors": []}
    # DXY
    dxy = yh("DX-Y.NYB")
    dxy_sma100 = dxy.rolling(100).mean().iloc[-1]
    s1 = dxy.iloc[-1] < dxy_sma100
    s2 = dxy.iloc[-1] < dxy.iloc[-21]
    out["dxy"] = round(float(dxy.iloc[-1]), 2); out["dxy_sma100"] = round(float(dxy_sma100), 2)
    out["signals"] += [("DXY < 100-day SMA (dollar weak)", bool(s1), f"{dxy.iloc[-1]:.1f} vs {dxy_sma100:.1f}"),
                        ("DXY falling (20d)", bool(s2), f"{dxy.iloc[-1]:.1f} vs {dxy.iloc[-21]:.1f} 20d ago")]
    # US 10Y Treasury yield via Yahoo ^TNX (reliable). Falling yields = gold-bullish.
    try:
        tnx = yh("^TNX")
        s3 = tnx.iloc[-1] < tnx.iloc[-21]
        out["us10y"] = round(float(tnx.iloc[-1]), 3)
        out["signals"].append(("US 10Y yield falling (20d)", bool(s3),
                               f"{tnx.iloc[-1]:.2f} vs {tnx.iloc[-21]:.2f} 20d ago"))
    except Exception as e:
        out["errors"].append(f"10Y yield: {e}")
    # Bonus: real yield from FRED (fast-fail, optional)
    try:
        ry = fred("DFII10", retries=1)
        s4 = ry.iloc[-1] < 2.0
        out["real_yield_10y"] = round(float(ry.iloc[-1]), 3)
        out["signals"].append(("10Y real yield < 2.0% (low headwind)", bool(s4), f"{ry.iloc[-1]:.2f}%"))
    except Exception as e:
        out["errors"].append(f"FRED real yield (optional): {str(e)[:50]}")
    # Gold structure
    try:
        gold = yh("GC=F")
        ma200 = gold.rolling(200).mean().iloc[-1]; ma50 = gold.rolling(50).mean().iloc[-1]
        s5 = gold.iloc[-1] > ma200
        s6 = ma50 > ma200
        out["gold"] = round(float(gold.iloc[-1]), 1)
        out["signals"] += [("Gold above 200-day MA", bool(s5), f"{gold.iloc[-1]:.0f} vs {ma200:.0f}"),
                            ("Gold 50DMA > 200DMA (golden)", bool(s6), f"{ma50:.0f} vs {ma200:.0f}")]
    except Exception as e:
        out["errors"].append(f"gold: {e}")

    bull = sum(1 for _, ok, _ in out["signals"] if ok)
    total = len(out["signals"])
    out["bullish_signals"] = bull; out["total_signals"] = total
    if total == 0:
        out["regime"] = "UNKNOWN"
    else:
        frac = bull / total
        out["regime"] = "BULLISH" if frac >= 0.66 else ("BEARISH" if frac <= 0.34 else "NEUTRAL")
    out["gate_method1"] = "ON (long allowed)" if out["signals"][0][1] else "OFF (sit out)"
    return out


def render(o):
    L = [f"GOLD FUNDAMENTAL REGIME — {o['as_of']}",
         f"Verdict: {o['regime']}  ({o['bullish_signals']}/{o['total_signals']} sinyal bullish)",
         f"Method #1 DXY-gate: {o['gate_method1']}", "-" * 40]
    for name, ok, detail in o["signals"]:
        L.append(f"  [{'BULL' if ok else 'bear'}] {name}  ({detail})")
    if o.get("errors"):
        L.append("errors: " + "; ".join(o["errors"]))
    L.append("-" * 40)
    L.append("Pemicu balik ke BULLISH: DXY tembus < 100-SMA, real yield turun < 2%, emas di atas 200DMA.")
    L.append("Riset/edukasi. Bukan saran keuangan.")
    return "\n".join(L)


def main():
    o = build()
    txt = render(o)
    print(txt)
    os.makedirs(os.path.join(ROOT, "results", "reports"), exist_ok=True)
    json.dump(o, open(os.path.join(ROOT, "results", "reports", "fundamental_regime.json"), "w"), indent=2)
    if "--telegram" in sys.argv:
        try:
            import telegram_signal_bot as bot
            cfg = bot.load_cfg()
            emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(o["regime"], "⚪")
            msg = f"{emoji} <b>GOLD FUNDAMENTAL: {o['regime']}</b>\n<pre>{txt}</pre>"
            if cfg["bot_token"] and cfg["chat_id"]:
                bot.send_telegram(cfg["bot_token"], cfg["chat_id"], msg); print("\n[sent to Telegram]")
            else:
                print("\n[no token — skip send]")
        except Exception as e:
            print("telegram send failed:", e)


if __name__ == "__main__":
    main()
