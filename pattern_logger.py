"""Perekam pattern XAUUSD M5/M15 -> jurnal JSONL (bahan belajar lintas sesi).

Daemon kecil di VPS (terpisah dari bot sinyal agar jalur sinyal tak tersentuh):
tiap bar M5 close, deteksi pattern candle (engulfing/pin/harami/inside) di M5
dan M15 (saat bar M15 baru close) + konteks (tren M15/M30 kausal, rasio ATR,
jarak ke EMA20, sesi) lalu append ke pattern_journal.jsonl. Outcome tiap entri
(pergerakan 6 & 12 bar berikutnya, dalam satuan ATR) diisi otomatis oleh pass
berikutnya begitu bar masa depannya tersedia — jadi jurnal berlabel sendiri.

Jurnal ini yang dianalisis Claude lintas sesi ("pattern X sudah muncul N kali,
lanjutannya begini") dan kelak jadi dataset backtest detektor pattern klasik.

Pemakaian: python3 pattern_logger.py --once | --loop
Env: PATTERN_JOURNAL (default /root/xauusd-bot/pattern_journal.jsonl)
"""
from __future__ import annotations
import os, sys, json, time, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import angelfing_signal as ang

JPATH = os.environ.get("PATTERN_JOURNAL", "/root/xauusd-bot/pattern_journal.jsonl")


def candle_patterns(o, h, l, c):
    """Flag pattern untuk bar TERAKHIR dari frame OHLC."""
    po, pc = o.iloc[-2], c.iloc[-2]
    oo, hh, ll, cc = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    body = abs(cc - oo); rng = max(hh - ll, 1e-9)
    upw = hh - max(oo, cc); dnw = min(oo, cc) - ll
    tags = []
    if (pc < po) and (cc > oo) and (cc > po) and (oo <= pc): tags.append("engulf_bull")
    if (pc > po) and (cc < oo) and (cc < po) and (oo >= pc): tags.append("engulf_bear")
    if dnw > 2 * body and dnw > 0.6 * rng: tags.append("pin_bawah")
    if upw > 2 * body and upw > 0.6 * rng: tags.append("pin_atas")
    if (pc < po) and max(oo, cc) <= po and min(oo, cc) >= pc: tags.append("harami_bull")
    if (pc > po) and max(oo, cc) <= pc and min(oo, cc) >= po: tags.append("harami_bear")
    if hh <= h.iloc[-2] and ll >= l.iloc[-2]: tags.append("inside")
    return tags


def sesi(ts):
    hh = ts.hour + ts.minute / 60.0
    if 8 <= hh < 13: return "LONDON"
    if 13 <= hh < 16.5: return "OVERLAP"
    if 16.5 <= hh < 22: return "NY"
    return "ASIA"


def load_journal():
    if not os.path.exists(JPATH): return []
    return [json.loads(x) for x in open(JPATH, encoding="utf-8") if x.strip()]


def save_journal(rows):
    tmp = JPATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, JPATH)


def process(df, rows):
    """Tambah entri utk bar M5 terakhir (jika ada pattern) + isi outcome entri lama."""
    L, S, m = ang.signal_masks(df)
    changed = False
    ts = m.index[-1]
    key = str(ts)
    if not any(r["t"] == key and r["tf"] == "M5" for r in rows[-40:]):
        tags = candle_patterns(m["open"], m["high"], m["low"], m["close"])
        if L.iloc[-1]: tags.append("ANGELFING_LONG")
        if S.iloc[-1]: tags.append("ANGELFING_SHORT")
        if tags:
            last = m.iloc[-1]
            rows.append({"t": key, "tf": "M5", "pat": tags, "px": round(float(last["close"]), 2),
                         "t15": int(last["t15"]), "t30": int(last["t30"]),
                         "atr": round(float(last["atr"]), 2),
                         "atr_r": round(float(last["atr"] / last["amed"]), 2) if last["amed"] > 0 else None,
                         "d_ema": round(float((last["close"] - last["ema20"]) / last["atr"]), 2),
                         "sesi": sesi(ts)})
            changed = True
    # M15: log saat bar M15 baru saja close (bar M5 :10/:25/:40/:55)
    if ts.minute % 15 == 10:
        m15 = pd.DataFrame({"open": m["open"].resample("15min").first(),
                            "high": m["high"].resample("15min").max(),
                            "low": m["low"].resample("15min").min(),
                            "close": m["close"].resample("15min").last()}).dropna()
        k15 = str(m15.index[-1])
        if len(m15) > 2 and not any(r["t"] == k15 and r["tf"] == "M15" for r in rows[-40:]):
            tags = candle_patterns(m15["open"], m15["high"], m15["low"], m15["close"])
            if tags:
                last = m.iloc[-1]
                rows.append({"t": k15, "tf": "M15", "pat": tags,
                             "px": round(float(m15["close"].iloc[-1]), 2),
                             "t15": int(last["t15"]), "t30": int(last["t30"]),
                             "atr": round(float(last["atr"]), 2), "sesi": sesi(ts)})
                changed = True
    # isi outcome yang bar masa depannya sudah ada
    closes = m["close"]
    for r in rows:
        if "f12" in r: continue
        step = pd.Timedelta(minutes=5 if r["tf"] == "M5" else 15)
        t0 = pd.Timestamp(r["t"])
        atr = r.get("atr") or 1.0
        for horizon, kkey in ((6, "f6"), (12, "f12")):
            tN = t0 + step * horizon
            if kkey in r: continue
            idx = closes.index[closes.index >= tN]
            if len(idx):
                r[kkey] = round(float((closes.loc[idx[0]] - r["px"]) / atr), 2)
                changed = True
    return changed


def main():
    args = sys.argv[1:]
    once = "--once" in args
    print(f"[{ang.utcnow():%Y-%m-%d %H:%M}] pattern logger start (jurnal={JPATH})", flush=True)
    while True:
        if not once:
            now = ang.utcnow()
            wait = (5 - now.minute % 5) * 60 - now.second + 40   # +40 dtk: setelah bot sinyal
            time.sleep(max(5, wait))
        try:
            now = ang.utcnow()
            bar_start = pd.Timestamp(now.replace(second=0, microsecond=0))
            bar_start = bar_start - pd.Timedelta(minutes=bar_start.minute % 5 + 5)
            if ang.market_open(bar_start):
                df = ang.get_m5(8)
                if len(df) > 350:
                    rows = load_journal()
                    if process(df, rows):
                        save_journal(rows)
                        print(f"[{now:%Y-%m-%d %H:%M}] jurnal: {len(rows)} entri", flush=True)
        except Exception as e:
            print(f"[{ang.utcnow():%Y-%m-%d %H:%M}] logger error: {e}", flush=True)
        if once:
            break


if __name__ == "__main__":
    main()
