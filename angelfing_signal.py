"""ANGELFING_M5 — engulfing M5 Telegram signal bot (MANUAL execution, never trades).

Metode dari project "Angelfing Confirmation" (D:\\Bryan Stuff\\Claude Trading\\
Angelfing Confirmation\\xauusd-backtest, lihat CLAUDE.md di sana):
  * M5, sesi London 03:00-11:30 EST = 08:00-16:30 UTC, BUY & SELL.
  * Sinyal di bar M5 yang SUDAH CLOSE: tren M15 & M30 searah (EMA20>50 pada
    resample close, KAUSAL: hanya bar HTF yang sudah close, shift 1) + bar
    menyentuh EMA20(M5) + candle engulfing searah tren + ATR14 > median-288.
  * Entry = open bar berikutnya (praktik: market order saat alert masih segar).
    SL 1.5xATR(14), TP 1.5R. Konservatif: SL & TP di bar sama = loss.

Angka jujur (logika KAUSAL identik dengan file ini; SL 1.5xATR di keduanya):
  * TP1 = 1.5R: 2026 Jan-14Jul PF 1.46 +30.7R (132 trade, WR 50%, streak<=5);
    2025 PF 1.06; 2024 PF 0.93; 2023 PF 0.79. Unggul di rezim bear-cepat 2026.
  * TP2 = 3R (TP 4.5xATR): 2025 PF 1.20 +42.2R; 2024 PF 1.02; 2023 PF 0.94;
    2026 PF 1.18. Satu-satunya positif full 3.5 th (+50R, PF 1.07) TAPI WR ~28%
    dan loss beruntun 10-14x. Pesan menampilkan KEDUA TP - pilih sesuai rezim.
  -> Edge hanya hidup saat pasar TRENDING. Pantau "Gold fundamental watch";
     saat ranging, harapkan hasil impas minus biaya.
  * Dedup posisi memakai TP1 (siklus tercepat): sinyal baru bisa muncul saat
    posisi TP2 masih terbuka - sesuaikan dengan posisimu sendiri.

Live data = Yahoo GC=F 5m (futures, proxy spot; terapkan JARAK SL/TP ke harga
broker). Kuota Yahoo 5m ~60 hari; kita ambil 30 hari (cukup warmup EMA/median).

Pemakaian:
  python angelfing_signal.py --once            # cek bar terakhir, kirim Telegram
  python angelfing_signal.py --once --dry-run  # print saja
  python angelfing_signal.py --test            # kirim pesan uji koneksi
State dedup: angelfing_state.json (terpisah dari bot H1 agar tidak bentrok git).
"""
from __future__ import annotations
import os, sys, json, datetime as dt
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import requests
from src import data_loader as dl
from signal_engine import lot_for_risk, risk_for_lot

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(ROOT, "angelfing_state.json")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

SESSION_UTC = (8.0, 16.5)      # 03:00-11:30 EST (GMT-5 tanpa DST) dalam UTC
ATR_P, AMED_N, SL_MULT = 14, 288, 1.5
RR1, RR2 = 1.5, 3.0            # dua pilihan TP di satu pesan; dedup pakai TP1
FRESH_MIN = float(os.environ.get("ANGELFING_FRESH_MIN", 20))   # menit; sinyal lebih tua di-skip
MAX_HOLD_M5 = 1440             # jaring pengaman state: anggap selesai setelah ~5 hari


def signal_masks(df: pd.DataFrame):
    """Vektorisasi sinyal, identik dengan backtest (varian KAUSAL).
    df: M5 OHLC UTC-naive, kolom open/high/low/close. Return (L, S, meta-df)."""
    df = df[~df.index.duplicated()].sort_index().copy()
    c = df["close"]
    for tf, name in (("15min", "t15"), ("30min", "t30")):
        s = c.resample(tf).last().dropna()
        t = pd.Series(np.where(s.ewm(span=20, adjust=False).mean()
                               > s.ewm(span=50, adjust=False).mean(), 1, -1), index=s.index)
        # KAUSAL: nilai bar HTF baru dipakai setelah bar itu CLOSE (shift 1 bar HTF)
        df[name] = t.shift(1).reindex(df.index, method="ffill")
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()))
    df["atr"] = tr.rolling(ATR_P).mean()
    df["amed"] = df["atr"].rolling(AMED_N).median()
    o, h, l = df["open"], df["high"], df["low"]
    po, pc = o.shift(), c.shift()
    bull = (pc < po) & (c > o) & (c > po) & (o <= pc)
    bear = (pc > po) & (c < o) & (c < po) & (o >= pc)
    touch = (l <= df["ema20"]) & (h >= df["ema20"])
    hh = df.index.hour + df.index.minute / 60.0
    sess = (hh >= SESSION_UTC[0]) & (hh < SESSION_UTC[1])
    vol = df["atr"] > df["amed"]
    L = (df["t15"] == 1) & (df["t30"] == 1) & touch & bull & vol & sess
    S = (df["t15"] == -1) & (df["t30"] == -1) & touch & bear & vol & sess
    return L.fillna(False), S.fillna(False), df


def get_m5(lookback_days: int = 30) -> pd.DataFrame:
    df = dl.fetch_yahoo("GC=F", interval="5m", range_=f"{lookback_days}d")
    df = df[~df.index.duplicated()].sort_index()
    # buang bar yang masih berjalan: (a) belum 5 menit penuh, atau (b) stempel di luar
    # grid 5-menit (Yahoo menstempel bar berjalan dengan waktu trade terakhir, mis. 20:59:54)
    now = utcnow()
    while len(df) and (df.index[-1] + pd.Timedelta(minutes=5) > now
                       or df.index[-1].second != 0 or df.index[-1].minute % 5 != 0):
        df = df.iloc[:-1]
    return df


def evaluate_last_bar(df: pd.DataFrame, *, balance: float, risk_pct: float,
                      round_lot_step: float = 0.01):
    L, S, m = signal_masks(df)
    if len(m) < AMED_N + ATR_P + 5 or not (L.iloc[-1] or S.iloc[-1]):
        return None
    direction = "BUY" if L.iloc[-1] else "SELL"
    atr = float(m["atr"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None
    entry = float(m["close"].iloc[-1])
    sl_dist = SL_MULT * atr
    d = 1 if direction == "BUY" else -1
    sl = entry - d * sl_dist
    tp1 = entry + d * RR1 * sl_dist
    tp2 = entry + d * RR2 * sl_dist
    lot = lot_for_risk(balance, risk_pct, sl_dist)
    lot = max(round_lot_step, np.floor(lot / round_lot_step) * round_lot_step)
    return {"key": "ANGELFING_M5", "signal": direction, "bar_time": str(m.index[-1]),
            "entry": round(entry, 2), "sl": round(sl, 2),
            "tp": round(tp1, 2),                       # state dedup pakai TP1
            "tp2": round(tp2, 2),
            "sl_distance": round(sl_dist, 2),
            "tp1_distance": round(RR1 * sl_dist, 2), "tp2_distance": round(RR2 * sl_dist, 2),
            "atr": round(atr, 2), "lot": round(float(lot), 2),
            "risk_money": round(balance * risk_pct / 100.0, 2),
            "risk_actual": round(risk_for_lot(lot, sl_dist), 2),
            "risk_pct": risk_pct, "balance": balance}


def format_msg(sig: dict, age_min: float) -> str:
    arrow = "🟢 BUY" if sig["signal"] == "BUY" else "🔴 SELL"
    tol = round(0.3 * sig["sl_distance"], 2)
    if sig["signal"] == "BUY":
        lim = round(sig["entry"] + tol, 2); lim_txt = f"sudah DI ATAS {lim}"
    else:
        lim = round(sig["entry"] - tol, 2); lim_txt = f"sudah DI BAWAH {lim}"
    return (
        f"<b>XAUUSD M5 — {arrow} (ANGELFING)</b>\n"
        f"📊 Setup: <b>ANGELFING_M5</b> — engulfing + tren M15/M30 + sentuh EMA20 + ATR tinggi, sesi London\n"
        f"🕒 Bar M5 closed: {sig['bar_time']} UTC (umur sinyal ~{age_min:.0f} mnt)\n"
        f"———————————————\n"
        f"➡️ <b>Entry</b>: ~{sig['entry']} (market SEKARANG — sinyal M5 cepat basi)\n"
        f"🛑 <b>Stop Loss</b>: {sig['sl']}  (jarak {sig['sl_distance']} = 1.5×ATR)\n"
        f"🎯 <b>TP1</b>: {sig['tp']}  (RR 1.5 — unggul di rezim 2026: PF 1.46, WR 50%, streak ≤5)\n"
        f"🎯 <b>TP2</b>: {sig['tp2']}  (RR 3 — konsisten antar tahun: +50R/3.5th, TAPI WR 28%, streak s/d 14)\n"
        f"👉 Pilih SATU sesuai rezim Gold fundamental watch (tren kuat searah → TP2 layak; bear cepat/rebound → TP1)\n"
        f"📦 <b>Lot</b>: {sig['lot']}  (risk {sig['risk_pct']}% = ${sig['risk_money']} dari ${sig['balance']:.0f}; "
        f"risk riil setelah pembulatan ≈ ${sig['risk_actual']})\n"
        f"———————————————\n"
        f"⏱️ ATURAN BASI: kalau harga {lim_txt} — <b>SKIP</b>, jarak ke TP sudah termakan.\n"
        f"⚠️ Harga = GC=F (futures). Terapkan JARAK SL/TP ke harga XAUUSD broker-mu.\n"
        f"📉 Edge hanya saat TREN jelas; tahun sideways ≈ impas. Sinyal baru bisa datang "
        f"saat posisi TP2 masih jalan — sesuaikan sendiri.\n"
        f"DEMO/manual. Backtest ≠ hasil masa depan."
    )


def send_telegram(token: str, chat_id: str, text: str):
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
    if not r.ok:
        print(f"Telegram API {r.status_code}: {r.text}")
    r.raise_for_status()
    return r.json()


def load_state():
    return json.load(open(STATE_PATH)) if os.path.exists(STATE_PATH) else {}


def save_state(s):
    json.dump(s, open(STATE_PATH, "w"))


def position_closed(pos: dict, df_after: pd.DataFrame):
    """SL & TP di bar sama -> SL dulu (konservatif, sama dengan backtest)."""
    d = 1 if pos["signal"] == "BUY" else -1
    held = 0
    for _, row in df_after.iterrows():
        held += 1
        hit_sl = (row["low"] <= pos["sl"]) if d > 0 else (row["high"] >= pos["sl"])
        hit_tp = (row["high"] >= pos["tp"]) if d > 0 else (row["low"] <= pos["tp"])
        if hit_sl:
            return True, "SL"
        if hit_tp:
            return True, "TP"
        if held >= MAX_HOLD_M5:
            return True, "TIME"
    return False, "open"


def check_once(dry_run=False):
    token = os.environ.get("TG_BOT_TOKEN", "")
    chat = os.environ.get("TG_CHAT_ID", "")
    balance = float(os.environ.get("TG_BALANCE", 10000.0))
    risk_pct = float(os.environ.get("TG_RISK", 2.0))

    df = get_m5()
    if len(df) < AMED_N + ATR_P + 10:
        print(f"data kurang: {len(df)} bar"); return
    state = load_state()

    # 1) kelola posisi hipotetis yang masih terbuka (dedup ala backtest: satu posisi)
    pos = state.get("position")
    if pos:
        after = df[df.index > pd.Timestamp(pos["bar_time"])]
        closed, why = position_closed(pos, after)
        if closed:
            print(f"posisi {pos['signal']} {pos['bar_time']} selesai: {why}")
            state["position"] = None
            save_state(state)
        else:
            print(f"posisi {pos['signal']} {pos['bar_time']} masih open -> tidak cari sinyal baru")
            return

    # 2) sinyal di bar terakhir yang sudah close
    sig = evaluate_last_bar(df, balance=balance, risk_pct=risk_pct)
    last_bar = str(df.index[-1])
    if sig is None:
        print(f"[{utcnow():%Y-%m-%d %H:%M} UTC] tidak ada sinyal | bar {last_bar}")
        return
    if state.get("last_signal_bar") == sig["bar_time"]:
        print(f"sinyal bar {sig['bar_time']} sudah pernah dikirim"); return
    bar_close = pd.Timestamp(sig["bar_time"]) + pd.Timedelta(minutes=5)
    age_min = (utcnow() - bar_close.to_pydatetime()).total_seconds() / 60.0
    if age_min > FRESH_MIN:
        print(f"sinyal {sig['signal']} bar {sig['bar_time']} BASI ({age_min:.0f} mnt) -> skip")
        state["last_signal_bar"] = sig["bar_time"]; save_state(state)
        return

    msg = format_msg(sig, age_min)
    if dry_run or not token or not chat:
        print("--- DRY ANGELFING_M5 ---\n" + msg)
    else:
        send_telegram(token, chat, msg)
        print(f"sent ANGELFING_M5 {sig['signal']} bar {sig['bar_time']} (umur {age_min:.0f} mnt)")
    state["position"] = sig
    state["last_signal_bar"] = sig["bar_time"]
    save_state(state)


def main():
    args = sys.argv[1:]
    if "--test" in args:
        token = os.environ.get("TG_BOT_TOKEN", ""); chat = os.environ.get("TG_CHAT_ID", "")
        txt = ("✅ <b>ANGELFING_M5 bot AKTIF</b>\nJadwal: tiap 15 mnt, sesi London (08:00-16:30 UTC, Sen-Jum).\n"
               "Sinyal engulfing M5 + tren M15/M30, SL 1.5×ATR, TP 1.5R.\n"
               f"🕒 {utcnow():%Y-%m-%d %H:%M} UTC")
        if token and chat:
            send_telegram(token, chat, txt); print("test message sent")
        else:
            print(txt)
        return
    check_once(dry_run="--dry-run" in args)


if __name__ == "__main__":
    main()
