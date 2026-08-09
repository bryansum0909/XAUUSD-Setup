"""ANGELFING_M5 — engulfing M5 Telegram signal bot (MANUAL execution, never trades).

Metode dari project "Angelfing Confirmation" (D:\\Bryan Stuff\\Claude Trading\\
Angelfing Confirmation\\xauusd-backtest, lihat CLAUDE.md di sana):
  * M5, sesi LONDON+NY+ASIA: 03:00-17:00 + 18:00-02:00 EST
    = 08:00-22:00 + 23:00-07:00 UTC, BUY & SELL. (Keputusan user 2026-08-02.)
    Sweep sesi (kausal): NY menambah +24% R di rezim 2026 (PF/streak sama);
    Asia menambah lagi +12R di 2026 (PF 1.46->1.40, streak TP1 5->6) TAPI
    memberatkan TP2 (streak 2026 13->19; histori s/d 24) dan tahun sideways
    (2023: -80.9R). Pesan sinyal sesi ASIA membawa peringatan "TP1 saja".
    Full-history semua perpanjangan dilutif; London-saja terbaik utk RR3.
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
  python angelfing_signal.py --loop            # REALTIME daemon (VPS): bangun tiap
                                               # bar M5 close +25 dtk, retry fetch
                                               # sampai bar baru muncul di Yahoo
  python angelfing_signal.py --test            # kirim pesan uji koneksi
State dedup: angelfing_state.json (terpisah dari bot H1 agar tidak bentrok git);
path bisa dioverride via env ANGELFING_STATE_PATH (dipakai di VPS agar git pull
tidak konflik dengan state yang di-commit workflow GitHub).
"""
from __future__ import annotations
import os, sys, json, time, datetime as dt
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
STATE_PATH = os.environ.get("ANGELFING_STATE_PATH", os.path.join(ROOT, "angelfing_state.json"))


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

# Sesi dalam UTC: London+NY 08:00-22:00, Asia 23:00-07:00 (wrap tengah malam).
# Gap 22-23 & 07-08 UTC (=17-18 & 02-03 EST) memang di luar definisi sesi backtest.
def in_session(hh):
    return ((hh >= 8.0) & (hh < 22.0)) | (hh >= 23.0) | (hh < 7.0)


def is_asia_hour(hour_utc: int) -> bool:
    return hour_utc >= 23 or hour_utc < 7
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
    sess = in_session(hh)
    vol = df["atr"] > df["amed"]
    L = (df["t15"] == 1) & (df["t30"] == 1) & touch & bull & vol & sess
    S = (df["t15"] == -1) & (df["t30"] == -1) & touch & bear & vol & sess
    return L.fillna(False), S.fillna(False), df


def spot_basis(gc_close: pd.Series):
    """Selisih GC=F (futures) - XAUUSD spot, dihitung ULANG tiap sinyal.

    WAJIB dinamis: basis melompat ~$60 saat GC=F ganti kontrak (terukur 29 Jul 2026:
    +$7 -> +$58), jadi angka tetap akan menyesatkan. Sumber spot = Dukascopy jam
    penuh terakhir (gratis, tanpa key; sama dengan sumber data backtest 2026).
    Basis sendiri bergerak lambat (std per jam $0.33) sehingga acuan 1-2 jam lalu
    tetap akurat. Return (basis, waktu_referensi) atau (None, None) kalau gagal.
    """
    import requests
    sess = requests.Session()
    now = utcnow().replace(minute=0, second=0, microsecond=0)
    # kandidat jam: 4 jam terakhir, lalu (utk akhir pekan / Senin dini hari yang
    # semua jam mundurnya kosong) langsung lompat ke jam perdagangan akhir Jumat.
    cands = [now - dt.timedelta(hours=k) for k in (1, 2, 3, 4)]
    fri = now
    while fri.weekday() != 4:                       # mundur ke Jumat terakhir
        fri -= dt.timedelta(days=1)
    fri = fri.replace(hour=21)
    if fri < now:
        cands += [fri, fri - dt.timedelta(hours=1), fri - dt.timedelta(hours=2)]
    for hr in cands:
        try:
            ticks = dl.fetch_dukascopy_hour("XAUUSD", hr, sess)
        except Exception as e:
            # 503 transien pada SATU jam tidak boleh membatalkan semuanya
            print(f"basis: jam {hr:%d/%m %H}h gagal ({str(e)[:60]}), coba jam lain")
            continue
        if not len(ticks):
            continue
        t_ref = ticks.index[-1]
        spot = float((ticks["ask"].iloc[-1] + ticks["bid"].iloc[-1]) / 2)
        near = gc_close[gc_close.index <= t_ref]
        if not len(near):
            continue
        return round(float(near.iloc[-1]) - spot, 2), t_ref
    print("basis spot gagal: semua kandidat jam kosong/error")
    return None, None


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


def evaluate_recent(df: pd.DataFrame, *, balance: float, risk_pct: float,
                    round_lot_step: float = 0.01):
    """Sinyal TERBARU di seluruh riwayat (bar mana pun, bukan hanya bar terakhir).
    Freshness dinilai oleh pemanggil. Penting: cron 15-menit + bar 5-menit berarti
    bar sinyal yang close di antara dua run (menit :05/:10) BUKAN bar terakhir
    saat run berikutnya — kalau hanya cek iloc[-1], 2 dari 3 bar sinyal bisa lolos."""
    L, S, m = signal_masks(df)
    if len(m) < AMED_N + ATR_P + 5:
        return None
    hits = m.index[(L | S).values]
    if len(hits) == 0:
        return None
    ts = hits[-1]
    direction = "BUY" if bool(L.loc[ts]) else "SELL"
    m = m.loc[:ts]                      # nilai indikator & harga pada bar sinyal itu
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


def spot_block(sig: dict, basis, t_ref) -> str:
    """Blok level SPOT (setara harga broker) — inilah yang dipakai user untuk entry."""
    if basis is None:
        return ("⚠️ Konversi spot gagal (feed Dukascopy) — pakai JARAK di bawah, "
                "diterapkan dari harga broker-mu saat ini.\n")
    d = 1 if sig["signal"] == "BUY" else -1
    e = sig["entry"] - basis
    return (f"💱 <b>LEVEL SPOT (pakai ini di broker-mu)</b> — basis futures {basis:+.2f} "
            f"per {t_ref:%d/%m %H:%M} UTC\n"
            f"   ➡️ Entry ~<b>{e:.2f}</b>  🛑 SL <b>{e - d*sig['sl_distance']:.2f}</b>  "
            f"🎯 TP1 <b>{e + d*sig['tp1_distance']:.2f}</b>  🎯 TP2 <b>{e + d*sig['tp2_distance']:.2f}</b>\n"
            f"   (broker-mu bisa beda ±$1; kalau meleset jauh, pakai JARAK dari harga broker)\n")


def format_msg(sig: dict, age_min: float, basis=None, t_ref=None) -> str:
    arrow = "🟢 BUY" if sig["signal"] == "BUY" else "🔴 SELL"
    asia = is_asia_hour(int(sig["bar_time"][11:13]))
    asia_line = ("🌏 <b>Sinyal sesi ASIA — sebaiknya TP1 SAJA.</b> Backtest: TP2 dengan Asia "
                 "streak s/d 19-24; TP1 hanya 6.\n") if asia else ""
    tol = round(0.3 * sig["sl_distance"], 2)
    # batas basi ditampilkan dalam harga SPOT bila konversi tersedia (itu yang dilihat user)
    ref = sig["entry"] - basis if basis is not None else sig["entry"]
    unit = "spot" if basis is not None else "GC=F"
    if sig["signal"] == "BUY":
        lim_txt = f"sudah DI ATAS {ref + tol:.2f} ({unit})"
    else:
        lim_txt = f"sudah DI BAWAH {ref - tol:.2f} ({unit})"
    return (
        f"<b>XAUUSD M5 — {arrow} (ANGELFING)</b>\n"
        f"📊 Setup: <b>ANGELFING_M5</b> — engulfing + tren M15/M30 + sentuh EMA20 + ATR tinggi, sesi London+NY+Asia\n"
        f"🕒 Bar M5 closed: {sig['bar_time']} UTC (umur sinyal ~{age_min:.0f} mnt)\n"
        f"{asia_line}"
        f"———————————————\n"
        f"{spot_block(sig, basis, t_ref)}"
        f"———————————————\n"
        f"<i>Acuan data (GC=F futures):</i>\n"
        f"➡️ <b>Entry</b>: ~{sig['entry']} (market SEKARANG — sinyal M5 cepat basi)\n"
        f"🛑 <b>Stop Loss</b>: {sig['sl']}  (jarak {sig['sl_distance']} = 1.5×ATR)\n"
        f"🎯 <b>TP1</b>: {sig['tp']}  (RR 1.5 — unggul di rezim 2026: PF 1.46, WR 50%, streak ≤5)\n"
        f"🎯 <b>TP2</b>: {sig['tp2']}  (RR 3 — konsisten antar tahun: +50R/3.5th, TAPI WR 28%, streak s/d 14)\n"
        f"👉 Pilih SATU sesuai rezim Gold fundamental watch (tren kuat searah → TP2 layak; bear cepat/rebound → TP1)\n"
        f"📦 <b>Lot</b>: {sig['lot']}  (risk {sig['risk_pct']}% = ${sig['risk_money']} dari ${sig['balance']:.0f}; "
        f"risk riil setelah pembulatan ≈ ${sig['risk_actual']})\n"
        f"———————————————\n"
        f"⏱️ ATURAN BASI: kalau harga {lim_txt} — <b>SKIP</b>, jarak ke TP sudah termakan.\n"
        f"⚠️ Sinyal dibaca dari GC=F; level spot di atas sudah dikonversi untuk broker-mu.\n"
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


def check_once(dry_run=False, expected_last=None, retries=2):
    """expected_last (mode loop): stempel bar M5 yang baru saja close — retry fetch
    singkat. KENYATAAN FEED (diukur 2026-08-03): GC=F Yahoo = data CME tertunda
    ~10 menit by-license, bar baru biasanya muncul SATU siklus kemudian — jadi
    retry panjang sia-sia; evaluate_recent menangkap bar telat di siklus berikut
    (latensi sinyal efektif ~6-11 mnt, masih dalam gerbang segar 20 mnt)."""
    token = os.environ.get("TG_BOT_TOKEN", "")
    chat = os.environ.get("TG_CHAT_ID", "")
    balance = float(os.environ.get("TG_BALANCE", 10000.0))
    risk_pct = float(os.environ.get("TG_RISK", 2.0))

    df = get_m5()
    if expected_last is not None:
        for _ in range(retries):
            if len(df) and df.index[-1] >= expected_last:
                break
            time.sleep(20)
            df = get_m5()
        else:
            print(f"[{utcnow():%Y-%m-%d %H:%M}] FEED LAG: bar {expected_last} belum muncul "
                  f"setelah {retries}x retry (terakhir di feed: {df.index[-1] if len(df) else '-'}) "
                  f"— evaluasi pakai data yang ada", flush=True)
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

    # 2) sinyal terbaru di bar mana pun yang sudah close (freshness dicek di bawah)
    sig = evaluate_recent(df, balance=balance, risk_pct=risk_pct)
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

    basis, t_ref = spot_basis(df["close"])
    msg = format_msg(sig, age_min, basis, t_ref)
    if dry_run or not token or not chat:
        print("--- DRY ANGELFING_M5 ---\n" + msg)
    else:
        send_telegram(token, chat, msg)
        print(f"sent ANGELFING_M5 {sig['signal']} bar {sig['bar_time']} (umur {age_min:.0f} mnt)")
    state["position"] = sig
    state["last_signal_bar"] = sig["bar_time"]
    save_state(state)


def market_open(bar_start: pd.Timestamp) -> bool:
    """Pasar emas: buka Minggu 23:00 UTC, tutup Jumat 22:00 UTC."""
    wd, hh = bar_start.weekday(), bar_start.hour + bar_start.minute / 60.0
    if wd == 5: return False                 # Sabtu
    if wd == 6: return hh >= 23.0            # Minggu: baru buka 23:00
    if wd == 4 and hh >= 22.0: return False  # Jumat malam
    return True


def loop():
    """Daemon realtime (VPS): tiap bar M5 close (+25 dtk buffer feed) evaluasi sekali.
    Di luar sesi/akhir pekan hanya tidur — tanpa fetch."""
    print(f"[{utcnow():%Y-%m-%d %H:%M}] ANGELFING_M5 realtime loop start "
          f"(sesi UTC 08-22 & 23-07, state={STATE_PATH})", flush=True)
    while True:
        now = utcnow()
        wait = (5 - now.minute % 5) * 60 - now.second + 25
        time.sleep(max(5, wait))
        boundary = utcnow().replace(second=0, microsecond=0)
        boundary = boundary - dt.timedelta(minutes=boundary.minute % 5)
        bar_start = pd.Timestamp(boundary) - pd.Timedelta(minutes=5)
        hh = bar_start.hour + bar_start.minute / 60.0
        if not (market_open(bar_start) and in_session(np.array([hh]))[0]):
            continue
        try:
            check_once(expected_last=bar_start)
        except Exception as e:
            print(f"[{utcnow():%Y-%m-%d %H:%M}] loop error: {e}", flush=True)


def main():
    args = sys.argv[1:]
    if "--test" in args:
        token = os.environ.get("TG_BOT_TOKEN", ""); chat = os.environ.get("TG_CHAT_ID", "")
        txt = ("✅ <b>ANGELFING_M5 bot AKTIF</b>\nJadwal: tiap 15 mnt, sesi London+NY+Asia "
               "(08:00-22:00 &amp; 23:00-07:00 UTC).\n"
               "Sinyal engulfing M5 + tren M15/M30, SL 1.5×ATR, TP 1.5R.\n"
               f"🕒 {utcnow():%Y-%m-%d %H:%M} UTC")
        if token and chat:
            send_telegram(token, chat, txt); print("test message sent")
        else:
            print(txt)
        return
    if "--loop" in args:
        loop(); return
    check_once(dry_run="--dry-run" in args)


if __name__ == "__main__":
    main()
