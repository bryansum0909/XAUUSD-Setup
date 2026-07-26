"""XAUUSD H1 Donchian-breakout Telegram SIGNAL bot (for MANUAL demo execution).

It NEVER trades. It watches the last closed H1 bar and, on a fresh breakout,
sends you a Telegram message with entry / SL / TP / LOT so you place the order
yourself on your demo account.

Setup (you do this once — I never see your token):
  1. In Telegram, message @BotFather -> /newbot -> copy the bot TOKEN.
  2. Message your new bot once (say "hi"), then open
     https://api.telegram.org/bot<TOKEN>/getUpdates and copy your chat "id".
  3. Copy telegram_config.example.json -> telegram_config.json and fill in
     token, chat_id, balance, risk_pct.

Run:
  python telegram_signal_bot.py --replay 800     # show signals on history (no token needed)
  python telegram_signal_bot.py --once --dry-run  # check current bar, print only
  python telegram_signal_bot.py --once            # check current bar, send to Telegram
  python telegram_signal_bot.py --loop            # keep running, check each hour
Schedule '--once' hourly (Windows Task Scheduler) a few minutes after each H1 close.
"""
import os, sys, json, time, datetime as dt
try:
    sys.stdout.reconfigure(encoding="utf-8")   # so emoji print on Windows consoles
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import requests
import signal_engine as se
import multi_signal_engine as mse
from src import data_loader as dl
from src.resampler import resample_ohlc

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "processed")
CFG_PATH = os.path.join(ROOT, "telegram_config.json")
STATE_PATH = os.path.join(ROOT, "telegram_bot_state.json")


def load_cfg():
    if os.path.exists(CFG_PATH):
        c = json.load(open(CFG_PATH))
    else:
        c = {}
    # env vars win when present (used by the cloud/GitHub-Actions deployment)
    c["bot_token"] = os.environ.get("TG_BOT_TOKEN") or c.get("bot_token", "")
    c["chat_id"] = os.environ.get("TG_CHAT_ID") or c.get("chat_id", "")
    c.setdefault("balance", 10000.0)
    c.setdefault("risk_pct", 2.0)   # user-set 2% per trade (lot sizing)
    c.setdefault("source", "dukascopy")   # 'dukascopy' (spot, matches backtest) | 'yahoo' | 'parquet'
    c.setdefault("n", 40); c.setdefault("rr", 1.5); c.setdefault("max_hold", 300)
    # which setups to send. Drop "PSAR" (noisy) or keep KELT_M1 (patented RR1:3) as you like.
    # focus on the 4 validated main methods; old basket (DONCH_H4/MACD_H4/PSAR) disabled
    c.setdefault("enabled_setups", ["KELT_M1", "DONCH_M1", "ORB_M15", "MA_M30", "TREND_M30"])
    if os.environ.get("TG_SETUPS"):
        c["enabled_setups"] = [s.strip() for s in os.environ["TG_SETUPS"].split(",") if s.strip()]
    if os.environ.get("TG_BALANCE"):
        c["balance"] = float(os.environ["TG_BALANCE"])
    if os.environ.get("TG_RISK"):
        c["risk_pct"] = float(os.environ["TG_RISK"])
    if os.environ.get("TG_SOURCE"):
        c["source"] = os.environ["TG_SOURCE"]
    return c


def get_h1(source: str, lookback_days: int = 8) -> pd.DataFrame:
    if source == "parquet":
        return pd.read_parquet(os.path.join(PROC, "XAUUSD_H1_spot.parquet")).sort_index()
    if source == "yahoo":
        df = dl.fetch_yahoo("GC=F", interval="1h", range_=f"{lookback_days}d")
        return df.sort_index()
    # dukascopy spot (matches the backtest exactly); drop the still-forming hour
    end = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(days=lookback_days)
    m1 = dl.download_dukascopy_range("XAUUSD", start, end, progress=lambda *_: None)
    return resample_ohlc(m1, "H1")


def get_m15(lookback_days: int = 55) -> pd.DataFrame:
    """M15 bars for ORB_M15 (Yahoo caps 15m history at ~60 days — enough for today's OR)."""
    df = dl.fetch_yahoo("GC=F", interval="15m", range_=f"{lookback_days}d")
    return df[~df.index.duplicated()].sort_index()


def get_m30(lookback_days: int = 58) -> pd.DataFrame:
    """M30 bars for MA_M30 golden cross (needs ~200 bars for EMA200 — 58d gives ~2800)."""
    df = dl.fetch_yahoo("GC=F", interval="30m", range_=f"{lookback_days}d")
    return df[~df.index.duplicated()].sort_index()


def format_ma_m30(s: dict, event: str) -> str:
    """event: 'GOLDEN' (enter long) or 'DEATH' (exit)."""
    if event == "GOLDEN":
        stop = round(s["close"] - 3 * s["atr"], 2) if s.get("atr") else "-"
        return (
            f"<b>🟢 GOLDEN CROSS M30 — MASUK LONG XAUUSD</b>\n"
            f"📊 Setup: <b>MA_M30</b> (EMA50 potong ke ATAS EMA200)\n"
            f"🕒 Bar M30: {s['bar_time']} UTC\n"
            f"———————————————\n"
            f"➡️ <b>Entry</b>: ~{s['close']} (market)\n"
            f"📈 EMA50 {s['ema50']} &gt; EMA200 {s['ema200']}\n"
            f"🛑 <b>Stop pengaman</b>: ~{stop} (3×ATR — hanya jaring pengaman)\n"
            f"🚪 <b>Exit utama</b>: saat DEATH CROSS (EMA50 balik di bawah EMA200), BUKAN TP tetap\n"
            f"———————————————\n"
            f"⏳ Ini POSISI-TREN — tahan berhari-hari, biarkan profit berjalan (RR efektif ~4,4, WR ~32%).\n"
            f"⚠️ DEMO/manual. Backtest ≠ hasil masa depan."
        )
    return (
        f"<b>🔴 DEATH CROSS M30 — KELUAR LONG XAUUSD</b>\n"
        f"📊 Setup: <b>MA_M30</b> (EMA50 potong ke BAWAH EMA200)\n"
        f"🕒 Bar M30: {s['bar_time']} UTC\n"
        f"———————————————\n"
        f"🚪 <b>Tutup posisi MA_M30</b> di ~{s['close']}.\n"
        f"📉 EMA50 {s['ema50']} &lt; EMA200 {s['ema200']}\n"
        f"Tunggu golden cross berikutnya untuk masuk lagi."
    )


def format_trend_m30(s: dict, event: str) -> str:
    """event: 'ENTER' (M30 EMA10>30 & H1 uptrend both on) or 'EXIT'."""
    if event == "ENTER":
        stop = round(s["close"] - 3 * s["atr"], 2) if s.get("atr") else "-"
        return (
            f"<b>🟢 TREND_M30 — MASUK LONG XAUUSD</b>\n"
            f"📊 Setup: <b>TREND_M30</b> (M30 EMA10&gt;EMA30 + tren H1 naik)\n"
            f"🕒 Bar M30: {s['bar_time']} UTC\n"
            f"———————————————\n"
            f"➡️ <b>Entry</b>: ~{s['close']} (market)\n"
            f"📈 M30 EMA10 {s['ema10']} &gt; EMA30 {s['ema30']} · H1 uptrend ✅\n"
            f"🛑 <b>Stop pengaman</b>: ~{stop} (3×ATR)\n"
            f"🚪 <b>Exit</b>: saat M30 EMA10&lt;EMA30 ATAU tren H1 patah\n"
            f"———————————————\n"
            f"⏳ Posisi-tren (~1 entry per 2 hari). Validasi: untung tiap tahun, PF 1,45.\n"
            f"⚠️ DEMO/manual. Backtest ≠ hasil masa depan."
        )
    return (
        f"<b>🔴 TREND_M30 — KELUAR LONG XAUUSD</b>\n"
        f"📊 Setup: <b>TREND_M30</b>\n"
        f"🕒 Bar M30: {s['bar_time']} UTC\n"
        f"———————————————\n"
        f"🚪 <b>Tutup posisi TREND_M30</b> di ~{s['close']}.\n"
        f"📉 Pemicu: M30 EMA10&lt;EMA30 atau tren H1 patah.\n"
        f"Tunggu sinyal masuk berikutnya."
    )


def format_msg(sig: dict) -> str:
    arrow = "🟢 BUY" if sig["signal"] == "BUY" else "🔴 SELL"
    tf = "M15" if sig.get("key") == "ORB_M15" else "H1"
    return (
        f"<b>XAUUSD {tf} — {arrow}</b>\n"
        f"📊 Setup: <b>{sig['setup']}</b>\n"
        f"🕒 Bar closed: {sig['bar_time']} UTC\n"
        f"———————————————\n"
        f"➡️ <b>Entry</b>: ~{sig['entry']} (market now / next H1 open)\n"
        f"🛑 <b>Stop Loss</b>: {sig['sl']}  ({sig['sl_distance']} = 1.5×ATR)\n"
        f"🎯 <b>Take Profit</b>: {sig['tp']}  (RR {sig['rr']}, {sig['tp_distance']})\n"
        f"📦 <b>Lot</b>: {sig['lot']}  (risk {sig['risk_pct']}% = ${sig['risk_money']} on ${sig['balance']:.0f})\n"
        f"🚪 Exit: tutup saat kena SL atau TP (tidak ada TP ke-2)\n"
        f"———————————————\n"
        f"⚠️ DEMO / manual. Terapkan JARAK SL/TP di atas ke harga XAUUSD broker-mu (level bisa beda tipis).\n"
        f"Backtest ≠ future results. Not financial advice."
    )


def send_telegram(token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
    if not r.ok:
        # surface Telegram's exact 'description' (e.g. chat not found / can't parse entities)
        print(f"Telegram API {r.status_code}: {r.text}")
    r.raise_for_status()
    return r.json()


def load_state():
    return json.load(open(STATE_PATH)) if os.path.exists(STATE_PATH) else {"positions": {}}


def save_state(s):
    json.dump(s, open(STATE_PATH, "w"))


def position_closed(pos: dict, df_after: pd.DataFrame, max_hold=60):
    """Has a hypothetical position closed within df_after (bars AFTER entry)? Returns
    (closed: bool, reason). Conservative: if SL & TP in same bar -> SL first."""
    d = 1 if pos["signal"] == "BUY" else -1
    sl, tp = pos["sl"], pos["tp"]
    held = 0
    for _, row in df_after.iterrows():
        held += 1
        hit_sl = (row["low"] <= sl) if d > 0 else (row["high"] >= sl)
        hit_tp = (row["high"] >= tp) if d > 0 else (row["low"] <= tp)
        if hit_sl:
            return True, "SL"
        if hit_tp:
            return True, "TP"
        if held >= max_hold:
            return True, "TIME"
    return False, "open"


def get_dxy():
    try:
        return dl.fetch_yahoo("DX-Y.NYB", interval="1d", range_="400d")["close"]
    except Exception as e:
        print("DXY fetch failed (Method#1 gate skipped):", e)
        return None


def format_regime(st: dict, flipped=None) -> str:
    yn = lambda b: "✅" if b else "❌"
    if flipped == "ON":
        head = "🚨🟢 REGIME BARU AKTIF — uptrend + dolar lemah.\nMethod#1 & ORB_M15 mulai bisa entry LONG."
    elif flipped == "OFF":
        head = "🚨🔴 REGIME MATI — dolar menguat / tren patah.\nSTOP cari LONG (semua setup tidak akan trigger)."
    else:
        head = "🟢 REGIME AKTIF — boleh cari LONG" if st["regime_on"] else "🔴 REGIME OFF — jangan LONG dulu"
    dxy_line = ""
    if st["dxy"] is not None and st["dxy_sma100"] is not None:
        tag = "LEMAH ✅" if st["dxy_weak"] else "KUAT ❌"
        dxy_line = (f"💵 DXY {st['dxy']:.2f} vs SMA100 {st['dxy_sma100']:.2f} "
                    f"(selisih {st['dxy_gap']:+.2f} → dolar {tag})\n")
    return (
        f"<b>{head}</b>\n"
        f"———————————————\n"
        f"📈 Tren Harian (EMA50/200): {yn(st['daily_up'])}  ← wajib\n"
        f"{dxy_line}"
        f"📆 Tren Mingguan (EMA20/50): {yn(st['weekly_up'])}  ← tambahan utk Method#1\n"
        f"———————————————\n"
        f"REGIME ON = Tren Harian ✅ + Dolar LEMAH ✅ → aktifkan breakout (KELT/DONCH/ORB).\n"
        f"Method#1 juga butuh Mingguan ✅ + ADX&lt;25. Saat regime OFF, breakout diam; setup tren "
        f"(MA_M30/TREND_M30) tetap jalan pada logika trennya sendiri.\n"
        f"🕒 {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC"
    )


def check_once(cfg, dry_run=False):
    # 730d of H1 -> enough for Weekly EMA (Method#1) and H4 EMA200 (basket).
    df = get_h1("yahoo", lookback_days=730)
    df = df[~df.index.duplicated()].sort_index()
    dxy = get_dxy()
    state = load_state()
    positions = state.get("positions") or {}
    enabled = cfg.get("enabled_setups", mse.SETUP_KEYS)

    # --- TIGHT REGIME MONITOR: alert the moment the regime flips ON/OFF, plus a daily heartbeat.
    #     Both Method #1 family and ORB_M15 only fire while regime is ON, so this is the
    #     single most important thing for the user to know is up-to-date. ---
    st = mse.regime_status(df, dxy)
    prev_on = state.get("regime_on")
    flipped = "ON" if (prev_on is False and st["regime_on"]) else ("OFF" if (prev_on is True and not st["regime_on"]) else None)
    today = f"{dt.datetime.utcnow():%Y-%m-%d}"
    if flipped or state.get("regime_hb_date") != today:
        rmsg = format_regime(st, flipped)
        if dry_run or not cfg["bot_token"] or not cfg["chat_id"]:
            print("--- REGIME ---\n" + rmsg + "\n")
        else:
            send_telegram(cfg["bot_token"], cfg["chat_id"], rmsg)
            print(f"sent regime {'FLIP ' + flipped if flipped else 'heartbeat'}")
            state["regime_hb_date"] = today   # only mark 'done today' when ACTUALLY sent
    state["regime_on"] = st["regime_on"]

    # --- H1 basket + Method #1 family ---
    active = {s["key"]: s for s in mse.evaluate_all(
        df, dxy_close=dxy, balance=cfg["balance"], risk_pct=cfg["risk_pct"])}

    # --- ORB_M15 (breakout read from M15, regime read from the long H1 series) ---
    if "ORB_M15" in enabled:
        try:
            m15 = get_m15()
            for s in mse.evaluate_orb_m15(m15, df, dxy_close=dxy,
                                          balance=cfg["balance"], risk_pct=cfg["risk_pct"]):
                active[s["key"]] = s
        except Exception as e:
            print("ORB_M15 skipped:", e)

    # --- M30 trend-POSITION signals (state-based enter/exit; fetch M30 once) ---
    def _position_flip(status, state_key, fmt, enter_ev, exit_ev):
        if not status:
            return
        prev = state.get(state_key)
        if prev is None:
            state[state_key] = status["long"]          # baseline on first run, no alert
        elif status["long"] != prev:
            ev = enter_ev if status["long"] else exit_ev
            msg = fmt(status, ev)
            if dry_run or not cfg["bot_token"] or not cfg["chat_id"]:
                print(f"--- {state_key} {ev} ---\n" + msg + "\n")
            else:
                send_telegram(cfg["bot_token"], cfg["chat_id"], msg)
                print(f"sent {state_key} {ev}")
            state[state_key] = status["long"]

    if "MA_M30" in enabled or "TREND_M30" in enabled:
        try:
            m30 = get_m30()
        except Exception as e:
            print("M30 fetch skipped:", e); m30 = None
        if m30 is not None:
            if "MA_M30" in enabled:
                _position_flip(mse.ma_m30_status(m30), "ma_m30_long", format_ma_m30, "GOLDEN", "DEATH")
            if "TREND_M30" in enabled:
                _position_flip(mse.trend_m30f_status(m30), "trend_m30f_long", format_trend_m30, "ENTER", "EXIT")

    sent = 0
    for key in list(mse.SETUP_KEYS) + ["ORB_M15"]:
        if key not in enabled:
            continue
        pos = positions.get(key)
        if pos:  # a hypothetical trade for THIS setup is still open -> no new alert
            after = df[df.index > pd.Timestamp(pos["bar_time"])]
            closed, _ = position_closed(pos, after, max_hold=cfg.get("max_hold", 60))
            if not closed:
                continue
            positions[key] = None
        sig = active.get(key)
        if sig:
            msg = format_msg(sig)
            if dry_run or not cfg["bot_token"] or not cfg["chat_id"]:
                print(f"--- DRY [{key}] ---\n" + msg + "\n")
            else:
                send_telegram(cfg["bot_token"], cfg["chat_id"], msg)
                print(f"sent {key} alert for {sig['bar_time']}")
            positions[key] = sig
            sent += 1
    state["positions"] = positions
    save_state(state)
    if sent == 0:
        print(f"[{dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC] no new signal | regime_on={st['regime_on']}")


def replay(cfg, n_bars=800):
    """Replay the 3-setup basket over the last n_bars (spot parquet), per-setup
    one-position discipline. Precomputes indicators once for speed."""
    from src import indicators as ind
    from src import filters as flt
    df = pd.read_parquet(os.path.join(PROC, "XAUUSD_H1_spot.parquet")).sort_index()
    df = df[~df.index.duplicated()]
    c = df["close"]
    up = df["high"].rolling(40, min_periods=40).max().shift(1)
    lo = df["low"].rolling(40, min_periods=40).min().shift(1)
    m, s, _ = ind.macd(c)
    sar = ind.parabolic_sar(df); above = c > sar
    h4 = flt.htf_trend(df, "H4", 50, 200).to_numpy()
    atr = ind.atr(df, 14).to_numpy()
    # Method #1 pre-compute: Keltner upper, Daily & Weekly trend, DXY dollar-weak mask
    _, kelt_up, _ = ind.keltner(df, 20, 2.0)
    d1 = flt.htf_trend(df, "D1", 50, 200).to_numpy()
    wk = flt.htf_trend(df, "W", 20, 50).to_numpy()
    adxv = ind.adx(df, 14)[0].to_numpy()
    keltv = kelt_up.to_numpy()
    dxy = pd.read_parquet(os.path.join(PROC, "DXY_daily.parquet"))["close"]
    dxy = dxy[~dxy.index.duplicated()].sort_index()
    dxy_weak = (dxy < dxy.rolling(100, min_periods=100).mean()).shift(1)
    dxy_weak.index = dxy_weak.index.normalize()
    _day = df.index.normalize()
    dxy_mask = pd.Series(dxy_weak.reindex(_day.unique()).reindex(_day).fillna(False).to_numpy(), index=df.index).to_numpy()
    cv, upv, lov = c.to_numpy(), up.to_numpy(), lo.to_numpy()
    mv, sv, av = m.to_numpy(), s.to_numpy(), above.to_numpy()
    hi, low_ = df["high"].to_numpy(), df["low"].to_numpy()

    def donch(i):
        if cv[i] > upv[i] and cv[i-1] <= upv[i-1]: return "BUY"
        if cv[i] < lov[i] and cv[i-1] >= lov[i-1]: return "SELL"
        return None
    def macd_s(i):
        if mv[i] > sv[i] and mv[i-1] <= sv[i-1]: return "BUY"
        if mv[i] < sv[i] and mv[i-1] >= sv[i-1]: return "SELL"
        return None
    def psar(i):
        if av[i] and not av[i-1]: return "BUY"
        if not av[i] and av[i-1]: return "SELL"
        return None

    start = max(300, len(df) - n_bars)   # warmup for H4 EMA(200)
    pos = {k: None for k in mse.SETUP_KEYS}
    counts = {k: 0 for k in mse.SETUP_KEYS}
    shown = 0
    for i in range(start, len(df)):
        for k in mse.SETUP_KEYS:  # manage open trades
            p = pos[k]
            if p:
                d = 1 if p["dir"] == "BUY" else -1
                hit_sl = (low_[i] <= p["sl"]) if d > 0 else (hi[i] >= p["sl"])
                hit_tp = (hi[i] >= p["tp"]) if d > 0 else (low_[i] <= p["tp"])
                p["held"] += 1
                if hit_sl or hit_tp or p["held"] >= cfg.get("max_hold", 60):
                    pos[k] = None
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        sl_d = 1.5 * atr[i]
        cand = {}
        d = donch(i)
        if d and ((d == "BUY" and h4[i] == 1) or (d == "SELL" and h4[i] == -1)): cand["DONCH_H4"] = d
        mm = macd_s(i)
        if mm and ((mm == "BUY" and h4[i] == 1) or (mm == "SELL" and h4[i] == -1)): cand["MACD_H4"] = mm
        pp = psar(i)
        if pp: cand["PSAR"] = pp
        # Method #1: Keltner LONG breakout + Daily up + Weekly up + dollar weak + ADX<25 (RR 1:3)
        if (cv[i] > keltv[i] and cv[i-1] <= keltv[i-1] and d1[i] == 1 and wk[i] == 1
                and dxy_mask[i] and np.isfinite(adxv[i]) and adxv[i] < 25):
            cand["KELT_M1"] = "BUY"
        for k, dirn in cand.items():
            if pos[k] is None:
                rr_k = 3.0 if k == "KELT_M1" else 1.5
                tp_d = rr_k * sl_d
                entry = cv[i]
                sl = entry - sl_d if dirn == "BUY" else entry + sl_d
                tp = entry + tp_d if dirn == "BUY" else entry - tp_d
                pos[k] = {"dir": dirn, "sl": sl, "tp": tp, "held": 0}
                counts[k] += 1
                if shown < 9:
                    shown += 1
                    print(f"[{df.index[i]}] {k:9s} {dirn:4s} entry {entry:.2f} "
                          f"SL {sl:.2f} TP {tp:.2f}")
    total = sum(counts.values()); span = max(1, (df.index[-1] - df.index[start]).days)
    print(f"\nper-setup alerts (last {n_bars} bars, ~{span} days): {counts}")
    print(f"TOTAL {total} = ~{total/span*7:.1f} alerts/week across the basket")


def main():
    cfg = load_cfg()
    args = sys.argv[1:]
    if "--replay" in args:
        k = args.index("--replay")
        n = int(args[k + 1]) if k + 1 < len(args) and args[k + 1].isdigit() else 800
        cfg["source"] = "parquet"; replay(cfg, n); return
    dry = "--dry-run" in args
    if "--loop" in args:
        print("Loop mode: checking each hour. Ctrl-C to stop.")
        while True:
            try:
                check_once(cfg, dry_run=dry)
            except Exception as e:
                print("check failed:", e)
            time.sleep(3600)
    else:
        check_once(cfg, dry_run=dry)   # default: --once


if __name__ == "__main__":
    main()
