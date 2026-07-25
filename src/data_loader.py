"""Real-market data acquisition for XAUUSD.

Sources (priority order for a *spot* dataset):
  1. Dukascopy tick data  -> true XAUUSD spot ticks -> resample to M1/M5/M30/H1.
     Free, ~3-5y available, but many small HTTP files (build incrementally).
  2. Yahoo GC=F (COMEX gold futures) -> real intraday PROXY for spot gold.
     Only a few requests; H1 ~730d, M5/M30 ~60d. Clearly a proxy, not spot.
  3. Generic CSV import (broker export / MT5 / HistData M1).

NO synthetic data is ever produced here — every function returns real ticks/bars
or raises. Synthetic series live only in tests (clearly named).
"""
from __future__ import annotations
import io
import lzma
import struct
import time
import datetime as dt
import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"}

# ----------------------------------------------------------------------------
# 1. Dukascopy tick data
# ----------------------------------------------------------------------------
DUKA_URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
# XAUUSD prices in Dukascopy are stored as int * 1000 (3 decimals).
DUKA_POINT = {"XAUUSD": 0.001}


def _decompress_bi5(raw: bytes) -> bytes:
    if not raw:
        return b""
    for fmt in (lzma.FORMAT_AUTO, lzma.FORMAT_ALONE):
        try:
            return lzma.LZMADecompressor(format=fmt).decompress(raw)
        except lzma.LZMAError:
            continue
    raise lzma.LZMAError("could not decompress bi5")


def _parse_bi5(data: bytes, hour_start: dt.datetime, point: float) -> pd.DataFrame:
    # 20-byte big-endian records: ms(uint32), ask(uint32), bid(uint32),
    #                              askvol(float32), bidvol(float32)
    n = len(data) // 20
    if n == 0:
        return pd.DataFrame(columns=["ask", "bid", "askvol", "bidvol"])
    rec = struct.unpack(">" + "IIIff" * n, data[: n * 20])
    arr = np.array(rec).reshape(n, 5)
    ms = arr[:, 0]
    ts = pd.to_datetime(hour_start) + pd.to_timedelta(ms, unit="ms")
    df = pd.DataFrame({
        "ask": arr[:, 1] * point,
        "bid": arr[:, 2] * point,
        "askvol": arr[:, 3],
        "bidvol": arr[:, 4],
    }, index=ts)
    return df


def fetch_dukascopy_hour(symbol: str, when: dt.datetime, session: requests.Session,
                         retries: int = 3) -> pd.DataFrame:
    point = DUKA_POINT[symbol]
    url = DUKA_URL.format(sym=symbol, y=when.year, m=when.month - 1, d=when.day, h=when.hour)
    for attempt in range(retries):
        try:
            r = session.get(url, headers=UA, timeout=30)
            if r.status_code == 404:
                return pd.DataFrame(columns=["ask", "bid", "askvol", "bidvol"])
            r.raise_for_status()
            data = _decompress_bi5(r.content)
            return _parse_bi5(data, when.replace(minute=0, second=0, microsecond=0), point)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def ticks_to_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    """Mid-price M1 OHLCV from tick ask/bid."""
    if ticks.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    mid = (ticks["ask"] + ticks["bid"]) / 2
    vol = ticks["askvol"] + ticks["bidvol"]
    o = mid.resample("1min").first()
    h = mid.resample("1min").max()
    l = mid.resample("1min").min()
    c = mid.resample("1min").last()
    v = vol.resample("1min").sum()
    m1 = pd.concat([o, h, l, c, v], axis=1)
    m1.columns = ["open", "high", "low", "close", "volume"]
    return m1.dropna(subset=["open"])


def download_dukascopy_range(symbol: str, start: dt.datetime, end: dt.datetime,
                             progress=print) -> pd.DataFrame:
    """Download ticks hour-by-hour and return concatenated M1 bars.
    Skips weekends (Sat all day, Sun before 21:00 UTC, Fri after 21:00 UTC)."""
    sess = requests.Session()
    cur = start.replace(minute=0, second=0, microsecond=0)
    all_m1 = []
    count = 0
    while cur < end:
        wd = cur.weekday()  # Mon=0 .. Sun=6
        closed = (wd == 5) or (wd == 6 and cur.hour < 21) or (wd == 4 and cur.hour >= 21)
        if not closed:
            try:
                ticks = fetch_dukascopy_hour(symbol, cur, sess)
                if not ticks.empty:
                    all_m1.append(ticks)
            except Exception as e:
                progress(f"  ! {cur} failed: {e}")
            count += 1
            if count % 200 == 0:
                progress(f"  ...{count} hours fetched, at {cur}")
        cur += dt.timedelta(hours=1)
    if not all_m1:
        raise RuntimeError("no tick data downloaded")
    ticks = pd.concat(all_m1).sort_index()
    ticks = ticks[~ticks.index.duplicated(keep="first")]
    return ticks_to_m1(ticks)


# ----------------------------------------------------------------------------
# 2. Yahoo GC=F futures proxy (immediate, few requests)
# ----------------------------------------------------------------------------
YF_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"


def fetch_yahoo(symbol: str = "GC=F", interval: str = "1h", range_: str = "730d",
                retries: int = 4) -> pd.DataFrame:
    params = {"interval": interval, "range": range_, "includePrePost": "false"}
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(YF_URL.format(sym=symbol), params=params, headers=UA, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); last = "429"; continue
            r.raise_for_status()
            j = r.json()["chart"]["result"][0]
            ts = pd.to_datetime(j["timestamp"], unit="s", utc=True)
            q = j["indicators"]["quote"][0]
            df = pd.DataFrame({
                "open": q["open"], "high": q["high"], "low": q["low"],
                "close": q["close"], "volume": q.get("volume"),
            }, index=ts)
            df.index = df.index.tz_convert("UTC").tz_localize(None)
            return df.dropna(subset=["open", "high", "low", "close"])
        except Exception as e:
            last = str(e); time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"yahoo fetch failed: {last}")


# ----------------------------------------------------------------------------
# 3. Generic CSV / MT5 export
# ----------------------------------------------------------------------------
def load_csv(path: str, tz: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in df.columns if c in ("time", "datetime", "date", "timestamp")), None)
    if tcol is None:
        raise ValueError("no time column found")
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol).sort_index()
    if tz:
        df.index = df.index.tz_localize(tz).tz_convert("UTC").tz_localize(None)
    return df


def save_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(path)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
