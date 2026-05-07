"""
data.py — yfinance 실시간 최대화

- 장중 30초 캐시 (최대한 빠른 갱신)
- prepost=True (프리/애프터마켓 포함)
- 장외 10분 캐시
- 재시도 1회
"""
from typing import Optional
import time
import datetime as dt
import pandas as pd
import streamlit as st
import yfinance as yf

def _is_market_open() -> bool:
    try:
        import pytz
        et = pytz.timezone("America/New_York")
        now_et = dt.datetime.now(et)
        if now_et.weekday() >= 5: return False
        t = now_et.time()
        return dt.time(9, 25) <= t <= dt.time(16, 5)
    except ImportError:
        now_utc = dt.datetime.utcnow()
        if now_utc.weekday() >= 5: return False
        t_utc = now_utc.time()
        return dt.time(13, 25) <= t_utc <= dt.time(21, 5)

def _download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    for attempt in range(2):
        try:
            df = yf.download(
                ticker, period=period, interval=interval,
                auto_adjust=False, prepost=True,
                progress=False, threads=False,
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            if attempt == 0: time.sleep(0.5)
    return pd.DataFrame()

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.rename_axis("Date").reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    keep = ["Date","Open","High","Low","Close","Volume"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["Close"]).copy()
    df.set_index("Date", inplace=True)
    if "Volume" in df.columns: df["Volume"] = df["Volume"].clip(lower=0)
    df = df[df["Close"] > 0]
    return df

@st.cache_data(ttl=30, show_spinner=False)
def _fetch_realtime(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return _clean(_download(ticker, period, interval))

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return _clean(_download(ticker, period, interval))

def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return _fetch_realtime(ticker, period, interval) if _is_market_open() else _fetch_cached(ticker, period, interval)

@st.cache_data(ttl=30, show_spinner=False)
def fetch_vix() -> Optional[float]:
    df = _clean(_download("^VIX", "10d", "1d"))
    if df.empty: return None
    try: return float(df["Close"].dropna().iloc[-1])
    except: return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_spy() -> pd.DataFrame:
    return fetch_ohlcv("SPY", "2y", "1d")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "beta": info.get("beta"),
        }
    except: return {"name": ticker}
