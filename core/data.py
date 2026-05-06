from typing import Optional
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=600, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename_axis("Date").reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["Close"]).copy()
    df.set_index("Date", inplace=True)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_vix() -> Optional[float]:
    df = yf.download("^VIX", period="10d", interval="1d", progress=False)
    if df is None or df.empty:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_spy() -> pd.DataFrame:
    return fetch_ohlcv("SPY", "2y", "1d")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "name":         info.get("longName", ticker),
            "sector":       info.get("sector", ""),
            "industry":     info.get("industry", ""),
            "market_cap":   info.get("marketCap"),
            "pe_ratio":     info.get("trailingPE"),
            "beta":         info.get("beta"),
            "avg_volume":   info.get("averageVolume"),
            "short_ratio":  info.get("shortRatio"),
            "float_shares": info.get("floatShares"),
        }
    except Exception:
        return {"name": ticker}
