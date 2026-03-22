from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename_axis("Date").reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[keep].dropna(subset=["Close"]).copy()
    df.set_index("Date", inplace=True)
    return df


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_vix() -> Optional[float]:
    df = yf.download("^VIX", period="10d", interval="1d", progress=False)
    if df is None or df.empty:
        return None
    try:
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return None
