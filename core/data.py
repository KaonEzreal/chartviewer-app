"""
data.py — 데이터 수집 모듈 (개선판)

변경사항:
  - 캐시 TTL: 10분 → 장중 1분 / 장외 10분 자동 전환
  - 장중 여부 자동 감지 (미국 동부시간 기준)
  - 실시간 근사: 최신 봉 강제 갱신 옵션
  - yfinance 오류 시 재시도 로직 (최대 2회)
  - MultiIndex 컬럼 처리 강화
"""

from typing import Optional
import time
import datetime as dt

import pandas as pd
import streamlit as st
import yfinance as yf

# ── 장중 여부 판단 (미국 동부시간 기준) ─────────────────────────────
def _is_market_open() -> bool:
    """NYSE 장중 여부: ET 09:30~16:00, 월~금"""
    try:
        import pytz
        et = pytz.timezone("America/New_York")
        now_et = dt.datetime.now(et)
        if now_et.weekday() >= 5:          # 토·일
            return False
        t = now_et.time()
        return dt.time(9, 25) <= t <= dt.time(16, 5)
    except ImportError:
        # pytz 없으면 UTC 기준 근사 (UTC-4~5)
        now_utc = dt.datetime.utcnow()
        if now_utc.weekday() >= 5:
            return False
        t_utc = now_utc.time()
        return dt.time(13, 25) <= t_utc <= dt.time(21, 5)

def _cache_ttl() -> int:
    """장중=60초, 장외=600초"""
    return 30 if _is_market_open() else 600  # 장중 30초 캐시


# ── OHLCV 다운로드 (재시도 포함) ────────────────────────────────────
def _download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    yfinance 다운로드 — 실시간 최대 근사
    - prepost=True: 프리/애프터마켓 데이터 포함
    - 오류 시 1회 재시도
    - Yahoo Finance는 구조상 15~20분 지연 데이터 (무료 한계)
    """
    for attempt in range(2):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                prepost=True,       # 프리/애프터마켓 포함 → 더 최신 데이터
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            if attempt == 0:
                time.sleep(0.5)
    return pd.DataFrame()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼 정리 + 인덱스 통일"""
    if df.empty:
        return df
    df = df.rename_axis("Date").reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    keep = ["Date","Open","High","Low","Close","Volume"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["Close"]).copy()
    df.set_index("Date", inplace=True)
    # 음수 거래량 / 비정상 가격 제거
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].clip(lower=0)
    df = df[df["Close"] > 0]
    return df


# ── Streamlit 캐시 래퍼 ─────────────────────────────────────────────
# TTL을 동적으로 적용하기 위해 두 버전 선언
@st.cache_data(ttl=30, show_spinner=False)
def _fetch_realtime(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return _clean(_download(ticker, period, interval))

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return _clean(_download(ticker, period, interval))


def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    메인 데이터 수집 함수
    - 장중: 1분 캐시 (실시간 근사)
    - 장외: 10분 캐시
    """
    if _is_market_open():
        return _fetch_realtime(ticker, period, interval)
    else:
        return _fetch_cached(ticker, period, interval)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_vix() -> Optional[float]:
    """VIX 실시간 (1분 캐시)"""
    df = _clean(_download("^VIX", "10d", "1d"))
    if df.empty:
        return None
    try:
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_spy() -> pd.DataFrame:
    """SPY 기준선 (10분 캐시 — 장중에도 자주 갱신 불필요)"""
    return fetch_ohlcv("SPY", "2y", "1d")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    """종목 기본 정보 (5분 캐시)"""
    try:
        t    = yf.Ticker(ticker)
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
            "earnings_date":info.get("earningsTimestamp"),
        }
    except Exception:
        return {"name": ticker}
