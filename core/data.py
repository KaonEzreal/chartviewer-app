"""
data.py — yfinance 실시간 최대화 (가능한 최선)

Yahoo Finance 무료 한계:
  - 일봉/주봉: 15~20분 지연 (구조적 한계, 코드로 해결 불가)
  - 1분봉: 상대적으로 더 최신 데이터 제공

최적화 전략:
  1. fast_info → 가장 빠른 현재가 조회
  2. 1분봉 오버레이 → 당일 장중 최신 틱 반영
  3. 장중 캐시 15초 (최소화)
  4. prepost=True → 프리/애프터마켓 포함
  5. 강제 갱신: st.cache_data.clear() 지원
"""

from typing import Optional, Tuple
import time
import datetime as dt
import pandas as pd
import streamlit as st
import yfinance as yf


# ── 장중 여부 ─────────────────────────────────────────────────────
def _is_market_open() -> bool:
    """NYSE 장중 09:30~16:00 ET"""
    try:
        import pytz
        et  = pytz.timezone("America/New_York")
        now = dt.datetime.now(et)
        if now.weekday() >= 5: return False
        t = now.time()
        return dt.time(9, 25) <= t <= dt.time(16, 5)
    except ImportError:
        now_utc = dt.datetime.utcnow()
        if now_utc.weekday() >= 5: return False
        t = now_utc.time()
        return dt.time(13, 25) <= t <= dt.time(21, 5)


def _is_premarket() -> bool:
    """프리마켓 04:00~09:30 ET"""
    try:
        import pytz
        et  = pytz.timezone("America/New_York")
        now = dt.datetime.now(et)
        if now.weekday() >= 5: return False
        t = now.time()
        return dt.time(4, 0) <= t < dt.time(9, 30)
    except Exception:
        return False


def _now_et_str() -> str:
    """현재 ET 시간 문자열"""
    try:
        import pytz
        et = pytz.timezone("America/New_York")
        return dt.datetime.now(et).strftime("%H:%M:%S ET")
    except Exception:
        return dt.datetime.utcnow().strftime("%H:%M:%S UTC")


# ── 기본 다운로드 ─────────────────────────────────────────────────
def _download(ticker: str, period: str, interval: str,
              prepost: bool = True) -> pd.DataFrame:
    """yfinance 다운로드 — 실패 시 1회 재시도"""
    for attempt in range(2):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                prepost=prepost,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            if attempt == 0:
                time.sleep(0.3)
    return pd.DataFrame()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼 정리 + 이상값 제거"""
    if df.empty: return df
    df = df.rename_axis("Date").reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["Close"]).copy()
    df.set_index("Date", inplace=True)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].clip(lower=0)
    df = df[df["Close"] > 0]
    return df


# ── fast_info: 가장 빠른 현재가 ──────────────────────────────────
@st.cache_data(ttl=10, show_spinner=False)   # 10초 캐시 (최소한)
def fetch_realtime_price(ticker: str) -> dict:
    """
    yfinance fast_info 사용 — 일봉보다 더 최신 현재가
    장중 10초 캐시
    """
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = getattr(fi, "last_price", None)
        prev  = getattr(fi, "previous_close", None)
        high  = getattr(fi, "day_high", None)
        low   = getattr(fi, "day_low", None)
        vol   = getattr(fi, "last_volume", None)
        if price and price > 0:
            chg_pct = ((price - prev) / prev * 100) if prev and prev > 0 else 0.0
            return {
                "price":    round(float(price), 2),
                "prev":     round(float(prev), 2) if prev else None,
                "chg_pct":  round(float(chg_pct), 2),
                "day_high": round(float(high), 2) if high else None,
                "day_low":  round(float(low), 2) if low else None,
                "volume":   int(vol) if vol else None,
                "source":   "fast_info",
                "time":     _now_et_str(),
            }
    except Exception:
        pass
    return {}


# ── 1분봉: 당일 최신 틱 ──────────────────────────────────────────
@st.cache_data(ttl=15, show_spinner=False)   # 15초 캐시
def fetch_intraday_1m(ticker: str) -> pd.DataFrame:
    """
    1분봉 데이터 (당일) — 일봉보다 더 최신
    장중에만 의미 있음
    """
    df = _clean(_download(ticker, "1d", "1m", prepost=True))
    return df


# ── OHLCV 메인 함수 ───────────────────────────────────────────────
@st.cache_data(ttl=15, show_spinner=False)
def _fetch_realtime(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return _clean(_download(ticker, period, interval, prepost=True))

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return _clean(_download(ticker, period, interval, prepost=True))


def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    메인 OHLCV 수집
    - 장중: 15초 캐시
    - 장외: 10분 캐시
    - 일봉이면 1분봉으로 마지막 봉 업데이트 시도
    """
    if _is_market_open() or _is_premarket():
        df = _fetch_realtime(ticker, period, interval)
    else:
        df = _fetch_cached(ticker, period, interval)

    # 일봉 + 장중인 경우: 1분봉으로 오늘 마지막 봉 업데이트
    if interval == "1d" and _is_market_open() and not df.empty:
        df = _patch_with_intraday(df, ticker)

    return df


def _patch_with_intraday(df_daily: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    일봉 마지막 봉을 1분봉 집계로 업데이트
    → 오늘의 실시간 High/Low/Close/Volume 반영
    """
    try:
        df_1m = fetch_intraday_1m(ticker)
        if df_1m.empty: return df_daily

        today = dt.date.today()
        # 오늘 날짜 데이터만
        today_1m = df_1m[pd.to_datetime(df_1m.index).date == today]
        if today_1m.empty: return df_daily

        # 오늘 집계
        today_open   = float(today_1m["Open"].iloc[0])
        today_high   = float(today_1m["High"].max())
        today_low    = float(today_1m["Low"].min())
        today_close  = float(today_1m["Close"].iloc[-1])
        today_vol    = float(today_1m["Volume"].sum())

        df_out = df_daily.copy()
        last_idx = df_out.index[-1]

        # 마지막 봉이 오늘이면 업데이트, 아니면 추가
        if pd.to_datetime(last_idx).date() == today:
            df_out.loc[last_idx, "Open"]   = today_open
            df_out.loc[last_idx, "High"]   = today_high
            df_out.loc[last_idx, "Low"]    = today_low
            df_out.loc[last_idx, "Close"]  = today_close
            df_out.loc[last_idx, "Volume"] = today_vol
        else:
            new_row = pd.DataFrame({
                "Open": [today_open], "High": [today_high],
                "Low": [today_low],  "Close": [today_close],
                "Volume": [today_vol],
            }, index=[pd.Timestamp(today)])
            df_out = pd.concat([df_out, new_row])

        return df_out
    except Exception:
        return df_daily


# ── VIX ───────────────────────────────────────────────────────────
@st.cache_data(ttl=15, show_spinner=False)
def fetch_vix() -> Optional[float]:
    """VIX 15초 캐시"""
    df = _clean(_download("^VIX", "5d", "1d", prepost=False))
    if df.empty: return None
    try: return float(df["Close"].dropna().iloc[-1])
    except: return None


# ── SPY ───────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_spy() -> pd.DataFrame:
    """SPY 1분 캐시 (장중 국면 판단용)"""
    return fetch_ohlcv("SPY", "2y", "1d")


# ── 종목 기본 정보 ───────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    """종목 기본 정보 5분 캐시"""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":       info.get("longName", ticker),
            "sector":     info.get("sector", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio":   info.get("trailingPE"),
            "beta":       info.get("beta"),
        }
    except Exception:
        return {"name": ticker}


# ── 캐시 강제 초기화 ─────────────────────────────────────────────
def clear_all_cache():
    """모든 캐시 삭제 — 강제 새로고침 버튼용"""
    st.cache_data.clear()
