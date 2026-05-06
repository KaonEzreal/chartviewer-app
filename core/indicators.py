"""
indicators.py — 전설적 트레이더들의 기법을 구현한 지표 라이브러리

포함 기법:
- 기본 지표 (RSI, MACD, ATR, OBV, MFI, BB, SMA, EMA)
- Minervini SEPA / Trend Template / VCP 감지
- Weinstein Stage Analysis (30주 MA 기반)
- O'Neil Cup-with-Handle / RS 점수
- Paul Tudor Jones 200MA 방어 규칙
- Livermore Pivot 포인트
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ══════════════════════════════════════════════════════════════════
# 기본 지표
# ══════════════════════════════════════════════════════════════════

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).clip(0, 100)

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    fl = ema(close, fast); sl = ema(close, slow)
    line = fl - sl; sig = ema(line, signal)
    return line, sig, line - sig

def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift(1)
    return pd.concat([(df["High"]-df["Low"]).abs(),
                      (df["High"]-pc).abs(),
                      (df["Low"]-pc).abs()], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n=14) -> pd.Series:
    return true_range(df).ewm(alpha=1/n, adjust=False).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    d = np.sign(df["Close"].diff()).fillna(0)
    return (d * df["Volume"].fillna(0)).cumsum()

def mfi(df: pd.DataFrame, n=14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"].fillna(0)
    pos = mf.where(tp.diff() > 0, 0.0)
    neg = mf.where(tp.diff() < 0, 0.0)
    pmf = pos.rolling(n).sum(); nmf = neg.abs().rolling(n).sum()
    r = pmf / nmf.replace(0, np.nan)
    return (100 - 100/(1+r)).clip(0, 100)

def bbands(close: pd.Series, n=20, k=2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    return mid - k*std, mid, mid + k*std

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0)
    return (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)

def stochastic(df: pd.DataFrame, k=14, d=3):
    lo = df["Low"].rolling(k).min()
    hi = df["High"].rolling(k).max()
    pk = (100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)).clip(0,100)
    return pk, pk.rolling(d).mean()

def adx(df: pd.DataFrame, n=14) -> pd.Series:
    tr  = true_range(df)
    up  = df["High"].diff(); dn = -df["Low"].diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    atr14 = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / atr14.replace(0, np.nan)
    ndi = 100 * ndm.ewm(alpha=1/n, adjust=False).mean() / atr14.replace(0, np.nan)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def cci(df: pd.DataFrame, n=20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def williams_r(df: pd.DataFrame, n=14) -> pd.Series:
    hi = df["High"].rolling(n).max(); lo = df["Low"].rolling(n).min()
    return (-100 * (hi - df["Close"]) / (hi - lo).replace(0, np.nan))

def pivot_levels(df: pd.DataFrame, lookback=60):
    r = df.tail(lookback)
    return float(r["Low"].min()), float(r["High"].max())

def week52(df: pd.DataFrame):
    r = df.tail(252) if len(df) >= 252 else df
    h, l = float(r["High"].max()), float(r["Low"].min())
    last = float(df["Close"].iloc[-1])
    pos  = (last - l) / (h - l) if h != l else 0.5
    return h, l, pos


# ══════════════════════════════════════════════════════════════════
# ★ Minervini Trend Template (SEPA 8조건)
# ══════════════════════════════════════════════════════════════════

def minervini_trend_template(df: pd.DataFrame) -> dict:
    """
    Mark Minervini의 SEPA Trend Template — 8가지 필수 조건
    모든 조건 통과 = Stage 2 상승 추세 확인

    조건:
    1. 현재가 > 150MA & 200MA
    2. 150MA > 200MA
    3. 200MA가 최소 1개월(20봉) 이상 상승 중
    4. 50MA > 150MA & 200MA
    5. 현재가 > 50MA
    6. 현재가가 52주 저점 대비 25% 이상 위
    7. 현재가가 52주 고점 대비 25% 이내 (고점 근접)
    8. RS 점수 ≥ 70 (시장 대비 상대강도 — 여기선 근사치 사용)
    """
    close = df["Close"]
    n = len(df)

    ma50  = sma(close, 50)
    ma150 = sma(close, 150)
    ma200 = sma(close, min(200, n))

    last   = float(close.iloc[-1])
    m50    = float(ma50.iloc[-1])
    m150   = float(ma150.iloc[-1])
    m200   = float(ma200.iloc[-1])
    h52, l52, pos52 = week52(df)

    # 200MA 기울기 (최근 20봉 평균 기울기)
    ma200_slope = float(ma200.diff().tail(20).mean()) if n >= 220 else 0.0

    c1 = last > m150 and last > m200
    c2 = m150 > m200
    c3 = ma200_slope > 0
    c4 = m50 > m150 and m50 > m200
    c5 = last > m50
    c6 = l52 > 0 and (last / l52 - 1) >= 0.25
    c7 = h52 > 0 and (last / h52) >= 0.75   # 52주 고점의 75% 이상 = 고점 25% 이내
    # RS 근사: 최근 12개월 수익률 (SPY 비교는 별도 fetch 필요, 여기선 절대 모멘텀)
    perf12m = (last / float(close.iloc[max(0, n-252)]) - 1) * 100 if n >= 50 else 0.0
    c8 = perf12m > 10  # 연간 10% 이상 상승 = RS 양호 근사

    conditions = [c1, c2, c3, c4, c5, c6, c7, c8]
    passed = sum(conditions)

    labels = [
        "현재가 > MA150 & MA200",
        "MA150 > MA200",
        "MA200 상승 기울기",
        "MA50 > MA150 & MA200",
        "현재가 > MA50",
        "52주 저점 +25% 이상",
        "52주 고점 -25% 이내",
        "12개월 모멘텀 양호",
    ]

    return {
        "passed": passed,
        "total": 8,
        "conditions": list(zip(labels, conditions)),
        "is_stage2": passed >= 7,        # 7개 이상 통과 = Stage 2 강한 상승
        "is_qualified": passed >= 5,     # 5개 이상 = 관심 구간
        "ma50": m50, "ma150": m150, "ma200": m200,
        "ma200_slope": ma200_slope,
        "perf12m": perf12m,
    }


# ══════════════════════════════════════════════════════════════════
# ★ VCP (Volatility Contraction Pattern) 감지
# ══════════════════════════════════════════════════════════════════

def detect_vcp(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Minervini VCP 감지:
    - 연속 수축하는 가격 조정 폭 (각 조정이 전 조정의 ~50%)
    - 조정 시 거래량 감소
    - 최종 수축 후 거래량 폭발 = 돌파 신호
    """
    if len(df) < lookback:
        return {"detected": False, "contractions": 0, "tightness": 0.0,
                "breakout_ready": False, "vol_dry_up": False}

    d = df.tail(lookback).copy()
    close = d["Close"]
    vol   = d["Volume"]
    atr14 = atr(d, 14)

    # 변동성 수축 측정: ATR / Close (정규화)
    atr_pct = atr14 / close
    recent_atr  = float(atr_pct.tail(10).mean())
    base_atr    = float(atr_pct.head(30).mean())
    tightness   = 1 - (recent_atr / base_atr) if base_atr > 0 else 0.0  # 0~1, 높을수록 수축

    # 거래량 감소 (수축기 거래량)
    recent_vol = float(vol.tail(10).mean())
    base_vol   = float(vol.mean())
    vol_dry_up = recent_vol < base_vol * 0.7  # 평균 대비 30% 이상 감소

    # 조정 횟수 (고점 → 저점 cycle 카운트)
    highs = (close == close.rolling(5, center=True).max())
    lows  = (close == close.rolling(5, center=True).min())
    contractions = min(int(highs.sum() / 2), 4)   # 최대 4회 조정

    # 돌파 준비: 수축 심하고 거래량 말랐으며 최근 가격이 고점 근처
    h52, l52, pos52 = week52(df)
    last = float(df["Close"].iloc[-1])
    near_pivot = pos52 > 0.85   # 52주 고점의 85% 이상

    breakout_ready = (
        tightness > 0.3 and
        vol_dry_up and
        contractions >= 2 and
        float(df["Close"].tail(5).std()) < float(df["Close"].tail(20).std()) * 0.6
    )

    return {
        "detected":       tightness > 0.25 and contractions >= 2,
        "contractions":   contractions,
        "tightness":      round(tightness, 3),
        "vol_dry_up":     vol_dry_up,
        "breakout_ready": breakout_ready,
        "near_pivot":     near_pivot,
        "vol_ratio":      round(recent_vol / base_vol if base_vol else 1.0, 2),
    }


# ══════════════════════════════════════════════════════════════════
# ★ Weinstein Stage Analysis (30주 = 150일 MA 기반)
# ══════════════════════════════════════════════════════════════════

def weinstein_stage(df: pd.DataFrame) -> dict:
    """
    Stan Weinstein의 4단계 주가 사이클 분석
    - Stage 1: Basing (횡보/바닥 다지기) — 150MA 평탄
    - Stage 2: Advancing (상승 추세)   — 150MA 우상향, 가격 위
    - Stage 3: Topping (천장 분배)    — 150MA 평탄화, 변동성 증가
    - Stage 4: Declining (하락 추세)  — 150MA 하향, 가격 아래
    """
    close = df["Close"]
    n = len(df)
    if n < 30:
        return {"stage": 0, "stage_name": "데이터 부족", "label": "—",
                "ma150": None, "ma150_slope": 0.0, "above_ma150": False}

    ma150 = sma(close, min(150, n))
    last  = float(close.iloc[-1])
    m150  = float(ma150.iloc[-1])

    # MA 기울기 (최근 20봉 평균)
    slope = float(ma150.diff().tail(20).mean())

    # 변동성 (가격 진동 폭)
    vol_ratio = float(close.tail(20).std() / close.tail(60).std()) if n >= 60 else 1.0

    above = last > m150
    rising = slope > 0
    falling = slope < -abs(slope * 0.1)  # 명확한 하향

    # 가격과 MA의 거리
    distance_pct = (last - m150) / m150 * 100 if m150 else 0

    if above and rising and distance_pct > 0:
        stage = 2
        stage_name = "Stage 2 — 상승 추세 (매수 최적 구간)"
        label = "🚀 상승 추세"
    elif above and not rising and vol_ratio > 1.2:
        stage = 3
        stage_name = "Stage 3 — 분배/천장 (수익 실현 구간)"
        label = "⚠️ 천장 분배"
    elif not above and falling:
        stage = 4
        stage_name = "Stage 4 — 하락 추세 (관망 필수)"
        label = "❌ 하락 추세"
    else:
        stage = 1
        stage_name = "Stage 1 — 바닥 다지기 (돌파 대기)"
        label = "⏳ 바닥 다지기"

    return {
        "stage": stage,
        "stage_name": stage_name,
        "label": label,
        "ma150": m150,
        "ma150_slope": slope,
        "above_ma150": above,
        "distance_pct": round(distance_pct, 2),
        "vol_ratio": round(vol_ratio, 2),
    }


# ══════════════════════════════════════════════════════════════════
# ★ O'Neil RS (Relative Strength) 점수 — 시장 대비 상대강도
# ══════════════════════════════════════════════════════════════════

def oneil_rs_score(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> dict:
    """
    William O'Neil의 RS Rating — IBD 방식 근사
    3개월(63봉), 6개월(126봉), 9개월(189봉), 12개월(252봉) 성과의 가중 평균
    IBD 원본: (63일 성과×2 + 126일 성과×1 + 189일 성과×1 + 252일 성과×1) / 5
    """
    close = df["Close"]
    n = len(close)

    def perf(bars):
        if n >= bars + 1:
            return (float(close.iloc[-1]) / float(close.iloc[-(bars+1)]) - 1) * 100
        return None

    p3  = perf(63)
    p6  = perf(126)
    p9  = perf(189)
    p12 = perf(252)

    # 가중 합계
    values = [v for v in [p3, p6, p9, p12] if v is not None]
    if not values:
        return {"rs_score": 50, "momentum_rank": "N/A", "perfs": {}}

    # IBD 가중: 3개월에 2배 가중
    weights = [2 if i == 0 else 1 for i in range(len(values))]
    weighted = sum(v*w for v, w in zip(values, weights)) / sum(weights)

    # SPY 대비 정규화 (SPY가 있는 경우)
    spy_ref = 0.0
    if spy_df is not None and len(spy_df) >= 63:
        spy_3m = (float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-64]) - 1) * 100
        spy_ref = spy_3m

    alpha = (p3 or 0) - spy_ref   # 초과 수익률

    # RS 점수 0~100 변환 (간이)
    rs_score = max(0, min(100, int(50 + weighted * 1.5)))

    if rs_score >= 90: rank = "🌟 Elite (상위 10%)"
    elif rs_score >= 80: rank = "💪 Strong (상위 20%)"
    elif rs_score >= 70: rank = "✅ Good (상위 30%)"
    elif rs_score >= 50: rank = "➡️ Average"
    else: rank = "⚠️ Weak (하위 50%)"

    return {
        "rs_score": rs_score,
        "alpha_3m": round(alpha, 2),
        "momentum_rank": rank,
        "perfs": {
            "3M": round(p3, 2) if p3 else None,
            "6M": round(p6, 2) if p6 else None,
            "9M": round(p9, 2) if p9 else None,
            "12M": round(p12, 2) if p12 else None,
        }
    }


# ══════════════════════════════════════════════════════════════════
# ★ Cup-with-Handle 패턴 감지 (O'Neil)
# ══════════════════════════════════════════════════════════════════

def detect_cup_handle(df: pd.DataFrame) -> dict:
    """
    O'Neil Cup-with-Handle 근사 감지
    - 컵: U자형 가격 패턴 (7주 이상)
    - 핸들: 컵 이후 5~15% 소폭 하락 후 수렴
    - 돌파: 핸들 고점을 거래량 50% 이상 증가로 돌파
    """
    if len(df) < 50:
        return {"detected": False, "pattern": "N/A"}

    close = df["Close"]
    n = len(df)

    # 최근 60일 패턴 분석
    window = min(60, n)
    d = close.tail(window)

    # 고점, 저점, 회복 여부
    left_high = float(d.iloc[:window//3].max())
    cup_low   = float(d.iloc[window//4:3*window//4].min())
    right_high = float(d.iloc[2*window//3:].max())
    last = float(close.iloc[-1])

    cup_depth = (left_high - cup_low) / left_high if left_high > 0 else 0
    recovery  = (right_high - cup_low) / (left_high - cup_low) if left_high != cup_low else 0

    # 컵 깊이 (일반적으로 12~35%)
    is_cup = 0.12 <= cup_depth <= 0.45 and recovery >= 0.85

    # 핸들: 최근 15봉의 변동성이 컵 대비 작고, 약간 하락
    handle_vol = float(d.tail(15).std()) / float(d.std()) if float(d.std()) > 0 else 1.0
    is_handle  = handle_vol < 0.7 and last > cup_low

    # 돌파 조건
    vol_now  = float(df["Volume"].tail(5).mean())
    vol_base = float(df["Volume"].tail(50).mean())
    vol_expansion = vol_now / vol_base if vol_base > 0 else 1.0
    breakout = last >= right_high * 0.98 and vol_expansion >= 1.5

    if is_cup and is_handle:
        pattern = "🏆 Cup-with-Handle 형성 중"
        if breakout:
            pattern = "🚀 Cup-with-Handle 돌파!"
    elif is_cup:
        pattern = "🥤 컵 형성 (핸들 대기)"
    else:
        pattern = "—"

    return {
        "detected":       is_cup,
        "handle":         is_handle,
        "breakout":       breakout,
        "cup_depth_pct":  round(cup_depth * 100, 1),
        "recovery_pct":   round(recovery * 100, 1),
        "vol_expansion":  round(vol_expansion, 2),
        "pattern":        pattern,
    }


# ══════════════════════════════════════════════════════════════════
# ★ Paul Tudor Jones 200MA 방어 규칙
# ══════════════════════════════════════════════════════════════════

def ptj_defense_rule(df: pd.DataFrame) -> dict:
    """
    Paul Tudor Jones: "200MA 아래 = 즉시 방어"
    5:1 리스크/리워드 원칙
    """
    close = df["Close"]
    n = len(df)
    ma200 = sma(close, min(200, n))
    last  = float(close.iloc[-1])
    m200  = float(ma200.iloc[-1])

    above_200 = last > m200
    dist_pct  = (last - m200) / m200 * 100 if m200 else 0

    # PTJ 룰: 200MA 아래면 무조건 방어
    if above_200:
        if dist_pct > 20:
            status = "⚠️ 200MA 대폭 이탈 — 과열 조정 주의"
            defense = False
        else:
            status = f"✅ 200MA 위 (+{dist_pct:.1f}%) — PTJ 매수 허용 구간"
            defense = False
    else:
        status = f"🛡️ 200MA 하회 ({dist_pct:.1f}%) — PTJ 즉시 방어/관망"
        defense = True

    return {
        "above_200ma": above_200,
        "ma200": m200,
        "dist_pct": round(dist_pct, 2),
        "defense_required": defense,
        "status": status,
    }


# ══════════════════════════════════════════════════════════════════
# ★ Livermore Pivot Point (주요 지지/저항 포인트)
# ══════════════════════════════════════════════════════════════════

def livermore_pivots(df: pd.DataFrame, sensitivity: int = 5) -> dict:
    """
    Jesse Livermore의 '저항의 최소선' 피벗 포인트
    - 주요 고점 = 저항선
    - 주요 저점 = 지지선
    - 이 레벨 돌파가 핵심 진입 시그널
    """
    if len(df) < sensitivity * 3:
        return {"support": None, "resistance": None, "near_breakout": False}

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    n     = len(df)
    last  = float(close.iloc[-1])

    # 로컬 고점/저점 식별
    pivot_highs = []
    pivot_lows  = []
    for i in range(sensitivity, n - sensitivity):
        if all(float(high.iloc[i]) >= float(high.iloc[i-j]) and
               float(high.iloc[i]) >= float(high.iloc[i+j])
               for j in range(1, sensitivity+1)):
            pivot_highs.append(float(high.iloc[i]))
        if all(float(low.iloc[i]) <= float(low.iloc[i-j]) and
               float(low.iloc[i]) <= float(low.iloc[i+j])
               for j in range(1, sensitivity+1)):
            pivot_lows.append(float(low.iloc[i]))

    # 현재가 아래 최고 지지선, 현재가 위 최저 저항선
    supports    = sorted([p for p in pivot_lows  if p < last], reverse=True)
    resistances = sorted([p for p in pivot_highs if p > last])

    support    = supports[0]    if supports    else float(low.tail(30).min())
    resistance = resistances[0] if resistances else float(high.tail(30).max())

    # 저항선 돌파 임박 여부 (0.5% 이내)
    near_breakout = resistance > 0 and (resistance - last) / resistance < 0.005
    near_support  = support > 0    and (last - support) / last < 0.005

    rr_ratio = (resistance - last) / (last - support) if (last - support) > 0 else 0

    return {
        "support":        round(support, 2),
        "resistance":     round(resistance, 2),
        "near_breakout":  near_breakout,
        "near_support":   near_support,
        "rr_ratio":       round(rr_ratio, 2),
        "all_resistances": resistances[:3],
        "all_supports":    supports[:3],
    }


# ══════════════════════════════════════════════════════════════════
# ★ 종합 시장 국면 판단
# ══════════════════════════════════════════════════════════════════

def market_breadth_score(df: pd.DataFrame) -> dict:
    """
    O'Neil의 'M' — Market Direction + Minervini의 시장 환경 평가
    추세, 모멘텀, 변동성 종합
    """
    close = df["Close"]
    n = len(df)
    if n < 50:
        return {"score": 50, "environment": "중립", "suitable": True}

    ma20  = float(sma(close, 20).iloc[-1])
    ma50  = float(sma(close, 50).iloc[-1])
    ma200 = float(sma(close, min(200,n)).iloc[-1])
    last  = float(close.iloc[-1])
    rsi14 = float(rsi(close, 14).iloc[-1])
    adx14 = float(adx(df, 14).iloc[-1])

    score = 50
    score += 15 if last > ma200 else -15
    score += 10 if last > ma50  else -10
    score += 5  if last > ma20  else -5
    score += 10 if adx14 > 25   else 0
    if   rsi14 > 70: score -= 5
    elif rsi14 < 40: score -= 5

    score = max(0, min(100, score))

    if score >= 70: env = "📈 강세장 — 공격적 진입 가능"
    elif score >= 50: env = "➡️ 중립장 — 선별적 진입"
    else: env = "📉 약세장 — 현금 비중 확대"

    return {
        "score": score,
        "environment": env,
        "suitable": score >= 50,
    }
