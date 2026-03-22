from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from .indicators import atr, bbands, macd, mfi, rsi, sma


@dataclass
class ScoreBreakdown:
    total: int
    grade: str
    reasons: List[str]
    trend: int
    momentum: int
    flow: int
    volatility: int
    volume: int
    trigger: int
    last: float
    ma20: float
    ma50: float
    ma200: float
    rsi: float
    mfi: float
    atr_pct: float
    vol_ratio: float
    macd_hist: float
    bb_position: float



def clamp_int(x: float, lo: int = 0, hi: int = 100) -> int:
    return int(max(lo, min(hi, round(x))))



def grade_from_score(score: int) -> str:
    if score >= 92:
        return "SSS"
    if score >= 84:
        return "SS"
    if score >= 75:
        return "S"
    if score >= 63:
        return "A"
    if score >= 52:
        return "B"
    if score >= 40:
        return "C"
    return "D"



def compute_score(df: pd.DataFrame) -> ScoreBreakdown:
    close = df["Close"]
    last = float(close.iloc[-1])

    ma20 = float(sma(close, 20).iloc[-1])
    ma50 = float(sma(close, 50).iloc[-1])
    ma200 = float(sma(close, 200).iloc[-1]) if len(df) >= 200 else float(sma(close, 100).iloc[-1])

    rsi_val = float(rsi(close, 14).iloc[-1])
    mfi_val = float(mfi(df, 14).iloc[-1])
    atr_val = float(atr(df, 14).iloc[-1])
    atr_pct = (atr_val / last) if last else 0.0

    lower, mid, upper = bbands(close, 20, 2.0)
    lb, mb, ub = float(lower.iloc[-1]), float(mid.iloc[-1]), float(upper.iloc[-1])
    width = max(ub - lb, 1e-9)
    bb_position = (last - lb) / width

    vol_now = float(df["Volume"].tail(20).mean())
    vol_base = float(df["Volume"].tail(80).mean()) if len(df) >= 80 else float(df["Volume"].mean())
    vol_ratio = (vol_now / vol_base) if vol_base else 1.0

    macd_line, signal_line, hist = macd(close)
    macd_hist = float(hist.iloc[-1])
    macd_cross_up = bool(macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2])

    trend = 0
    trend += 12 if last > ma20 else 0
    trend += 14 if last > ma50 else 0
    trend += 16 if last > ma200 else 0
    trend += 8 if ma20 > ma50 else 0
    trend += 10 if ma50 > ma200 else 0

    momentum = 0
    if rsi_val < 30:
        momentum += 18
    elif rsi_val < 40:
        momentum += 14
    elif rsi_val < 55:
        momentum += 10
    elif rsi_val < 65:
        momentum += 7
    elif rsi_val < 75:
        momentum += 4
    else:
        momentum += 1

    flow = 0
    if mfi_val < 20:
        flow += 12
    elif mfi_val < 40:
        flow += 9
    elif mfi_val < 60:
        flow += 7
    elif mfi_val < 80:
        flow += 4
    else:
        flow += 1

    volatility = 0
    if atr_pct < 0.012:
        volatility += 12
    elif atr_pct < 0.025:
        volatility += 9
    elif atr_pct < 0.045:
        volatility += 5
    else:
        volatility += 2

    volume = 0
    if vol_ratio >= 1.3:
        volume += 10
    elif vol_ratio >= 1.1:
        volume += 7
    elif vol_ratio >= 0.9:
        volume += 5
    else:
        volume += 3

    trigger = 0
    if macd_cross_up:
        trigger += 10
    elif macd_hist > 0:
        trigger += 6
    elif macd_hist > -0.1:
        trigger += 3
    if bb_position < 0.18:
        trigger += 6
    elif bb_position < 0.35:
        trigger += 3

    total = clamp_int(trend + momentum + flow + volatility + volume + trigger)

    reasons: List[str] = []
    if last > ma20 and last > ma50 and last > ma200:
        reasons.append("주요 이동평균선 위에서 추세가 유지되고 있어요")
    elif last < ma20 and last < ma50:
        reasons.append("가격이 단기 평균 아래라 추세 신뢰도가 약해요")
    else:
        reasons.append("추세는 혼합 구간이라 확인 매매가 좋아요")

    if rsi_val < 35:
        reasons.append("RSI가 과매도권이라 단기 반등 여지가 있어요")
    elif rsi_val < 60:
        reasons.append("RSI가 중립권이라 추세 확인이 더 필요해요")
    else:
        reasons.append("RSI가 양호하지만 과열 추격은 주의가 필요해요")

    if mfi_val >= 65:
        reasons.append("MFI 기준 수급 흐름이 강한 편이에요")
    elif mfi_val >= 40:
        reasons.append("자금 유입은 중립 수준이에요")
    else:
        reasons.append("자금 유입이 약해서 탄력은 제한될 수 있어요")

    if vol_ratio >= 1.2:
        reasons.append("평균 대비 거래량이 붙어서 신호 신뢰도가 올라갔어요")
    else:
        reasons.append("거래량 확인이 약해서 확정 신호로 보기엔 이릅니다")

    if macd_cross_up:
        reasons.append("MACD 골든크로스가 발생해 트리거 점수가 추가됐어요")
    elif macd_hist > 0:
        reasons.append("MACD 히스토그램이 플러스로 전환돼 모멘텀이 살아 있어요")

    return ScoreBreakdown(
        total=total,
        grade=grade_from_score(total),
        reasons=reasons,
        trend=trend,
        momentum=momentum,
        flow=flow,
        volatility=volatility,
        volume=volume,
        trigger=trigger,
        last=last,
        ma20=ma20,
        ma50=ma50,
        ma200=ma200,
        rsi=rsi_val,
        mfi=mfi_val,
        atr_pct=atr_pct,
        vol_ratio=vol_ratio,
        macd_hist=macd_hist,
        bb_position=bb_position,
    )
