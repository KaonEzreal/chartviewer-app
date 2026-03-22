import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from .config import TF_OPTIONS
from .data import fetch_ohlcv, fetch_vix
from .indicators import atr, bbands, obv, pivot_levels, rsi, sma
from .scoring import ScoreBreakdown, compute_score


@dataclass
class Signal:
    score: int
    grade: str
    score_reasons: list[str]
    bias: str
    wave: str
    energy: str
    pattern: str
    obv_ratio: Optional[float]
    rsi: float
    mfi: float
    weekly_perf: float
    target: float
    stop: float
    vix: Optional[float]
    vix_text: Optional[str]
    asof: str
    breakdown: ScoreBreakdown



def vix_warning(vix: Optional[float]) -> Optional[str]:
    if vix is None or (isinstance(vix, float) and math.isnan(vix)):
        return None
    if vix >= 30:
        return f"VIX 경고 ({vix:.1f}): 변동성이 매우 커서 포지션 축소가 유리해요"
    if vix >= 25:
        return f"VIX 경고 ({vix:.1f}): 손절 기준을 짧게 잡는 편이 좋아요"
    if vix >= 20:
        return f"VIX 주의 ({vix:.1f}): 추격 매수보다 눌림 확인이 우선이에요"
    return f"VIX 안정 ({vix:.1f})"



def label_wave(df: pd.DataFrame) -> str:
    close = df["Close"]
    if len(df) < 60:
        return "데이터 부족"
    ma20 = sma(close, 20)
    ma50 = sma(close, 50)
    slope20 = float(ma20.diff().tail(5).mean())
    slope50 = float(ma50.diff().tail(5).mean())
    rsi_val = float(rsi(close, 14).iloc[-1])
    if slope20 > 0 and slope50 > 0 and rsi_val > 55:
        return "상승 파동"
    if slope20 < 0 and rsi_val < 45:
        return "조정/횡보 파동"
    if rsi_val < 35:
        return "반등 준비"
    return "혼합"



def label_energy(df: pd.DataFrame) -> Tuple[str, Optional[float]]:
    o = obv(df)
    if len(o) < 80:
        return "보통", None
    recent = o.diff().tail(10).mean()
    base = o.diff().tail(60).mean()
    if base == 0 or math.isnan(base):
        return "보통", None
    ratio = float(recent / base)
    if ratio > 1.15:
        return "매수세 증가", ratio
    if ratio > 0.95:
        return "매수/매도 균형", ratio
    return "매도세 우위", ratio



def label_pattern(df: pd.DataFrame) -> str:
    lower, mid, upper = bbands(df["Close"], 20, 2.0)
    last = float(df["Close"].iloc[-1])
    lb = float(lower.iloc[-1])
    mb = float(mid.iloc[-1])
    ub = float(upper.iloc[-1])
    if last < lb:
        return "BB 하단 이탈 반등 후보"
    if last > ub:
        return "BB 상단 돌파 과열 주의"
    if last > mb:
        return "상승 흐름 유지"
    return "조정/관망"



def calc_target_stop(df: pd.DataFrame, style: str, tf_choice: str) -> Tuple[float, float]:
    last = float(df["Close"].iloc[-1])
    atr_val = float(atr(df, 14).iloc[-1])
    support, resistance = pivot_levels(df, 60)
    interval = TF_OPTIONS[tf_choice]["interval"]
    if interval == "15m":
        tf_stop_mul, tf_tgt_mul = 0.9, 1.4
    elif interval == "1h":
        tf_stop_mul, tf_tgt_mul = 1.0, 1.6
    else:
        tf_stop_mul, tf_tgt_mul = 1.2, 2.2

    if style == "단타":
        stop = max(support, last - (1.0 * tf_stop_mul) * atr_val)
        target = min(resistance, last + (1.7 * tf_tgt_mul) * atr_val)
    else:
        stop = max(support, last - (1.4 * tf_stop_mul) * atr_val)
        target = min(resistance, last + (2.6 * tf_tgt_mul) * atr_val)

    stop = min(stop, last * 0.999)
    target = max(target, last * 1.001)
    return float(stop), float(target)



def final_action_line(score: int, bias: str, rsi_val: float, vix: Optional[float], last_price: float, ma20: float, tf_label: str) -> str:
    trend_up = ("상승" in bias)
    vix_high = vix is not None and vix >= 25
    if score >= 84 and trend_up and last_price >= ma20 and rsi_val <= 68 and not vix_high:
        return f"추세 추종 진입 후보 · {tf_label}"
    if rsi_val < 35 and score >= 63:
        return f"단기 반등 매매 후보 · {tf_label}"
    if vix_high:
        return f"변동성 주의 구간 · {tf_label}"
    if score < 52 or (last_price < ma20 and rsi_val < 45):
        return f"관망 우선 · {tf_label}"
    return f"눌림목 대기 후 분할 접근 · {tf_label}"



def build_signal(ticker: str, style: str, tf_choice: str) -> Optional[Tuple[Signal, pd.DataFrame]]:
    tf = TF_OPTIONS[tf_choice]
    df = fetch_ohlcv(ticker, period=tf["period"], interval=tf["interval"])
    if df is None or df.empty or len(df) < tf["min_bars"]:
        return None

    breakdown = compute_score(df)
    close = float(df["Close"].iloc[-1])
    ma50 = float(sma(df["Close"], 50).iloc[-1])
    ma200 = float(sma(df["Close"], 200).iloc[-1]) if len(df) >= 200 else float(sma(df["Close"], 100).iloc[-1])

    if close > ma50 and ma50 > ma200:
        bias = "상승장 (강함)"
    elif close < ma50 and ma50 < ma200:
        bias = "하락장 (주의)"
    else:
        bias = "횡보장"

    wave = label_wave(df)
    energy, obv_ratio = label_energy(df)
    pattern = label_pattern(df)

    interval = tf["interval"]
    if interval == "1d":
        steps = 5
    elif interval == "1h":
        steps = 30
    else:
        steps = 130

    if len(df) > steps:
        weekly_perf = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-(steps + 1)]) - 1.0) * 100
    else:
        weekly_perf = float("nan")

    stop, target = calc_target_stop(df, style, tf_choice)
    vix = fetch_vix()
    asof = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    signal = Signal(
        score=breakdown.total,
        grade=breakdown.grade,
        score_reasons=breakdown.reasons,
        bias=bias,
        wave=wave,
        energy=energy,
        pattern=pattern,
        obv_ratio=obv_ratio,
        rsi=breakdown.rsi,
        mfi=breakdown.mfi,
        weekly_perf=float(weekly_perf),
        target=float(target),
        stop=float(stop),
        vix=vix,
        vix_text=vix_warning(vix),
        asof=asof,
        breakdown=breakdown,
    )
    return signal, df
