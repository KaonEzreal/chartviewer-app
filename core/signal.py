"""
signal.py v4 — 진입가 / 손절가 / 1·2·3차 익절 시스템

기법별 로직:
  진입가   = Livermore 피벗 돌파점 또는 현재가 기반 (VCP/Cup 셋업 시 피벗 우선)
  손절가   = ATR × 배수 vs Livermore 지지선 → 더 타이트한 쪽 (O'Neil: -7~8% 룰)
  1차 익절 = 1R (손절폭 × 1) — 초기 리스크 회수 (단타: 빠른 절반 수익 실현)
  2차 익절 = Livermore 1차 저항선 or 2R — 주요 저항 돌파 전 절반 추가 실현
  3차 익절 = Minervini 목표가 or 3~5R — 추세 추종 잔량 극대화 (PTJ 5:1 원칙)

포지션 관리:
  1차 익절: 30% 물량 청산
  2차 익절: 40% 물량 청산
  3차 익절: 30% 물량 청산 (추세 끝까지 보유)
"""

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import pandas as pd

from .config import TF_OPTIONS
from .data import fetch_ohlcv, fetch_vix, fetch_spy
from .indicators import atr, sma, bbands, week52
from .scoring import MasterScorecard, master_score


@dataclass
class TradePlan:
    """완전한 매매 계획 — 진입 ~ 3차 익절"""
    # 진입
    entry_price:   float    # 진입가 (Livermore 피벗 or 현재가)
    entry_reason:  str      # 진입 근거
    entry_type:    str      # "즉시 진입" / "피벗 돌파 대기" / "눌림목 대기"

    # 손절
    stop_price:    float    # 손절가
    stop_reason:   str      # 손절 근거
    stop_pct:      float    # 손절 % (진입가 대비)
    risk_1r:       float    # 1R = 진입가 - 손절가

    # 익절 3단계
    tp1_price:     float    # 1차 익절가
    tp1_pct:       float    # 1차 익절 % (진입가 대비)
    tp1_qty:       int      # 1차 익절 수량 % (30%)
    tp1_reason:    str

    tp2_price:     float    # 2차 익절가
    tp2_pct:       float    # 2차 익절 %
    tp2_qty:       int      # 2차 익절 수량 % (40%)
    tp2_reason:    str

    tp3_price:     float    # 3차 익절가
    tp3_pct:       float    # 3차 익절 %
    tp3_qty:       int      # 3차 익절 수량 % (30%)
    tp3_reason:    str

    # R:R 비율
    rr1: float   # 1차 R:R
    rr2: float   # 2차 R:R
    rr3: float   # 3차 R:R

    # 기대 수익 (전체 포지션 기준)
    weighted_rr:   float    # 가중 평균 R:R
    max_loss_pct:  float    # 최대 손실 %

    # 포지션 사이즈 가이드
    position_note: str


@dataclass
class Signal:
    sc:            MasterScorecard
    vix:           Optional[float]
    vix_text:      str
    action:        str
    action_detail: str
    action_color:  str
    trade_plan:    TradePlan
    weekly_perf:   float
    bias:          str
    asof:          str


# ──────────────────────────────────────────────────────────────────
def _fmt_pct(entry, price):
    return round((price / entry - 1) * 100, 2)

def _rr(entry, stop, target):
    risk = entry - stop
    reward = target - entry
    return round(reward / risk, 2) if risk > 0 else 0.0


def vix_label(vix: Optional[float]) -> str:
    if vix is None: return ""
    if vix >= 35: return f"🔴 VIX {vix:.1f} — 극도의 공포. 포지션 최소화 (PTJ: play defense)"
    if vix >= 28: return f"🟠 VIX {vix:.1f} — 변동성 경고. 손절 축소 & 분할 접근"
    if vix >= 22: return f"🟡 VIX {vix:.1f} — 변동성 주의. 추격 매수 자제"
    return f"🟢 VIX {vix:.1f} — 시장 안정. 정상 리스크 관리"


def decide_action(sc: MasterScorecard, vix: Optional[float]) -> Tuple[str, str, str]:
    score     = sc.total
    vix_high  = vix is not None and vix >= 28
    defense   = sc.ptj["defense_required"]
    stage4    = sc.weinstein["stage"] == 4
    stage2    = sc.weinstein["stage"] == 2
    is_vcp    = sc.vcp["breakout_ready"]
    is_cup    = sc.cup_handle["breakout"]
    tt_ok     = sc.trend_template["is_stage2"]

    if stage4 or defense:
        return "관망/회피", "Stage 4 하락 추세 또는 PTJ 200MA 방어 원칙 — 신규 매수 금지", "red"
    if score >= 85 and stage2 and tt_ok and (is_vcp or is_cup):
        return "강력 매수", "Minervini TT ✓ + Stage 2 ✓ + VCP/Cup 완성 — 최고 등급 시그널", "cyan"
    if score >= 78 and stage2 and not vix_high:
        return "적극 매수", "추세·모멘텀·수급 정렬 — 추세 추종 진입 적합", "green"
    if score >= 65 and not vix_high:
        return "분할 매수", "긍정 시그널 다수 — 눌림목 분할 진입 권장", "green"
    if 50 <= score < 65:
        return "대기 관찰", "시그널 혼재 — 추가 확인 후 진입 검토", "yellow"
    return "신중 관찰", "부정 시그널 우세 또는 고변동성 — 현금 유지 권장", "red"


def build_trade_plan(
    df: pd.DataFrame,
    sc: MasterScorecard,
    style: str,
    tf_choice: str,
) -> TradePlan:
    """
    진입가, 손절가, 1·2·3차 익절가 계산

    기법 우선순위:
    ① VCP / Cup 셋업 감지 → Livermore 피벗 돌파점을 진입가로
    ② 일반 상승 추세    → 현재가 = 즉시 진입 or 눌림목 대기
    ③ 손절 = O'Neil -7~8% 룰 vs ATR × 1.5 vs Livermore 지지선 → 가장 타이트
    ④ 익절 = 1R / 2R / 저항선+3R 구조 (PTJ 5:1 원칙)
    """
    last      = float(df["Close"].iloc[-1])
    atr14     = float(atr(df, 14).iloc[-1])
    interval  = TF_OPTIONS[tf_choice]["interval"]

    # ── ATR 배수 (타임프레임별) ────────────────────────────────────
    stop_atr_mul = {"15m": 0.8, "1h": 1.0, "1d": 1.5}.get(interval, 1.2)
    # 스타일 조정
    if style == "단타":
        stop_atr_mul *= 0.85

    # ── 피벗/지지/저항 ─────────────────────────────────────────────
    pivot_support    = sc.pivots.get("support",    last * 0.95)
    pivot_resistance = sc.pivots.get("resistance", last * 1.10)
    all_resistances  = sc.pivots.get("all_resistances", [])
    all_supports     = sc.pivots.get("all_supports",    [])

    # 2차, 3차 저항선 추출
    r1 = all_resistances[0] if len(all_resistances) > 0 else pivot_resistance
    r2 = all_resistances[1] if len(all_resistances) > 1 else r1 * 1.04
    r3 = all_resistances[2] if len(all_resistances) > 2 else r2 * 1.05

    # ── 진입가 결정 ───────────────────────────────────────────────
    vcp_ready = sc.vcp["breakout_ready"]
    cup_ready = sc.cup_handle["breakout"]
    near_pivot = sc.pivots.get("near_breakout", False)

    if vcp_ready or cup_ready:
        # VCP / Cup 돌파: Livermore 피벗 위 0.5~1% 돌파 진입
        entry_price  = round(pivot_resistance * 1.005, 2)
        entry_type   = "피벗 돌파 진입"
        entry_reason = ("VCP 셋업 완성 — 저항선 돌파 + 거래량 확인 후 진입 (Minervini 방식)"
                        if vcp_ready else
                        "Cup-with-Handle 돌파 — 핸들 고점 0.5% 상방 진입 (O'Neil 방식)")
    elif near_pivot:
        # 피벗 근처: 돌파 확인 대기
        entry_price  = round(pivot_resistance * 1.003, 2)
        entry_type   = "피벗 돌파 대기"
        entry_reason = f"Livermore 저항선 ${pivot_resistance:.2f} 돌파 임박 — 거래량 동반 돌파 확인 후 진입"
    else:
        # 일반: 현재가 진입 (눌림목 이라면 약간 할인)
        if sc.rsi_val < 50 and last < sc.ma20:
            entry_price  = round(last * 0.999, 2)  # 눌림목: 현재가 또는 소폭 아래
            entry_type   = "눌림목 진입"
            entry_reason = "MA20 아래 눌림목 — 지지선 확인 후 분할 진입"
        else:
            entry_price  = round(last, 2)
            entry_type   = "즉시 진입"
            entry_reason = "추세 추종 — 현재가 진입 (모멘텀 유지 구간)"

    # ── 손절가 결정 ───────────────────────────────────────────────
    # 후보 3가지
    atr_stop    = entry_price - stop_atr_mul * atr14           # ATR 기반
    oneil_stop  = entry_price * (1 - 0.07)                     # O'Neil -7% 룰
    pivot_stop  = pivot_support                                 # Livermore 지지선

    # 3가지 중 가장 타이트(가장 높은 값) 선택 → 리스크 최소화
    raw_stop = max(atr_stop, oneil_stop, pivot_stop)

    # 안전장치: 진입가의 95% 이상
    stop_price = round(max(raw_stop, entry_price * 0.93), 2)
    stop_price = min(stop_price, entry_price * 0.999)  # 반드시 진입가 아래

    # 손절 근거
    used_method = (
        "ATR 기반" if atr_stop >= oneil_stop and atr_stop >= pivot_stop else
        "O'Neil -7% 룰" if oneil_stop >= pivot_stop else
        "Livermore 지지선"
    )
    stop_reason = f"{used_method} (ATR×{stop_atr_mul:.1f}=${atr_stop:.2f} / O'Neil=${oneil_stop:.2f} / 피벗=${pivot_stop:.2f})"
    stop_pct    = _fmt_pct(entry_price, stop_price)

    # 1R
    risk_1r = entry_price - stop_price

    # ── 익절가 결정 ───────────────────────────────────────────────
    # 1차 익절: 1R (손절폭만큼 수익 = 빠른 회수)
    tp1_raw   = entry_price + risk_1r * 1.0
    tp1_price = round(tp1_raw, 2)
    tp1_pct   = _fmt_pct(entry_price, tp1_price)
    tp1_rr    = _rr(entry_price, stop_price, tp1_price)
    tp1_qty   = 30  # 30% 청산
    tp1_reason = f"1R 달성 — 초기 리스크 회수 ({tp1_qty}% 청산, 나머지 손익분기 이동)"

    # 2차 익절: 2R 또는 1차 저항선 (더 가까운 쪽)
    tp2_raw_2r = entry_price + risk_1r * 2.2
    tp2_raw_r1 = r1 * 0.998   # 저항선 직전 (O'Neil: 저항선 못 미처 청산)
    tp2_price  = round(min(tp2_raw_2r, tp2_raw_r1) if r1 > entry_price else tp2_raw_2r, 2)
    tp2_price  = max(tp2_price, tp1_price * 1.005)  # 반드시 1차 위
    tp2_pct    = _fmt_pct(entry_price, tp2_price)
    tp2_rr     = _rr(entry_price, stop_price, tp2_price)
    tp2_qty    = 40  # 40% 청산
    tp2_reason = f"2R / 1차 저항선 ${r1:.2f} 근접 ({tp2_qty}% 청산, 잔량 추세 극대화)"

    # 3차 익절: PTJ 5:1 R:R 목표 또는 3차 저항선
    tp3_raw_5r = entry_price + risk_1r * 4.5
    tp3_raw_r2 = r2 * 0.998
    tp3_price  = round(max(tp3_raw_5r, tp3_raw_r2) if r2 > tp2_price else tp3_raw_5r, 2)
    tp3_price  = max(tp3_price, tp2_price * 1.01)   # 반드시 2차 위
    tp3_pct    = _fmt_pct(entry_price, tp3_price)
    tp3_rr     = _rr(entry_price, stop_price, tp3_price)
    tp3_qty    = 30  # 나머지 30% 청산
    tp3_reason = f"PTJ 5:1 R:R 목표 / 2차 저항선 ${r2:.2f} ({tp3_qty}% 청산, 추세 종료 시 전량)"

    # 가중 평균 R:R (물량 가중)
    weighted_rr = round(
        (tp1_rr * tp1_qty + tp2_rr * tp2_qty + tp3_rr * tp3_qty) / 100, 2
    )

    # 포지션 사이즈 가이드 (리스크 기반)
    if stop_pct < -8:
        pos_note = "손절폭 큼 — 포지션 소량 (계좌의 0.5~1%만 리스크)"
    elif stop_pct < -5:
        pos_note = "일반 리스크 — 계좌의 1~2% 리스크 기준 포지션"
    else:
        pos_note = "타이트 손절 — 계좌의 2% 리스크 기준 포지션 (비교적 크게 가능)"

    return TradePlan(
        entry_price=entry_price, entry_reason=entry_reason, entry_type=entry_type,
        stop_price=stop_price,   stop_reason=stop_reason,   stop_pct=stop_pct,
        risk_1r=round(risk_1r, 4),
        tp1_price=tp1_price, tp1_pct=tp1_pct, tp1_qty=tp1_qty,
        tp1_reason=tp1_reason, rr1=tp1_rr,
        tp2_price=tp2_price, tp2_pct=tp2_pct, tp2_qty=tp2_qty,
        tp2_reason=tp2_reason, rr2=tp2_rr,
        tp3_price=tp3_price, tp3_pct=tp3_pct, tp3_qty=tp3_qty,
        tp3_reason=tp3_reason, rr3=tp3_rr,
        weighted_rr=weighted_rr,
        max_loss_pct=round(abs(stop_pct), 2),
        position_note=pos_note,
    )


def build_signal(ticker: str, style: str, tf_choice: str) -> Optional[Tuple[Signal, pd.DataFrame]]:
    tf = TF_OPTIONS[tf_choice]
    df = fetch_ohlcv(ticker, period=tf["period"], interval=tf["interval"])
    if df is None or df.empty or len(df) < tf["min_bars"]:
        return None

    spy_df = fetch_spy()
    sc = master_score(df, spy_df if not spy_df.empty else None)

    close = df["Close"]
    last  = float(close.iloc[-1])
    ma50  = float(sma(close, 50).iloc[-1])
    ma200 = float(sma(close, min(200, len(df))).iloc[-1])
    bias  = ("상승장" if last > ma50 > ma200 else
             "하락장" if last < ma50 < ma200 else "횡보장")

    interval = tf["interval"]
    steps    = {"1d": 5, "1h": 30, "15m": 130}.get(interval, 5)
    weekly_perf = (
        (float(close.iloc[-1]) / float(close.iloc[-(steps+1)]) - 1) * 100
        if len(df) > steps else float("nan")
    )

    trade_plan = build_trade_plan(df, sc, style, tf_choice)
    vix        = fetch_vix()
    action, action_detail, action_color = decide_action(sc, vix)
    asof       = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    return Signal(
        sc=sc, vix=vix, vix_text=vix_label(vix),
        action=action, action_detail=action_detail, action_color=action_color,
        trade_plan=trade_plan,
        weekly_perf=float(weekly_perf), bias=bias, asof=asof,
    ), df
