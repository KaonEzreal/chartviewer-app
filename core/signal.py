"""
signal.py v5 — v2 백테스트 전략과 완전 일치

v2 백테스트 기준:
  손절 = ATR×1.2 vs 진입가의 -8% → 더 타이트한(높은) 쪽
  1차 익절(30%) = +18% (부분익절)
  2차 익절(40%) = 트레일링 스톱 구간 (고점-10%)
  3차 익절(30%) = 추세 종료 or 고점-10% 트레일링
  브레이크이븐 = +10% 달성 시 손절 → 진입가+1%

9개 진입 조건 (v2 백테스트와 동일):
  1. 시장 국면 (SPY MA50·MA200 위)
  2. Weinstein Stage 2
  3. Minervini TT 6/8 이상
  4. PTJ 200MA 룰
  5. O'Neil RS 65 이상
  6. 거래량 1.4배 이상
  7. VCP / BB 수축
  8. RSI 80 이하
  9. 데드크로스 없음
"""

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd

from .config import TF_OPTIONS
from .data import fetch_ohlcv, fetch_vix, fetch_spy
from .indicators import atr, sma
from .scoring import MasterScorecard, master_score


# ──────────────────────────────────────────────────────────────────
# v2 백테스트 기준 상수
# ──────────────────────────────────────────────────────────────────
V2_STOP_ATR_MUL  = 1.2    # ATR × 1.2
V2_STOP_MAX_PCT  = 0.08   # 최대 손절 -8%
V2_TP1_PCT       = 0.18   # 1차 익절 +18% (30% 청산)
V2_TP2_PCT       = 0.28   # 2차 익절 +28% 근사 (40% 청산)
V2_TP3_PCT       = 0.45   # 3차 익절 +45% 근사 (30% 청산)
V2_BREAKEVEN_PCT = 0.10   # +10% 달성 시 손절 → 진입가+1%
V2_TRAIL_PCT     = 0.10   # 트레일링 10%
V2_MIN_RS        = 65
V2_MIN_ADX       = 18
V2_MIN_VOL       = 1.4
V2_MIN_TT        = 6
V2_MIN_SCORE_S   = 73
V2_MIN_SCORE_A   = 62


@dataclass
class TradePlan:
    # 진입
    entry_price:  float
    entry_type:   str
    entry_reason: str

    # 손절 (v2: ATR×1.2 vs -8% 타이트한 쪽)
    stop_price:   float
    stop_pct:     float
    stop_reason:  str
    risk_1r:      float

    # 1차 익절 +18%, 30% 청산
    tp1_price: float
    tp1_pct:   float
    tp1_qty:   int
    tp1_reason: str
    rr1:       float

    # 2차 익절 ~ +28%, 40% 청산
    tp2_price: float
    tp2_pct:   float
    tp2_qty:   int
    tp2_reason: str
    rr2:       float

    # 3차 익절 ~ +45%, 30% 청산
    tp3_price: float
    tp3_pct:   float
    tp3_qty:   int
    tp3_reason: str
    rr3:       float

    weighted_rr:  float
    max_loss_pct: float
    position_note: str
    breakeven_note: str  # 브레이크이븐 안내


@dataclass
class EntryCondition:
    """9개 진입 조건 체크 결과"""
    name:      str
    passed:    bool
    current:   str   # 현재 값 표시
    required:  str   # 필요 조건
    meaning:   str   # 왜 중요한지 한 줄 설명


@dataclass
class Signal:
    sc:             MasterScorecard
    vix:            Optional[float]
    vix_text:       str
    action:         str
    action_detail:  str
    action_color:   str   # green / cyan / yellow / red
    trade_plan:     TradePlan
    conditions:     List[EntryCondition]   # 9개 조건
    pass_count:     int                    # 충족 조건 수
    weekly_perf:    float
    bias:           str
    asof:           str


# ──────────────────────────────────────────────────────────────────
# 9개 진입 조건 평가 (v2 백테스트 기준 그대로)
# ──────────────────────────────────────────────────────────────────
def evaluate_conditions(sc: MasterScorecard,
                        spy_bias: str) -> List[EntryCondition]:
    conds = []

    # 1. 시장 국면
    spy_ok = spy_bias == "상승장"
    conds.append(EntryCondition(
        name="시장 국면 (SPY)",
        passed=spy_ok,
        current=spy_bias,
        required="상승장 (SPY MA50·MA200 위)",
        meaning="시장 전체가 오르는 환경이어야 개별주도 오른다",
    ))

    # 2. Weinstein Stage 2
    stage = sc.weinstein["stage"]
    conds.append(EntryCondition(
        name="Weinstein Stage 2",
        passed=stage == 2,
        current=f"Stage {stage}",
        required="Stage 2 (상승 추세 구간)",
        meaning="Stage 2만 매수 — Stage 4는 절대 금지",
    ))

    # 3. Minervini TT
    tt = sc.trend_template["passed"]
    conds.append(EntryCondition(
        name="Minervini TT",
        passed=tt >= V2_MIN_TT,
        current=f"{tt}/8",
        required=f"{V2_MIN_TT}/8 이상",
        meaning="MA 정렬·52주 위치 등 8개 조건으로 Stage 2 품질 검증",
    ))

    # 4. PTJ 200MA
    ptj_ok = not sc.ptj["defense_required"]
    dist   = sc.ptj.get("dist_pct", 0)
    conds.append(EntryCondition(
        name="PTJ 200MA 룰",
        passed=ptj_ok,
        current=f"200MA {'위' if ptj_ok else '아래'} ({dist:+.1f}%)",
        required="현재가 > 200MA",
        meaning="Paul Tudor Jones: 200MA 아래는 즉시 관망",
    ))

    # 5. O'Neil RS
    rs = sc.rs["rs_score"]
    conds.append(EntryCondition(
        name="O'Neil RS 점수",
        passed=rs >= V2_MIN_RS,
        current=f"RS {rs}",
        required=f"RS {V2_MIN_RS} 이상",
        meaning="시장 대비 강한 종목만 — 약한 종목은 시장 빠질 때 더 빠짐",
    ))

    # 6. 거래량
    vol = sc.vol_ratio
    conds.append(EntryCondition(
        name="거래량 돌파",
        passed=vol >= V2_MIN_VOL,
        current=f"{vol:.2f}x",
        required=f"평균 대비 {V2_MIN_VOL}배 이상",
        meaning="거래량 없는 상승은 가짜 — 기관이 들어와야 진짜",
    ))

    # 7. VCP / BB 수축
    vcp_ok = sc.vcp["breakout_ready"] or sc.vcp["detected"] or sc.bb_squeeze
    vcp_label = ("VCP 돌파 준비" if sc.vcp["breakout_ready"]
                 else "VCP 진행중" if sc.vcp["detected"]
                 else "BB 수축" if sc.bb_squeeze else "미감지")
    conds.append(EntryCondition(
        name="VCP / BB 수축",
        passed=vcp_ok,
        current=vcp_label,
        required="VCP 또는 BB 수축 감지",
        meaning="눌림 후 폭발 직전 패턴 — Minervini의 핵심 진입 셋업",
    ))

    # 8. RSI 과열 아님
    rsi_v = sc.rsi_val
    conds.append(EntryCondition(
        name="RSI 과열 아님",
        passed=rsi_v <= 80,
        current=f"RSI {rsi_v:.0f}",
        required="RSI 80 이하",
        meaning="RSI 80 초과 = 단기 극단 과열, 추격 매수 위험",
    ))

    # 9. 데드크로스 없음
    no_death = not sc.death_cross
    conds.append(EntryCondition(
        name="데드크로스 없음",
        passed=no_death,
        current="✓ 없음" if no_death else "💀 발생",
        required="MA50 > MA200 (데드크로스 없음)",
        meaning="데드크로스 = 중장기 하락 신호, 이 상태에서 매수는 역추세",
    ))

    return conds


# ──────────────────────────────────────────────────────────────────
# v2 기준 매매 계획 계산
# ──────────────────────────────────────────────────────────────────
def build_trade_plan(df: pd.DataFrame, sc: MasterScorecard,
                     style: str, tf_choice: str) -> TradePlan:
    last     = float(df["Close"].iloc[-1])
    atr14    = float(atr(df, 14).iloc[-1])
    interval = TF_OPTIONS[tf_choice]["interval"]

    # ── 진입가 ────────────────────────────────────────────────────
    vcp_r = sc.vcp["breakout_ready"]
    pivs  = sc.pivots
    resistance = pivs.get("resistance", last * 1.05)
    support    = pivs.get("support",    last * 0.95)

    if vcp_r and resistance > last:
        entry_price  = round(resistance * 1.003, 2)
        entry_type   = "피벗 돌파 진입"
        entry_reason = "VCP 셋업 완성 — 저항선 0.3% 상방 돌파 확인 후 진입"
    elif sc.rsi_val < 50 and last < sc.ma20:
        entry_price  = round(last, 2)
        entry_type   = "눌림목 진입"
        entry_reason = "MA20 아래 눌림목 — 지지 확인 후 분할 진입"
    else:
        entry_price  = round(last, 2)
        entry_type   = "현재가 진입"
        entry_reason = "추세 추종 — 현재가 진입 (모멘텀 유지 구간)"

    # ── 손절가 (v2 백테스트 기준 그대로) ─────────────────────────
    # ATR×1.2 vs -8% → 더 타이트한(높은) 쪽
    atr_stop = entry_price - V2_STOP_ATR_MUL * atr14
    pct_stop = entry_price * (1 - V2_STOP_MAX_PCT)
    stop_price = round(max(atr_stop, pct_stop), 2)
    stop_price = min(stop_price, entry_price * 0.999)

    method = "ATR×1.2" if atr_stop >= pct_stop else "-8% 룰"
    stop_pct   = round((stop_price / entry_price - 1) * 100, 2)
    stop_reason = (f"{method} 적용 "
                   f"(ATR=${atr_stop:.2f} / -8%=${pct_stop:.2f} → "
                   f"더 타이트한 쪽 선택)")
    risk_1r = entry_price - stop_price

    def rr(tp):
        return round((tp - entry_price) / risk_1r, 2) if risk_1r > 0 else 0

    # ── 익절 (v2 백테스트 기준) ───────────────────────────────────
    # 1차: +18% (30% 청산) — v2 부분익절 기준
    tp1 = round(entry_price * (1 + V2_TP1_PCT), 2)

    # 2차: +28% 근사 (40% 청산) — 트레일링 구간 진입점
    tp2 = round(entry_price * (1 + V2_TP2_PCT), 2)
    tp2 = max(tp2, tp1 * 1.005)

    # 3차: +45% (30% 청산) — 트레일링 최종 목표
    tp3 = round(entry_price * (1 + V2_TP3_PCT), 2)
    tp3 = max(tp3, tp2 * 1.01)

    tp1_pct = round((tp1 / entry_price - 1) * 100, 2)
    tp2_pct = round((tp2 / entry_price - 1) * 100, 2)
    tp3_pct = round((tp3 / entry_price - 1) * 100, 2)

    weighted_rr = round(
        (rr(tp1)*30 + rr(tp2)*40 + rr(tp3)*30) / 100, 2
    )

    # 브레이크이븐 가격
    be_price = round(entry_price * (1 + 0.01), 2)  # 진입가+1%
    be_trigger = round(entry_price * (1 + V2_BREAKEVEN_PCT), 2)

    # 포지션 사이즈
    if abs(stop_pct) > 7:
        pos_note = "손절폭 큼 → 계좌의 1% 이하 리스크"
    elif abs(stop_pct) > 5:
        pos_note = "계좌의 1~1.5% 리스크 기준 (v2: 1.5%)"
    else:
        pos_note = "타이트 손절 → 계좌의 1.5~2% 리스크 가능"

    return TradePlan(
        entry_price=entry_price,
        entry_type=entry_type,
        entry_reason=entry_reason,
        stop_price=stop_price,
        stop_pct=stop_pct,
        stop_reason=stop_reason,
        risk_1r=round(risk_1r, 4),
        tp1_price=tp1, tp1_pct=tp1_pct, tp1_qty=30,
        tp1_reason=f"+18% 달성 → 30% 청산, 손절 진입가+1%로 상향 (브레이크이븐)",
        rr1=rr(tp1),
        tp2_price=tp2, tp2_pct=tp2_pct, tp2_qty=40,
        tp2_reason=f"+28% 구간 → 40% 청산, 트레일링 10% 적용",
        rr2=rr(tp2),
        tp3_price=tp3, tp3_pct=tp3_pct, tp3_qty=30,
        tp3_reason=f"+45% 구간 → 나머지 30% 전량 청산",
        rr3=rr(tp3),
        weighted_rr=weighted_rr,
        max_loss_pct=round(abs(stop_pct), 2),
        position_note=pos_note,
        breakeven_note=(f"+10% ({be_trigger:.2f}) 도달 시 "
                        f"손절을 {be_price:.2f}로 자동 상향"),
    )


# ──────────────────────────────────────────────────────────────────
# 액션 판정 (9개 조건 기반)
# ──────────────────────────────────────────────────────────────────
def decide_action(sc: MasterScorecard, pass_count: int,
                  conds: List[EntryCondition],
                  vix: Optional[float]) -> Tuple[str, str, str]:
    vix_high  = vix is not None and vix >= 28
    # 필수 차단 조건 (이것만 걸려도 진입 불가)
    critical_fail = [c for c in conds
                     if not c.passed and c.name in
                     ("PTJ 200MA 룰", "Weinstein Stage 2", "데드크로스 없음")]

    if critical_fail:
        names = " · ".join(c.name for c in critical_fail)
        return "진입 금지", f"필수 조건 미충족: {names}", "red"

    if sc.total >= V2_MIN_SCORE_S and pass_count >= 8:
        return "매수 적극 추천", f"9개 조건 {pass_count}개 충족 · S등급+ · 최우선 진입 검토", "cyan"
    if sc.total >= V2_MIN_SCORE_S and pass_count >= 7:
        return "매수 추천", f"9개 조건 {pass_count}개 충족 · S등급+ · 진입 적합", "green"
    if sc.total >= V2_MIN_SCORE_S and pass_count >= 6:
        return "분할 매수", f"9개 조건 {pass_count}개 충족 · 분할 진입 권장", "green"
    if pass_count >= 5 and sc.total >= V2_MIN_SCORE_A:
        return "조건 대기", f"9개 조건 {pass_count}개 충족 · 미충족 조건 개선 대기", "yellow"
    return "진입 보류", f"9개 조건 {pass_count}개만 충족 · 다른 종목 탐색 권장", "red"


def vix_label(vix: Optional[float]) -> str:
    if vix is None: return ""
    if vix >= 35: return f"🔴 VIX {vix:.1f} — 극단 공포. 포지션 최소화"
    if vix >= 28: return f"🟠 VIX {vix:.1f} — 변동성 경고. 손절 타이트하게"
    if vix >= 22: return f"🟡 VIX {vix:.1f} — 변동성 주의. 추격 매수 자제"
    return f"🟢 VIX {vix:.1f} — 시장 안정. 정상 매매 가능"


# ──────────────────────────────────────────────────────────────────
# 메인 빌드 함수
# ──────────────────────────────────────────────────────────────────
def build_signal(ticker: str, style: str,
                 tf_choice: str) -> Optional[Tuple[Signal, pd.DataFrame]]:
    tf = TF_OPTIONS[tf_choice]
    df = fetch_ohlcv(ticker, period=tf["period"], interval=tf["interval"])
    if df is None or df.empty or len(df) < tf["min_bars"]:
        return None

    spy_df = fetch_spy()
    sc     = master_score(df, spy_df if not spy_df.empty else None)

    close  = df["Close"]
    last   = float(close.iloc[-1])
    ma50   = float(sma(close, 50).iloc[-1])
    ma200  = float(sma(close, min(200, len(df))).iloc[-1])
    bias   = ("상승장" if last > ma50 > ma200 else
              "하락장" if last < ma50 < ma200 else "횡보장")

    interval    = tf["interval"]
    steps       = {"1d": 5, "1h": 30, "15m": 130}.get(interval, 5)
    weekly_perf = (
        (float(close.iloc[-1]) / float(close.iloc[-(steps+1)]) - 1) * 100
        if len(df) > steps else float("nan")
    )

    conds      = evaluate_conditions(sc, bias)
    pass_count = sum(1 for c in conds if c.passed)
    trade_plan = build_trade_plan(df, sc, style, tf_choice)
    vix        = fetch_vix()
    action, action_detail, action_color = decide_action(
        sc, pass_count, conds, vix
    )
    asof = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    return Signal(
        sc=sc, vix=vix, vix_text=vix_label(vix),
        action=action, action_detail=action_detail,
        action_color=action_color,
        trade_plan=trade_plan,
        conditions=conds,
        pass_count=pass_count,
        weekly_perf=float(weekly_perf),
        bias=bias, asof=asof,
    ), df
