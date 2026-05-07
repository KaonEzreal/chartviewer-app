"""
signal.py — v2 백테스트 전략 완전 구현

v2 백테스트 (PF 1.85, 승률 46.1%) 파라미터 그대로:
  손절:    ATR×1.2 vs -8% 중 더 타이트한(높은) 쪽
  브레이크이븐: +10% 달성 시 손절 → 진입가+1% 자동 상향
  1차 익절: +18% 달성 시 50% 청산
  잔량:    트레일링 스톱 (SS:8%, S:10%, A:12%) 고점 추적
  트레일   부분익절 후 7%로 더 타이트하게
  진입:    S등급(73점+) / A등급(62점+추가조건)
  쿨다운:  3연속 손절 후 7일 휴식
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

# ══════════════════════════════════════════════════════════════════
# v2 백테스트 파라미터 (backtest.py와 완전 동일)
# ══════════════════════════════════════════════════════════════════
V2_STOP_ATR_MUL   = 1.2    # 손절: ATR × 1.2
V2_STOP_MAX_PCT   = 0.08   # 손절: 최대 -8%
V2_BREAKEVEN_PCT  = 0.10   # +10% 달성 시 브레이크이븐
V2_BREAKEVEN_RAISE= 0.01   # 브레이크이븐 후 손절 → 진입가+1%
V2_PARTIAL_PCT    = 0.18   # +18% 달성 시 50% 청산
V2_PARTIAL_TRAIL  = 0.07   # 부분익절 후 트레일링 7%
V2_TRAIL_PCT      = {      # 등급별 기본 트레일링
    "SSS": 0.08, "SS": 0.08,
    "S":   0.10,
    "A":   0.12,
    "default": 0.10,
}
V2_MIN_SCORE_S    = 73
V2_MIN_SCORE_A    = 62
V2_MIN_RS         = 65
V2_MIN_ADX        = 18
V2_VOL_BREAKOUT   = 1.4
V2_RISK_PCT       = 0.015  # 트레이드당 리스크 1.5%


# ══════════════════════════════════════════════════════════════════
# 데이터 클래스
# ══════════════════════════════════════════════════════════════════
@dataclass
class EntryCondition:
    """v2 진입 조건 1개"""
    name:       str   # 조건명
    passed:     bool  # 통과 여부
    current:    str   # 현재 값
    required:   str   # 필요 조건
    meaning:    str   # 왜 중요한지 설명
    importance: str   # "critical" / "high" / "medium"


@dataclass
class TradePlan:
    """v2 백테스트 기준 매매 계획"""
    # 진입
    entry_price:  float
    entry_type:   str    # "VCP 돌파 진입" / "눌림목 진입" / "현재가 진입"
    entry_reason: str

    # 손절 (v2: ATR×1.2 vs -8%)
    stop_price:   float
    stop_pct:     float
    stop_reason:  str
    risk_1r:      float

    # 1차 익절 (v2: +18%, 50% 청산)
    tp1_price:    float
    tp1_pct:      float
    tp1_qty:      int     # 50
    tp1_reason:   str
    rr1:          float

    # 트레일링 (1차 익절 후 잔량 50% 처리)
    trail_pct:    float   # 등급별 (SS:8%, S:10%, A:12%)
    trail_note:   str     # 트레일링 설명

    # 브레이크이븐
    be_trigger:   float   # +10% 가격
    be_stop:      float   # 진입가+1%

    # 포지션 관리
    weighted_rr:  float
    max_loss_pct: float
    position_note: str


@dataclass
class Signal:
    sc:           MasterScorecard
    vix:          Optional[float]
    vix_text:     str
    action:       str
    action_detail: str
    action_color: str
    trade_plan:   TradePlan
    entry_conds:  List[EntryCondition]   # v2 9개 조건
    weekly_perf:  float
    bias:         str
    asof:         str


# ══════════════════════════════════════════════════════════════════
# v2 9개 진입 조건 계산
# ══════════════════════════════════════════════════════════════════
def calc_entry_conditions(sc: MasterScorecard, bias: str) -> List[EntryCondition]:
    """v2 백테스트 진입 조건 9개 — 순서대로"""
    conds = []

    # 1. 시장 국면
    conds.append(EntryCondition(
        name="시장 국면 (SPY)",
        passed=bias == "상승장",
        current=f"SPY {bias}",
        required="SPY 상승장 (MA50·MA200 위)",
        meaning="SPY가 MA50·MA200 위에 있어야 매수 가능. 하락장에서 백테스트 진입 차단.",
        importance="critical",
    ))

    # 2. Weinstein Stage 2
    stage = sc.weinstein["stage"]
    conds.append(EntryCondition(
        name="Weinstein Stage 2",
        passed=stage == 2,
        current=f"Stage {stage}",
        required="Stage 2 (상승 추세)",
        meaning="Stage 2만 매수. Stage 4 = 절대 금지. v2 백테스트 핵심 필터.",
        importance="critical",
    ))

    # 3. PTJ 200MA
    ptj_ok = not sc.ptj["defense_required"]
    conds.append(EntryCondition(
        name="PTJ 200MA 방어",
        passed=ptj_ok,
        current="200MA 위 ✓" if ptj_ok else "200MA 하회 ✗",
        required="현재가 > 200일 이동평균",
        meaning="Paul Tudor Jones 원칙: 200MA 아래면 즉시 관망. v2 백테스트 필수 조건.",
        importance="critical",
    ))

    # 4. Minervini TT
    tt = sc.trend_template["passed"]
    conds.append(EntryCondition(
        name="Minervini TT (6/8+)",
        passed=tt >= 6,
        current=f"TT {tt}/8",
        required="TT 6/8 이상",
        meaning="MA 정렬·52주 위치 등 8개 기술 조건 중 6개 이상. v2: TT 미달 시 A등급 진입 차단.",
        importance="high",
    ))

    # 5. O'Neil RS
    rs = sc.rs["rs_score"]
    conds.append(EntryCondition(
        name="O'Neil RS (65+)",
        passed=rs >= V2_MIN_RS,
        current=f"RS {rs}",
        required=f"RS {V2_MIN_RS} 이상",
        meaning="시장 대비 상대 강도. v2 S등급 진입 기준: RS 65+. 낮으면 시장 대비 약세.",
        importance="high",
    ))

    # 6. ADX 추세 강도
    adx_v = sc.adx_val
    conds.append(EntryCondition(
        name=f"ADX 추세강도 ({V2_MIN_ADX}+)",
        passed=adx_v >= V2_MIN_ADX,
        current=f"ADX {adx_v:.0f}",
        required=f"ADX {V2_MIN_ADX} 이상",
        meaning="추세의 강도 지표. 18 미만이면 방향성 없는 박스권일 가능성 높음.",
        importance="medium",
    ))

    # 7. 종합 점수
    score = sc.total
    conds.append(EntryCondition(
        name=f"종합 점수 ({V2_MIN_SCORE_S}점+)",
        passed=score >= V2_MIN_SCORE_S,
        current=f"{score}점 [{sc.grade}]",
        required=f"{V2_MIN_SCORE_S}점 이상 (S등급)",
        meaning="5대 트레이더 기법 통합 점수. v2 기본 진입 기준: S등급(73점) 이상.",
        importance="high",
    ))

    # 8. VCP / BB수축
    vcp_r     = sc.vcp["breakout_ready"]
    vcp_d     = sc.vcp["detected"]
    bb_sq     = sc.bb_squeeze
    perf_vcp  = sc.is_perfect_vcp
    vcp_sc    = sc.vcp_score
    conds.append(EntryCondition(
        name="VCP / BB수축 셋업",
        passed=perf_vcp or vcp_r or vcp_d or bb_sq,
        current=(f"🔥VCP완전체({vcp_sc}/10)" if perf_vcp
                 else "🚀VCP돌파준비" if vcp_r
                 else "⚡VCP진행중" if vcp_d
                 else "⚡BB수축" if bb_sq else "미감지"),
        required="VCP 감지 또는 BB수축 (완전체 = 거래량+돌파확인+기관 동시)",
        meaning=f"Minervini VCP 패턴. 점수 {vcp_sc}/10. 완전체(10점)일수록 백테스트 성과 높음.",
        importance="medium",
    ))

    # 9. 데드크로스 없음
    no_death = not sc.death_cross
    conds.append(EntryCondition(
        name="데드크로스 없음",
        passed=no_death,
        current="✓ 없음" if no_death else ("🌟골든크로스" if sc.golden_cross else "💀데드크로스"),
        required="MA50 > MA200 (데드크로스 없음)",
        meaning="데드크로스 = 중장기 하락 신호. 이 상태에서 매수는 역추세 매매.",
        importance="high",
    ))

    return conds


# ══════════════════════════════════════════════════════════════════
# v2 백테스트 기준 매매 계획
# ══════════════════════════════════════════════════════════════════
def build_trade_plan(df: pd.DataFrame, sc: MasterScorecard,
                     style: str, tf_choice: str) -> TradePlan:
    """
    v2 백테스트와 동일한 손절/익절 계산
    - 손절: ATR×1.2 vs -8% 타이트한 쪽
    - 브레이크이븐: +10% 달성 → 손절 진입가+1%
    - 1차 익절: +18% 달성 → 50% 청산
    - 잔량 50%: 등급별 트레일링 스톱 (SS:8%, S:10%, A:12%)
    - 부분익절 후 트레일링 7%로 더 타이트
    """
    last     = float(df["Close"].iloc[-1])
    atr14    = float(atr(df, 14).iloc[-1])
    grade    = sc.grade

    # ── 진입가 ────────────────────────────────────────────────────
    pivs       = sc.pivots
    resistance = pivs.get("resistance", last * 1.05)
    vcp_r      = sc.vcp["breakout_ready"]

    if vcp_r and resistance > last:
        entry_price  = round(resistance * 1.003, 2)
        entry_type   = "VCP 돌파 진입"
        entry_reason = "VCP 완성 — 저항선 0.3% 상방 거래량 동반 돌파 확인 후 진입"
    elif sc.rsi_val < 50 and last < sc.ma20:
        entry_price  = round(last, 2)
        entry_type   = "눌림목 진입"
        entry_reason = "MA20 아래 눌림목 — 지지 확인 후 분할 진입"
    else:
        entry_price  = round(last, 2)
        entry_type   = "현재가 진입"
        entry_reason = "추세 추종 — 현재가 진입 (모멘텀 유지 구간)"

    # ── 손절가 (v2: ATR×1.2 vs -8% 타이트한 쪽) ─────────────────
    atr_stop  = entry_price - V2_STOP_ATR_MUL * atr14
    pct_stop  = entry_price * (1 - V2_STOP_MAX_PCT)
    stop_price = round(max(atr_stop, pct_stop), 2)   # 더 높은(타이트한) 쪽
    stop_price = min(stop_price, entry_price * 0.999)

    used = "ATR×1.2" if atr_stop >= pct_stop else "-8% 룰"
    stop_pct    = round((stop_price / entry_price - 1) * 100, 2)
    stop_reason = (f"{used} 적용 (ATR×1.2=${atr_stop:.2f} / -8%=${pct_stop:.2f} "
                   f"→ 더 타이트한 ${stop_price:.2f} 선택)")
    risk_1r     = max(entry_price - stop_price, 0.01)

    # ── 브레이크이븐 ──────────────────────────────────────────────
    be_trigger = round(entry_price * (1 + V2_BREAKEVEN_PCT), 2)  # +10% 가격
    be_stop    = round(entry_price * (1 + V2_BREAKEVEN_RAISE), 2) # 진입가+1%

    # ── 1차 익절 (v2: +18% 달성 시 50% 청산) ─────────────────────
    tp1_price = round(entry_price * (1 + V2_PARTIAL_PCT), 2)
    tp1_pct   = round((tp1_price / entry_price - 1) * 100, 2)
    rr1       = round((tp1_price - entry_price) / risk_1r, 2)
    tp1_reason = f"+{V2_PARTIAL_PCT*100:.0f}% 달성 → 50% 청산, 손절 진입가+1%로 상향 (브레이크이븐)"

    # ── 트레일링 스톱 (잔량 50% 처리) ────────────────────────────
    trail_pct  = V2_TRAIL_PCT.get(grade, V2_TRAIL_PCT["default"])
    trail_note = (
        f"잔량 50%는 고점 대비 -{trail_pct*100:.0f}% 트레일링 스톱 자동 추적\n"
        f"• +{V2_BREAKEVEN_PCT*100:.0f}% 달성 시 → 손절 ${be_stop:.2f}(진입가+1%)로 상향\n"
        f"• +{V2_PARTIAL_PCT*100:.0f}% 1차 청산 후 → 트레일링 7%로 더 타이트\n"
        f"• 고점 갱신될 때마다 자동 상향 → 추세 끝까지 보유"
    )

    # 참고용 트레일링 예상 청산가 (현재 고점 기준)
    # 실제 청산가는 시장 움직임에 따라 달라짐

    # ── 가중 R:R (1차 50% + 트레일링 50% 평균 2.5R 가정) ─────────
    # weighted_rr: 1차 50%(rr1) + 잔량 50%(트레일링으로 추가 수익)
    # 트레일링 잔량의 기대 R:R = (trail_pct 반비례) 
    # 타이트한 트레일(SS:8%) = 빨리 청산 가능성, 느슨한(A:12%) = 더 오래 보유
    trail_rr_estimate = round(1.0 / trail_pct, 1)  # 8%→12.5R, 10%→10R, 12%→8.3R (이론값)
    trail_rr_capped   = min(trail_rr_estimate, 5.0)  # 현실적 5R 캡
    weighted_rr = round(rr1 * 0.5 + trail_rr_capped * 0.5, 2)

    # ── 포지션 사이즈 ────────────────────────────────────────────
    if abs(stop_pct) > 7:
        pos_note = "손절폭 큼 → 계좌의 1% 이하 리스크 (v2: 트레이드당 1.5%)"
    elif abs(stop_pct) > 5:
        pos_note = "계좌의 1~1.5% 리스크 기준 (v2 기본: 1.5%)"
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
        tp1_price=tp1_price,
        tp1_pct=tp1_pct,
        tp1_qty=50,
        tp1_reason=tp1_reason,
        rr1=rr1,
        trail_pct=trail_pct,
        trail_note=trail_note,
        be_trigger=be_trigger,
        be_stop=be_stop,
        weighted_rr=weighted_rr,
        max_loss_pct=round(abs(stop_pct), 2),
        position_note=pos_note,
    )


# ══════════════════════════════════════════════════════════════════
# 진입 판정 (v2 기준)
# ══════════════════════════════════════════════════════════════════
def decide_action(sc: MasterScorecard, vix: Optional[float],
                  conds: List[EntryCondition]) -> Tuple[str, str, str]:
    passed      = sum(1 for c in conds if c.passed)
    total       = len(conds)
    critical_ok = all(c.passed for c in conds if c.importance == "critical")
    stage       = sc.weinstein["stage"]
    vix_high    = vix is not None and vix >= 28

    if not critical_ok or stage == 4:
        fails = [c.name for c in conds if not c.passed and c.importance == "critical"]
        return "진입 보류", f"핵심 조건 미충족: {' · '.join(fails[:2])}", "red"

    if passed == total and sc.vcp["breakout_ready"]:
        return "강력 매수", f"9/9 조건 + VCP 돌파 — v2 백테스트 최고 성과 구간", "cyan"

    if passed == total:
        return "적극 매수", f"9/9 모든 조건 충족 — 하라는 대로 진입", "green"

    if passed >= 7 and critical_ok and not vix_high:
        return "매수 검토", f"{passed}/{total} 조건 충족 — 추가 확인 후 진입", "green"

    if passed >= 5 and critical_ok:
        return "조건부 매수", f"{passed}/{total} 조건 충족 — 소량 또는 대기", "yellow"

    return "관망", f"{passed}/{total} 조건 — 조건 개선 대기", "red"


def vix_label(vix: Optional[float]) -> str:
    if vix is None: return ""
    if vix >= 35: return f"🔴 VIX {vix:.1f} — 극도의 공포. 포지션 최소화"
    if vix >= 28: return f"🟠 VIX {vix:.1f} — 변동성 경고. 손절 축소"
    if vix >= 22: return f"🟡 VIX {vix:.1f} — 변동성 주의. 추격 매수 자제"
    return f"🟢 VIX {vix:.1f} — 시장 안정. 정상 매매 가능"


# ══════════════════════════════════════════════════════════════════
# 메인 시그널 빌더
# ══════════════════════════════════════════════════════════════════
def build_signal(ticker: str, style: str,
                 tf_choice: str) -> Optional[Tuple[Signal, pd.DataFrame]]:
    tf = TF_OPTIONS[tf_choice]
    df = fetch_ohlcv(ticker, period=tf["period"], interval=tf["interval"])
    if df is None or df.empty or len(df) < tf["min_bars"]:
        return None

    spy_df = fetch_spy()
    sc     = master_score(df, spy_df if not spy_df.empty else None)

    close = df["Close"]
    last  = float(close.iloc[-1])
    ma50  = float(sma(close, 50).iloc[-1])
    ma200 = float(sma(close, min(200, len(df))).iloc[-1])

    # ── 시장 국면: 종목 자체 MA가 아닌 SPY MA 기준으로 판단 ─────
    if not spy_df.empty and len(spy_df) >= 50:
        spy_close  = spy_df["Close"]
        spy_last   = float(spy_close.iloc[-1])
        spy_ma50   = float(sma(spy_close, 50).iloc[-1])
        spy_ma200  = float(sma(spy_close, min(200, len(spy_df))).iloc[-1])
        bias = ("상승장" if spy_last > spy_ma50 > spy_ma200 else
                "하락장" if spy_last < spy_ma50 and spy_last < spy_ma200 else "횡보장")
    else:
        # SPY 데이터 없으면 종목 자체 기준 폴백
        bias = ("상승장" if last > ma50 > ma200 else
                "하락장" if last < ma50 < ma200 else "횡보장")

    interval = tf["interval"]
    steps    = {"1d": 5, "1h": 30, "15m": 130}.get(interval, 5)
    weekly_perf = (
        (float(close.iloc[-1]) / float(close.iloc[-(steps+1)]) - 1) * 100
        if len(df) > steps else float("nan")
    )

    entry_conds = calc_entry_conditions(sc, bias)
    trade_plan  = build_trade_plan(df, sc, style, tf_choice)
    vix         = fetch_vix()
    action, action_detail, action_color = decide_action(sc, vix, entry_conds)
    asof        = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    return Signal(
        sc=sc,
        vix=vix,
        vix_text=vix_label(vix),
        action=action,
        action_detail=action_detail,
        action_color=action_color,
        trade_plan=trade_plan,
        entry_conds=entry_conds,
        weekly_perf=float(weekly_perf),
        bias=bias,
        asof=asof,
    ), df
