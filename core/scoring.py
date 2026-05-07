import math
"""
scoring.py v3 — 5대 트레이더 통합 점수 (완전 재설계)

핵심 철학:
  Minervini: 강한 추세 + VCP 셋업 = 높은 점수
  Weinstein: Stage 2 = 최고점, Stage 4 = 최저점
  O'Neil:    RS 높을수록 + 거래량 터질수록 = 좋음
  PTJ:       200MA 위 = 안전, 아래 = 즉시 감점
  Livermore: 피벗 돌파 근처 = 최적 진입

점수 분해 (100점):
  추세 구조    35점  (MA 정렬, Stage, TT, PTJ)
  모멘텀       25점  (MACD, RS, ADX, 이격도)
  패턴/셋업    20점  (VCP, Cup, 피벗, BB수축)
  수급/거래량  20점  (MFI, OBV, Vol 배수)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np

from .indicators import (
    sma, rsi, macd, atr, mfi, bbands, stochastic, adx, cci,
    vwap, williams_r, week52,
    minervini_trend_template, detect_vcp, weinstein_stage,
    oneil_rs_score, detect_cup_handle,
    ptj_defense_rule, livermore_pivots,
    relative_volume, breakout_quality, institutional_proxy, enhanced_vcp,
)


@dataclass
class MasterScorecard:
    total: int
    grade: str
    signal_strength: str

    trend_score: int
    momentum_score: int
    pattern_score: int
    flow_score: int
    risk_adj: int

    trend_template: dict
    vcp: dict
    weinstein: dict
    rs: dict
    cup_handle: dict
    ptj: dict
    pivots: dict

    last: float
    ma20: float
    ma50: float
    ma200: float
    rsi_val: float
    mfi_val: float
    macd_hist: float
    atr_pct: float
    vol_ratio: float
    bb_position: float
    adx_val: float
    stoch_k: float
    stoch_d: float
    cci_val: float
    williams_r: float
    vwap_val: float
    vwap_above: bool
    week52_h: float
    week52_l: float
    week52_pos: float
    bb_squeeze: bool
    golden_cross: bool
    death_cross: bool
    # 고급 거래량/돌파 분석
    rvol:           float
    rvol_category:  str
    vol_confirm:    bool
    fake_breakout:  bool
    close_above_pivot: bool
    inst_signal:    str
    inst_score:     int
    vcp_score:      int
    is_perfect_vcp: bool

    reasons: List[str]
    warnings: List[str]


def clamp(x, lo=0, hi=100):
    return int(max(lo, min(hi, round(x))))

def grade(s):
    if s >= 90: return "SSS"
    if s >= 82: return "SS"
    if s >= 73: return "S"
    if s >= 62: return "A"
    if s >= 50: return "B"
    if s >= 38: return "C"
    return "D"

def signal_strength(s):
    if s >= 85: return "극강 매수"
    if s >= 72: return "매수 우위"
    if s >= 58: return "중립 관찰"
    if s >= 45: return "매수 자제"
    return "관망/회피"


def master_score(df: pd.DataFrame, spy_df=None) -> MasterScorecard:
    close = df["Close"]
    n = len(df)
    last = float(close.iloc[-1])

    # ── 기본 지표 ─────────────────────────────────────────────────
    ma20  = float(sma(close, 20).iloc[-1])
    ma50  = float(sma(close, 50).iloc[-1])
    ma200 = float(sma(close, min(200, n)).iloc[-1])

    rsi14  = float(rsi(close, 14).iloc[-1])
    _mfi   = mfi(df, 14).iloc[-1]
    mfi14  = float(_mfi) if not (hasattr(_mfi, "__float__") and __import__("math").isnan(float(_mfi))) else 50.0
    atr14  = float(atr(df, 14).iloc[-1])
    atr_pct = atr14 / last if last else 0
    adx14  = float(adx(df, 14).iloc[-1])

    lb, mb, ub = bbands(close, 20, 2.0)
    lb, mb, ub = float(lb.iloc[-1]), float(mb.iloc[-1]), float(ub.iloc[-1])
    bb_width   = max(ub - lb, 1e-9)
    bb_pos     = (last - lb) / bb_width

    # BB Squeeze: 현재 밴드폭이 20봉 전보다 좁으면
    if n >= 25:
        lb_p, mb_p, ub_p = bbands(close, 20, 2.0)
        bw_now  = (ub - lb) / mb if mb else 0
        bw_prev = (float(ub_p.iloc[-20]) - float(lb_p.iloc[-20])) / float(mb_p.iloc[-20]) if float(mb_p.iloc[-20]) > 0 else bw_now
        bb_squeeze = bw_now < bw_prev * 0.75
    else:
        bb_squeeze = False

    ml, sl2, hist = macd(close)
    macd_h     = float(hist.iloc[-1])
    macd_cu    = bool(n >= 2 and ml.iloc[-1] > sl2.iloc[-1] and ml.iloc[-2] <= sl2.iloc[-2])
    macd_cd    = bool(n >= 2 and ml.iloc[-1] < sl2.iloc[-1] and ml.iloc[-2] >= sl2.iloc[-2])

    sk_s, sd_s = stochastic(df)
    sk_v = float(sk_s.iloc[-1])
    sd_v = float(sd_s.iloc[-1])
    cci_v = float(cci(df).iloc[-1])
    wr_v  = float(williams_r(df).iloc[-1])

    vwap_v    = float(vwap(df).iloc[-1])
    vwap_ab   = last > vwap_v

    h52, l52, pos52 = week52(df)
    vol_now  = float(df["Volume"].tail(10).mean())
    vol_base = float(df["Volume"].tail(60).mean()) if n >= 60 else float(df["Volume"].mean())
    vol_ratio = vol_now / vol_base if vol_base else 1.0

    golden = bool(n >= 202 and
                  sma(close,50).iloc[-1] > sma(close,200).iloc[-1] and
                  sma(close,50).iloc[-2] <= sma(close,200).iloc[-2])
    death  = bool(n >= 202 and
                  sma(close,50).iloc[-1] < sma(close,200).iloc[-1] and
                  sma(close,50).iloc[-2] >= sma(close,200).iloc[-2])

    # ── 전설 트레이더 기법 ─────────────────────────────────────────
    tt   = minervini_trend_template(df)
    vcp   = enhanced_vcp(df)
    _rvol = relative_volume(df)
    _bq   = breakout_quality(df)
    _inst = institutional_proxy(df)
    # 보정값 미리 계산 (패턴/수급/리스크 섹션에서 사용)
    rvol_val = _rvol["rvol"]
    if   rvol_val >= 2.0: vol_quality_bonus = 4
    elif rvol_val >= 1.5: vol_quality_bonus = 3
    elif rvol_val >= 1.2: vol_quality_bonus = 1
    else:                 vol_quality_bonus = 0
    fake_bk_penalty = -5 if _bq["fake_breakout"] else 0
    inst_bonus = max(-4, min(4, _inst["inst_score"] * 2))
    ws   = weinstein_stage(df)
    rs   = oneil_rs_score(df, spy_df)
    cph  = detect_cup_handle(df)
    ptj  = ptj_defense_rule(df)
    pivs = livermore_pivots(df)

    # ══════════════════════════════════════════════════════════════
    # 1. 추세 구조 점수 (35점)
    #    철학: 추세가 강할수록 높은 점수 (Minervini/Weinstein/PTJ)
    # ══════════════════════════════════════════════════════════════
    trend = 0

    # (a) MA 정렬 상태 (15점) — 위에 있을수록 좋음
    if last > ma20:   trend += 3
    if last > ma50:   trend += 4
    if last > ma200:  trend += 5
    if ma20 > ma50:   trend += 2
    if ma50 > ma200:  trend += 1
    # 완전 정렬 보너스
    if last > ma20 > ma50 > ma200: trend += 2  # 총 17 가능, cap은 clamp로

    # (b) Weinstein Stage (10점)
    stage = ws["stage"]
    stage_pts = {2: 10, 1: 4, 3: 1, 4: 0}
    trend += stage_pts.get(stage, 0)

    # (c) Minervini Trend Template (7점)
    trend += int(tt["passed"] / 8 * 7)

    # (d) PTJ 200MA (3점)
    if ptj["above_200ma"]:
        trend += 3

    # ══════════════════════════════════════════════════════════════
    # 2. 모멘텀 점수 (25점)
    #    철학: 추세 추종 → 강한 모멘텀(높은 RSI)도 긍정적
    #          역발상 → 과매도에서 반등 기대
    #    둘 다 반영: RSI 40~70 최고점, 30 이하/80 이상도 상황 따라 점수
    # ══════════════════════════════════════════════════════════════
    momentum = 0

    # (a) RSI (6점) — 추세 추종 친화적으로 수정
    #     40~70 = 건전한 상승 모멘텀 (최고)
    #     30~40 = 과매도 반등 가능 (중간)
    #     70~80 = 강한 추세 유지 (중간)
    #     <30 또는 >80 = 극단 (낮음, 리스크)
    if   40 <= rsi14 <= 70:  momentum += 6   # 건전한 모멘텀
    elif 30 <= rsi14 < 40:   momentum += 4   # 과매도 반등 기대
    elif 70 < rsi14 <= 80:   momentum += 4   # 강한 추세
    elif rsi14 < 30:         momentum += 2   # 극단 과매도
    else:                    momentum += 1   # 극단 과매수 (>80)

    # (b) MACD (8점) — 크로스 최우선
    if   macd_cu:              momentum += 8
    elif macd_h > 0 and not macd_cd: momentum += 5
    elif macd_h > 0:           momentum += 3
    if   macd_cd:              momentum -= 3  # 데드크로스 패널티

    # (c) O'Neil RS Score (7점) — 높을수록 좋음
    rs_s = rs["rs_score"]
    if   rs_s >= 90: momentum += 7
    elif rs_s >= 80: momentum += 6
    elif rs_s >= 70: momentum += 4
    elif rs_s >= 50: momentum += 2
    else:            momentum += 0

    # (d) ADX (추세 강도) (4점)
    if   adx14 >= 40: momentum += 4
    elif adx14 >= 25: momentum += 3
    elif adx14 >= 15: momentum += 1
    else:             momentum += 0

    # ══════════════════════════════════════════════════════════════
    # 3. 패턴/셋업 점수 (20점)
    #    VCP + Cup + 피벗 + BB Squeeze
    # ══════════════════════════════════════════════════════════════
    pattern = 0

    # (a) VCP (8점)
    if vcp["breakout_ready"]:  pattern += 8
    elif vcp["detected"]:      pattern += 5
    elif vcp["tightness"] > 0.15: pattern += 2

    # (b) Cup-with-Handle (6점)
    if cph["breakout"]:       pattern += 6
    elif cph["handle"]:       pattern += 4
    elif cph["detected"]:     pattern += 2

    # (c) Livermore 피벗 (4점)
    if pivs["near_breakout"]:  pattern += 4
    elif pivs["rr_ratio"] >= 2: pattern += 2
    elif pivs["rr_ratio"] >= 1: pattern += 1

    # (d) BB Squeeze (2점) — 폭발 준비
    if bb_squeeze:             pattern += 2

    # ══════════════════════════════════════════════════════════════
    # 4. 수급/거래량 점수 (20점)
    #    MFI, Vol ratio, VWAP
    # ══════════════════════════════════════════════════════════════
    flow = 0

    # (a) MFI (8점) — 50 중립, 높을수록 수급 강함, 너무 높으면 과열
    if   50 <= mfi14 <= 75:   flow += 8   # 건전한 수급
    elif 40 <= mfi14 < 50:    flow += 6   # 중립 근처
    elif 75 < mfi14 <= 85:    flow += 5   # 강한 수급 (약간 과열)
    elif mfi14 < 40:          flow += 3   # 수급 약함
    else:                     flow += 2   # MFI > 85 극단 과열

    # (b) Volume Ratio (8점) — 거래량 터질수록 좋음 (O'Neil: 150% 이상)
    if   vol_ratio >= 2.0:    flow += 8
    elif vol_ratio >= 1.5:    flow += 7
    elif vol_ratio >= 1.2:    flow += 5
    elif vol_ratio >= 1.0:    flow += 3
    elif vol_ratio >= 0.8:    flow += 1
    else:                     flow += 0
    flow += vol_quality_bonus  # RVOL 보너스 (+0~+4)

    # (c) VWAP 위치 (4점)
    if vwap_ab:               flow += 4
    else:                     flow += 0

    # ══════════════════════════════════════════════════════════════
    # 5. 리스크 조정 (가감)
    # ══════════════════════════════════════════════════════════════
    risk = 0

    risk += fake_bk_penalty        # fake breakout -5
    risk += inst_bonus // 2        # 기관수급 ±2
    # 골든/데드크로스
    if golden:    risk += 5
    if death:     risk -= 7

    # Stage 4 = 강한 패널티
    if stage == 4: risk -= 15

    # PTJ 200MA 방어
    if ptj["defense_required"]: risk -= 8

    # 52주 위치 보정 (너무 많이 올랐으면 소폭 감점, 저점 근처면 가점)
    if pos52 < 0.10:   risk += 3    # 52주 극저점 근처 역발상
    elif pos52 > 0.95: risk -= 2    # 52주 극고점 추격 소폭 주의

    # ══════════════════════════════════════════════════════════════
    # 최종 합산
    # ══════════════════════════════════════════════════════════════
    raw   = trend + momentum + pattern + flow + risk
    total = clamp(raw, 0, 100)

    # ── 이유 & 경고 생성 ──────────────────────────────────────────
    reasons  = []
    warnings = []

    # Minervini TT
    if tt["is_stage2"]:
        reasons.append(f"🌟 [Minervini] Trend Template {tt['passed']}/8 통과 → Stage 2 강한 상승 구조 확인")
    elif tt["is_qualified"]:
        reasons.append(f"✅ [Minervini] Trend Template {tt['passed']}/8 — 추세 형성 중")
    else:
        warnings.append(f"❌ [Minervini] Trend Template {tt['passed']}/8 — Stage 2 조건 미달")

    # VCP
    if vcp["breakout_ready"]:
        reasons.append(f"🚀 [Minervini] VCP 돌파 준비! 수축율 {vcp['tightness']*100:.0f}%, 거래량 건조 확인")
    elif vcp["detected"]:
        reasons.append(f"⚡ [Minervini] VCP 패턴 감지 ({vcp['contractions']}회 수축) — 돌파 대기")

    # Weinstein Stage
    reasons.append(f"📊 [Weinstein] {ws['stage_name']}")
    if stage == 4:
        warnings.append("🚨 [Weinstein] Stage 4 하락 추세 — 매수 금지")
    elif stage == 3:
        warnings.append("⚠️ [Weinstein] Stage 3 분배 구간 — 수익 실현 구간")

    # O'Neil RS
    reasons.append(f"📈 [O'Neil] RS Score {rs['rs_score']} — {rs['momentum_rank']}")
    if rs["rs_score"] < 70:
        warnings.append("⚠️ [O'Neil] RS < 70 — 시장 대비 약세, 매수 적합성 낮음")

    # Cup-with-Handle
    if cph["breakout"]:
        reasons.append(f"🏆 [O'Neil] Cup-with-Handle 돌파! 거래량 {cph['vol_expansion']:.1f}배")
    elif cph["detected"]:
        reasons.append(f"🥤 [O'Neil] Cup-with-Handle 형성 중 (컵 깊이 {cph['cup_depth_pct']:.0f}%)")

    # PTJ
    reasons.append(f"🛡️ [PTJ] {ptj['status']}")
    if ptj["defense_required"]:
        warnings.append("🚨 [PTJ] 200MA 하회 → 즉시 관망 원칙 적용")

    # Livermore
    if pivs["near_breakout"]:
        reasons.append(f"🎯 [Livermore] 저항선 ${pivs['resistance']:.2f} 돌파 임박!")
    reasons.append(f"📐 [Livermore] 지지 ${pivs['support']:.2f} / 저항 ${pivs['resistance']:.2f} (R:R {pivs['rr_ratio']:.1f}:1)")

    # MACD
    if macd_cu:
        reasons.append("🚀 MACD 골든크로스 발생 — 강력 진입 트리거")
    elif macd_cd:
        warnings.append("⛔ MACD 데드크로스 — 하락 모멘텀 전환 경고")
    elif macd_h > 0:
        reasons.append("✅ MACD 히스토그램 양수 — 상승 모멘텀 유지")

    # 골든/데드
    if golden: reasons.append("🌟 MA50/200 골든크로스 — 중장기 상승 전환")
    if death:  warnings.append("💀 MA50/200 데드크로스 — 중장기 하락 구조 전환")

    # BB Squeeze
    if bb_squeeze:
        reasons.append("⚡ 볼린저 밴드 수축 — 큰 움직임 준비 (VCP 병행 시 최강)")

    # RSI 상태
    if rsi14 < 30:
        reasons.append(f"🔄 RSI {rsi14:.1f} 과매도 — 역발상 반등 셋업")
    elif 40 <= rsi14 <= 70:
        reasons.append(f"✅ RSI {rsi14:.1f} 건전한 모멘텀 구간 (추세 추종 유리)")
    elif rsi14 > 80:
        warnings.append(f"🔥 RSI {rsi14:.1f} 극단 과매수 — 단기 조정 가능성")

    # 52주 위치
    if pos52 < 0.15:
        reasons.append(f"📍 52주 저점 근처 ({pos52*100:.0f}%) — 역발상 바닥 기회")
    elif pos52 > 0.90:
        reasons.append(f"📍 52주 고점 근처 ({pos52*100:.0f}%) — 강한 상승 추세 (신고가 돌파 확인 필요)")

    return MasterScorecard(
        total=total, grade=grade(total), signal_strength=signal_strength(total),
        trend_score=clamp(trend, 0, 35),
        momentum_score=clamp(momentum, 0, 25),
        pattern_score=clamp(pattern, 0, 20),
        flow_score=clamp(flow, 0, 20),
        risk_adj=risk,
        trend_template=tt, vcp=vcp, weinstein=ws,
        rs=rs, cup_handle=cph, ptj=ptj, pivots=pivs,
        last=last, ma20=ma20, ma50=ma50, ma200=ma200,
        rsi_val=rsi14, mfi_val=mfi14, macd_hist=macd_h,
        atr_pct=atr_pct, vol_ratio=vol_ratio,
        bb_position=bb_pos, adx_val=adx14,
        stoch_k=sk_v, stoch_d=sd_v, cci_val=cci_v,
        williams_r=wr_v,
        vwap_val=vwap_v, vwap_above=vwap_ab,
        week52_h=h52, week52_l=l52, week52_pos=pos52,
        bb_squeeze=bb_squeeze, golden_cross=golden, death_cross=death,
        rvol=_rvol["rvol"],
        rvol_category=_rvol["rvol_category"],
        vol_confirm=_bq["vol_confirmation"],
        fake_breakout=_bq["fake_breakout"],
        close_above_pivot=_bq["close_above_breakout"],
        inst_signal=_inst["inst_signal"],
        inst_score=_inst["inst_score"],
        vcp_score=vcp.get("vcp_score", 0),
        is_perfect_vcp=vcp.get("is_perfect_vcp", False),
        reasons=reasons, warnings=warnings,
    )
