"""
backtest.py v4 — StockEdge Pro 균형 최적화

v2 기반 유지 + 선별적 개선:
  - 진입 조건: v2 그대로 (S:73점+, A:62점+추가조건)
  - 데이터 기간: 350일 버퍼 (v3의 400일 → 복구)
  - 부분익절: 22% → 18% (적당히 낮춤)
  - 트레일링: 12% 고정 → 등급별 SS:8% / S:10% / A:12%
  - 브레이크이븐: +10% 달성 시 손절 → 진입가+1% 자동
  - 보유기간: 25일 → 40일 (강세장) / 15일 (중립장)
  - 중립장: 조건 강화 X, 보유기간만 단축
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta

from core.indicators import sma, rsi, atr, adx
from core.scoring import master_score

# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
TICKERS = [
    "NVDA","AMD","AVGO","QCOM","ARM","MU","AMAT","LRCX","KLAC",
    "AAPL","MSFT","GOOG","AMZN","META","NFLX","CRM","ADBE",
    "PLTR","CRWD","AXON","NET","DDOG","SNOW","UBER","ABNB",
    "TSLA","SMR","LLY","V","MA","JPM","GS",
]

BT_START        = "2022-01-01"
BT_END          = "2024-12-31"
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE  = 0.015
MAX_POSITIONS   = 6

# ── 진입 필터 (v2 그대로) ──────────────────────────────────────
MIN_SCORE_S  = 73
MIN_SCORE_A  = 62
VOL_BREAKOUT = 1.4
MIN_RS       = 65
MIN_ADX      = 18
MARKET_FILTER = True

# ── 손절 (v2 그대로) ───────────────────────────────────────────
STOP_MAX_PCT = 0.08
STOP_ATR_MUL = 1.2

# ── 트레일링: 등급별 차등 (핵심 개선) ─────────────────────────
TRAIL_PCT = {
    "SSS": 0.08, "SS": 0.08,
    "S":   0.10,
    "A":   0.12,
    "default": 0.10,
}

# ── 부분 익절 (22% → 18%) ─────────────────────────────────────
TP_PARTIAL_PCT  = 0.18   # 18%에서 절반 청산
TP_PARTIAL_TRAIL = 0.07  # 부분 익절 후 트레일링 7%

# ── 브레이크이븐 (신규) ────────────────────────────────────────
BREAKEVEN_PCT   = 0.10   # 10% 수익 달성 시
BREAKEVEN_RAISE = 0.01   # 손절 → 진입가+1%로 상향

# ── 보유기간 (국면별 차등) ─────────────────────────────────────
MAX_HOLD_BULL    = 40    # 강세장: 40일 (v2 25 → v3 60 → v4 40)
MAX_HOLD_NEUTRAL = 15    # 중립장: 15일 (빠른 청산)
MAX_HOLD_BEAR    = 10


# ══════════════════════════════════════════════════════════════════
# 데이터 (350일 버퍼 — v2 수준 복구)
# ══════════════════════════════════════════════════════════════════
def fetch(ticker: str, start_buf: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start_buf, end=end,
                         interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df = df[df["Close"] > 0]
        df["Volume"] = df["Volume"].clip(lower=0)
        return df
    except: return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# 시장 국면 (bull_strong / bull / neutral / bear)
# ══════════════════════════════════════════════════════════════════
class MarketRegime:
    def __init__(self, spy_df: pd.DataFrame):
        self.regimes: Dict[str, str] = {}
        close = spy_df["Close"]
        ma20  = sma(close, 20)
        ma50  = sma(close, 50)
        ma200 = sma(close, 200)
        rsi14 = rsi(close, 14)
        ret_std = close.pct_change().rolling(20).std() * math.sqrt(252)

        for i in range(200, len(spy_df)):
            date = str(spy_df.index[i].date())
            p    = float(close.iloc[i])
            m20  = float(ma20.iloc[i])
            m50  = float(ma50.iloc[i])
            m200 = float(ma200.iloc[i])
            r    = float(rsi14.iloc[i])
            vol  = float(ret_std.iloc[i]) if not math.isnan(float(ret_std.iloc[i])) else 0.25

            if p > m20 > m50 > m200 and r > 50 and vol < 0.25:
                self.regimes[date] = "bull_strong"
            elif p > m50 > m200 and r > 45:
                self.regimes[date] = "bull"
            elif p < m200 or (p < m50 and r < 38):
                self.regimes[date] = "bear"
            else:
                self.regimes[date] = "neutral"

    def get(self, date: str) -> str:
        return self.regimes.get(date, "neutral")


# ══════════════════════════════════════════════════════════════════
# 진입 필터 (v2 그대로)
# ══════════════════════════════════════════════════════════════════
def check_entry(sc, regime: str) -> Tuple[bool, str]:
    score = sc.total
    rs    = sc.rs["rs_score"]
    adx_v = sc.adx_val
    vol_r = sc.vol_ratio
    stage = sc.weinstein["stage"]
    tt_ok = sc.trend_template["is_stage2"]
    ptj   = not sc.ptj["defense_required"]

    # 필수 차단
    if not ptj:          return False, "PTJ:200MA하회"
    if stage == 4:       return False, "Stage4"
    if stage == 3:       return False, "Stage3"
    if regime == "bear" and MARKET_FILTER:
        return False, "약세장"

    # S등급 이상 (v2 그대로)
    if score >= MIN_SCORE_S:
        if rs < MIN_RS:    return False, f"RS부족:{rs}"
        if adx_v < MIN_ADX: return False, f"ADX부족:{adx_v:.0f}"
        return True, f"S+({score}점)"

    # A등급 (v2 그대로)
    if score >= MIN_SCORE_A:
        if not tt_ok:      return False, "TT미달"
        if rs < 75:        return False, f"RS부족:{rs}"
        if vol_r < VOL_BREAKOUT: return False, f"거래량부족:{vol_r:.2f}"
        if stage != 2:     return False, "Stage2아님"
        if regime == "neutral" and not tt_ok:
            return False, "중립장+TT미달"
        return True, f"A+조건충족({score}점)"

    return False, f"점수부족:{score}"


# ══════════════════════════════════════════════════════════════════
# 트레이드 시뮬레이션 (v4 핵심 개선)
# ══════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    ticker:        str
    entry_date:    str
    exit_date:     str
    entry_price:   float
    exit_price:    float
    pnl:           float
    pnl_pct:       float
    exit_reason:   str
    score:         int
    grade:         str
    regime:        str
    hold_days:     int
    entry_reason:  str
    partial_taken: bool
    max_gain_pct:  float


def simulate_trade(df: pd.DataFrame, idx: int,
                   entry_px: float, sc,
                   regime: str, entry_reason: str) -> Optional[Trade]:
    n     = len(df)
    grade = sc.grade

    # 손절가 (v2 그대로)
    atr14    = float(atr(df.iloc[max(0, idx-20):idx+1], 14).iloc[-1])
    atr_stop = entry_px - STOP_ATR_MUL * atr14
    pct_stop = entry_px * (1 - STOP_MAX_PCT)
    stop_px  = max(atr_stop, pct_stop)
    stop_px  = min(stop_px, entry_px * 0.999)
    if entry_px - stop_px <= 0:
        return None

    # 국면별 보유기간
    if regime in ("bull_strong", "bull"):
        max_hold = MAX_HOLD_BULL
    elif regime == "neutral":
        max_hold = MAX_HOLD_NEUTRAL
    else:
        max_hold = MAX_HOLD_BEAR

    # 등급별 트레일링
    trail_pct = TRAIL_PCT.get(grade, TRAIL_PCT["default"])

    # 상태 변수
    partial_taken   = False
    partial_exit_px = 0.0
    trailing_stop   = stop_px
    high_water      = entry_px
    breakeven_done  = False
    max_gain        = 0.0
    exit_price      = None
    exit_reason     = "timeout"
    exit_dt         = str(df.index[min(idx + max_hold, n-1)].date())
    hold_days       = 0

    for j in range(1, max_hold + 1):
        bar_idx = idx + j
        if bar_idx >= n: break

        bar   = df.iloc[bar_idx]
        high  = float(bar["High"])
        low   = float(bar["Low"])
        close = float(bar["Close"])
        date  = str(df.index[bar_idx].date())
        hold_days = j
        max_gain  = max(max_gain, (high - entry_px) / entry_px)

        # 고점 갱신 → 트레일링 상향
        if high > high_water:
            high_water = high
            new_trail  = high_water * (1 - trail_pct)
            if new_trail > trailing_stop:
                trailing_stop = new_trail

        # 브레이크이븐: +10% 달성 → 손절 진입가+1%
        if not breakeven_done and high >= entry_px * (1 + BREAKEVEN_PCT):
            breakeven_done = True
            be_stop = entry_px * (1 + BREAKEVEN_RAISE)
            stop_px = max(stop_px, be_stop)
            trailing_stop = max(trailing_stop, stop_px)

        # 손절/트레일링 확인
        eff_stop = max(stop_px, trailing_stop)
        if low <= eff_stop:
            exit_price  = eff_stop
            exit_reason = ("trail_stop"
                           if (trailing_stop > stop_px * 1.005 or breakeven_done)
                           else "stop")
            exit_dt = date
            break

        # 부분 익절: 18% 달성 시 50% 청산
        if not partial_taken and high >= entry_px * (1 + TP_PARTIAL_PCT):
            partial_taken   = True
            partial_exit_px = entry_px * (1 + TP_PARTIAL_PCT)
            # 부분 익절 후 트레일링 7%로 타이트
            trail_pct = TP_PARTIAL_TRAIL
            stop_px   = max(stop_px, entry_px * (1 + BREAKEVEN_RAISE))
            tighter   = high_water * (1 - trail_pct)
            trailing_stop = max(trailing_stop, tighter)

        # 최대 보유일
        if j == max_hold:
            exit_price  = close
            exit_reason = "timeout"
            exit_dt     = date
            break

    if exit_price is None:
        exit_price = float(df.iloc[min(idx + max_hold, n-1)]["Close"])
        hold_days  = max_hold

    # PnL
    avg_exit = (partial_exit_px * 0.5 + exit_price * 0.5) if partial_taken else exit_price
    pnl_pct  = (avg_exit / entry_px - 1) * 100
    risk_sz  = entry_px - min(atr_stop, pct_stop)
    pnl      = ((avg_exit - entry_px)
                * (INITIAL_CAPITAL * RISK_PER_TRADE / risk_sz)
                if risk_sz > 0 else 0)

    return Trade(
        ticker="?",
        entry_date=(str(df.index[idx+1].date()) if idx+1 < n
                    else str(df.index[idx].date())),
        exit_date=exit_dt,
        entry_price=round(entry_px, 2),
        exit_price=round(avg_exit, 2),
        pnl=round(pnl, 2),
        pnl_pct=round(pnl_pct, 2),
        exit_reason=exit_reason,
        score=sc.total,
        grade=sc.grade,
        regime=regime,
        hold_days=hold_days,
        entry_reason=entry_reason,
        partial_taken=partial_taken,
        max_gain_pct=round(max_gain * 100, 2),
    )


# ══════════════════════════════════════════════════════════════════
# 메인 백테스트
# ══════════════════════════════════════════════════════════════════
def run_backtest(tickers: List[str],
                 regime_cache: MarketRegime) -> List[Trade]:
    all_trades: List[Trade] = []

    for ticker in tickers:
        print(f"  [{ticker}]", end=" ", flush=True)

        # 350일 버퍼 (v2 수준)
        start_buf = (pd.to_datetime(BT_START)
                     - timedelta(days=350)).strftime("%Y-%m-%d")
        df = fetch(ticker, start_buf, BT_END)

        if df is None or df.empty or len(df) < 250:
            print("❌ 스킵"); continue

        bt_idx_list = [i for i, d in enumerate(df.index)
                       if str(d.date()) >= BT_START]
        if not bt_idx_list:
            print("❌ 기간없음"); continue

        consecutive_losses = 0
        cooldown_until     = ""
        last_entry_date    = ""
        trade_count        = 0

        for i in bt_idx_list[200:]:
            date = str(df.index[i].date())
            if date <= cooldown_until:    continue
            if date == last_entry_date:   continue

            df_slice = df.iloc[i-200:i+1].copy()
            try:
                sc = master_score(df_slice)
            except:
                continue

            regime = regime_cache.get(date)
            ok, reason = check_entry(sc, regime)
            if not ok: continue
            if i + 1 >= len(df): continue

            entry_px = float(df.iloc[i+1]["Open"])
            if entry_px <= 0: continue

            trade = simulate_trade(df, i, entry_px, sc, regime, reason)
            if trade is None: continue

            trade.ticker = ticker
            all_trades.append(trade)
            trade_count    += 1
            last_entry_date = date

            if trade.exit_reason == "stop":
                consecutive_losses += 1
                if consecutive_losses >= 3:
                    cooldown_until = str(
                        (pd.to_datetime(trade.exit_date)
                         + timedelta(days=7)).date()
                    )
                    consecutive_losses = 0
            else:
                consecutive_losses = 0

        print(f"✅ {trade_count}건")

    return all_trades


# ══════════════════════════════════════════════════════════════════
# 분석
# ══════════════════════════════════════════════════════════════════
def analyze(trades: List[Trade]) -> dict:
    if not trades: return {}

    df = pd.DataFrame([t.__dict__ for t in trades])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["month"]      = df["entry_date"].dt.to_period("M").astype(str)
    df["quarter"]    = df["entry_date"].dt.to_period("Q").astype(str)

    def stats(sub):
        if len(sub) == 0: return {}
        wins   = sub[sub["pnl_pct"] > 0]
        loses  = sub[sub["pnl_pct"] <= 0]
        trails = sub[sub["exit_reason"] == "trail_stop"]
        pf     = (wins["pnl"].sum() / abs(loses["pnl"].sum())
                  if len(loses) > 0 and loses["pnl"].sum() != 0
                  else float("inf"))
        return {
            "count":          len(sub),
            "win_rate":       round(len(wins)/len(sub)*100, 1),
            "avg_pnl_pct":    round(sub["pnl_pct"].mean(), 2),
            "median_pnl_pct": round(sub["pnl_pct"].median(), 2),
            "avg_win":        round(wins["pnl_pct"].mean(), 2) if len(wins) else 0,
            "avg_loss":       round(loses["pnl_pct"].mean(), 2) if len(loses) else 0,
            "max_win":        round(sub["pnl_pct"].max(), 2),
            "max_loss":       round(sub["pnl_pct"].min(), 2),
            "avg_max_gain":   round(sub["max_gain_pct"].mean(), 2),
            "total_pnl":      round(sub["pnl"].sum(), 2),
            "avg_hold":       round(sub["hold_days"].mean(), 1),
            "profit_factor":  round(pf, 2),
            "partial_pct":    round(sub["partial_taken"].mean()*100, 1),
            "trail_pct":      round(len(trails)/len(sub)*100, 1),
        }

    # MDD
    df_s    = df.sort_values("entry_date")
    cum_pnl = df_s["pnl"].cumsum()
    max_dd  = round(float((cum_pnl - cum_pnl.cummax()).min()), 2)

    # 월별
    monthly = {}
    for m in sorted(df["month"].unique()):
        sub = df[df["month"] == m]
        monthly[m] = {
            "count":     len(sub),
            "win_rate":  round((sub["pnl_pct"]>0).mean()*100, 1),
            "total_pnl": round(sub["pnl"].sum(), 2),
            "avg_pnl":   round(sub["pnl_pct"].mean(), 2),
        }

    return {
        "all":       stats(df),
        "by_grade":  {g: stats(df[df["grade"]==g])
                      for g in ["SSS","SS","S","A","B"]},
        "by_exit":   {r: stats(df[df["exit_reason"]==r])
                      for r in ["stop","trail_stop","timeout"]},
        "by_regime": {r: stats(df[df["regime"]==r])
                      for r in ["bull_strong","bull","neutral","bear"]},
        "by_score":  {
            "A(62-72)": stats(df[(df["score"]>=62)&(df["score"]<73)]),
            "S(73-81)": stats(df[(df["score"]>=73)&(df["score"]<82)]),
            "SS+(82+)": stats(df[df["score"]>=82]),
        },
        "monthly":   monthly,
        "max_dd":    max_dd,
        "df":        df,
    }


def print_report(result: dict):
    if not result: return
    a   = result["all"]
    sep = "═" * 65

    print(f"\n{sep}")
    print("  StockEdge Pro v4 — 균형 최적화 백테스트 리포트")
    print(f"  기간: {BT_START} ~ {BT_END}")
    print(f"  진입: v2 그대로 (S:{MIN_SCORE_S}점+ / A:{MIN_SCORE_A}점+추가조건)")
    print(f"  트레일링: SS/SS:{TRAIL_PCT['SS']*100:.0f}% / "
          f"S:{TRAIL_PCT['S']*100:.0f}% / A:{TRAIL_PCT['A']*100:.0f}%")
    print(f"  부분익절: +{TP_PARTIAL_PCT*100:.0f}% | "
          f"브레이크이븐: +{BREAKEVEN_PCT*100:.0f}%→+{BREAKEVEN_RAISE*100:.0f}%")
    print(f"  보유: 강세 최대{MAX_HOLD_BULL}일 | 중립 최대{MAX_HOLD_NEUTRAL}일")
    print(sep)

    print(f"\n【 전체 성과 】")
    print(f"  총 트레이드:      {a['count']}건")
    print(f"  승률:             {a['win_rate']}%      "
          f"← v2: 46.1% / v3: 45.1%")
    print(f"  평균 수익률:      {a['avg_pnl_pct']:+.2f}%    "
          f"← v2: +3.41% / v3: +1.32%")
    print(f"  중앙값 수익률:    {a['median_pnl_pct']:+.2f}%")
    print(f"  평균 수익:        {a['avg_win']:+.2f}%")
    print(f"  평균 손실:        {a['avg_loss']:+.2f}%")
    print(f"  보유 중 최대수익:  {a['avg_max_gain']:+.2f}% (평균)")
    print(f"  최대 수익:        {a['max_win']:+.2f}%")
    print(f"  최대 손실:        {a['max_loss']:+.2f}%")
    print(f"  Profit Factor:    {a['profit_factor']}      "
          f"← v2: 1.85 / v3: 1.70")
    print(f"  총 PnL:           ${a['total_pnl']:,.0f}")
    print(f"  최대 낙폭(MDD):   ${result['max_dd']:,.0f}")
    print(f"  평균 보유일:      {a['avg_hold']}일    ← v2: 14.5일")
    print(f"  부분익절 비율:    {a['partial_pct']}%    ← v2: 9.1%")
    print(f"  트레일링 청산:    {a['trail_pct']}%    ← v2: 1.8%")

    print(f"\n【 등급별 성과 】")
    print(f"  {'등급':<6}{'건수':>5}{'승률':>8}{'평균수익':>9}"
          f"{'중앙값':>8}{'PF':>7}{'부분익절':>9}{'트레일':>8}")
    print(f"  {'-'*60}")
    for g in ["SSS","SS","S","A"]:
        d = result["by_grade"].get(g, {})
        if d and d.get("count", 0) > 0:
            print(f"  {g:<6}{d['count']:>5}{d['win_rate']:>7.1f}%"
                  f"{d['avg_pnl_pct']:>+8.2f}%{d['median_pnl_pct']:>+7.2f}%"
                  f"{d['profit_factor']:>6.2f}"
                  f"{d['partial_pct']:>7.1f}%{d['trail_pct']:>6.1f}%")

    print(f"\n【 시장 국면별 성과 】")
    labels = {
        "bull_strong": "강한강세",
        "bull":        "일반강세",
        "neutral":     "중립장  ",
        "bear":        "약세장  ",
    }
    for r, label in labels.items():
        d = result["by_regime"].get(r, {})
        if d and d.get("count", 0) > 0:
            print(f"  {label}  건수:{d['count']:>4}  승률:{d['win_rate']:>5.1f}%"
                  f"  평균:{d['avg_pnl_pct']:>+6.2f}%  PF:{d['profit_factor']:>5.2f}")

    print(f"\n【 점수 구간별 성과 】")
    for label, d in result["by_score"].items():
        if d and d.get("count", 0) > 0:
            print(f"  {label:<12} 건수:{d['count']:>4}  승률:{d['win_rate']:>5.1f}%"
                  f"  평균:{d['avg_pnl_pct']:>+6.2f}%  PF:{d['profit_factor']:>5.2f}")

    print(f"\n【 청산 사유별 】")
    for r, label in {"stop":"손절","trail_stop":"트레일스톱","timeout":"기간종료"}.items():
        d = result["by_exit"].get(r, {})
        if d and d.get("count", 0) > 0:
            print(f"  {label:<8} {d['count']:>4}건  "
                  f"승률:{d['win_rate']:>5.1f}%  평균:{d['avg_pnl_pct']:>+6.2f}%")

    print(f"\n【 분기별 성과 】")
    df_all = result["df"]
    for q in sorted(df_all["quarter"].unique()):
        sub  = df_all[df_all["quarter"] == q]
        wr   = round((sub["pnl_pct"]>0).mean()*100, 1)
        avg  = round(sub["pnl_pct"].mean(), 2)
        pnl  = round(sub["pnl"].sum(), 2)
        sign = "+" if pnl >= 0 else ""
        print(f"  {q}  건수:{len(sub):>4}  승률:{wr:>5.1f}%  "
              f"평균:{avg:>+5.2f}%  PnL:{sign}${abs(pnl):>9,.0f}")

    print(f"\n{sep}")


def save_outputs(result: dict):
    if "df" not in result: return
    df = result["df"].copy()
    df.to_csv("backtest_trades_v4.csv", index=False, encoding="utf-8-sig")
    print("📄 backtest_trades_v4.csv 저장")

    df2       = df.sort_values("entry_date").copy()
    df2["cum_pnl"] = df2["pnl"].cumsum()
    final_pnl = float(df2["cum_pnl"].iloc[-1])
    line_c    = "#22c55e" if final_pnl > 0 else "#ef4444"
    fill_c    = ("rgba(34,197,94,0.1)" if final_pnl > 0
                 else "rgba(239,68,68,0.1)")

    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=[
            "누적 PnL ($)", "트레이드별 수익률 (%)",
            "등급별 승률 (%)", "국면별 평균 수익률 (%)", "월별 PnL ($)",
        ],
        row_heights=[0.28, 0.18, 0.18, 0.18, 0.18],
        vertical_spacing=0.07,
    )

    # 누적 PnL
    fig.add_trace(go.Scatter(
        x=df2["entry_date"], y=df2["cum_pnl"],
        fill="tozeroy", fillcolor=fill_c,
        line=dict(color=line_c, width=2), name="누적PnL",
    ), row=1, col=1)

    # 개별 수익률
    bar_c = ["#22c55e" if p > 0 else "#ef4444" for p in df2["pnl_pct"]]
    fig.add_trace(go.Bar(
        x=df2["entry_date"], y=df2["pnl_pct"],
        marker_color=bar_c,
    ), row=2, col=1)

    # 등급별 승률
    gstats = []
    for g in ["SSS","SS","S","A"]:
        sub = df2[df2["grade"] == g]
        if len(sub) > 5:
            wr  = round((sub["pnl_pct"]>0).mean()*100, 1)
            lp  = sub[sub["pnl_pct"]<=0]["pnl"].sum()
            wp  = sub[sub["pnl_pct"]>0]["pnl"].sum()
            pf  = round(wp/abs(lp), 2) if lp != 0 else 0
            gstats.append((f"{g}\n(n={len(sub)},PF={pf})", wr))
    if gstats:
        glbls, gwr = zip(*gstats)
        gc = ["#22c55e" if w>=55 else "#f59e0b" if w>=45 else "#ef4444"
              for w in gwr]
        fig.add_trace(go.Bar(
            x=list(glbls), y=list(gwr), marker_color=gc,
            text=[f"{w:.1f}%" for w in gwr], textposition="outside",
        ), row=3, col=1)
        fig.add_hline(y=50, line=dict(color="white", dash="dot", width=1),
                      row=3, col=1)

    # 국면별 평균 수익률
    rl_map = {"bull_strong":"강한강세","bull":"일반강세","neutral":"중립장"}
    rlbls, ravgs, rcs = [], [], []
    for r, lbl in rl_map.items():
        sub = df2[df2["regime"] == r]
        if len(sub) > 0:
            avg = round(sub["pnl_pct"].mean(), 2)
            rlbls.append(f"{lbl}\n(n={len(sub)})")
            ravgs.append(avg)
            rcs.append("#22c55e" if avg > 0 else "#ef4444")
    fig.add_trace(go.Bar(
        x=rlbls, y=ravgs, marker_color=rcs,
        text=[f"{v:+.2f}%" for v in ravgs], textposition="outside",
    ), row=4, col=1)

    # 월별 PnL
    monthly = result.get("monthly", {})
    if monthly:
        months = sorted(monthly.keys())
        mpnl   = [monthly[m]["total_pnl"] for m in months]
        mc     = ["#22c55e" if v > 0 else "#ef4444" for v in mpnl]
        fig.add_trace(go.Bar(x=months, y=mpnl, marker_color=mc),
                      row=5, col=1)

    fig.update_layout(
        height=1200,
        paper_bgcolor="#0c1220", plot_bgcolor="#0c1220",
        font=dict(color="#d8e8ff", size=11),
        showlegend=False,
        title=dict(
            text=(f"StockEdge Pro v4  |  {BT_START}~{BT_END}  |  "
                  f"{len(df2)}건  |  "
                  f"PF {result['all']['profit_factor']}  |  "
                  f"승률 {result['all']['win_rate']}%"),
            font=dict(size=14, color="#06b6d4"),
        ),
    )
    for i in range(1, 6):
        fig.update_xaxes(showgrid=True, gridcolor="#1a2840", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#1a2840", row=i, col=1)

    fig.write_html("backtest_chart_v4.html")
    print("📊 backtest_chart_v4.html 저장")


# ══════════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  StockEdge Pro v4 — 균형 최적화 백테스트")
    print(f"  종목: {len(TICKERS)}개  |  {BT_START}~{BT_END}")
    print(f"  진입: v2 그대로 | 트레일링 등급별 차등")
    print(f"  부분익절: +{TP_PARTIAL_PCT*100:.0f}% | "
          f"브레이크이븐: +{BREAKEVEN_PCT*100:.0f}%")
    print(f"  보유: 강세 {MAX_HOLD_BULL}일 | 중립 {MAX_HOLD_NEUTRAL}일")
    print("=" * 65)

    print("\n📊 SPY 시장 국면 분석 중...")
    spy_start = (pd.to_datetime(BT_START)
                 - timedelta(days=350)).strftime("%Y-%m-%d")
    spy_df = fetch("SPY", spy_start, BT_END)
    if spy_df.empty:
        print("❌ SPY 데이터 실패.")
        sys.exit(1)

    regime_cache = MarketRegime(spy_df)
    counts: Dict[str, int] = {}
    for v in regime_cache.regimes.values():
        counts[v] = counts.get(v, 0) + 1
    for k in sorted(counts, key=lambda x: -counts[x]):
        print(f"  {k}: {counts[k]}일")

    print(f"\n🔄 종목별 백테스트 실행 중...")
    trades = run_backtest(TICKERS, regime_cache)
    if not trades:
        print("\n❌ 트레이드 없음.")
        sys.exit(1)

    print(f"\n✅ 총 {len(trades)}건 트레이드 완료")
    result = analyze(trades)
    print_report(result)
    save_outputs(result)

    print("\n🎯 완료!")
    print("   backtest_trades_v4.csv")
    print("   backtest_chart_v4.html")
