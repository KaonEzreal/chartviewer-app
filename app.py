"""
StockEdge Pro — 미국주식 극한 수익 차트 분석 엔진

통합 기법:
• Mark Minervini — SEPA Trend Template + VCP 패턴
• Stan Weinstein — Stage 1-4 분석
• William O'Neil  — CANSLIM + Cup-with-Handle + RS 점수
• Paul Tudor Jones — 200MA 방어 + 5:1 R:R
• Jesse Livermore  — 피벗 포인트 / 저항의 최소선
"""

import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.config import TF_OPTIONS
from core.data import fetch_spy, fetch_ticker_info, fetch_vix
from core.indicators import (
    bbands, ema, macd, rsi, sma, stochastic, vwap,
    minervini_trend_template, detect_vcp, weinstein_stage,
    oneil_rs_score, detect_cup_handle, ptj_defense_rule,
    livermore_pivots,
)
from core.signal import build_signal

# ─── 페이지 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="StockEdge Pro — 미국주식 극한 분석",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --bg:      #070b12;
  --card:    #0c1220;
  --card2:   #0f1828;
  --border:  #1a2840;
  --border2: #253555;
  --muted:   #5a7299;
  --text:    #d8e8ff;
  --green:   #22c55e;
  --red:     #ef4444;
  --yellow:  #f59e0b;
  --blue:    #3b82f6;
  --cyan:    #06b6d4;
  --purple:  #a855f7;
  --orange:  #f97316;
}
*, body, html { font-family: 'Inter', sans-serif !important; }
html, body, [class*="css"], .stApp { background: var(--bg) !important; color: var(--text) !important; }
.block-container { padding: 1rem 1.5rem 3rem !important; max-width: 1500px !important; }

/* ── 헤더 ── */
.app-header {
  background: linear-gradient(135deg, #0c1220 0%, #111e35 100%);
  border: 1px solid var(--border); border-radius: 18px;
  padding: 16px 24px; margin-bottom: 18px;
  display: flex; align-items: center; justify-content: space-between;
}
.app-title { font-size: 22px; font-weight: 900; color: var(--cyan); letter-spacing: -0.04em; }
.app-sub   { font-size: 11px; color: var(--muted); margin-top: 2px; }
.app-legend { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
.legend-item { font-size:11px; color: var(--muted); }
.legend-item span { font-weight:700; }

/* ── 카드 ── */
.card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 16px; padding: 16px; margin-bottom: 12px;
}
.card-sm { background: var(--card2); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }
.card-highlight {
  background: linear-gradient(135deg, rgba(6,182,212,0.08), rgba(59,130,246,0.05));
  border: 1px solid rgba(6,182,212,0.3); border-radius: 16px; padding: 18px;
}

/* ── 점수 ── */
.score-big { font-size: 96px; font-weight: 900; line-height: 1; letter-spacing: -0.06em; }
.score-grade { font-size: 24px; font-weight: 700; letter-spacing: 3px; }

/* ── 액션 뱃지 ── */
.action-badge {
  display: inline-flex; align-items:center; gap:6px;
  padding: 10px 24px; border-radius: 999px;
  font-size: 16px; font-weight: 800; letter-spacing: 0.02em;
}
.badge-cyan   { background:rgba(6,182,212,0.12); color:#06b6d4; border:1.5px solid rgba(6,182,212,0.4); }
.badge-green  { background:rgba(34,197,94,0.12); color:#22c55e; border:1.5px solid rgba(34,197,94,0.4); }
.badge-yellow { background:rgba(245,158,11,0.12); color:#f59e0b; border:1.5px solid rgba(245,158,11,0.4); }
.badge-red    { background:rgba(239,68,68,0.12); color:#ef4444; border:1.5px solid rgba(239,68,68,0.4); }
.badge-gray   { background:rgba(90,114,153,0.12); color:#5a7299; border:1.5px solid rgba(90,114,153,0.4); }

/* ── KV 행 ── */
.kv { display:flex; justify-content:space-between; align-items:center;
      padding: 5px 0; border-bottom: 1px solid rgba(26,40,64,0.6); }
.kv:last-child { border-bottom: none; }
.k { color: var(--muted); font-size: 12px; font-weight: 500; }
.v { font-weight: 700; font-size: 13px; }

/* ── 게이지 ── */
.gauge-wrap { margin: 5px 0; }
.gauge-label { display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-bottom:3px; }
.gauge-bar { height:6px; border-radius:99px; background:var(--border); }
.gauge-fill { height:100%; border-radius:99px; }

/* ── 가격 타겟 ── */
.target-box { border-radius:14px; padding:14px; text-align:center; }
.target-up  { background:rgba(34,197,94,0.07);  border:1px solid rgba(34,197,94,0.25); }
.target-dn  { background:rgba(239,68,68,0.07);  border:1px solid rgba(239,68,68,0.25); }
.target-label { font-size:11px; color:var(--muted); margin-bottom:4px; }
.target-price { font-size:26px; font-weight:800; }
.target-pct   { font-size:13px; font-weight:600; margin-top:3px; }

/* ── 이유 / 경고 박스 ── */
.reason-item {
  background: rgba(6,182,212,0.04); border: 1px solid rgba(6,182,212,0.15);
  border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
  font-size: 13px; line-height: 1.5;
}
.warn-item {
  background: rgba(239,68,68,0.05); border: 1px solid rgba(239,68,68,0.2);
  border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
  font-size: 13px; line-height: 1.5;
}

/* ── 레전드 트레이더 뱃지 ── */
.trader-badge {
  display:inline-block; border-radius:6px; padding:2px 8px;
  font-size:10px; font-weight:700; margin-right:4px; vertical-align:middle;
}
.tb-min { background:rgba(6,182,212,0.2);  color:#06b6d4; }
.tb-wei { background:rgba(168,85,247,0.2); color:#a855f7; }
.tb-one { background:rgba(245,158,11,0.2); color:#f59e0b; }
.tb-ptj { background:rgba(34,197,94,0.2);  color:#22c55e; }
.tb-liv { background:rgba(249,115,22,0.2); color:#f97316; }

/* ── 섹션 타이틀 ── */
.sec-title {
  font-size:11px; font-weight:700; color:var(--muted);
  letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;
}

/* ── VIX 배너 ── */
.vix-ok   { background:rgba(34,197,94,0.07);  border:1px solid rgba(34,197,94,0.2);  color:#22c55e; }
.vix-warn { background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.2); color:#f59e0b; }
.vix-bad  { background:rgba(239,68,68,0.07);  border:1px solid rgba(239,68,68,0.2);  color:#ef4444; }
.vix-banner { border-radius:12px; padding:10px 16px; margin-bottom:12px; font-size:13px; font-weight:600; }

/* ── Weinstein Stage ── */
.stage-2 { background:rgba(34,197,94,0.1);   border:1px solid rgba(34,197,94,0.3);   border-radius:10px; padding:10px 14px; }
.stage-1 { background:rgba(245,158,11,0.1);  border:1px solid rgba(245,158,11,0.3);  border-radius:10px; padding:10px 14px; }
.stage-3 { background:rgba(249,115,22,0.1);  border:1px solid rgba(249,115,22,0.3);  border-radius:10px; padding:10px 14px; }
.stage-4 { background:rgba(239,68,68,0.1);   border:1px solid rgba(239,68,68,0.3);   border-radius:10px; padding:10px 14px; }

/* ── Minervini TT 체크리스트 ── */
.tt-check { display:flex; align-items:center; gap:8px; padding:4px 0;
            font-size:12px; border-bottom:1px solid rgba(26,40,64,0.5); }
.tt-check:last-child { border:none; }
.tt-pass { color: var(--green); font-weight:700; }
.tt-fail { color: var(--red);   font-weight:700; }

/* ── Streamlit 오버라이드 ── */
.stTextInput>div>div>input {
  background:var(--card2)!important; border:1px solid var(--border)!important;
  border-radius:10px!important; color:var(--text)!important; font-size:15px!important;
}
.stSelectbox>div>div {
  background:var(--card2)!important; border:1px solid var(--border)!important;
  border-radius:10px!important;
}
.stButton>button {
  background:linear-gradient(135deg,#06b6d4,#3b82f6)!important;
  color:white!important; border:none!important; border-radius:10px!important;
  font-weight:800!important; font-size:14px!important; padding:10px 24px!important;
}
div[data-testid="stExpander"] {
  background:var(--card2)!important; border:1px solid var(--border)!important;
  border-radius:12px!important;
}
footer,header,#MainMenu { display:none!important; }

.green{color:var(--green);} .red{color:var(--red);} .yellow{color:var(--yellow);}
.cyan{color:var(--cyan);}   .muted{color:var(--muted);} .purple{color:var(--purple);}

/* 태그 */
.tag { display:inline-block; border:1px solid var(--border); border-radius:999px;
       padding:3px 10px; font-size:11px; margin-right:5px; color:var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─── 유틸 ─────────────────────────────────────────────────────────
def money(x): return f"${x:,.2f}"
def fmt_mcap(x):
    if not x: return "—"
    if x>=1e12: return f"${x/1e12:.2f}T"
    if x>=1e9:  return f"${x/1e9:.2f}B"
    return f"${x/1e6:.2f}M"
def sc_color(s):
    if s>=85: return "#06b6d4"
    if s>=72: return "#22c55e"
    if s>=55: return "#f59e0b"
    return "#ef4444"
def badge_cls(color):
    return {"cyan":"badge-cyan","green":"badge-green",
            "yellow":"badge-yellow","red":"badge-red"}.get(color,"badge-gray")
def gauge_color(pct):
    if pct>=0.68: return "#22c55e"
    if pct>=0.38: return "#f59e0b"
    return "#ef4444"
def gauge_html(label, val, lo, hi, fmt=".1f", suffix=""):
    pct = max(0, min(1, (val-lo)/(hi-lo) if hi!=lo else 0.5))
    c = gauge_color(pct); bar = int(pct*100)
    return f"""<div class='gauge-wrap'>
      <div class='gauge-label'><span>{label}</span>
      <span style='color:{c};font-weight:700'>{val:{fmt}}{suffix}</span></div>
      <div class='gauge-bar'><div class='gauge-fill' style='width:{bar}%;background:{c};'></div></div>
    </div>"""

PLOT_BG = "rgba(0,0,0,0)"; GRID = "#1a2840"; TC = "#5a7299"

# ─── 차트 함수 ────────────────────────────────────────────────────
def make_chart(df, ticker, tf_choice, sig):
    d = df.tail(220)
    close = d["Close"]
    ma20 = sma(close, 20); ma50 = sma(close, 50)
    ma150 = sma(close, min(150, len(d))); ma200 = sma(close, min(200, len(d)))
    vwap_l = vwap(d)
    rsi_l  = rsi(close, 14)
    ml, sl2, hist = macd(close)
    lb, mb, ub = bbands(close, 20)
    sk, sd = stochastic(d)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.17, 0.17, 0.16],
        vertical_spacing=0.015,
        specs=[[{"secondary_y": True}],[{}],[{}],[{}]])

    # ── 캔들 ──────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        increasing=dict(line=dict(color="#22c55e", width=1), fillcolor="#22c55e"),
        decreasing=dict(line=dict(color="#ef4444", width=1), fillcolor="#ef4444"),
        name="OHLC", showlegend=False), row=1, col=1)

    # BB
    fig.add_trace(go.Scatter(x=d.index, y=ub, line=dict(color="#3b82f6", width=1, dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=lb, line=dict(color="#3b82f6", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(59,130,246,0.04)", showlegend=False), row=1, col=1)

    # MA Lines
    for ma, col, nm in [(ma20,"#f59e0b","MA20"),(ma50,"#a855f7","MA50"),
                        (ma150,"#f97316","MA150"),(ma200,"#ef4444","MA200")]:
        fig.add_trace(go.Scatter(x=d.index, y=ma, line=dict(color=col, width=1.3),
            name=nm, showlegend=True), row=1, col=1)

    # VWAP
    fig.add_trace(go.Scatter(x=d.index, y=vwap_l, line=dict(color="#06b6d4", width=1.8, dash="dash"),
        name="VWAP", showlegend=True), row=1, col=1)

    # Livermore 지지/저항선
    sc_obj = sig.sc
    if sc_obj.pivots.get("support"):
        fig.add_hline(y=sc_obj.pivots["support"],
            line=dict(color="#22c55e", dash="dot", width=1.2), row=1, col=1)
    if sc_obj.pivots.get("resistance"):
        fig.add_hline(y=sc_obj.pivots["resistance"],
            line=dict(color="#ef4444", dash="dot", width=1.2), row=1, col=1)

    # 거래량
    vc = ["#22c55e" if float(d["Close"].iloc[i])>=float(d["Open"].iloc[i]) else "#ef4444" for i in range(len(d))]
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vc, opacity=0.35,
        name="Vol", showlegend=False), row=1, col=1, secondary_y=True)

    # ── RSI ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=d.index, y=rsi_l, line=dict(color="#06b6d4", width=1.5),
        showlegend=False), row=2, col=1)
    for level, col in [(70,"#ef4444"),(50,"#5a7299"),(30,"#22c55e")]:
        fig.add_hline(y=level, line=dict(color=col, dash="dot", width=0.8), row=2, col=1)

    # ── MACD ──────────────────────────────────────────────────────
    hc = ["#22c55e" if float(v)>=0 else "#ef4444" for v in hist]
    fig.add_trace(go.Bar(x=d.index, y=hist, marker_color=hc, opacity=0.7, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=ml,  line=dict(color="#3b82f6", width=1.2), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=sl2, line=dict(color="#f97316", width=1.2), showlegend=False), row=3, col=1)

    # ── Stochastic ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=d.index, y=sk, line=dict(color="#a855f7", width=1.2), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=sd, line=dict(color="#f59e0b", width=1.2), showlegend=False), row=4, col=1)
    for level, col in [(80,"#ef4444"),(20,"#22c55e")]:
        fig.add_hline(y=level, line=dict(color=col, dash="dot", width=0.8), row=4, col=1)

    axis_s = dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                  showline=False, tickfont=dict(color=TC, size=10))

    fig.update_layout(
        height=700, margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        title=dict(text=f"<b>{ticker}</b>  ·  {tf_choice}  ·  캔들 + BB + MA20/50/150/200 + VWAP + Livermore 피벗",
                   x=0.01, y=0.99, font=dict(color="#d8e8ff", size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(color=TC, size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
        xaxis=dict(**axis_s, showticklabels=False),
        yaxis=dict(**axis_s),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, showticklabels=False),
        xaxis2=dict(**axis_s, showticklabels=False),
        yaxis3=dict(**axis_s, range=[0,100]),
        xaxis3=dict(**axis_s, showticklabels=False),
        yaxis4=dict(**axis_s),
        xaxis4=dict(**axis_s, showticklabels=False),
        yaxis5=dict(**axis_s, range=[0,100]),
        xaxis5=dict(**axis_s),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
#  앱 시작
# ══════════════════════════════════════════════════════════════════

# ── 헤더 ──────────────────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
  <div>
    <div class='app-title'>📈 StockEdge Pro</div>
    <div class='app-sub'>
      <span class='trader-badge tb-min'>Minervini</span>
      <span class='trader-badge tb-wei'>Weinstein</span>
      <span class='trader-badge tb-one'>O'Neil</span>
      <span class='trader-badge tb-ptj'>P.T.Jones</span>
      <span class='trader-badge tb-liv'>Livermore</span>
      &nbsp;5대 전설 트레이더 기법 통합 분석 엔진
    </div>
  </div>
  <div style='text-align:right'>
    <div style='font-size:11px; color:#5a7299;'>실시간 Yahoo Finance · 10분 캐시</div>
    <div style='font-size:11px; color:#5a7299; margin-top:2px;'>⚠️ 본 도구는 투자 조언이 아닙니다</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  메인 탭 구조
# ══════════════════════════════════════════════════════════════════
_tab_chart, _tab_scan = st.tabs(["📈  차트 분석", "🔭  빠른 티커 스캔  (등급별 분류)"])

# ══════════════════════════════════════════════════════════════════
#  탭 1 — 차트 분석 (기존 그대로)
# ══════════════════════════════════════════════════════════════════
with _tab_chart:

    # ── 입력 컨트롤 ──────────────────────────────────────────────
    ic1, ic2, ic3, ic4 = st.columns([2.5, 1.4, 1.4, 1])
    with ic1:
        ticker = st.text_input("티커", value="AAPL", placeholder="AAPL, NVDA, TSLA…",
                               label_visibility="collapsed")
    with ic2:
        style = st.selectbox("스타일", ["스윙", "단타"], label_visibility="collapsed")
    with ic3:
        tf_default = "스윙 (1D)" if style == "스윙" else "단타 (1H)"
        tf_choice = st.selectbox("타임프레임", list(TF_OPTIONS.keys()),
                                  index=list(TF_OPTIONS.keys()).index(tf_default),
                                  label_visibility="collapsed")
    with ic4:
        st.button("🔍 분석", use_container_width=True)

    ticker = ticker.strip().upper()
    if not ticker:
        st.info("티커를 입력하세요.")
        st.stop()

    # ── 데이터 로딩 ──────────────────────────────────────────────
    with st.spinner("데이터 수집 중…"):
        result = build_signal(ticker, style, tf_choice)
        info   = fetch_ticker_info(ticker)

    if not result:
        st.error("❌ 데이터 부족 또는 유효하지 않은 티커. 다른 타임프레임을 시도하세요.")
        st.stop()

    sig, df = result
    sc  = sig.sc
    last_price = float(df["Close"].iloc[-1])

    # ── VIX 배너 ──────────────────────────────────────────────────
    if sig.vix_text:
        vv  = sig.vix or 0
        cls = "vix-bad" if vv >= 28 else ("vix-warn" if vv >= 22 else "vix-ok")
        st.markdown(f"<div class='vix-banner {cls}'>{sig.vix_text}</div>", unsafe_allow_html=True)

    # ── 메인 레이아웃 ─────────────────────────────────────────────
    left, right = st.columns([2.1, 1], gap="medium")

    # ── 왼쪽 패널 ─────────────────────────────────────────────────
    with left:
        name   = info.get("name", ticker)
        sector = info.get("sector", "")
        pe     = info.get("pe_ratio")
        beta   = info.get("beta")
        mcap   = info.get("market_cap")
        chg    = sig.weekly_perf
        chg_c  = "#22c55e" if chg >= 0 else "#ef4444"
        chg_s  = "▲" if chg >= 0 else "▼"

        st.markdown(f"""
        <div class='card'>
          <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
            <div>
              <div style='font-size:28px; font-weight:900; color:#06b6d4; letter-spacing:-0.04em;'>{ticker}</div>
              <div style='font-size:13px; color:#5a7299; margin-top:2px;'>{name}</div>
              <div style='margin-top:10px;'>
                <span class='tag'>{sector or "—"}</span>
                <span class='tag'>시총 {fmt_mcap(mcap)}</span>
                {"<span class='tag'>P/E "+f"{pe:.1f}"+"</span>" if pe else ""}
                {"<span class='tag'>β "+f"{beta:.2f}"+"</span>" if beta else ""}
              </div>
            </div>
            <div style='text-align:right;'>
              <div style='font-size:34px; font-weight:800;'>{money(last_price)}</div>
              <div style='font-size:15px; font-weight:700; color:{chg_c};'>{chg_s} {abs(chg):.2f}%</div>
              <div style='font-size:11px; color:#5a7299; margin-top:4px;'>{tf_choice} · {sig.asof}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        fig = make_chart(df, ticker, tf_choice, sig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='sec-title'>기술 지표 상세</div>", unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            wr_adj = sc.williams_r + 100
            st.markdown(f"""
            <div class='card-sm'>
              <div class='sec-title'>모멘텀 오실레이터</div>
              {gauge_html("RSI (14)", sc.rsi_val, 0, 100)}
              {gauge_html("Stoch %K", sc.stoch_k, 0, 100)}
              {gauge_html("MFI (14)", sc.mfi_val, 0, 100)}
              {gauge_html("Williams %R", wr_adj, 0, 100)}
              {gauge_html("CCI (20) [역산]", max(0,-sc.cci_val/3+50), 0, 100)}
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class='card-sm'>
              <div class='sec-title'>추세 & 강도</div>
              {gauge_html("ADX (14)", sc.adx_val, 0, 60)}
              {gauge_html("Minervini TT", sc.trend_template['passed'], 0, 8, ".0f", "/8")}
              {gauge_html("O'Neil RS", sc.rs['rs_score'], 0, 100)}
              {gauge_html("52주 위치", sc.week52_pos*100, 0, 100, ".0f", "%")}
              {gauge_html("BB 위치", sc.bb_position*100, 0, 100, ".0f", "%")}
            </div>
            """, unsafe_allow_html=True)
        with mc3:
            macd_c = "#22c55e" if sc.macd_hist > 0 else "#ef4444"
            vwap_c = "#22c55e" if sc.vwap_above else "#ef4444"
            vwap_diff = (last_price / sc.vwap_val - 1)*100 if sc.vwap_val else 0
            st.markdown(f"""
            <div class='card-sm'>
              <div class='sec-title'>수급 & 특수 지표</div>
              <div class='kv'><div class='k'>MACD Hist</div>
                <div class='v' style='color:{macd_c};'>{sc.macd_hist:+.4f}</div></div>
              <div class='kv'><div class='k'>ATR%</div>
                <div class='v'>{sc.atr_pct*100:.2f}%</div></div>
              <div class='kv'><div class='k'>Volume Ratio</div>
                <div class='v' style='color:{"#22c55e" if sc.vol_ratio>=1.2 else "#5a7299"};'>{sc.vol_ratio:.2f}x</div></div>
              <div class='kv'><div class='k'>VWAP 대비</div>
                <div class='v' style='color:{vwap_c};'>{vwap_diff:+.2f}%</div></div>
              <div class='kv'><div class='k'>BB Squeeze</div>
                <div class='v' style='color:{"#f59e0b" if sc.bb_squeeze else "#5a7299"};'>{"⚡수축 감지" if sc.bb_squeeze else "일반"}</div></div>
              <div class='kv'><div class='k'>VCP 상태</div>
                <div class='v' style='color:{"#06b6d4" if sc.vcp["breakout_ready"] else "#5a7299"};'>{"🚀돌파준비" if sc.vcp["breakout_ready"] else ("⚡감지" if sc.vcp["detected"] else "—")}</div></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><div class='sec-title'>이동평균 현황</div>", unsafe_allow_html=True)
        mav_cols = st.columns(5)
        for col, lbl, val in zip(mav_cols, ["MA20","MA50","MA150","MA200","VWAP"],
                                  [sc.ma20, sc.ma50,
                                   sc.trend_template.get("ma150", sc.ma50),
                                   sc.ma200, sc.vwap_val]):
            d_pct = (last_price/val-1)*100 if val else 0
            dc = "#22c55e" if d_pct >= 0 else "#ef4444"
            with col:
                st.markdown(f"""
                <div class='card-sm' style='text-align:center;'>
                  <div class='k'>{lbl}</div>
                  <div style='font-size:13px; font-weight:700; margin-top:3px;'>{money(val)}</div>
                  <div style='font-size:12px; font-weight:600; color:{dc};'>{d_pct:+.1f}%</div>
                </div>""", unsafe_allow_html=True)

    # ── 오른쪽 패널 ───────────────────────────────────────────────
    with right:
        col_c = sc_color(sc.total)
        bc    = badge_cls(sig.action_color)
        st.markdown(f"""
        <div class='card-highlight' style='text-align:center;'>
          <div class='k' style='margin-bottom:8px; font-size:11px;'>5대 트레이더 통합 점수</div>
          <div class='score-big' style='color:{col_c};'>{sc.total}</div>
          <div class='score-grade' style='color:{col_c}; margin-top:6px;'>[{sc.grade}]</div>
          <div style='margin-top:14px;'>
            <span class='action-badge {bc}'>{sig.action}</span>
          </div>
          <div style='font-size:12px; color:#5a7299; margin-top:8px; line-height:1.4;'>{sig.action_detail}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>점수 분해</div>", unsafe_allow_html=True)
        cats = [
            ("추세 (Minervini+Weinstein+PTJ)", sc.trend_score, 35),
            ("모멘텀 (RSI+RS+MACD+ADX)",       sc.momentum_score, 25),
            ("패턴 (VCP+Cup+Livermore)",        sc.pattern_score, 20),
            ("수급/거래량 (MFI+Vol+VWAP)",      sc.flow_score, 20),
            ("리스크 조정",                     sc.risk_adj, 15),
        ]
        html = ""
        for lbl, val, mx in cats:
            pct = max(0, min(1, val/mx)) if mx else 0
            c = gauge_color(pct); bar = int(abs(val)/mx*100 if mx else 0)
            sign = "+" if val >= 0 else ""
            html += f"""<div class='gauge-wrap'>
              <div class='gauge-label'><span style='font-size:11px;'>{lbl}</span>
              <span style='color:{c};font-weight:700;'>{sign}{val} / {mx}</span></div>
              <div class='gauge-bar'><div class='gauge-fill' style='width:{bar}%;background:{c};'></div></div>
            </div>"""
        st.markdown(html + "</div>", unsafe_allow_html=True)

        # 매매 계획
        tp = sig.trade_plan
        wrr_c = "#22c55e" if tp.weighted_rr >= 2.5 else ("#f59e0b" if tp.weighted_rr >= 1.5 else "#ef4444")
        etype_c = {"즉시 진입":"#06b6d4","피벗 돌파 진입":"#22c55e",
                   "피벗 돌파 대기":"#f59e0b","눌림목 진입":"#a855f7"}.get(tp.entry_type,"#5a7299")

        st.markdown(f"""
        <div class='card' style='border:1px solid rgba(6,182,212,0.3); background:linear-gradient(135deg,rgba(6,182,212,0.05),rgba(0,0,0,0));'>
          <div class='sec-title'>📋 매매 계획 (Minervini + O'Neil + PTJ + Livermore)</div>
          <div style='background:rgba(6,182,212,0.08); border:1px solid rgba(6,182,212,0.25);
                      border-radius:10px; padding:10px 14px; margin-bottom:8px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
              <div>
                <div style='font-size:10px; color:#5a7299; font-weight:600; letter-spacing:0.1em;'>🎯 진입가</div>
                <div style='font-size:22px; font-weight:900; color:#06b6d4;'>{money(tp.entry_price)}</div>
              </div>
              <div style='text-align:right;'>
                <div style='font-size:11px; font-weight:700; color:{etype_c}; margin-bottom:3px;'>{tp.entry_type}</div>
                <div style='font-size:10px; color:#5a7299;'>1R = ${tp.risk_1r:.2f}</div>
              </div>
            </div>
            <div style='font-size:11px; color:#5a7299; margin-top:5px;'>{tp.entry_reason}</div>
          </div>
          <div style='background:rgba(239,68,68,0.07); border:1px solid rgba(239,68,68,0.25);
                      border-radius:10px; padding:10px 14px; margin-bottom:8px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
              <div>
                <div style='font-size:10px; color:#5a7299; font-weight:600; letter-spacing:0.1em;'>🛑 손절가</div>
                <div style='font-size:22px; font-weight:900; color:#ef4444;'>{money(tp.stop_price)}</div>
              </div>
              <div style='text-align:right;'>
                <div style='font-size:14px; font-weight:700; color:#ef4444;'>{tp.stop_pct:.2f}%</div>
                <div style='font-size:10px; color:#5a7299;'>O'Neil -7% 룰 적용</div>
              </div>
            </div>
            <div style='font-size:10px; color:#5a7299; margin-top:5px;'>{tp.stop_reason}</div>
          </div>
          <div style='background:rgba(34,197,94,0.06); border:1px solid rgba(34,197,94,0.2);
                      border-radius:10px; padding:10px 14px; margin-bottom:6px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
              <div>
                <div style='font-size:10px; color:#5a7299; font-weight:600; letter-spacing:0.1em;'>✅ 1차 익절 ({tp.tp1_qty}% 청산)</div>
                <div style='font-size:20px; font-weight:800; color:#22c55e;'>{money(tp.tp1_price)}</div>
              </div>
              <div style='text-align:right;'>
                <div style='font-size:13px; font-weight:700; color:#22c55e;'>{tp.tp1_pct:+.2f}%</div>
                <div style='font-size:11px; color:#5a7299;'>R:R {tp.rr1:.1f} : 1</div>
              </div>
            </div>
            <div style='font-size:10px; color:#5a7299; margin-top:4px;'>{tp.tp1_reason}</div>
          </div>
          <div style='background:rgba(34,197,94,0.09); border:1px solid rgba(34,197,94,0.3);
                      border-radius:10px; padding:10px 14px; margin-bottom:6px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
              <div>
                <div style='font-size:10px; color:#5a7299; font-weight:600; letter-spacing:0.1em;'>✅✅ 2차 익절 ({tp.tp2_qty}% 청산)</div>
                <div style='font-size:20px; font-weight:800; color:#22c55e;'>{money(tp.tp2_price)}</div>
              </div>
              <div style='text-align:right;'>
                <div style='font-size:13px; font-weight:700; color:#22c55e;'>{tp.tp2_pct:+.2f}%</div>
                <div style='font-size:11px; color:#5a7299;'>R:R {tp.rr2:.1f} : 1</div>
              </div>
            </div>
            <div style='font-size:10px; color:#5a7299; margin-top:4px;'>{tp.tp2_reason}</div>
          </div>
          <div style='background:rgba(6,182,212,0.08); border:1px solid rgba(6,182,212,0.3);
                      border-radius:10px; padding:10px 14px; margin-bottom:8px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
              <div>
                <div style='font-size:10px; color:#5a7299; font-weight:600; letter-spacing:0.1em;'>🚀 3차 익절 ({tp.tp3_qty}% 청산)</div>
                <div style='font-size:20px; font-weight:800; color:#06b6d4;'>{money(tp.tp3_price)}</div>
              </div>
              <div style='text-align:right;'>
                <div style='font-size:13px; font-weight:700; color:#06b6d4;'>{tp.tp3_pct:+.2f}%</div>
                <div style='font-size:11px; color:#5a7299;'>R:R {tp.rr3:.1f} : 1</div>
              </div>
            </div>
            <div style='font-size:10px; color:#5a7299; margin-top:4px;'>{tp.tp3_reason}</div>
          </div>
          <div style='display:flex; justify-content:space-between; align-items:center;
                      background:rgba(255,255,255,0.02); border-radius:8px; padding:8px 12px;'>
            <div style='text-align:center;'>
              <div style='font-size:10px; color:#5a7299;'>가중 R:R</div>
              <div style='font-size:18px; font-weight:900; color:{wrr_c};'>{tp.weighted_rr:.2f} : 1</div>
            </div>
            <div style='text-align:center;'>
              <div style='font-size:10px; color:#5a7299;'>최대 손실</div>
              <div style='font-size:18px; font-weight:900; color:#ef4444;'>-{tp.max_loss_pct:.2f}%</div>
            </div>
            <div style='text-align:center;'>
              <div style='font-size:10px; color:#5a7299;'>물량 배분</div>
              <div style='font-size:12px; font-weight:700; color:#d8e8ff;'>{tp.tp1_qty}/{tp.tp2_qty}/{tp.tp3_qty}%</div>
            </div>
          </div>
          <div style='font-size:11px; color:#5a7299; margin-top:8px; text-align:center;'>
            💡 {tp.position_note}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Weinstein Stage
        ws = sc.weinstein
        stage_cls = {1:"stage-1",2:"stage-2",3:"stage-3",4:"stage-4"}.get(ws["stage"],"stage-1")
        st.markdown(f"""
        <div class='{stage_cls}' style='margin-bottom:12px;'>
          <div class='sec-title'>Weinstein Stage Analysis</div>
          <div style='font-size:14px; font-weight:700;'>{ws['stage_name']}</div>
          <div style='font-size:12px; color:#5a7299; margin-top:6px;'>
            MA150 대비: {ws['distance_pct']:+.1f}% &nbsp;|&nbsp;
            기울기: {"↑상승" if ws['ma150_slope']>0 else "↓하락" if ws['ma150_slope']<0 else "→횡보"}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Minervini TT
        tt = sc.trend_template
        tt_color = "#06b6d4" if tt["is_stage2"] else ("#f59e0b" if tt["is_qualified"] else "#ef4444")
        check_html = "".join([
            f"<div class='tt-check'>"
            f"<span class='{'tt-pass' if ok else 'tt-fail'}'>{'✓' if ok else '✗'}</span>"
            f"<span style='font-size:11px; color:#{'d8e8ff' if ok else '5a7299'};'>{lbl}</span>"
            f"</div>"
            for lbl, ok in tt["conditions"]
        ])
        st.markdown(f"""
        <div class='card'>
          <div class='sec-title'>Minervini Trend Template</div>
          <div style='display:flex; align-items:center; gap:10px; margin-bottom:10px;'>
            <div style='font-size:28px; font-weight:900; color:{tt_color};'>{tt["passed"]}/8</div>
            <div>
              <div style='font-size:12px; font-weight:700; color:{tt_color};'>
                {"✅ Stage 2 확인" if tt["is_stage2"] else ("⚡ 부분 적합" if tt["is_qualified"] else "❌ 미달")}
              </div>
              <div style='font-size:11px; color:#5a7299;'>12M 성과: {tt['perf12m']:+.1f}%</div>
            </div>
          </div>
          {check_html}
        </div>
        """, unsafe_allow_html=True)

        # O'Neil RS
        rs = sc.rs; cph = sc.cup_handle
        rs_c = "#22c55e" if rs["rs_score"]>=80 else ("#f59e0b" if rs["rs_score"]>=70 else "#ef4444")
        perfs = rs.get("perfs", {})
        st.markdown(f"""
        <div class='card'>
          <div class='sec-title'>O'Neil CANSLIM 분석</div>
          <div style='display:flex; gap:12px; margin-bottom:10px; align-items:center;'>
            <div style='text-align:center;'>
              <div style='font-size:11px; color:#5a7299;'>RS Score</div>
              <div style='font-size:26px; font-weight:900; color:{rs_c};'>{rs['rs_score']}</div>
            </div>
            <div style='flex:1;'>
              <div style='font-size:12px; font-weight:600; color:{rs_c};'>{rs['momentum_rank']}</div>
              <div style='font-size:11px; color:#5a7299; margin-top:3px;'>
                3M: {f"{perfs.get('3M',0):+.1f}%" if perfs.get('3M') else "—"} &nbsp;
                6M: {f"{perfs.get('6M',0):+.1f}%" if perfs.get('6M') else "—"} &nbsp;
                12M: {f"{perfs.get('12M',0):+.1f}%" if perfs.get('12M') else "—"}
              </div>
            </div>
          </div>
          <div class='kv'><div class='k'>Cup-with-Handle</div>
            <div class='v' style='color:{"#06b6d4" if cph["breakout"] else "#5a7299"};'>{cph['pattern']}</div></div>
          <div class='kv'><div class='k'>컵 깊이</div>
            <div class='v'>{cph['cup_depth_pct']}% {"✅" if 12<=cph['cup_depth_pct']<=35 else "⚠️"}</div></div>
          <div class='kv'><div class='k'>거래량 배수</div>
            <div class='v' style='color:{"#22c55e" if cph["vol_expansion"]>=1.5 else "#5a7299"};'>{cph['vol_expansion']}x</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Livermore 피벗
        pivs = sc.pivots
        st.markdown(f"""
        <div class='card'>
          <div class='sec-title'>Livermore 피벗 & 수급</div>
          <div class='kv'><div class='k'>지지선</div><div class='v green'>{money(pivs.get('support',0))}</div></div>
          <div class='kv'><div class='k'>저항선 (피벗)</div><div class='v red'>{money(pivs.get('resistance',0))}</div></div>
          <div class='kv'><div class='k'>피벗 돌파 임박</div>
            <div class='v' style='color:{"#06b6d4" if pivs.get("near_breakout") else "#5a7299"};'>{"🎯 임박!" if pivs.get("near_breakout") else "—"}</div></div>
          <div class='kv'><div class='k'>골든크로스</div>
            <div class='v' style='color:{"#22c55e" if sc.golden_cross else "#5a7299"};'>{"🌟 발생" if sc.golden_cross else "—"}</div></div>
          <div class='kv'><div class='k'>데드크로스</div>
            <div class='v' style='color:{"#ef4444" if sc.death_cross else "#5a7299"};'>{"💀 발생" if sc.death_cross else "—"}</div></div>
          <div class='kv'><div class='k'>52주 위치</div>
            <div class='v'>{sc.week52_pos*100:.0f}% (저: {money(sc.week52_l)} / 고: {money(sc.week52_h)})</div></div>
        </div>
        """, unsafe_allow_html=True)

        if sc.warnings:
            st.markdown("<div class='sec-title' style='color:#ef4444;'>⚠️ 경고</div>", unsafe_allow_html=True)
            for w in sc.warnings:
                st.markdown(f"<div class='warn-item'>{w}</div>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>점수 이유</div>", unsafe_allow_html=True)
        for r in sc.reasons:
            st.markdown(f"<div class='reason-item'>{r}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  탭 2 — 빠른 티커 스캔 (등급별 분류)
# ══════════════════════════════════════════════════════════════════
with _tab_scan:

    # 등급 색상 / 배경 정의
    GRADE_COLOR = {
        "SSS":"#06b6d4","SS":"#06b6d4","S":"#22c55e",
        "A":"#4ade80","B":"#f59e0b","C":"#f97316","D":"#ef4444",
    }
    GRADE_BG = {
        "SSS":"rgba(6,182,212,0.13)","SS":"rgba(6,182,212,0.09)",
        "S":"rgba(34,197,94,0.10)","A":"rgba(74,222,128,0.07)",
        "B":"rgba(245,158,11,0.08)","C":"rgba(249,115,22,0.07)",
        "D":"rgba(239,68,68,0.06)",
    }
    GRADE_ORDER = ["SSS","SS","S","A","B","C","D"]

    # 프리셋 티커 묶음
    PRESETS = {
        "🔥 AI·반도체":   "NVDA,AMD,AVGO,QCOM,ARM,MRVL,TSM,AMAT,LRCX,KLAC,MU,SMCI,ASML",
        "💻 빅테크":       "AAPL,MSFT,GOOG,AMZN,META,NFLX,CRM,ORCL,ADBE,IBM",
        "⚡ 에너지·원전":  "SMR,NNE,CEG,VST,ETR,NRG,IREN,FSLR,ENPH,CLNE",
        "🚀 고성장주":     "PLTR,AXON,CRWD,ZS,DDOG,SNOW,NET,ABNB,UBER,MSTR",
        "💊 헬스케어":     "LLY,NVO,ABBV,JNJ,UNH,MRNA,PFE,AMGN,GILD,REGN",
        "🏦 금융·핀테크":  "JPM,BAC,GS,MS,BLK,V,MA,PYPL,SQ,COIN",
        "🎮 소비·엔터":    "TSLA,DIS,ABNB,BKNG,CMG,MCD,SBUX,NKE,LULU,RBLX",
        "📦 내 위시리스트":"NVDA,TSLA,MSFT,AAPL,AMZN,META,GOOG,PLTR,SMR,IREN,AXON,CRWD,MSTR,SNOW,NET",
    }

    # ── 컨트롤 ────────────────────────────────────────────────────
    st.markdown("""
    <div style='margin-bottom:14px;'>
      <div style='font-size:15px; font-weight:800; color:#d8e8ff; margin-bottom:4px;'>
        종목 선택 → 스캔 → S/A/B/C/D 등급 카드로 자동 분류
      </div>
      <div style='font-size:11px; color:#5a7299;'>
        프리셋 선택 또는 직접 입력 · 최대 40종목 · 종목당 약 3~5초
      </div>
    </div>
    """, unsafe_allow_html=True)

    row1c1, row1c2, row1c3 = st.columns([2, 1.2, 1.2])
    with row1c1:
        sc_preset = st.selectbox("📂 프리셋", list(PRESETS.keys()), key="sc_preset")
    with row1c2:
        sc_tf = st.selectbox("타임프레임", list(TF_OPTIONS.keys()), key="sc_tf")
    with row1c3:
        sc_style = st.selectbox("스타일", ["스윙","단타"], key="sc_style")

    sc_custom = st.text_input(
        "✏️ 직접 입력 (비워두면 프리셋 사용)",
        value="", placeholder="AAPL,NVDA,TSLA,PLTR,SMR …",
        key="sc_custom"
    )

    sc_tickers = (
        [t.strip().upper() for t in sc_custom.replace("\n",",").split(",") if t.strip()]
        if sc_custom.strip()
        else [t.strip().upper() for t in PRESETS[sc_preset].split(",") if t.strip()]
    )
    sc_tickers = list(dict.fromkeys(sc_tickers))[:40]

    btn_col, info_col = st.columns([1, 4])
    with btn_col:
        do_scan = st.button("🚀 스캔 시작", use_container_width=True, key="sc_run")
    with info_col:
        st.markdown(
            f"<div style='padding:9px 0; font-size:12px; color:#5a7299;'>"
            f"분석 대상: <b style='color:#d8e8ff;'>{len(sc_tickers)}종목</b></div>",
            unsafe_allow_html=True
        )

    if do_scan:
        results = []
        bar  = st.progress(0)
        stat = st.empty()

        for i, tk in enumerate(sc_tickers):
            bar.progress((i+1)/len(sc_tickers))
            stat.markdown(
                f"<div style='font-size:12px;color:#5a7299;'>"
                f"분석 중 [{i+1}/{len(sc_tickers)}]: "
                f"<b style='color:#06b6d4;'>{tk}</b></div>",
                unsafe_allow_html=True
            )
            try:
                res = build_signal(tk, sc_style, sc_tf)
                if not res:
                    continue
                s, d = res
                tp2 = s.trade_plan
                results.append({
                    "ticker":    tk,
                    "score":     s.sc.total,
                    "grade":     s.sc.grade,
                    "action":    s.action,
                    "action_c":  s.action_color,
                    "stage":     s.sc.weinstein["stage"],
                    "stage_lbl": {1:"Stage1⏳",2:"Stage2🚀",3:"Stage3⚠️",4:"Stage4❌"}.get(s.sc.weinstein["stage"],"—"),
                    "stage_c":   {1:"#f59e0b",2:"#22c55e",3:"#f97316",4:"#ef4444"}.get(s.sc.weinstein["stage"],"#5a7299"),
                    "tt":        s.sc.trend_template["passed"],
                    "rs":        s.sc.rs["rs_score"],
                    "rsi":       s.sc.rsi_val,
                    "adx":       s.sc.adx_val,
                    "vol_r":     s.sc.vol_ratio,
                    "vcp":       s.sc.vcp["breakout_ready"],
                    "vcp_det":   s.sc.vcp["detected"],
                    "cup":       s.sc.cup_handle["breakout"],
                    "golden":    s.sc.golden_cross,
                    "squeeze":   s.sc.bb_squeeze,
                    "entry":     tp2.entry_price,
                    "stop":      tp2.stop_price,
                    "stop_pct":  tp2.stop_pct,
                    "tp1":       tp2.tp1_price,
                    "tp1_pct":   tp2.tp1_pct,
                    "tp2":       tp2.tp2_price,
                    "tp2_pct":   tp2.tp2_pct,
                    "tp3":       tp2.tp3_price,
                    "tp3_pct":   tp2.tp3_pct,
                    "wrr":       tp2.weighted_rr,
                    "entry_type":tp2.entry_type,
                    "weekly":    s.weekly_perf,
                    "mfi":       s.sc.mfi_val,
                })
            except Exception:
                continue

        bar.empty(); stat.empty()

        if not results:
            st.warning("스캔 결과가 없습니다. 네트워크를 확인하세요.")
        else:
            results.sort(key=lambda x: x["score"], reverse=True)

            # ── 등급 분포 요약 배너 ───────────────────────────────
            grade_cnt = {}
            for r in results:
                grade_cnt[r["grade"]] = grade_cnt.get(r["grade"], 0) + 1

            banner_html = "<div style='display:flex; gap:8px; margin:10px 0 20px; flex-wrap:wrap;'>"
            for g in GRADE_ORDER:
                cnt = grade_cnt.get(g, 0)
                c   = GRADE_COLOR.get(g, "#5a7299")
                bg  = GRADE_BG.get(g, "rgba(90,114,153,0.07)")
                banner_html += f"""
                <div style='background:{bg}; border:1px solid {c}40; border-radius:12px;
                            padding:10px 18px; text-align:center; min-width:56px;'>
                  <div style='font-size:15px; font-weight:900; color:{c};'>{g}</div>
                  <div style='font-size:24px; font-weight:900; color:#d8e8ff; line-height:1.1;'>{cnt}</div>
                  <div style='font-size:10px; color:#5a7299; margin-top:1px;'>종목</div>
                </div>"""
            # Stage 2 카운트
            stage2_cnt = sum(1 for r in results if r["stage"] == 2)
            banner_html += f"""
                <div style='background:rgba(34,197,94,0.07); border:1px solid rgba(34,197,94,0.3);
                            border-radius:12px; padding:10px 18px; text-align:center; min-width:56px; margin-left:auto;'>
                  <div style='font-size:11px; font-weight:700; color:#22c55e;'>Stage2</div>
                  <div style='font-size:24px; font-weight:900; color:#22c55e; line-height:1.1;'>{stage2_cnt}</div>
                  <div style='font-size:10px; color:#5a7299; margin-top:1px;'>리더</div>
                </div>
                <div style='background:rgba(90,114,153,0.07); border:1px solid #1a284040;
                            border-radius:12px; padding:10px 18px; text-align:center; min-width:56px;'>
                  <div style='font-size:11px; font-weight:700; color:#5a7299;'>총 분석</div>
                  <div style='font-size:24px; font-weight:900; color:#d8e8ff; line-height:1.1;'>{len(results)}</div>
                  <div style='font-size:10px; color:#5a7299; margin-top:1px;'>종목</div>
                </div>
            </div>"""
            st.markdown(banner_html, unsafe_allow_html=True)

            # ── 등급별 탭 ─────────────────────────────────────────
            def grade_label(g):
                cnt = grade_cnt.get(g, 0)
                icon = {"SSS":"⭐","SS":"🌟","S":"✅","A":"📈","B":"➡️","C":"⚠️","D":"📉"}.get(g,"")
                return f"{icon} {g} ({cnt})"

            tabs_grades = st.tabs(
                [grade_label(g) for g in GRADE_ORDER] + ["📋 전체 테이블"]
            )

            def render_cards(items):
                if not items:
                    st.markdown("<div style='padding:24px; text-align:center; color:#5a7299; font-size:13px;'>해당 등급 종목이 없습니다</div>", unsafe_allow_html=True)
                    return
                cols3 = st.columns(3)
                for idx, r in enumerate(items):
                    g   = r["grade"]
                    gc  = GRADE_COLOR.get(g, "#5a7299")
                    gbg = GRADE_BG.get(g, "rgba(90,114,153,0.07)")
                    ac  = {"cyan":"#06b6d4","green":"#22c55e","yellow":"#f59e0b",
                           "red":"#ef4444","gray":"#5a7299"}.get(r["action_c"],"#5a7299")
                    wc  = sig.weekly_perf  # will be overridden below
                    wc  = "#22c55e" if r["weekly"] >= 0 else "#ef4444"
                    ws  = "▲" if r["weekly"] >= 0 else "▼"
                    wrrc = "#22c55e" if r["wrr"] >= 2 else ("#f59e0b" if r["wrr"] >= 1.5 else "#ef4444")

                    badges = " ".join(b for b in [
                        "🚀VCP" if r["vcp"] else ("⚡" if r["vcp_det"] else ""),
                        "🏆Cup" if r["cup"] else "",
                        "🌟골든" if r["golden"] else "",
                        "⚡수축" if r["squeeze"] else "",
                    ] if b)

                    with cols3[idx % 3]:
                        st.markdown(f"""
                        <div style='background:{gbg}; border:1px solid {gc}35;
                                    border-radius:14px; padding:14px; margin-bottom:12px;'>

                          <!-- 헤더 -->
                          <div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;'>
                            <div>
                              <div style='font-size:22px; font-weight:900; color:{gc}; letter-spacing:-0.03em;'>{r["ticker"]}</div>
                              <div style='font-size:10px; color:{r["stage_c"]}; font-weight:700; margin-top:2px;'>{r["stage_lbl"]}</div>
                              {f"<div style='font-size:10px; color:#5a7299; margin-top:2px;'>{badges}</div>" if badges else ""}
                            </div>
                            <div style='text-align:right;'>
                              <div style='font-size:28px; font-weight:900; color:{gc}; line-height:1;'>{g}</div>
                              <div style='font-size:12px; font-weight:700; color:{gc};'>{r["score"]}점</div>
                            </div>
                          </div>

                          <!-- 시그널 -->
                          <div style='background:rgba(0,0,0,0.18); border-radius:7px; padding:4px 10px;
                                      font-size:12px; font-weight:700; color:{ac}; margin-bottom:8px;'>
                            {r["action"]}
                          </div>

                          <!-- 지표 그리드 -->
                          <div style='display:grid; grid-template-columns:1fr 1fr; gap:3px; margin-bottom:8px;'>
                            <div style='font-size:11px; color:#5a7299;'>TT <span style='color:#d8e8ff;font-weight:700;'>{r["tt"]}/8</span></div>
                            <div style='font-size:11px; color:#5a7299;'>RS <span style='color:#d8e8ff;font-weight:700;'>{r["rs"]}</span></div>
                            <div style='font-size:11px; color:#5a7299;'>RSI <span style='color:#d8e8ff;font-weight:700;'>{r["rsi"]:.0f}</span></div>
                            <div style='font-size:11px; color:#5a7299;'>ADX <span style='color:#d8e8ff;font-weight:700;'>{r["adx"]:.0f}</span></div>
                            <div style='font-size:11px; color:#5a7299;'>Vol <span style='color:#d8e8ff;font-weight:700;'>{r["vol_r"]:.2f}x</span></div>
                            <div style='font-size:11px; color:{wc};'>{ws} <span style='font-weight:700;'>{abs(r["weekly"]):.1f}%</span></div>
                          </div>

                          <!-- 매매 계획 -->
                          <div style='border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;'>
                            <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                              <span style='font-size:10px; color:#5a7299;'>진입가</span>
                              <span style='font-size:11px; font-weight:700; color:#06b6d4;'>${r["entry"]:.2f}</span>
                            </div>
                            <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                              <span style='font-size:10px; color:#5a7299;'>🛑 손절</span>
                              <span style='font-size:11px; font-weight:700; color:#ef4444;'>${r["stop"]:.2f} ({r["stop_pct"]:.1f}%)</span>
                            </div>
                            <div style='display:flex; justify-content:space-between; margin-bottom:2px;'>
                              <span style='font-size:10px; color:#5a7299;'>✅ 1차 TP</span>
                              <span style='font-size:11px; font-weight:600; color:#22c55e;'>${r["tp1"]:.2f} ({r["tp1_pct"]:+.1f}%)</span>
                            </div>
                            <div style='display:flex; justify-content:space-between; margin-bottom:2px;'>
                              <span style='font-size:10px; color:#5a7299;'>✅ 2차 TP</span>
                              <span style='font-size:11px; font-weight:600; color:#22c55e;'>${r["tp2"]:.2f} ({r["tp2_pct"]:+.1f}%)</span>
                            </div>
                            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                              <span style='font-size:10px; color:#5a7299;'>🚀 3차 TP</span>
                              <span style='font-size:11px; font-weight:700; color:#06b6d4;'>${r["tp3"]:.2f} ({r["tp3_pct"]:+.1f}%)</span>
                            </div>
                            <div style='display:flex; justify-content:space-between; border-top:1px solid rgba(255,255,255,0.06); padding-top:6px;'>
                              <span style='font-size:10px; color:#5a7299;'>가중 R:R</span>
                              <span style='font-size:13px; font-weight:900; color:{wrrc};'>{r["wrr"]:.1f} : 1</span>
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

            # 각 등급별 탭 렌더링
            for ti, g in enumerate(GRADE_ORDER):
                with tabs_grades[ti]:
                    items_g = [r for r in results if r["grade"] == g]
                    if items_g:
                        st.markdown(
                            f"<div style='font-size:12px;color:#5a7299;margin-bottom:10px;'>"
                            f"<b style='color:{GRADE_COLOR[g]};'>{g}등급</b> {len(items_g)}종목 — "
                            f"점수 {min(r['score'] for r in items_g)}~{max(r['score'] for r in items_g)}점</div>",
                            unsafe_allow_html=True
                        )
                    render_cards(items_g)

            # 전체 테이블 탭
            with tabs_grades[-1]:
                table_rows = []
                for r in results:
                    table_rows.append({
                        "티커":   r["ticker"],
                        "등급":   r["grade"],
                        "점수":   r["score"],
                        "시그널": r["action"],
                        "Stage":  r["stage"],
                        "TT":     f"{r['tt']}/8",
                        "RS":     r["rs"],
                        "RSI":    f"{r['rsi']:.0f}",
                        "ADX":    f"{r['adx']:.0f}",
                        "Vol":    f"{r['vol_r']:.2f}x",
                        "VCP":    "🚀" if r["vcp"] else ("⚡" if r["vcp_det"] else "—"),
                        "진입가": f"${r['entry']:.2f}",
                        "손절가": f"${r['stop']:.2f}({r['stop_pct']:.1f}%)",
                        "1차TP":  f"${r['tp1']:.2f}({r['tp1_pct']:+.1f}%)",
                        "2차TP":  f"${r['tp2']:.2f}({r['tp2_pct']:+.1f}%)",
                        "3차TP":  f"${r['tp3']:.2f}({r['tp3_pct']:+.1f}%)",
                        "R:R":    f"{r['wrr']:.1f}",
                        "주간":   f"{r['weekly']:+.1f}%",
                    })
                df_table = pd.DataFrame(table_rows)
                st.dataframe(
                    df_table, use_container_width=True, hide_index=True,
                    column_config={
                        "점수": st.column_config.ProgressColumn("점수", min_value=0, max_value=100, format="%d"),
                    }
                )
                # Stage 2 리더 강조
                leaders = [r for r in results if r["stage"] == 2 and r["score"] >= 62]
                if leaders:
                    st.success(f"🌟 Minervini Stage 2 리더 (A등급 이상): {', '.join(r['ticker'] for r in leaders[:10])}")


# ─── 푸터 ─────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:20px 0 8px; color:#5a7299; font-size:11px;'>
  ⚠️ 본 서비스는 투자 조언이 아닌 개인 참고용 도구입니다. 모든 투자 결정은 본인 책임입니다.<br>
  데이터 출처: Yahoo Finance · 기법 출처: Minervini(SEPA/VCP) · Weinstein(Stage Analysis) · O'Neil(CANSLIM) · P.T.Jones(200MA/R:R) · Livermore(Pivot)
</div>
""", unsafe_allow_html=True)
