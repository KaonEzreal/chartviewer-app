"""
StockEdge Pro — v2 백테스트 전략 완전 구현
PF 1.85 / 승률 46.1% 검증 전략
"""
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.config import TF_OPTIONS
from core.data import fetch_ticker_info
from core.indicators import bbands, macd, rsi, sma, stochastic, vwap
from core.signal import (
    build_signal, calc_entry_conditions, build_trade_plan,
    V2_STOP_ATR_MUL, V2_STOP_MAX_PCT, V2_BREAKEVEN_PCT,
    V2_BREAKEVEN_RAISE, V2_PARTIAL_PCT, V2_TRAIL_PCT,
    V2_MIN_SCORE_S, V2_MIN_SCORE_A, V2_MIN_RS, V2_MIN_ADX,
)

st.set_page_config(page_title="StockEdge Pro", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root{--bg:#070b12;--card:#0c1220;--card2:#0f1828;--border:#1a2840;--muted:#5a7299;--text:#d8e8ff;--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--blue:#3b82f6;--cyan:#06b6d4;--purple:#a855f7;--orange:#f97316;}
*,body,html{font-family:'Inter',sans-serif!important;}
html,body,[class*="css"],.stApp{background:var(--bg)!important;color:var(--text)!important;}
.block-container{padding:1rem 1.5rem 3rem!important;max-width:1500px!important;}
.app-header{background:linear-gradient(135deg,#0c1220,#111e35);border:1px solid var(--border);border-radius:18px;padding:16px 24px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;}
.app-title{font-size:22px;font-weight:900;color:var(--cyan);letter-spacing:-0.04em;}
.app-sub{font-size:11px;color:var(--muted);margin-top:2px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:16px;margin-bottom:12px;}
.card-sm{background:var(--card2);border:1px solid var(--border);border-radius:12px;padding:12px;}
.card-highlight{background:linear-gradient(135deg,rgba(6,182,212,0.08),rgba(59,130,246,0.05));border:1px solid rgba(6,182,212,0.3);border-radius:16px;padding:18px;}
.score-big{font-size:96px;font-weight:900;line-height:1;letter-spacing:-0.06em;}
.score-grade{font-size:24px;font-weight:700;letter-spacing:3px;}
.action-badge{display:inline-flex;align-items:center;gap:6px;padding:10px 24px;border-radius:999px;font-size:16px;font-weight:800;}
.badge-cyan{background:rgba(6,182,212,0.12);color:#06b6d4;border:1.5px solid rgba(6,182,212,0.4);}
.badge-green{background:rgba(34,197,94,0.12);color:#22c55e;border:1.5px solid rgba(34,197,94,0.4);}
.badge-yellow{background:rgba(245,158,11,0.12);color:#f59e0b;border:1.5px solid rgba(245,158,11,0.4);}
.badge-red{background:rgba(239,68,68,0.12);color:#ef4444;border:1.5px solid rgba(239,68,68,0.4);}
.badge-gray{background:rgba(90,114,153,0.12);color:#5a7299;border:1.5px solid rgba(90,114,153,0.4);}
.kv{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(26,40,64,0.6);}
.kv:last-child{border-bottom:none;}
.k{color:var(--muted);font-size:12px;font-weight:500;}
.v{font-weight:700;font-size:13px;}
.gauge-wrap{margin:5px 0;}
.gauge-label{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:3px;}
.gauge-bar{height:6px;border-radius:99px;background:var(--border);}
.gauge-fill{height:100%;border-radius:99px;}
.stage-2{background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);border-radius:10px;padding:10px 14px;}
.stage-1{background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);border-radius:10px;padding:10px 14px;}
.stage-3{background:rgba(249,115,22,0.1);border:1px solid rgba(249,115,22,0.3);border-radius:10px;padding:10px 14px;}
.stage-4{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:10px 14px;}
.tt-check{display:flex;align-items:center;gap:8px;padding:4px 0;font-size:12px;border-bottom:1px solid rgba(26,40,64,0.5);}
.tt-check:last-child{border:none;}
.tt-pass{color:var(--green);font-weight:700;}
.tt-fail{color:var(--red);font-weight:700;}
.vix-ok{background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);color:#22c55e;}
.vix-warn{background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.2);color:#f59e0b;}
.vix-bad{background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.2);color:#ef4444;}
.vix-banner{border-radius:12px;padding:10px 16px;margin-bottom:12px;font-size:13px;font-weight:600;}
.trader-badge{display:inline-block;border-radius:6px;padding:2px 8px;font-size:10px;font-weight:700;margin-right:4px;}
.tb-min{background:rgba(6,182,212,0.2);color:#06b6d4;}
.tb-wei{background:rgba(168,85,247,0.2);color:#a855f7;}
.tb-one{background:rgba(245,158,11,0.2);color:#f59e0b;}
.tb-ptj{background:rgba(34,197,94,0.2);color:#22c55e;}
.tb-liv{background:rgba(249,115,22,0.2);color:#f97316;}
.sec-title{font-size:11px;font-weight:700;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;}
.tag{display:inline-block;border:1px solid var(--border);border-radius:999px;padding:3px 10px;font-size:11px;margin-right:5px;color:var(--muted);}
.reason-item{background:rgba(6,182,212,0.04);border:1px solid rgba(6,182,212,0.15);border-radius:10px;padding:10px 14px;margin-bottom:6px;font-size:13px;line-height:1.5;}
.warn-item{background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.2);border-radius:10px;padding:10px 14px;margin-bottom:6px;font-size:13px;line-height:1.5;}
.stTextInput>div>div>input{background:var(--card2)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important;font-size:15px!important;}
.stSelectbox>div>div{background:var(--card2)!important;border:1px solid var(--border)!important;border-radius:10px!important;}
.stButton>button{background:linear-gradient(135deg,#06b6d4,#3b82f6)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:800!important;font-size:14px!important;padding:10px 24px!important;}
div[data-testid="stExpander"]{background:var(--card2)!important;border:1px solid var(--border)!important;border-radius:12px!important;}
footer,header,#MainMenu{display:none!important;}
* {
    line-height: 1.5 !important;
}

div, span {
    word-break: keep-all;
}

.reason-item, .warn-item {
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)

# ── 유틸 함수 ────────────────────────────────────────────────────
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
def badge_cls(c):
    return {"cyan":"badge-cyan","green":"badge-green","yellow":"badge-yellow","red":"badge-red"}.get(c,"badge-gray")
def gc(pct):
    if pct>=0.68: return "#22c55e"
    if pct>=0.38: return "#f59e0b"
    return "#ef4444"
def gh(label, val, lo, hi, fmt=".1f", suffix=""):
    p=max(0,min(1,(val-lo)/(hi-lo) if hi!=lo else 0.5))
    c=gc(p); b=int(p*100)
    return f"<div class='gauge-wrap'><div class='gauge-label'><span>{label}</span><span style='color:{c};font-weight:700'>{val:{fmt}}{suffix}</span></div><div class='gauge-bar'><div class='gauge-fill' style='width:{b}%;background:{c};'></div></div></div>"
def safe_mfi(v):
    try:
        f=float(v)
        return 50.0 if math.isnan(f) else f
    except: return 50.0

# ── 9개 조건 체크리스트 HTML 생성 ─────────────────────────────
def build_checklist_html(conds):
    html = ""
    for c in conds:
        icon = "✅" if c.passed else "❌"
        col  = "#22c55e" if c.passed else "#ef4444"
        imp_tag = ""
        if c.importance == "critical":
            imp_tag = "<span style='background:rgba(239,68,68,0.2);color:#ef4444;border-radius:4px;padding:1px 5px;font-size:9px;font-weight:700;margin-left:5px;'>핵심</span>"
        elif c.importance == "high":
            imp_tag = "<span style='background:rgba(245,158,11,0.2);color:#f59e0b;border-radius:4px;padding:1px 5px;font-size:9px;font-weight:700;margin-left:5px;'>중요</span>"
        status = c.current if c.passed else c.current
        html += f"""<div style='display:grid;grid-template-columns:18px 1fr;gap:8px;align-items:start;padding:7px 0;border-bottom:1px solid rgba(30,58,95,0.5);'>
<div style='font-size:14px;'>{icon}</div>
<div>
<div style='font-size:11px;font-weight:700;color:#d8e8ff;'>{c.name}{imp_tag}</div>
<div style='font-size:11px;color:{col};font-weight:600;margin-top:1px;'>{status} — {c.current if c.passed else c.required}</div>
<div style='font-size:10px;color:#5a7299;margin-top:2px;line-height:1.4;'>{c.meaning}</div>
</div>
</div>"""
    return html

# ── 추천 등급 계산 ────────────────────────────────────────────
def get_recommendation(conds):
    total       = len(conds)
    passed      = sum(1 for c in conds if c.passed)
    critical_ok = all(c.passed for c in conds if c.importance == "critical")
    if not critical_ok:
        fails = [c.name for c in conds if not c.passed and c.importance == "critical"]
        return "red", "🔴 진입 보류", f"핵심 조건 미충족: {' · '.join(fails[:2])}", passed, total
    if passed == total:
        return "cyan", "🔥 매수 강력 추천", "9/9 모든 조건 충족 — 하라는 대로 진입하세요", passed, total
    if passed >= 7:
        return "green", "✅ 매수 적극 추천", f"{passed}/{total} 조건 충족 — v2 진입 기준 통과", passed, total
    if passed >= 5:
        return "yellow", "🟡 조건부 매수", f"{passed}/{total} 조건 충족 — 추가 확인 후 소량", passed, total
    return "red", "🔴 진입 보류", f"{passed}/{total} 조건 충족 — 조건 개선 대기", passed, total

PBG="rgba(0,0,0,0)"; GRD="#1a2840"; TC="#5a7299"

# ── 차트 생성 ─────────────────────────────────────────────────
def make_chart(df, ticker, tf_choice, sig):
    d=df.tail(220); close=d["Close"]
    ma20=sma(close,20); ma50=sma(close,50)
    ma150=sma(close,min(150,len(d))); ma200=sma(close,min(200,len(d)))
    vwap_l=vwap(d); rsi_l=rsi(close,14)
    ml,sl2,hist=macd(close); lb,mb,ub=bbands(close,20); sk,sd=stochastic(d)

    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,
        row_heights=[0.50,0.17,0.17,0.16],vertical_spacing=0.015,
        specs=[[{"secondary_y":True}],[{}],[{}],[{}]])

    fig.add_trace(go.Candlestick(x=d.index,open=d["Open"],high=d["High"],low=d["Low"],close=d["Close"],
        increasing=dict(line=dict(color="#22c55e",width=1),fillcolor="#22c55e"),
        decreasing=dict(line=dict(color="#ef4444",width=1),fillcolor="#ef4444"),
        name="OHLC",showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=ub,line=dict(color="#3b82f6",width=1,dash="dot"),showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=lb,line=dict(color="#3b82f6",width=1,dash="dot"),
        fill="tonexty",fillcolor="rgba(59,130,246,0.04)",showlegend=False),row=1,col=1)
    for ma,col,nm in [(ma20,"#f59e0b","MA20"),(ma50,"#a855f7","MA50"),(ma150,"#f97316","MA150"),(ma200,"#ef4444","MA200")]:
        fig.add_trace(go.Scatter(x=d.index,y=ma,line=dict(color=col,width=1.3),name=nm,showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=vwap_l,line=dict(color="#06b6d4",width=1.8,dash="dash"),name="VWAP",showlegend=True),row=1,col=1)
    pivs=sig.sc.pivots
    if pivs.get("support"): fig.add_hline(y=pivs["support"],line=dict(color="#22c55e",dash="dot",width=1.2),row=1,col=1)
    if pivs.get("resistance"): fig.add_hline(y=pivs["resistance"],line=dict(color="#ef4444",dash="dot",width=1.2),row=1,col=1)
    vc=["#22c55e" if float(d["Close"].iloc[i])>=float(d["Open"].iloc[i]) else "#ef4444" for i in range(len(d))]
    fig.add_trace(go.Bar(x=d.index,y=d["Volume"],marker_color=vc,opacity=0.35,showlegend=False),row=1,col=1,secondary_y=True)
    fig.add_trace(go.Scatter(x=d.index,y=rsi_l,line=dict(color="#06b6d4",width=1.5),showlegend=False),row=2,col=1)
    for lv,col in [(70,"#ef4444"),(50,"#5a7299"),(30,"#22c55e")]:
        fig.add_hline(y=lv,line=dict(color=col,dash="dot",width=0.8),row=2,col=1)
    hc=["#22c55e" if float(v)>=0 else "#ef4444" for v in hist]
    fig.add_trace(go.Bar(x=d.index,y=hist,marker_color=hc,opacity=0.7,showlegend=False),row=3,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=ml,line=dict(color="#3b82f6",width=1.2),showlegend=False),row=3,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=sl2,line=dict(color="#f97316",width=1.2),showlegend=False),row=3,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=sk,line=dict(color="#a855f7",width=1.2),showlegend=False),row=4,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=sd,line=dict(color="#f59e0b",width=1.2),showlegend=False),row=4,col=1)
    for lv,col in [(80,"#ef4444"),(20,"#22c55e")]:
        fig.add_hline(y=lv,line=dict(color=col,dash="dot",width=0.8),row=4,col=1)

    ax=dict(showgrid=True,gridcolor=GRD,gridwidth=1,zeroline=False,showline=False,tickfont=dict(color=TC,size=10))
    fig.update_layout(height=700,margin=dict(l=8,r=8,t=40,b=8),paper_bgcolor=PBG,plot_bgcolor=PBG,
        title=dict(text=f"<b>{ticker}</b>  ·  {tf_choice}",x=0.01,y=0.99,font=dict(color="#d8e8ff",size=13)),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,font=dict(color=TC,size=11),bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
        xaxis=dict(**ax,showticklabels=False),yaxis=dict(**ax),
        yaxis2=dict(overlaying="y",side="right",showgrid=False,showticklabels=False),
        xaxis2=dict(**ax,showticklabels=False),yaxis3=dict(**ax,range=[0,100]),
        xaxis3=dict(**ax,showticklabels=False),yaxis4=dict(**ax),
        xaxis4=dict(**ax,showticklabels=False),yaxis5=dict(**ax,range=[0,100]),xaxis5=dict(**ax))
    return fig


# ══════════════════════════════════════════════════════════════════
# 앱 시작
# ══════════════════════════════════════════════════════════════════
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
&nbsp;v2 백테스트 검증 전략 (PF 1.85 / 승률 46.1%) — 하라는 대로만 하세요
</div>
</div>
<div style='text-align:right'>
<div style='font-size:11px;color:#5a7299;'>Yahoo Finance · 장중 1분 / 장외 10분 캐시</div>
<div style='font-size:11px;color:#5a7299;margin-top:2px;'>⚠️ 본 도구는 투자 조언이 아닙니다</div>
</div>
</div>
""", unsafe_allow_html=True)

# 탭 3개
_tabs = st.tabs(["📈  차트 분석", "🔭  빠른 티커 스캔  (등급 + 조건 충족률)", "🎯  만점 티커 찾기  (9/9 + TT 완전 충족)"])

# ══════════════════════════════════════════════════════════════════
# 탭 1 — 차트 분석
# ══════════════════════════════════════════════════════════════════
with _tabs[0]:
    ic1,ic2,ic3,ic4=st.columns([2.5,1.4,1.4,1])
    with ic1: ticker=st.text_input("티커",value="AAPL",placeholder="AAPL, NVDA, TSLA…",label_visibility="collapsed")
    with ic2: style=st.selectbox("스타일",["스윙","단타"],label_visibility="collapsed")
    with ic3:
        tf_default="스윙 (1D)" if style=="스윙" else "단타 (1H)"
        tf_choice=st.selectbox("타임프레임",list(TF_OPTIONS.keys()),index=list(TF_OPTIONS.keys()).index(tf_default),label_visibility="collapsed")
    with ic4: st.button("🔍 분석",use_container_width=True)

    ticker=ticker.strip().upper()
    if not ticker:
        st.info("티커를 입력하세요.")
        st.stop()

    with st.spinner("데이터 수집 중…"):
        result=build_signal(ticker,style,tf_choice)
        info=fetch_ticker_info(ticker)

    if not result:
        st.error("❌ 데이터 부족 또는 유효하지 않은 티커.")
        st.stop()

    sig,df=result
    sc=sig.sc
    last_price=float(df["Close"].iloc[-1])

    if sig.vix_text:
        vv=sig.vix or 0
        cls="vix-bad" if vv>=28 else("vix-warn" if vv>=22 else "vix-ok")
        st.markdown(f"<div class='vix-banner {cls}'>{sig.vix_text}</div>",unsafe_allow_html=True)

    # 추천 등급
    conds=sig.entry_conds
    rec_color,rec_label,rec_desc,passed_cnt,total_cnt=get_recommendation(conds)
    pct_bar=int(passed_cnt/total_cnt*100)
    bar_c={"cyan":"#06b6d4","green":"#22c55e","yellow":"#f59e0b","red":"#ef4444"}.get(rec_color,"#5a7299")
    tp=sig.trade_plan

    # ── v2 매매 가이드 패널 ───────────────────────────────────────
    with st.expander("📋 v2 백테스트 실전 매매 가이드 — 하라는 대로만 하세요", expanded=True):

        # 상단: 추천 + 충족률 바
        st.markdown(f"""
<div style='background:linear-gradient(135deg,#0c1828,#0f1e30);border:1px solid #1e3a5f;border-radius:16px;padding:18px 20px;'>
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;'>
<div>
<div style='font-size:16px;font-weight:900;color:#06b6d4;'>{ticker} — v2 백테스트 실전 가이드</div>
<div style='font-size:11px;color:#5a7299;margin-top:3px;'>분석 시점: {sig.asof} · 조건 충족 시 손절/익절대로만 하면 됩니다</div>
</div>
<div style='background:rgba(255,255,255,0.04);border:2px solid {bar_c}50;border-radius:14px;padding:10px 20px;text-align:center;'>
<div style='font-size:20px;font-weight:900;color:{bar_c};'>{rec_label}</div>
<div style='font-size:12px;color:#5a7299;margin-top:3px;'>{passed_cnt}/{total_cnt} 조건 충족</div>
</div>
</div>
<div style='margin-bottom:14px;'>
<div style='display:flex;justify-content:space-between;font-size:11px;color:#5a7299;margin-bottom:5px;'>
<span>v2 백테스트 진입 조건 충족률</span>
<span style='color:{bar_c};font-weight:700;'>{passed_cnt}/{total_cnt} ({pct_bar}%)</span>
</div>
<div style='height:10px;background:#1e3a5f;border-radius:99px;overflow:hidden;'>
<div style='height:100%;width:{pct_bar}%;background:linear-gradient(90deg,{bar_c},{bar_c}cc);border-radius:99px;'></div>
</div>
<div style='font-size:11px;color:{bar_c};font-weight:600;margin-top:5px;'>{rec_desc}</div>
</div>
</div>
""", unsafe_allow_html=True)

        # 3단 그리드
        g1, g2, g3 = st.columns(3)
        with g1:
            spy_c={"상승장":"#22c55e","횡보장":"#f59e0b","하락장":"#ef4444"}.get(sig.bias,"#5a7299")
            spy_txt={"상승장":"🟢 상승장 — 매수 가능 환경","횡보장":"🟡 횡보장 — 선별 매수","하락장":"🔴 하락장 — 매수 중단"}.get(sig.bias,"—")
            st.markdown(f"""
<div style='background:rgba(255,255,255,0.02);border:1px solid #1e3a5f;border-radius:12px;padding:13px;'>
<div style='font-size:10px;font-weight:700;color:#5a7299;letter-spacing:0.12em;margin-bottom:8px;'>STEP 1 · 시장 국면</div>
<div style='font-size:14px;font-weight:800;color:{spy_c};margin-bottom:6px;'>{spy_txt}</div>
<div style='font-size:11px;color:#5a7299;line-height:1.7;'>
매일 아침 SPY 먼저 분석<br>
MA50·MA200 위 → 매수 가능<br>
MA200 하회 → 신규 매수 중단
</div>
</div>""", unsafe_allow_html=True)

        with g2:
            if sc.vcp["breakout_ready"]:
                timing_txt="🚀 지금 진입 — VCP 돌파 확인"
            elif sc.vcp["detected"] or sc.bb_squeeze:
                timing_txt="⏳ 돌파 대기 — VCP/수축 완성 중"
            elif passed_cnt >= 7:
                timing_txt="📈 눌림목 확인 후 분할 진입"
            else:
                timing_txt="👀 조건 개선 대기"
            st.markdown(f"""
<div style='background:rgba(255,255,255,0.02);border:1px solid #1e3a5f;border-radius:12px;padding:13px;'>
<div style='font-size:10px;font-weight:700;color:#5a7299;letter-spacing:0.12em;margin-bottom:8px;'>STEP 2 · 진입 타이밍</div>
<div style='font-size:12px;font-weight:700;color:#d8e8ff;margin-bottom:6px;line-height:1.5;'>{timing_txt}</div>
<div style='font-size:11px;color:#5a7299;line-height:1.7;'>
진입가: <b style='color:#06b6d4;'>{money(tp.entry_price)}</b> [{tp.entry_type}]<br>
점수: <b style='color:{"#22c55e" if sc.total>=73 else "#f59e0b"};'>{sc.total}점 [{sc.grade}]</b><br>
Stage: <b style='color:{"#22c55e" if sc.weinstein["stage"]==2 else "#ef4444"};'>Stage {sc.weinstein["stage"]}</b>
</div>
</div>""", unsafe_allow_html=True)

        with g3:
            st.markdown(f"""
<div style='background:rgba(239,68,68,0.04);border:1px solid rgba(239,68,68,0.2);border-radius:12px;padding:13px;'>
<div style='font-size:10px;font-weight:700;color:#ef4444;letter-spacing:0.12em;margin-bottom:8px;'>⚠️ 절대 원칙 5가지</div>
<div style='font-size:11px;line-height:1.9;color:#d8e8ff;'>
<span style='color:#ef4444;font-weight:800;'>①</span> 손절가 → <b>이유 없이 즉시 매도</b><br>
<span style='font-size:10px;color:#5a7299;margin-left:14px;'>"조금만 더" = 가장 큰 손실 원인</span><br>
<span style='color:#ef4444;font-weight:800;'>②</span> 한 종목 = <b>계좌 최대 10%</b><br>
<span style='color:#ef4444;font-weight:800;'>③</span> SPY MA200 하회 = <b>매수 중단</b><br>
<span style='color:#ef4444;font-weight:800;'>④</span> 3연속 손절 = <b>1주일 휴식</b><br>
<span style='color:#ef4444;font-weight:800;'>⑤</span> 동시 보유 = <b>최대 5~6종목</b>
</div>
</div>""", unsafe_allow_html=True)

        # 9개 조건 체크리스트
        st.markdown("""
<div style='background:rgba(255,255,255,0.02);border:1px solid #1e3a5f;border-radius:12px;padding:14px 14px 8px;margin-top:10px;'>
<div style='font-size:10px;font-weight:700;color:#5a7299;letter-spacing:0.12em;margin-bottom:8px;'>
📋 v2 백테스트 9개 진입 조건 체크리스트
</div>
</div>""", unsafe_allow_html=True)
        st.markdown(build_checklist_html(conds), unsafe_allow_html=True)

        # v2 매매 계획 (손절 + 1차 익절 50% + 트레일링)
        st.markdown(f"""
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:12px;'>
<div style='background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.25);border-radius:12px;padding:14px;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:4px;font-weight:700;'>🛑 손절가 — 도달 즉시 매도</div>
<div style='font-size:24px;font-weight:900;color:#ef4444;'>{money(tp.stop_price)}</div>
<div style='font-size:12px;color:#ef4444;font-weight:600;margin-top:3px;'>{tp.stop_pct:.1f}% 손실</div>
<div style='height:5px;background:#1e3a5f;border-radius:99px;margin:8px 0;overflow:hidden;'>
<div style='width:{min(abs(tp.stop_pct)/8*100,100):.0f}%;height:100%;background:#ef4444;border-radius:99px;'></div>
</div>
<div style='font-size:10px;color:#5a7299;line-height:1.5;'>
{tp.stop_reason}<br>
<b style='color:#f59e0b;'>⚡ +{V2_BREAKEVEN_PCT*100:.0f}% 달성 시</b> 손절 → {money(tp.be_stop)} 자동 상향
</div>
</div>
<div style='background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);border-radius:12px;padding:14px;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:4px;font-weight:700;'>✅ 1차 익절 — 50% 청산</div>
<div style='font-size:24px;font-weight:900;color:#22c55e;'>{money(tp.tp1_price)}</div>
<div style='font-size:12px;color:#22c55e;font-weight:600;margin-top:3px;'>{tp.tp1_pct:+.1f}% · R:R {tp.rr1:.1f}</div>
<div style='background:#1e3a5f;border-radius:99px;height:5px;margin:8px 0;overflow:hidden;'>
<div style='width:50%;height:100%;background:#22c55e;border-radius:99px;'></div>
</div>
<div style='font-size:10px;color:#5a7299;line-height:1.5;'>
v2: +18% 달성 시 보유량 50% 청산<br>
<b style='color:#22c55e;'>이때 손절 → 진입가+1%로 상향</b><br>
잔량 50%는 트레일링으로 처리
</div>
</div>
<div style='background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.25);border-radius:12px;padding:14px;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:4px;font-weight:700;'>🔄 잔량 50% — 트레일링 스톱</div>
<div style='font-size:18px;font-weight:800;color:#06b6d4;'>고점 대비 -{tp.trail_pct*100:.0f}% 추적</div>
<div style='font-size:12px;color:#06b6d4;font-weight:600;margin-top:3px;'>{sc.grade}등급 트레일링 적용</div>
<div style='background:#1e3a5f;border-radius:99px;height:5px;margin:8px 0;overflow:hidden;'>
<div style='width:100%;height:100%;background:#06b6d4;border-radius:99px;'></div>
</div>
<div style='font-size:10px;color:#5a7299;line-height:1.5;'>
1차 익절 후 트레일링 → 7%로 타이트<br>
고점 갱신 시 자동 상향 추적<br>
<b style='color:#06b6d4;'>추세 끝까지 보유 극대화</b>
</div>
</div>
</div>
<div style='display:flex;gap:10px;margin-top:10px;'>
<div style='flex:1;background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.2);border-radius:10px;padding:10px 14px;'>
<span style='font-size:11px;font-weight:700;color:#06b6d4;'>💡 한줄 판단: </span>
<span style='font-size:12px;color:#d8e8ff;font-weight:600;'>{rec_desc}</span>
</div>
<div style='background:rgba(255,255,255,0.02);border:1px solid #1e3a5f;border-radius:10px;padding:10px 16px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>포지션 리스크</div>
<div style='font-size:13px;font-weight:700;color:#d8e8ff;'>계좌의 1.5%</div>
</div>
<div style='background:rgba(255,255,255,0.02);border:1px solid #1e3a5f;border-radius:10px;padding:10px 16px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>1R 손실</div>
<div style='font-size:13px;font-weight:700;color:#ef4444;'>${tp.risk_1r:.2f}</div>
</div>
</div>
""", unsafe_allow_html=True)

    # ── 메인 레이아웃 ─────────────────────────────────────────────
    left, right = st.columns([2.1, 1], gap="medium")

    with left:
        name=info.get("name",ticker); sector=info.get("sector","")
        pe=info.get("pe_ratio"); beta=info.get("beta"); mcap=info.get("market_cap")
        chg=sig.weekly_perf
        chg_c="#22c55e" if chg>=0 else "#ef4444"; chg_s="▲" if chg>=0 else "▼"
        st.markdown(f"""
<div class='card'>
<div style='display:flex;justify-content:space-between;align-items:flex-start;'>
<div>
<div style='font-size:28px;font-weight:900;color:#06b6d4;letter-spacing:-0.04em;'>{ticker}</div>
<div style='font-size:13px;color:#5a7299;margin-top:2px;'>{name}</div>
<div style='margin-top:10px;'>
<span class='tag'>{sector or "—"}</span>
<span class='tag'>시총 {fmt_mcap(mcap)}</span>
{"<span class='tag'>P/E "+f"{pe:.1f}"+"</span>" if pe else ""}
{"<span class='tag'>β "+f"{beta:.2f}"+"</span>" if beta else ""}
</div>
</div>
<div style='text-align:right;'>
<div style='font-size:34px;font-weight:800;'>{money(last_price)}</div>
<div style='font-size:15px;font-weight:700;color:{chg_c};'>{chg_s} {abs(chg):.2f}%</div>
<div style='font-size:11px;color:#5a7299;margin-top:4px;'>{tf_choice} · {sig.asof}</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

        fig=make_chart(df,ticker,tf_choice,sig)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        st.markdown("<div class='sec-title'>기술 지표 상세</div>",unsafe_allow_html=True)
        mc1,mc2,mc3=st.columns(3)
        with mc1:
            st.markdown(f"""
<div class='card-sm'>
<div class='sec-title'>모멘텀 오실레이터</div>
{gh("RSI (14)", sc.rsi_val, 0, 100)}
{gh("Stoch %K", sc.stoch_k, 0, 100)}
{gh("MFI (14)", safe_mfi(sc.mfi_val), 0, 100)}
{gh("Williams %R", sc.williams_r+100, 0, 100)}
{gh("CCI (20) [역산]", max(0,-sc.cci_val/3+50), 0, 100)}
</div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
<div class='card-sm'>
<div class='sec-title'>추세 & 강도</div>
{gh("ADX (14)", sc.adx_val, 0, 60)}
{gh("Minervini TT", sc.trend_template['passed'], 0, 8, ".0f", "/8")}
{gh("O'Neil RS", sc.rs['rs_score'], 0, 100)}
{gh("52주 위치", sc.week52_pos*100, 0, 100, ".0f", "%")}
{gh("BB 위치", sc.bb_position*100, 0, 100, ".0f", "%")}
</div>""", unsafe_allow_html=True)
        with mc3:
            macd_c="#22c55e" if sc.macd_hist>0 else "#ef4444"
            vwap_c="#22c55e" if sc.vwap_above else "#ef4444"
            vwap_diff=(last_price/sc.vwap_val-1)*100 if sc.vwap_val else 0
            st.markdown(f"""
<div class='card-sm'>
<div class='sec-title'>수급 & 특수 지표</div>
<div class='kv'><div class='k'>MACD Hist</div><div class='v' style='color:{macd_c};'>{sc.macd_hist:+.4f}</div></div>
<div class='kv'><div class='k'>ATR%</div><div class='v'>{sc.atr_pct*100:.2f}%</div></div>
<div class='kv'><div class='k'>Volume Ratio</div><div class='v' style='color:{"#22c55e" if sc.vol_ratio>=1.2 else "#5a7299"};'>{sc.vol_ratio:.2f}x</div></div>
<div class='kv'><div class='k'>VWAP 대비</div><div class='v' style='color:{vwap_c};'>{vwap_diff:+.2f}%</div></div>
<div class='kv'><div class='k'>BB Squeeze</div><div class='v' style='color:{"#f59e0b" if sc.bb_squeeze else "#5a7299"};'>{"⚡수축감지" if sc.bb_squeeze else "일반"}</div></div>
<div class='kv'><div class='k'>VCP</div><div class='v' style='color:{"#06b6d4" if sc.vcp["breakout_ready"] else "#5a7299"};'>{"🚀돌파준비" if sc.vcp["breakout_ready"] else("⚡감지" if sc.vcp["detected"] else "—")}</div></div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br><div class='sec-title'>이동평균 현황</div>",unsafe_allow_html=True)
        mav_cols=st.columns(5)
        for col,lbl,val in zip(mav_cols,["MA20","MA50","MA150","MA200","VWAP"],
                                [sc.ma20,sc.ma50,sc.trend_template.get("ma150",sc.ma50),sc.ma200,sc.vwap_val]):
            d_pct=(last_price/val-1)*100 if val else 0
            dc="#22c55e" if d_pct>=0 else "#ef4444"
            with col:
                st.markdown(f"""
<div class='card-sm' style='text-align:center;'>
<div class='k'>{lbl}</div>
<div style='font-size:13px;font-weight:700;margin-top:3px;'>{money(val)}</div>
<div style='font-size:12px;font-weight:600;color:{dc};'>{d_pct:+.1f}%</div>
</div>""", unsafe_allow_html=True)

    with right:
        col_c=sc_color(sc.total); bc=badge_cls(sig.action_color)
        st.markdown(f"""
<div class='card-highlight' style='text-align:center;'>
<div class='k' style='margin-bottom:8px;font-size:11px;'>5대 트레이더 통합 점수</div>
<div class='score-big' style='color:{col_c};'>{sc.total}</div>
<div class='score-grade' style='color:{col_c};margin-top:6px;'>[{sc.grade}]</div>
<div style='margin-top:12px;'>
<span class='action-badge {bc}'>{sig.action}</span>
</div>
<div style='font-size:12px;color:#5a7299;margin-top:8px;line-height:1.4;'>{sig.action_detail}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div class='card'>",unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>점수 분해</div>",unsafe_allow_html=True)
        cats=[("추세(Minervini+Weinstein+PTJ)",sc.trend_score,35),
              ("모멘텀(RSI+RS+MACD+ADX)",sc.momentum_score,25),
              ("패턴(VCP+Cup+Livermore)",sc.pattern_score,20),
              ("수급/거래량(MFI+Vol+VWAP)",sc.flow_score,20),
              ("리스크 조정",sc.risk_adj,15)]
        html=""
        for lbl,val,mx in cats:
            p=max(0,min(1,val/mx)) if mx else 0
            c=gc(p); b=int(abs(val)/mx*100 if mx else 0)
            sign="+" if val>=0 else ""
            html+=f"<div class='gauge-wrap'><div class='gauge-label'><span style='font-size:11px;'>{lbl}</span><span style='color:{c};font-weight:700;'>{sign}{val}/{mx}</span></div><div class='gauge-bar'><div class='gauge-fill' style='width:{b}%;background:{c};'></div></div></div>"
        st.markdown(html+"</div>",unsafe_allow_html=True)

        ws=sc.weinstein
        stage_cls={1:"stage-1",2:"stage-2",3:"stage-3",4:"stage-4"}.get(ws["stage"],"stage-1")
        st.markdown(f"""
<div class='{stage_cls}' style='margin-bottom:12px;'>
<div class='sec-title'>Weinstein Stage</div>
<div style='font-size:14px;font-weight:700;'>{ws['stage_name']}</div>
<div style='font-size:12px;color:#5a7299;margin-top:6px;'>MA150 대비: {ws['distance_pct']:+.1f}% · {"↑" if ws['ma150_slope']>0 else "↓" if ws['ma150_slope']<0 else "→"}</div>
</div>""", unsafe_allow_html=True)

        tt=sc.trend_template
        tt_color="#06b6d4" if tt["is_stage2"] else("#f59e0b" if tt["is_qualified"] else "#ef4444")
        check_html="".join([
            f"<div class='tt-check'><span class='{'tt-pass' if ok else 'tt-fail'}'>{'✓' if ok else '✗'}</span><span style='font-size:11px;color:#{'d8e8ff' if ok else '5a7299'};'>{lbl}</span></div>"
            for lbl,ok in tt["conditions"]
        ])
        st.markdown(f"""
<div class='card'>
<div class='sec-title'>Minervini Trend Template</div>
<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
<div style='font-size:28px;font-weight:900;color:{tt_color};'>{tt["passed"]}/8</div>
<div>
<div style='font-size:12px;font-weight:700;color:{tt_color};'>{"✅ Stage 2 확인" if tt["is_stage2"] else("⚡ 부분 적합" if tt["is_qualified"] else "❌ 미달")}</div>
<div style='font-size:11px;color:#5a7299;'>12M 성과: {tt['perf12m']:+.1f}%</div>
</div>
</div>
{check_html}
</div>""", unsafe_allow_html=True)

        rs=sc.rs; cph=sc.cup_handle
        rs_c="#22c55e" if rs["rs_score"]>=80 else("#f59e0b" if rs["rs_score"]>=70 else "#ef4444")
        perfs=rs.get("perfs",{})
        st.markdown(f"""
<div class='card'>
<div class='sec-title'>O'Neil RS & Cup-Handle</div>
<div style='display:flex;gap:12px;margin-bottom:10px;align-items:center;'>
<div style='text-align:center;'>
<div style='font-size:11px;color:#5a7299;'>RS Score</div>
<div style='font-size:26px;font-weight:900;color:{rs_c};'>{rs['rs_score']}</div>
</div>
<div style='flex:1;'>
<div style='font-size:12px;font-weight:600;color:{rs_c};'>{rs['momentum_rank']}</div>
<div style='font-size:11px;color:#5a7299;margin-top:3px;'>3M:{f"{perfs.get('3M',0):+.1f}%" if perfs.get('3M') else "—"} 6M:{f"{perfs.get('6M',0):+.1f}%" if perfs.get('6M') else "—"} 12M:{f"{perfs.get('12M',0):+.1f}%" if perfs.get('12M') else "—"}</div>
</div>
</div>
<div class='kv'><div class='k'>Cup-with-Handle</div><div class='v' style='color:{"#06b6d4" if cph["breakout"] else "#5a7299"};'>{cph['pattern']}</div></div>
<div class='kv'><div class='k'>컵깊이/거래량</div><div class='v'>{cph['cup_depth_pct']}% / {cph['vol_expansion']}x</div></div>
</div>""", unsafe_allow_html=True)

        pivs=sc.pivots
        st.markdown(f"""
<div class='card'>
<div class='sec-title'>Livermore 피벗 & 크로스</div>
<div class='kv'><div class='k'>지지선</div><div class='v' style='color:#22c55e;'>{money(pivs.get('support',0))}</div></div>
<div class='kv'><div class='k'>저항선</div><div class='v' style='color:#ef4444;'>{money(pivs.get('resistance',0))}</div></div>
<div class='kv'><div class='k'>돌파 임박</div><div class='v' style='color:{"#06b6d4" if pivs.get("near_breakout") else "#5a7299"};'>{"🎯 임박!" if pivs.get("near_breakout") else "—"}</div></div>
<div class='kv'><div class='k'>골든크로스</div><div class='v' style='color:{"#22c55e" if sc.golden_cross else "#5a7299"};'>{"🌟 발생" if sc.golden_cross else "—"}</div></div>
<div class='kv'><div class='k'>데드크로스</div><div class='v' style='color:{"#ef4444" if sc.death_cross else "#5a7299"};'>{"💀 발생" if sc.death_cross else "—"}</div></div>
<div class='kv'><div class='k'>52주 위치</div><div class='v'>{sc.week52_pos*100:.0f}% / 저:{money(sc.week52_l)} 고:{money(sc.week52_h)}</div></div>
</div>""", unsafe_allow_html=True)

        if sc.warnings:
            st.markdown("<div class='sec-title' style='color:#ef4444;'>⚠️ 경고</div>",unsafe_allow_html=True)
            for w in sc.warnings:
                st.markdown(f"<div class='warn-item'>{w}</div>",unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>점수 이유</div>",unsafe_allow_html=True)
        for r in sc.reasons:
            st.markdown(f"<div class='reason-item'>{r}</div>",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 탭 2 — 빠른 티커 스캔
# ══════════════════════════════════════════════════════════════════
with _tabs[1]:
    GRADE_COLOR={"SSS":"#06b6d4","SS":"#06b6d4","S":"#22c55e","A":"#4ade80","B":"#f59e0b","C":"#f97316","D":"#ef4444"}
    GRADE_BG={"SSS":"rgba(6,182,212,0.13)","SS":"rgba(6,182,212,0.09)","S":"rgba(34,197,94,0.10)","A":"rgba(74,222,128,0.07)","B":"rgba(245,158,11,0.08)","C":"rgba(249,115,22,0.07)","D":"rgba(239,68,68,0.06)"}
    GRADE_ORDER=["SSS","SS","S","A","B","C","D"]

    PRESETS={
        "🔥 AI·반도체":    "NVDA,AMD,AVGO,QCOM,ARM,MRVL,TSM,AMAT,LRCX,KLAC,MU,SMCI,ASML,INTC,MCHP,MPWR,ON",
        "💻 빅테크":        "AAPL,MSFT,GOOG,AMZN,META,NFLX,CRM,ORCL,ADBE,IBM,NOW,INTU,WDAY,TEAM",
        "⚡ 에너지·원전":   "SMR,NNE,CEG,VST,ETR,NRG,IREN,FSLR,ENPH,CLNE,NEE,AES,PCG,EXC",
        "🚀 고성장주":      "PLTR,AXON,CRWD,ZS,DDOG,SNOW,NET,ABNB,UBER,MSTR,AFRM,BILL,TTD",
        "💊 헬스케어":      "LLY,NVO,ABBV,JNJ,UNH,MRNA,PFE,AMGN,GILD,REGN,ISRG,BSX,DXCM",
        "🏦 금융·핀테크":   "JPM,BAC,GS,MS,BLK,V,MA,PYPL,SQ,COIN,HOOD,SOFI,NU",
        "🛡️ 사이버보안":    "CRWD,ZS,PANW,FTNT,S,OKTA,CYBR,QLYS,TENB,DDOG",
        "🤖 AI 소프트웨어": "PLTR,AI,PATH,BBAI,SOUN,GTLB,MNDY,ASAN,TTD,CELH",
        "🎮 소비·엔터":     "TSLA,DIS,ABNB,BKNG,CMG,MCD,SBUX,NKE,LULU,RBLX,SPOT",
        "📦 내 위시리스트":  "NVDA,TSLA,MSFT,AAPL,AMZN,META,GOOG,PLTR,SMR,IREN,AXON,CRWD,MSTR,SNOW,NET,AMD,ARM,AVGO,LLY,V",
    }

    st.markdown("""
<div style='margin-bottom:14px;'>
<div style='font-size:15px;font-weight:800;color:#d8e8ff;margin-bottom:4px;'>종목 스캔 → 등급 + v2 조건 충족률 + 매수 추천 자동 분류</div>
<div style='font-size:11px;color:#5a7299;'>최대 100종목 · 카드마다 X/9 조건 충족 + 매수 추천 등급 + 손절/익절 표시</div>
</div>""", unsafe_allow_html=True)

    r1c1,r1c2,r1c3=st.columns([2,1.2,1.2])
    with r1c1: sc_preset=st.selectbox("📂 프리셋",list(PRESETS.keys()),key="sc_preset")
    with r1c2: sc_tf=st.selectbox("타임프레임",list(TF_OPTIONS.keys()),key="sc_tf")
    with r1c3: sc_style=st.selectbox("스타일",["스윙","단타"],key="sc_style")

    sc_custom=st.text_input("✏️ 직접 입력 (비워두면 프리셋)",value="",placeholder="AAPL,NVDA,TSLA…",key="sc_custom")
    sc_tickers=(
        [t.strip().upper() for t in sc_custom.replace("\n",",").split(",") if t.strip()]
        if sc_custom.strip()
        else [t.strip().upper() for t in PRESETS[sc_preset].split(",") if t.strip()]
    )
    sc_tickers=list(dict.fromkeys(sc_tickers))[:100]

    bc1,bc2=st.columns([1,4])
    with bc1: do_scan=st.button("🚀 스캔 시작",use_container_width=True,key="sc_run")
    with bc2: st.markdown(f"<div style='padding:9px 0;font-size:12px;color:#5a7299;'>분석 대상: <b style='color:#d8e8ff;'>{len(sc_tickers)}종목</b></div>",unsafe_allow_html=True)

    if do_scan:
        results=[]
        bar=st.progress(0); stat=st.empty()
        for i,tk in enumerate(sc_tickers):
            bar.progress((i+1)/len(sc_tickers))
            stat.markdown(f"<div style='font-size:12px;color:#5a7299;'>분석중 [{i+1}/{len(sc_tickers)}]: <b style='color:#06b6d4;'>{tk}</b></div>",unsafe_allow_html=True)
            try:
                res=build_signal(tk,sc_style,sc_tf)
                if not res: continue
                s,d=res; tp2=s.trade_plan
                v2c=s.entry_conds
                rc,rl,rd,pc,tc=get_recommendation(v2c)
                results.append({
                    "ticker":tk,"score":s.sc.total,"grade":s.sc.grade,
                    "action":s.action,"action_c":s.action_color,
                    "stage":s.sc.weinstein["stage"],
                    "stage_lbl":{1:"Stage1⏳",2:"Stage2🚀",3:"Stage3⚠️",4:"Stage4❌"}.get(s.sc.weinstein["stage"],"—"),
                    "stage_c":{1:"#f59e0b",2:"#22c55e",3:"#f97316",4:"#ef4444"}.get(s.sc.weinstein["stage"],"#5a7299"),
                    "tt":s.sc.trend_template["passed"],"rs":s.sc.rs["rs_score"],
                    "rsi":s.sc.rsi_val,"adx":s.sc.adx_val,"vol_r":s.sc.vol_ratio,
                    "vcp":s.sc.vcp["breakout_ready"],"vcp_det":s.sc.vcp["detected"],
                    "cup":s.sc.cup_handle["breakout"],"golden":s.sc.golden_cross,"squeeze":s.sc.bb_squeeze,
                    "entry":tp2.entry_price,"stop":tp2.stop_price,"stop_pct":tp2.stop_pct,
                    "tp1":tp2.tp1_price,"tp1_pct":tp2.tp1_pct,
                    "trail_pct":tp2.trail_pct,"be_trigger":tp2.be_trigger,
                    "weighted_rr":tp2.weighted_rr,"weekly":s.weekly_perf,
                    "cond_passed":pc,"cond_total":tc,
                    "rec_color":rc,"rec_label":rl,"rec_desc":rd,
                })
            except Exception: continue
        bar.empty(); stat.empty()

        if not results:
            st.warning("스캔 결과가 없습니다.")
        else:
            results.sort(key=lambda x:(x["cond_passed"],x["score"]),reverse=True)
            grade_cnt={}; rec_cnt={"cyan":0,"green":0,"yellow":0,"red":0}
            for r in results:
                grade_cnt[r["grade"]]=grade_cnt.get(r["grade"],0)+1
                rec_cnt[r["rec_color"]]=rec_cnt.get(r["rec_color"],0)+1

            st.markdown(f"""
<div style='display:flex;gap:8px;margin:10px 0 16px;flex-wrap:wrap;'>
<div style='background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.4);border-radius:12px;padding:10px 16px;text-align:center;'>
<div style='font-size:11px;color:#5a7299;font-weight:700;'>🔥 강력추천</div>
<div style='font-size:24px;font-weight:900;color:#06b6d4;'>{rec_cnt["cyan"]}</div>
</div>
<div style='background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.4);border-radius:12px;padding:10px 16px;text-align:center;'>
<div style='font-size:11px;color:#5a7299;font-weight:700;'>✅ 적극추천</div>
<div style='font-size:24px;font-weight:900;color:#22c55e;'>{rec_cnt["green"]}</div>
</div>
<div style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.4);border-radius:12px;padding:10px 16px;text-align:center;'>
<div style='font-size:11px;color:#5a7299;font-weight:700;'>🟡 조건부</div>
<div style='font-size:24px;font-weight:900;color:#f59e0b;'>{rec_cnt["yellow"]}</div>
</div>
<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);border-radius:12px;padding:10px 16px;text-align:center;'>
<div style='font-size:11px;color:#5a7299;font-weight:700;'>🔴 보류</div>
<div style='font-size:24px;font-weight:900;color:#ef4444;'>{rec_cnt["red"]}</div>
</div>
<div style='margin-left:auto;background:rgba(90,114,153,0.07);border:1px solid #1a284040;border-radius:12px;padding:10px 16px;text-align:center;'>
<div style='font-size:11px;color:#5a7299;font-weight:700;'>총 분석</div>
<div style='font-size:24px;font-weight:900;color:#d8e8ff;'>{len(results)}</div>
</div>
</div>""", unsafe_allow_html=True)

            def grade_label(g):
                return f'{{"SSS":"⭐","SS":"🌟","S":"✅","A":"📈","B":"➡️","C":"⚠️","D":"📉"}}.get("{g}","") {g} ({grade_cnt.get(g,0)})'

            tabs_grades=st.tabs([f'{{"SSS":"⭐","SS":"🌟","S":"✅","A":"📈","B":"➡️","C":"⚠️","D":"📉"}}.get("{g}","") {g} ({grade_cnt.get(g,0)})' for g in GRADE_ORDER]+["📋 전체 테이블"])

            def render_cards(items):
                if not items:
                    st.markdown("<div style='padding:24px;text-align:center;color:#5a7299;'>해당 등급 없음</div>",unsafe_allow_html=True)
                    return
                cols3=st.columns(3)
                for idx,r in enumerate(items):
                    g=r["grade"]; gc_=GRADE_COLOR.get(g,"#5a7299"); gbg=GRADE_BG.get(g,"rgba(90,114,153,0.07)")
                    ac={"cyan":"#06b6d4","green":"#22c55e","yellow":"#f59e0b","red":"#ef4444","gray":"#5a7299"}.get(r["action_c"],"#5a7299")
                    wc="#22c55e" if r["weekly"]>=0 else "#ef4444"; ws_="▲" if r["weekly"]>=0 else "▼"
                    rc=r["rec_color"]
                    badge_c={"cyan":"#06b6d4","green":"#22c55e","yellow":"#f59e0b","red":"#ef4444"}.get(rc,"#5a7299")
                    badge_bg={"cyan":"rgba(6,182,212,0.12)","green":"rgba(34,197,94,0.12)","yellow":"rgba(245,158,11,0.12)","red":"rgba(239,68,68,0.12)"}.get(rc,"rgba(90,114,153,0.1)")
                    cond_bar=int(r["cond_passed"]/r["cond_total"]*100)
                    badges=" ".join(b for b in["🚀VCP" if r["vcp"] else("⚡" if r["vcp_det"] else ""),"🏆Cup" if r["cup"] else "","🌟골든" if r["golden"] else "","⚡수축" if r["squeeze"] else ""] if b)
                    with cols3[idx%3]:
                        st.markdown(f"""
<div style='background:{gbg};border:1px solid {gc_}35;border-radius:14px;padding:14px;margin-bottom:12px;'>
<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;'>
<div>
<div style='font-size:22px;font-weight:900;color:{gc_};'>{r["ticker"]}</div>
<div style='font-size:10px;color:{r["stage_c"]};font-weight:700;margin-top:2px;'>{r["stage_lbl"]}</div>
{f"<div style='font-size:10px;color:#5a7299;margin-top:2px;'>{badges}</div>" if badges else ""}
</div>
<div style='text-align:right;'>
<div style='font-size:26px;font-weight:900;color:{gc_};line-height:1;'>{g}</div>
<div style='font-size:12px;font-weight:700;color:{gc_};'>{r["score"]}점</div>
</div>
</div>
<div style='background:{badge_bg};border:1px solid {badge_c}40;border-radius:8px;padding:5px;text-align:center;margin-bottom:8px;'>
<div style='font-size:13px;font-weight:900;color:{badge_c};'>{r["rec_label"]}</div>
</div>
<div style='margin-bottom:8px;'>
<div style='display:flex;justify-content:space-between;font-size:10px;color:#5a7299;margin-bottom:3px;'>
<span>v2 진입조건</span><span style='color:{badge_c};font-weight:700;'>{r["cond_passed"]}/{r["cond_total"]}</span>
</div>
<div style='height:5px;background:#1e3a5f;border-radius:99px;overflow:hidden;'>
<div style='height:100%;width:{cond_bar}%;background:{badge_c};border-radius:99px;'></div>
</div>
</div>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-bottom:8px;'>
<div style='font-size:11px;color:#5a7299;'>TT <span style='color:#d8e8ff;font-weight:700;'>{r["tt"]}/8</span></div>
<div style='font-size:11px;color:#5a7299;'>RS <span style='color:#d8e8ff;font-weight:700;'>{r["rs"]}</span></div>
<div style='font-size:11px;color:#5a7299;'>RSI <span style='color:#d8e8ff;font-weight:700;'>{r["rsi"]:.0f}</span></div>
<div style='font-size:11px;color:#5a7299;'>ADX <span style='color:#d8e8ff;font-weight:700;'>{r["adx"]:.0f}</span></div>
<div style='font-size:11px;color:#5a7299;'>Vol <span style='color:#d8e8ff;font-weight:700;'>{r["vol_r"]:.2f}x</span></div>
<div style='font-size:11px;color:{wc};'>{ws_} <span style='font-weight:700;'>{abs(r["weekly"]):.1f}%</span></div>
</div>
<div style='border-top:1px solid rgba(255,255,255,0.06);padding-top:8px;'>
<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
<span style='font-size:10px;color:#5a7299;'>진입가</span>
<span style='font-size:11px;font-weight:700;color:#06b6d4;'>${r["entry"]:.2f}</span>
</div>
<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
<span style='font-size:10px;color:#5a7299;'>🛑 손절</span>
<span style='font-size:11px;font-weight:700;color:#ef4444;'>${r["stop"]:.2f} ({r["stop_pct"]:.1f}%)</span>
</div>
<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
<span style='font-size:10px;color:#5a7299;'>✅ +18% 50%청산</span>
<span style='font-size:11px;color:#22c55e;'>${r["tp1"]:.2f} ({r["tp1_pct"]:+.1f}%)</span>
</div>
<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
<span style='font-size:10px;color:#5a7299;'>🔄 잔량트레일링</span>
<span style='font-size:11px;color:#06b6d4;font-weight:700;'>고점 -{r["trail_pct"]*100:.0f}% 추적</span>
</div>
<div style='display:flex;justify-content:space-between;border-top:1px solid rgba(255,255,255,0.06);padding-top:6px;'>
<span style='font-size:10px;color:#5a7299;'>가중 R:R</span>
<span style='font-size:13px;font-weight:900;color:{"#22c55e" if r["weighted_rr"]>=2 else "#f59e0b"};'>{r["weighted_rr"]:.1f}:1</span>
</div>
</div>
</div>""", unsafe_allow_html=True)

            for ti,g in enumerate(GRADE_ORDER):
                with tabs_grades[ti]:
                    items_g=[r for r in results if r["grade"]==g]
                    if items_g:
                        st.markdown(f"<div style='font-size:12px;color:#5a7299;margin-bottom:10px;'><b style='color:{GRADE_COLOR[g]};'>{g}등급</b> {len(items_g)}종목</div>",unsafe_allow_html=True)
                    render_cards(items_g)

            with tabs_grades[-1]:
                table_rows=[{
                    "티커":r["ticker"],"등급":r["grade"],"점수":r["score"],
                    "추천":r["rec_label"],"v2조건":f"{r['cond_passed']}/{r['cond_total']}",
                    "Stage":r["stage"],"TT":f"{r['tt']}/8","RS":r["rs"],
                    "RSI":f"{r['rsi']:.0f}","ADX":f"{r['adx']:.0f}","Vol":f"{r['vol_r']:.2f}x",
                    "VCP":"🚀" if r["vcp"] else("⚡" if r["vcp_det"] else "—"),
                    "진입가":f"${r['entry']:.2f}","손절가":f"${r['stop']:.2f}({r['stop_pct']:.1f}%)",
                    "+18%청산":f"${r['tp1']:.2f}({r['tp1_pct']:+.1f}%)",
                    "트레일링":f"-{r['trail_pct']*100:.0f}%","R:R":f"{r['weighted_rr']:.1f}",
                } for r in results]
                df_table=pd.DataFrame(table_rows)
                st.dataframe(df_table,use_container_width=True,hide_index=True,
                    column_config={"점수":st.column_config.ProgressColumn("점수",min_value=0,max_value=100,format="%d")})
                leaders=[r for r in results if r["stage"]==2 and r["rec_color"] in ("cyan","green")]
                if leaders:
                    st.success(f"🌟 v2 조건 통과 Stage2: {', '.join(r['ticker'] for r in leaders[:10])}")


# ══════════════════════════════════════════════════════════════════
# 탭 3 — 만점 티커 찾기
# ══════════════════════════════════════════════════════════════════
with _tabs[2]:
    st.markdown("""
<div style='margin-bottom:16px;'>
<div style='font-size:16px;font-weight:900;color:#06b6d4;margin-bottom:4px;'>🎯 만점 티커 찾기 — v2 9/9 + TT 완전 충족 종목</div>
<div style='font-size:12px;color:#5a7299;line-height:1.6;'>
<b style='color:#d8e8ff;'>9/9 조건 충족 + TT 기준 이상 + Stage 2</b> 동시 충족 종목만 추출합니다.<br>
백테스트에서 가장 높은 수익을 낸 조건을 모두 충족하는 종목입니다.
</div>
</div>""", unsafe_allow_html=True)

    with st.spinner("SPY 시장 국면 확인 중…"):
        try:
            spy_res=build_signal("SPY","스윙","스윙 (1D)")
            spy_bias=spy_res[0].bias if spy_res else "알수없음"
        except: spy_bias="알수없음"

    spy_c={"상승장":"#22c55e","횡보장":"#f59e0b","하락장":"#ef4444"}.get(spy_bias,"#5a7299")
    spy_warn=spy_bias!="상승장"
    st.markdown(f"""
<div style='background:{"rgba(239,68,68,0.08)" if spy_warn else "rgba(34,197,94,0.07)"};
border:1px solid {"rgba(239,68,68,0.3)" if spy_warn else "rgba(34,197,94,0.2)"};
border-radius:12px;padding:12px 16px;margin-bottom:16px;display:flex;align-items:center;gap:12px;'>
<div style='font-size:24px;'>{"🔴" if spy_bias=="하락장" else "🟡" if spy_bias=="횡보장" else "🟢"}</div>
<div>
<div style='font-size:14px;font-weight:800;color:{spy_c};'>SPY {spy_bias} — {"만점 티커라도 신중하게" if spy_warn else "최적 매수 환경"}</div>
<div style='font-size:11px;color:#5a7299;margin-top:2px;'>{"약세/횡보장: 조건 충족 종목도 손실 가능 · 소량 또는 관망" if spy_warn else "강세장 + 9/9 조건 = v2 백테스트 최고 성과 구간"}</div>
</div>
</div>""", unsafe_allow_html=True)

    PERFECT_PRESETS={
        "🔥 AI·반도체":    "NVDA,AMD,AVGO,QCOM,ARM,MRVL,TSM,AMAT,LRCX,KLAC,MU,SMCI,ASML,INTC,MCHP,MPWR,ON",
        "💻 빅테크":        "AAPL,MSFT,GOOG,AMZN,META,NFLX,CRM,ORCL,ADBE,IBM,NOW,INTU,WDAY,TEAM",
        "⚡ 에너지·원전":   "SMR,NNE,CEG,VST,ETR,NRG,IREN,FSLR,ENPH,CLNE,NEE,AES,PCG,EXC",
        "🚀 고성장주":      "PLTR,AXON,CRWD,ZS,DDOG,SNOW,NET,ABNB,UBER,MSTR,AFRM,BILL,TTD",
        "💊 헬스케어":      "LLY,NVO,ABBV,JNJ,UNH,MRNA,PFE,AMGN,GILD,REGN,ISRG,BSX,DXCM",
        "🏦 금융·핀테크":   "JPM,BAC,GS,MS,BLK,V,MA,PYPL,SQ,COIN,HOOD,SOFI,NU",
        "🛡️ 사이버보안":    "CRWD,ZS,PANW,FTNT,S,OKTA,CYBR,QLYS,TENB,DDOG",
        "🤖 AI 소프트웨어": "PLTR,AI,PATH,BBAI,SOUN,GTLB,MNDY,ASAN,TTD,CELH",
        "🎮 소비·엔터":     "TSLA,DIS,ABNB,BKNG,CMG,MCD,SBUX,NKE,LULU,RBLX,SPOT",
        "📦 전체 유니버스":  "NVDA,AMD,AVGO,QCOM,ARM,AMAT,LRCX,KLAC,MU,AAPL,MSFT,GOOG,AMZN,META,NFLX,CRM,ADBE,PLTR,CRWD,AXON,NET,DDOG,SNOW,UBER,ABNB,TSLA,SMR,LLY,V,MA,JPM,GS,PANW,FTNT,ZS,TTD,MSTR,COIN,NOW,INTU",
    }

    pf_c1,pf_c2,pf_c3=st.columns([2,1.2,1])
    with pf_c1: pf_preset=st.selectbox("📂 스캔 대상",list(PERFECT_PRESETS.keys()),key="pf_preset")
    with pf_c2: pf_tf=st.selectbox("타임프레임",list(TF_OPTIONS.keys()),key="pf_tf")
    with pf_c3: pf_min_tt=st.selectbox("TT 최소",["6/8 이상","7/8 이상","8/8 만점"],key="pf_tt")

    pf_custom=st.text_input("✏️ 직접 입력",value="",placeholder="AAPL,NVDA,TSLA…",key="pf_custom")
    pf_tickers=(
        [t.strip().upper() for t in pf_custom.replace("\n",",").split(",") if t.strip()]
        if pf_custom.strip()
        else [t.strip().upper() for t in PERFECT_PRESETS[pf_preset].split(",") if t.strip()]
    )
    pf_tickers=list(dict.fromkeys(pf_tickers))[:80]
    pf_min_tt_num={"6/8 이상":6,"7/8 이상":7,"8/8 만점":8}[pf_min_tt]

    pb1,pb2=st.columns([1,4])
    with pb1: pf_run=st.button("🎯 만점 티커 찾기",use_container_width=True,key="pf_run")
    with pb2: st.markdown(f"<div style='padding:9px 0;font-size:12px;color:#5a7299;'>스캔: <b style='color:#d8e8ff;'>{len(pf_tickers)}종목</b> → <b style='color:#06b6d4;'>9/9 + TT {pf_min_tt} + Stage2 동시 충족만 추출</b></div>",unsafe_allow_html=True)

    if pf_run:
        perfect=[]; near=[]
        pf_bar=st.progress(0); pf_stat=st.empty()
        for i,tk in enumerate(pf_tickers):
            pf_bar.progress((i+1)/len(pf_tickers))
            pf_stat.markdown(f"<div style='font-size:12px;color:#5a7299;'>스캔 [{i+1}/{len(pf_tickers)}]: <b style='color:#06b6d4;'>{tk}</b></div>",unsafe_allow_html=True)
            try:
                res=build_signal(tk,"스윙",pf_tf)
                if not res: continue
                s,d=res; tp2=s.trade_plan
                v2c=s.entry_conds
                rc,rl,rd,pc,tc=get_recommendation(v2c)
                tt_passed=s.sc.trend_template["passed"]
                stage=s.sc.weinstein["stage"]
                row={
                    "ticker":tk,"score":s.sc.total,"grade":s.sc.grade,
                    "cond_passed":pc,"cond_total":tc,"tt_passed":tt_passed,"stage":stage,
                    "tt_conds":s.sc.trend_template["conditions"],"perf12m":s.sc.trend_template["perf12m"],
                    "rs":s.sc.rs["rs_score"],"rsi":s.sc.rsi_val,"adx":s.sc.adx_val,"vol_r":s.sc.vol_ratio,
                    "vcp":s.sc.vcp["breakout_ready"],"vcp_det":s.sc.vcp["detected"],
                    "golden":s.sc.golden_cross,"squeeze":s.sc.bb_squeeze,
                    "bias":s.bias,"weekly":s.weekly_perf,
                    "entry":tp2.entry_price,"stop":tp2.stop_price,"stop_pct":tp2.stop_pct,
                    "tp1":tp2.tp1_price,"tp1_pct":tp2.tp1_pct,
                    "trail_pct":tp2.trail_pct,"be_trigger":tp2.be_trigger,"be_stop":tp2.be_stop,
                    "weighted_rr":tp2.weighted_rr,"rec_label":rl,
                }
                if pc==tc and tt_passed>=pf_min_tt_num and stage==2:
                    perfect.append(row)
                elif pc>=tc-1 and tt_passed>=pf_min_tt_num-1 and stage==2:
                    near.append(row)
            except Exception: continue
        pf_bar.empty(); pf_stat.empty()

        st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(6,182,212,0.1),rgba(34,197,94,0.05));border:1px solid rgba(6,182,212,0.3);border-radius:14px;padding:16px 20px;margin-bottom:16px;display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;text-align:center;'>
<div><div style='font-size:11px;color:#5a7299;'>스캔 종목</div><div style='font-size:28px;font-weight:900;color:#d8e8ff;'>{len(pf_tickers)}</div></div>
<div style='border-left:1px solid #1e3a5f;'><div style='font-size:11px;color:#5a7299;'>🔥 만점 티커</div><div style='font-size:28px;font-weight:900;color:#06b6d4;'>{len(perfect)}</div></div>
<div style='border-left:1px solid #1e3a5f;'><div style='font-size:11px;color:#5a7299;'>⚡ 아깝게 탈락</div><div style='font-size:28px;font-weight:900;color:#f59e0b;'>{len(near)}</div></div>
<div style='border-left:1px solid #1e3a5f;'><div style='font-size:11px;color:#5a7299;'>TT 기준</div><div style='font-size:20px;font-weight:900;color:#a855f7;'>{pf_min_tt}</div></div>
</div>""", unsafe_allow_html=True)

        if not perfect and not near:
            st.warning(f"9/9 + TT {pf_min_tt} + Stage2 동시 충족 종목이 없습니다. TT 기준을 낮추거나 다른 프리셋을 시도해보세요.")
        else:
            if perfect:
                perfect.sort(key=lambda x:(x["tt_passed"],x["score"]),reverse=True)
                st.markdown(f"""
<div style='background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.3);border-radius:12px;padding:12px 16px;margin-bottom:12px;'>
<div style='font-size:13px;font-weight:800;color:#06b6d4;'>🔥 만점 티커 {len(perfect)}종목 — v2 9/9 + TT {pf_min_tt} + Stage2 전부 충족</div>
<div style='font-size:11px;color:#5a7299;margin-top:3px;'>{"⚠️ SPY "+spy_bias+" — 신중 접근 권장" if spy_warn else "✅ SPY 상승장 + 만점 조건 = v2 백테스트 최고 성과 구간"}</div>
</div>""", unsafe_allow_html=True)

                pf_cols=st.columns(min(3,len(perfect)))
                for i,r in enumerate(perfect):
                    with pf_cols[i%3]:
                        wc="#22c55e" if r["weekly"]>=0 else "#ef4444"
                        ws_="▲" if r["weekly"]>=0 else "▼"
                        wrrc="#22c55e" if r["weighted_rr"]>=2 else "#f59e0b"
                        tt_cond_html="".join([
                            f"<div style='font-size:10px;color:{'#22c55e' if ok else '#ef4444'};padding:1px 0;'>{'✓' if ok else '✗'} {lbl}</div>"
                            for lbl,ok in r["tt_conds"]
                        ])
                        vcp_b="🚀VCP준비" if r["vcp"] else("⚡VCP" if r["vcp_det"] else "")
                        gld_b="🌟골든크" if r["golden"] else ""
                        sq_b="⚡BB수축" if r["squeeze"] else ""
                        badges=" ".join(b for b in [vcp_b,gld_b,sq_b] if b)
                        st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(6,182,212,0.12),rgba(34,197,94,0.08));border:2px solid rgba(6,182,212,0.5);border-radius:16px;padding:16px;margin-bottom:14px;'>
<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;'>
<div>
<div style='font-size:24px;font-weight:900;color:#06b6d4;'>{r["ticker"]}</div>
<div style='font-size:11px;color:#22c55e;font-weight:700;margin-top:2px;'>Stage2🚀 · {r["grade"]}등급 {r["score"]}점</div>
{f"<div style='font-size:10px;color:#5a7299;margin-top:2px;'>{badges}</div>" if badges else ""}
</div>
<div style='text-align:center;background:rgba(6,182,212,0.15);border:1px solid rgba(6,182,212,0.4);border-radius:10px;padding:6px 12px;'>
<div style='font-size:10px;color:#5a7299;'>v2 조건</div>
<div style='font-size:20px;font-weight:900;color:#06b6d4;'>9/9</div>
</div>
</div>
<div style='background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.3);border-radius:8px;padding:6px;text-align:center;margin-bottom:10px;'>
<div style='font-size:14px;font-weight:900;color:#06b6d4;'>🔥 매수 강력 추천</div>
</div>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:10px;margin-bottom:10px;'>
<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
<div style='font-size:20px;font-weight:900;color:#06b6d4;'>TT {r["tt_passed"]}/8</div>
<div>
<div style='font-size:11px;font-weight:700;color:#22c55e;'>✅ Stage 2 확인</div>
<div style='font-size:10px;color:#5a7299;'>12M: {r["perf12m"]:+.1f}%</div>
</div>
</div>
{tt_cond_html}
</div>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:10px;'>
<div style='background:rgba(255,255,255,0.03);border-radius:7px;padding:5px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>RS</div>
<div style='font-size:14px;font-weight:700;color:{"#22c55e" if r["rs"]>=80 else "#f59e0b"};'>{r["rs"]}</div>
</div>
<div style='background:rgba(255,255,255,0.03);border-radius:7px;padding:5px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>RSI</div>
<div style='font-size:14px;font-weight:700;color:#d8e8ff;'>{r["rsi"]:.0f}</div>
</div>
<div style='background:rgba(255,255,255,0.03);border-radius:7px;padding:5px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>ADX</div>
<div style='font-size:14px;font-weight:700;color:{"#22c55e" if r["adx"]>=25 else "#f59e0b"};'>{r["adx"]:.0f}</div>
</div>
</div>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;'>
<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:7px;padding:6px;text-align:center;'>
<div style='font-size:9px;color:#5a7299;'>🛑 손절</div>
<div style='font-size:12px;font-weight:700;color:#ef4444;'>${r["stop"]:.2f}</div>
<div style='font-size:9px;color:#ef4444;'>{r["stop_pct"]:.1f}%</div>
</div>
<div style='background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);border-radius:7px;padding:6px;text-align:center;'>
<div style='font-size:9px;color:#5a7299;'>✅ +18% 50%</div>
<div style='font-size:12px;font-weight:700;color:#22c55e;'>${r["tp1"]:.2f}</div>
<div style='font-size:9px;color:#22c55e;'>{r["tp1_pct"]:+.1f}%</div>
</div>
<div style='background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.25);border-radius:7px;padding:6px;text-align:center;'>
<div style='font-size:9px;color:#5a7299;'>🔄 트레일링</div>
<div style='font-size:12px;font-weight:700;color:#06b6d4;'>-{r["trail_pct"]*100:.0f}%</div>
<div style='font-size:9px;color:#06b6d4;'>고점추적</div>
</div>
</div>
<div style='display:flex;justify-content:space-between;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06);'>
<div style='font-size:10px;color:#5a7299;'>진입 <b style='color:#06b6d4;'>${r["entry"]:.2f}</b></div>
<div style='font-size:10px;color:#5a7299;'>R:R <b style='color:{wrrc};'>{r["weighted_rr"]:.1f}:1</b></div>
<div style='font-size:10px;color:{wc};'>{ws_} {abs(r["weekly"]):.1f}%</div>
</div>
</div>""", unsafe_allow_html=True)

            if near:
                near.sort(key=lambda x:(x["cond_passed"],x["tt_passed"],x["score"]),reverse=True)
                with st.expander(f"⚡ 아깝게 탈락 {len(near)}종목 — 조건 1~2개 미달"):
                    nr_cols=st.columns(3)
                    for i,r in enumerate(near):
                        with nr_cols[i%3]:
                            st.markdown(f"""
<div style='background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.3);border-radius:12px;padding:12px;margin-bottom:10px;'>
<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
<div style='font-size:18px;font-weight:900;color:#f59e0b;'>{r["ticker"]}</div>
<div style='text-align:right;'>
<div style='font-size:11px;font-weight:700;color:#f59e0b;'>⚡ 거의 완벽</div>
<div style='font-size:10px;color:#5a7299;'>TT {r["tt_passed"]}/8 · {r["cond_passed"]}/{r["cond_total"]}</div>
</div>
</div>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px;'>
<div style='text-align:center;font-size:10px;'><div style='color:#5a7299;'>손절</div><div style='color:#ef4444;font-weight:700;'>${r["stop"]:.2f}</div></div>
<div style='text-align:center;font-size:10px;'><div style='color:#5a7299;'>+18%50%</div><div style='color:#22c55e;font-weight:700;'>{r["tp1_pct"]:+.1f}%</div></div>
<div style='text-align:center;font-size:10px;'><div style='color:#5a7299;'>트레일</div><div style='color:#06b6d4;font-weight:700;'>-{r["trail_pct"]*100:.0f}%</div></div>
</div>
</div>""", unsafe_allow_html=True)


st.markdown("""
<div style='text-align:center;padding:20px 0 8px;color:#5a7299;font-size:11px;'>
⚠️ 투자 조언 아닙니다 · v2 백테스트 검증 전략 · 모든 투자 결정은 본인 책임<br>
Minervini(SEPA/VCP) · Weinstein(Stage) · O'Neil(CANSLIM) · P.T.Jones(200MA) · Livermore(Pivot)
</div>""", unsafe_allow_html=True)
