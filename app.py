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
from core.data import fetch_ticker_info, fetch_realtime_price, clear_all_cache
from core.indicators import bbands, macd, rsi, sma, stochastic, vwap
from core.signal import (
    build_signal, calc_entry_conditions, build_trade_plan,
    V2_STOP_ATR_MUL, V2_STOP_MAX_PCT, V2_BREAKEVEN_PCT,
    V2_BREAKEVEN_RAISE, V2_PARTIAL_PCT, V2_TRAIL_PCT,
    V2_MIN_SCORE_S, V2_MIN_SCORE_A, V2_MIN_RS, V2_MIN_ADX,
)

st.set_page_config(page_title="Trader_TM", page_icon="📈",
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
/* 탭5 보유종목 상세보기는 정상 표시 */
div[data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    padding: 10px 14px !important;
}
div[data-testid="stExpander"] summary:hover {
    background: rgba(6,182,212,0.05) !important;
}
footer,header,#MainMenu{display:none!important;}
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
    except Exception: return 50.0

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
        html += f"""<div style='display:grid;grid-template-columns:18px 1fr;gap:8px;align-items:start;padding:7px 0;border-bottom:1px solid rgba(30,58,95,0.5);'>
<div style='font-size:14px;'>{icon}</div>
<div>
<div style='font-size:11px;font-weight:700;color:#d8e8ff;'>{c.name}{imp_tag}</div>
<div style='font-size:11px;color:{col};font-weight:600;margin-top:1px;'>{c.current} — {"통과" if c.passed else c.required}</div>
<div style='font-size:10px;color:#5a7299;margin-top:2px;line-height:1.4;'>{c.meaning}</div>
</div>
</div>"""
    return html

def get_recommendation(conds):
    total       = len(conds)
    passed      = sum(1 for c in conds if c.passed)
    critical_ok = all(c.passed for c in conds if c.importance == "critical")
    if not critical_ok:
        fails = [c.name for c in conds if not c.passed and c.importance == "critical"]
        return "red", "🔴 진입 보류", f"핵심 조건 미충족: {' · '.join(fails[:2])}", passed, total
    if passed == total:
        return "cyan", "🔥 매수 강력 추천", "9/9 모든 조건 충족 — 진입하세요", passed, total
    if passed >= 7:
        return "green", "✅ 매수 적극 추천", f"{passed}/{total} 조건 충족 — v2 진입 기준 통과", passed, total
    if passed >= 5:
        return "yellow", "🟡 조건부 매수", f"{passed}/{total} 조건 충족 — 추가 확인 후 소량", passed, total
    return "red", "🔴 진입 보류", f"{passed}/{total} 조건 충족 — 조건 개선 대기", passed, total

PBG="rgba(0,0,0,0)"; GRD="#1a2840"; TC="#5a7299"

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
# 헤더
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class='app-header'>
<div>
<div class='app-title'>📈 Trader_TM</div>
<div class='app-sub'>
<span class='trader-badge tb-min'>Minervini</span>
<span class='trader-badge tb-wei'>Weinstein</span>
<span class='trader-badge tb-one'>O'Neil</span>
<span class='trader-badge tb-ptj'>P.T.Jones</span>
<span class='trader-badge tb-liv'>Livermore</span>
&nbsp;v2 백테스트 검증 전략 (PF 1.85 / 승률 46.1%)
</div>
</div>
<div style='text-align:right'>
<div style='font-size:11px;color:#5a7299;'>Yahoo Finance · 장중 15초 캐시 · 1분봉 실시간 패치 · prepost 포함</div>
<div style='font-size:11px;color:#5a7299;margin-top:2px;'></div>
</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 탭 4개
# ══════════════════════════════════════════════════════════════════
_tabs = st.tabs([
    "📈  차트 분석",
    "🔭  빠른 티커 스캔",
    "🎯  만점 티커 찾기",
    "📖  실전 매매 가이드",
    "💼  내 보유 종목 관리",
])

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
    with ic4: analyze_btn = st.button("🔍 분석",use_container_width=True)

    # ── 장중 자동 새로고침 + 강제 새로고침 ───────────────────────
    _market_open = True  # data.py 함수 직접 참조 안 되므로 근사
    try:
        import pytz, datetime as _dt
        _et = pytz.timezone("America/New_York")
        _now = _dt.datetime.now(_et)
        _t = _now.time()
        _market_open = (_now.weekday() < 5 and
                        _dt.time(9,25) <= _t <= _dt.time(16,5))
    except Exception: pass

    _side1, _side2, _side3 = st.columns([1.2, 1, 2])
    with _side1:
        if st.button("🔄 강제 새로고침", use_container_width=True, key="force_refresh"):
            clear_all_cache()
            st.rerun()
    with _side2:
        _auto = st.toggle("⚡ 자동갱신", value=False, key="auto_refresh",
                          help="장중 30초마다 자동 갱신 (장중에만 의미 있음)")
    with _side3:
        if _market_open:
            st.markdown("<div style='padding:6px 0;font-size:11px;color:#22c55e;'>🟢 장중 — 15초 캐시 · 1분봉 실시간 패치 적용 중</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:6px 0;font-size:11px;color:#5a7299;'>🔴 장외 — 10분 캐시 · 마지막 종가 기준</div>", unsafe_allow_html=True)

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

    # ── 실시간 현재가 오버레이 (fast_info) ──────────────────────
    _rt = fetch_realtime_price(ticker)
    if _rt and _rt.get("price"):
        _p   = _rt["price"]
        _chg = _rt["chg_pct"]
        _tc  = "#22c55e" if _chg >= 0 else "#ef4444"
        _ts  = "▲" if _chg >= 0 else "▼"
        _dh  = f"  고:{_rt['day_high']:.2f}" if _rt.get("day_high") else ""
        _dl  = f"  저:{_rt['day_low']:.2f}"  if _rt.get("day_low")  else ""
        st.markdown(f"""
<div style='background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.25);border-radius:10px;padding:8px 16px;margin-bottom:8px;display:flex;align-items:center;gap:16px;'>
<div style='font-size:11px;color:#5a7299;font-weight:600;'>⚡ 실시간 현재가 (fast_info)</div>
<div style='font-size:20px;font-weight:900;color:#06b6d4;'>${_p:,.2f}</div>
<div style='font-size:14px;font-weight:700;color:{_tc};'>{_ts} {abs(_chg):.2f}%</div>
<div style='font-size:11px;color:#5a7299;'>{_dh}{_dl}</div>
<div style='margin-left:auto;font-size:10px;color:#5a7299;'>{_rt.get("time","")}</div>
</div>""", unsafe_allow_html=True)

    # ── 자동갱신 (장중) — time.sleep 블로킹 제거, rerun 주기 타이머 방식 ─
    if _auto and _market_open:
        import time as _time
        # session_state로 마지막 갱신 시각 추적
        _now_ts = _time.time()
        if "last_refresh" not in st.session_state:
            st.session_state["last_refresh"] = _now_ts
        _elapsed = _now_ts - st.session_state.get("last_refresh", 0)
        if _elapsed >= 30:
            st.session_state["last_refresh"] = _now_ts
            st.rerun()
        else:
            _remaining = int(30 - _elapsed)
            st.caption(f"⚡ 자동갱신 대기 중... {_remaining}초 후 갱신")

    if sig.vix_text:
        vv=sig.vix or 0
        cls="vix-bad" if vv>=28 else("vix-warn" if vv>=22 else "vix-ok")
        st.markdown(f"<div class='vix-banner {cls}'>{sig.vix_text}</div>",unsafe_allow_html=True)

    conds=sig.entry_conds
    rec_color,rec_label,rec_desc,passed_cnt,total_cnt=get_recommendation(conds)
    pct_bar=int(passed_cnt/total_cnt*100)
    bar_c={"cyan":"#06b6d4","green":"#22c55e","yellow":"#f59e0b","red":"#ef4444"}.get(rec_color,"#5a7299")
    tp=sig.trade_plan

    # ── 매매 가이드 패널 ─────────────────────────────────────────
    with st.container():  # 매매 가이드 패널 (항상 표시)

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
<div style='margin-bottom:4px;'>
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

        g1,g2,g3=st.columns(3)
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
            if sc.vcp["breakout_ready"]: timing_txt="🚀 지금 진입 — VCP 돌파 확인"
            elif sc.vcp["detected"] or sc.bb_squeeze: timing_txt="⏳ 돌파 대기 — VCP/수축 완성 중"
            elif passed_cnt >= 7: timing_txt="📈 눌림목 확인 후 분할 진입"
            else: timing_txt="👀 조건 개선 대기"
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

        st.markdown("""
<div style='background:rgba(255,255,255,0.02);border:1px solid #1e3a5f;border-radius:12px;padding:14px 14px 8px;margin-top:10px;'>
<div style='font-size:10px;font-weight:700;color:#5a7299;letter-spacing:0.12em;margin-bottom:8px;'>
📋 v2 백테스트 9개 진입 조건 체크리스트
</div>
</div>""", unsafe_allow_html=True)
        st.markdown(build_checklist_html(conds), unsafe_allow_html=True)

        # 손절 / 1차익절 / 트레일링
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
<div style='font-size:10px;color:#5a7299;margin-bottom:4px;font-weight:700;'>🔄 잔량 50% — 트레일링 스톱 ({sc.grade}등급 -{tp.trail_pct*100:.0f}%)</div>
<div style='font-size:14px;font-weight:800;color:#06b6d4;margin-bottom:4px;'>매일 저녁 손절가를 직접 올리세요</div>
<div style='background:#1e3a5f;border-radius:99px;height:5px;margin:6px 0 8px;overflow:hidden;'>
<div style='width:100%;height:100%;background:#06b6d4;border-radius:99px;'></div>
</div>
<div style='background:rgba(0,0,0,0.2);border-radius:8px;padding:8px 10px;'>
<div style='font-size:10px;font-weight:700;color:#06b6d4;margin-bottom:4px;'>📌 실전 방법 (매일 반복)</div>
<div style='font-size:10px;color:#d8e8ff;line-height:1.8;'>
<span style='color:#f59e0b;font-weight:700;'>①</span> 장 마감 후 오늘 고점 확인<br>
<span style='color:#f59e0b;font-weight:700;'>②</span> 새 손절가 = 고점 × {(1-tp.trail_pct):.2f}<br>
<span style='color:#f59e0b;font-weight:700;'>③</span> 어제보다 높으면 → 증권사 앱 손절가 변경<br>
<span style='color:#f59e0b;font-weight:700;'>④</span> 주가가 손절가 하회 → 즉시 전량 매도
</div>
</div>
<div style='margin-top:6px;background:rgba(6,182,212,0.07);border-radius:7px;padding:6px 8px;font-size:10px;color:#5a7299;'>
💡 예: 진입 $200, 고점 $220 → 손절 ${200*(1+tp.trail_pct):.0f}×{(1-tp.trail_pct):.2f}=${220*(1-tp.trail_pct):.1f}
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



    # ══════════════════════════════════════════════════════════════
    # 📊 내 매매 관리 — 실시간 현재가 반영 매매 가이드
    # ══════════════════════════════════════════════════════════════

    # ── 종목 바뀔 때 매매관리 입력값 초기화 ─────────────────────
    # session_state에 이전 종목 저장 → 다르면 관련 키 전부 삭제
    _prev_ticker_key = "mgmt_ticker"
    if st.session_state.get(_prev_ticker_key) != ticker:
        # 종목이 바뀌었으면 입력값 초기화
        for _k in [f"trail_entry_{st.session_state.get(_prev_ticker_key,'')}", 
                   f"trail_high_{st.session_state.get(_prev_ticker_key,'')}",
                   f"trail_grade_{st.session_state.get(_prev_ticker_key,'')}",
                   f"partial_done_{st.session_state.get(_prev_ticker_key,'')}",
                   f"holding_shares_{st.session_state.get(_prev_ticker_key,'')}"]:
            if _k in st.session_state:
                del st.session_state[_k]
        st.session_state[_prev_ticker_key] = ticker

    # 실시간 현재가 fetch (fast_info 10초 캐시)
    _rt_price_data = fetch_realtime_price(ticker)
    _current_price = _rt_price_data.get("price", last_price) if _rt_price_data else last_price
    _current_time  = _rt_price_data.get("time", sig.asof) if _rt_price_data else sig.asof

    st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(6,182,212,0.08),rgba(0,0,0,0));
border:1px solid rgba(6,182,212,0.35);border-radius:16px;padding:16px 20px;margin-bottom:14px;'>
<div style='display:flex;align-items:center;justify-content:space-between;'>
<div>
<div style='font-size:15px;font-weight:900;color:#06b6d4;'>📊 {ticker} — 내 매매 관리</div>
<div style='font-size:11px;color:#5a7299;margin-top:3px;'>
매수가 입력 → v2 백테스트 전략대로 1차 익절·트레일링·손절가 실시간 자동 계산
</div>
</div>
<div style='text-align:right;'>
<div style='font-size:11px;color:#5a7299;'>⚡ 실시간 현재가</div>
<div style='font-size:20px;font-weight:900;color:#06b6d4;'>${_current_price:,.2f}</div>
<div style='font-size:10px;color:#5a7299;'>{_current_time}</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ── 입력 섹션 ────────────────────────────────────────────────
    _mg1, _mg2, _mg3, _mg4, _mg5 = st.columns([1.3, 1.1, 1, 1, 1.1])
    with _mg1:
        _my_entry = st.number_input(
            "💵 내 매수가 ($)", value=float(round(_current_price, 2)),
            step=0.5, min_value=0.01, key=f"trail_entry_{ticker}",
            help="실제로 매수한 가격을 입력하세요"
        )
    with _mg2:
        # 오늘 고점 기본값: 실시간 day_high or 현재가
        _day_high_default = _rt_price_data.get("day_high", _current_price) if _rt_price_data else _current_price
        _day_high_default = max(_day_high_default or _current_price, _current_price)
        _my_high = st.number_input(
            "📈 오늘 고점 ($)", value=float(round(_day_high_default, 2)),
            step=0.5, min_value=0.01, key=f"trail_high_{ticker}",
            help="오늘 장중 최고가. 장중이면 자동으로 day_high 반영"
        )
    with _mg3:
        _trail_grade = st.selectbox(
            "등급", ["SS/SSS (8%)", "S (10%)", "A (12%)", "1차익절후 (7%)"],
            index={"SSS":0,"SS":0,"S":1,"A":2}.get(sc.grade, 1),
            key=f"trail_grade_{ticker}",
            help="종목 등급에 맞는 트레일링 비율 (분석 결과 자동 선택됨)"
        )
    with _mg4:
        _partial_done = st.checkbox(
            "1차익절(+18%) 완료", value=False, key=f"partial_done_{ticker}",
            help="체크 시 트레일링 7%로 자동 전환"
        )
    with _mg5:
        _holding_shares = st.number_input(
            "보유 수량 (주)", value=10, step=1, min_value=1, key=f"holding_shares_{ticker}",
            help="보유 수량 입력 시 예상 수익금 계산"
        )

    # 트레일링 % 결정
    if _partial_done:
        _trail_pct_val = 0.07
        _trail_label   = "7% (1차 익절 후)"
    else:
        _trail_map = {"SS/SSS (8%)":0.08, "S (10%)":0.10, "A (12%)":0.12, "1차익절후 (7%)":0.07}
        _trail_pct_val = _trail_map.get(_trail_grade, 0.10)
        _trail_label   = _trail_grade

    if _my_entry > 0:
        # 핵심 가격 계산 (v2 백테스트 그대로)
        _new_stop     = round(_my_high * (1 - _trail_pct_val), 2)
        _be_trigger   = round(_my_entry * 1.10, 2)   # +10% 브레이크이븐 트리거
        _be_stop      = round(_my_entry * 1.01, 2)   # 브레이크이븐 후 손절가
        _tp1_target   = round(_my_entry * 1.18, 2)   # +18% 1차 익절
        _tp1_after_trail = round(_tp1_target * (1 - 0.07), 2)  # 1차 익절 후 트레일 7% 손절

        # 현재가 기준 손익
        _gain_current = (_current_price / _my_entry - 1) * 100
        _gain_high    = (_my_high / _my_entry - 1) * 100
        _stop_from_high = (_new_stop / _my_high - 1) * 100

        # 현재 매매 단계 판정 (v2 백테스트 단계별 행동)
        if _my_high >= _tp1_target and _partial_done:
            _phase = "phase_after_tp1"
        elif _my_high >= _tp1_target:
            _phase = "phase_tp1_reached"
        elif _my_high >= _be_trigger:
            _phase = "phase_breakeven"
        elif _gain_current >= 0:
            _phase = "phase_holding"
        elif _current_price <= _new_stop:
            _phase = "phase_stop_hit"
        else:
            _phase = "phase_loss"

        # ── 단계별 색상 & 메시지 ────────────────────────────────
        _phase_cfg = {
            "phase_after_tp1":    ("#06b6d4", "🔄 잔량 트레일링 진행 중",   "1차 익절 완료 → 잔량 50% 트레일링 7% 적용 중"),
            "phase_tp1_reached":  ("#22c55e", "🎯 1차 익절 구간 도달!",     f"+18% 달성 → 지금 바로 보유량 50% 매도하세요"),
            "phase_breakeven":    ("#06b6d4", "✅ 브레이크이븐 구간",        f"+10% 달성 → 손절가를 ${_be_stop:.2f}로 상향하세요"),
            "phase_holding":      ("#22c55e", "📈 수익 보유 중",             f"+{_gain_current:.1f}% · {_be_trigger:.2f} 도달 시 브레이크이븐 전환"),
            "phase_stop_hit":     ("#ef4444", "🛑 손절가 도달!",             "지금 즉시 전량 매도하세요. 이유 불문"),
            "phase_loss":         ("#f59e0b", "📉 매수가 이하",              f"손절가 ${_new_stop:.2f} 이하 시 즉시 매도"),
        }
        _pc, _pt, _pd = _phase_cfg[_phase]

        # 예상 손익
        _profit_at_stop = (_new_stop - _my_entry) * _holding_shares
        _profit_at_tp1  = (_tp1_target - _my_entry) * (_holding_shares // 2)
        _profit_current = (_current_price - _my_entry) * _holding_shares

        # ── 현황 카드 5개 ─────────────────────────────────────────
        st.markdown(f"""
<div style='background:rgba({("239,68,68" if _phase=="phase_stop_hit" else "0,0,0")},0.2);border:1px solid {"rgba(239,68,68,0.4)" if _phase=="phase_stop_hit" else "#1e3a5f"};border-radius:14px;padding:14px;margin-bottom:10px;'>
<div style='display:flex;align-items:center;gap:12px;margin-bottom:12px;'>
<div style='font-size:16px;font-weight:900;color:{_pc};'>{_pt}</div>
<div style='font-size:11px;color:#5a7299;'>{_pd}</div>
{"<div style='margin-left:auto;background:rgba(239,68,68,0.2);border:1px solid rgba(239,68,68,0.5);border-radius:8px;padding:5px 12px;font-size:12px;font-weight:800;color:#ef4444;'>즉시 매도!</div>" if _phase=="phase_stop_hit" else ""}
</div>
<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:8px;'>
<div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:3px;'>🛑 손절가</div>
<div style='font-size:18px;font-weight:900;color:#ef4444;'>${_new_stop:.2f}</div>
<div style='font-size:10px;color:#ef4444;'>고점 -{int(_trail_pct_val*100)}%</div>
<div style='font-size:9px;color:#5a7299;margin-top:2px;'>= ${_my_high:.2f}×{1-_trail_pct_val:.2f}</div>
</div>
<div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:3px;'>⚡ 브레이크이븐</div>
<div style='font-size:18px;font-weight:900;color:{"#06b6d4" if _current_price>=_be_trigger else "#f59e0b"};'>${_be_trigger:.2f}</div>
<div style='font-size:10px;color:#f59e0b;'>{"✅ 달성!" if _current_price>=_be_trigger else "+10% 목표"}</div>
<div style='font-size:9px;color:#5a7299;margin-top:2px;'>→ 손절 ${_be_stop:.2f}</div>
</div>
<div style='background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:3px;'>✅ 1차 익절</div>
<div style='font-size:18px;font-weight:900;color:{"#06b6d4" if _current_price>=_tp1_target else "#22c55e"};'>${_tp1_target:.2f}</div>
<div style='font-size:10px;color:#22c55e;'>{"✅ 달성!" if _current_price>=_tp1_target else "+18% 50%청산"}</div>
<div style='font-size:9px;color:#5a7299;margin-top:2px;'>이후 트레일 7%</div>
</div>
<div style='background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.25);border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:3px;'>📊 현재 손익</div>
<div style='font-size:18px;font-weight:900;color:{"#22c55e" if _gain_current>=0 else "#ef4444"};'>{_gain_current:+.1f}%</div>
<div style='font-size:10px;color:#5a7299;'>현재가 ${_current_price:.2f}</div>
<div style='font-size:9px;color:{"#22c55e" if _profit_current>=0 else "#ef4444"};margin-top:2px;'>{"+" if _profit_current>=0 else ""}${abs(_profit_current):,.0f}</div>
</div>
<div style='background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.25);border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:3px;'>💰 트레일 청산시</div>
<div style='font-size:18px;font-weight:900;color:{"#22c55e" if _profit_at_stop>=0 else "#ef4444"};'>{"+" if _profit_at_stop>=0 else ""}${abs(_profit_at_stop):,.0f}</div>
<div style='font-size:10px;color:#5a7299;'>{_holding_shares}주 기준</div>
<div style='font-size:9px;color:#5a7299;margin-top:2px;'>손절 ${_new_stop:.2f}에서</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

        # ── 단계별 상세 행동 가이드라인 ──────────────────────────
        if _phase == "phase_stop_hit":
            st.markdown(f"""
<div style='background:rgba(239,68,68,0.1);border:2px solid rgba(239,68,68,0.5);border-radius:12px;padding:14px 16px;margin-bottom:8px;'>
<div style='font-size:13px;font-weight:900;color:#ef4444;margin-bottom:10px;'>🚨 손절가 도달 — 지금 당장 매도해야 합니다</div>
<div style='font-size:11px;color:#d8e8ff;line-height:2.0;'>
<span style='color:#ef4444;font-weight:800;'>①</span> 증권사 앱 열기 → <b>{ticker}</b> 검색 → <b style='color:#ef4444;'>전량 시장가 매도</b><br>
<span style='color:#ef4444;font-weight:800;'>②</span> "조금만 더 기다리면 오르겠지" → <b style='color:#ef4444;'>절대 금지</b>. 이게 가장 큰 손실 원인<br>
<span style='color:#ef4444;font-weight:800;'>③</span> 손절 후 → 같은 종목 즉시 재진입 금지 (최소 3일 대기)<br>
<span style='color:#f59e0b;font-weight:800;'>④</span> 3연속 손절 시 → 1주일 완전 휴식 (v2 백테스트 쿨다운 규칙)
</div>
</div>""", unsafe_allow_html=True)

        elif _phase == "phase_tp1_reached":
            st.markdown(f"""
<div style='background:rgba(34,197,94,0.08);border:2px solid rgba(34,197,94,0.4);border-radius:12px;padding:14px 16px;margin-bottom:8px;'>
<div style='font-size:13px;font-weight:900;color:#22c55e;margin-bottom:10px;'>🎯 1차 익절 구간 — 지금 바로 절반 팔아야 합니다</div>
<div style='font-size:11px;color:#d8e8ff;line-height:2.0;'>
<span style='color:#22c55e;font-weight:800;'>①</span> 증권사 앱 → <b>{ticker}</b> → 보유 수량 <b>{_holding_shares}주</b> 중 <b style='color:#22c55e;'>{_holding_shares//2}주 지정가 ${_tp1_target:.2f} 매도</b><br>
<span style='color:#22c55e;font-weight:800;'>②</span> 매도 완료 후 → 손절가를 <b style='color:#f59e0b;'>진입가+1% (${_be_stop:.2f})</b>로 즉시 상향<br>
<span style='color:#06b6d4;font-weight:800;'>③</span> 잔량 <b>{_holding_shares - _holding_shares//2}주</b>는 트레일링 7%로 전환 → 아래 "1차익절 완료" 체크<br>
<span style='color:#5a7299;font-weight:800;'>④</span> 예상 1차 익절 수익: <b style='color:#22c55e;'>+${_profit_at_tp1:,.0f}</b> ({_holding_shares//2}주 × ${_tp1_target-_my_entry:.2f})
</div>
</div>""", unsafe_allow_html=True)

        elif _phase == "phase_after_tp1":
            st.markdown(f"""
<div style='background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.3);border-radius:12px;padding:14px 16px;margin-bottom:8px;'>
<div style='font-size:13px;font-weight:900;color:#06b6d4;margin-bottom:10px;'>🔄 잔량 트레일링 진행 — 매일 저녁 손절가를 업데이트하세요</div>
<div style='font-size:11px;color:#d8e8ff;line-height:2.0;'>
<span style='color:#f59e0b;font-weight:800;'>①</span> 매일 장 마감 후 → <b>{ticker}</b> 오늘 <b>고가</b> 확인 (HTS/MTS)<br>
<span style='color:#f59e0b;font-weight:800;'>②</span> 새 손절가 = 오늘 고점 × 0.93 (7% 트레일링)<br>
<span style='color:#f59e0b;font-weight:800;'>③</span> 어제 손절가보다 높으면 → 증권사 앱 손절 주문 <b style='color:#06b6d4;'>즉시 변경</b><br>
<span style='color:#ef4444;font-weight:800;'>④</span> 현재 손절가: <b style='color:#ef4444;'>${_new_stop:.2f}</b> → 이 가격 이하 장중 도달 시 <b style='color:#ef4444;'>즉시 전량 매도</b>
</div>
</div>""", unsafe_allow_html=True)

        elif _phase == "phase_breakeven":
            st.markdown(f"""
<div style='background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.3);border-radius:12px;padding:14px 16px;margin-bottom:8px;'>
<div style='font-size:13px;font-weight:900;color:#06b6d4;margin-bottom:10px;'>✅ 브레이크이븐 전환 — 손절가를 올려야 합니다</div>
<div style='font-size:11px;color:#d8e8ff;line-height:2.0;'>
<span style='color:#06b6d4;font-weight:800;'>①</span> 지금 즉시 → 증권사 앱 손절 주문을 <b style='color:#f59e0b;'>${_be_stop:.2f} (진입가+1%)</b>로 변경<br>
<span style='color:#06b6d4;font-weight:800;'>②</span> 이제 이 종목은 <b style='color:#22c55e;'>최소 +1% 수익 확정</b> (손실 없음)<br>
<span style='color:#f59e0b;font-weight:800;'>③</span> 트레일링 손절가: <b style='color:#ef4444;'>${_new_stop:.2f}</b> (고점 대비 -{int(_trail_pct_val*100)}%)<br>
<span style='color:#5a7299;font-weight:800;'>④</span> 목표: <b style='color:#22c55e;'>${_tp1_target:.2f}</b> (+18%) 도달 시 50% 청산
</div>
</div>""", unsafe_allow_html=True)

        else:  # phase_holding, phase_loss
            st.markdown(f"""
<div style='background:rgba(255,255,255,0.02);border:1px solid #1e3a5f;border-radius:12px;padding:14px 16px;margin-bottom:8px;'>
<div style='font-size:13px;font-weight:900;color:{"#22c55e" if _gain_current>=0 else "#f59e0b"};margin-bottom:10px;'>
{"📈 보유 중 — 손절가 지키면서 기다리세요" if _gain_current>=0 else "📉 매수가 이하 — 손절가 확인"}
</div>
<div style='font-size:11px;color:#d8e8ff;line-height:2.0;'>
<span style='color:#f59e0b;font-weight:800;'>①</span> 현재 손절가: <b style='color:#ef4444;'>${_new_stop:.2f}</b> → 이 가격 이하 즉시 매도<br>
<span style='color:#f59e0b;font-weight:800;'>②</span> 목표 ①: <b style='color:#f59e0b;'>${_be_trigger:.2f}</b> (+10%) → 손절가 진입가+1%로 상향<br>
<span style='color:#22c55e;font-weight:800;'>③</span> 목표 ②: <b style='color:#22c55e;'>${_tp1_target:.2f}</b> (+18%) → 50% 매도 + 트레일링 전환<br>
<span style='color:#5a7299;font-weight:800;'>④</span> 매일 저녁 고점 확인 → 손절가가 올라가면 증권사 앱 업데이트
</div>
</div>""", unsafe_allow_html=True)

        # ── 오늘 손절가 계산 요약 ─────────────────────────────────
        _trail_pct_disp = int(_trail_pct_val * 100)
        st.markdown(f"""
<div style='background:rgba(6,182,212,0.04);border:1px solid rgba(6,182,212,0.15);border-radius:10px;padding:10px 14px;'>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;font-size:11px;'>
<div>
<span style='color:#5a7299;'>📐 계산식:</span>
<span style='color:#d8e8ff;'> 오늘 고점 ${_my_high:.2f} × {1-_trail_pct_val:.2f} = <b style="color:#ef4444;">${_new_stop:.2f}</b></span>
</div>
<div>
<span style='color:#5a7299;'>📊 매수가 대비:</span>
<span style='color:{"#22c55e" if _new_stop>=_my_entry else "#f59e0b"};'> {"손절 시 +" if _new_stop>=_my_entry else "손절 시 -"}${abs(_new_stop-_my_entry):.2f} ({abs((_new_stop/_my_entry-1)*100):.1f}%)</span>
</div>
<div>
<span style='color:#5a7299;'>⏰ 갱신 시점:</span>
<span style='color:#d8e8ff;'> 매일 장 마감 후 (오후 5시 이후)</span>
</div>
</div>
</div>""", unsafe_allow_html=True)

    else:
        st.info("매수가를 입력하면 손절가·익절가·트레일링 스톱이 자동 계산됩니다.")

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
<div class='kv'><div class='k'>VCP (점수)</div><div class='v' style='color:{"#06b6d4" if sc.is_perfect_vcp else "#22c55e" if sc.vcp["breakout_ready"] else "#5a7299"};'>{"🔥완전체" if sc.is_perfect_vcp else "🚀돌파준비" if sc.vcp["breakout_ready"] else ("⚡감지" if sc.vcp["detected"] else "—")} {sc.vcp_score}/10</div></div>
<div class='kv'><div class='k'>RVOL</div><div class='v' style='color:{"#22c55e" if sc.rvol>=1.5 else "#f59e0b" if sc.rvol>=1.2 else "#5a7299"};'>{sc.rvol:.2f}x {sc.rvol_category}</div></div>
<div class='kv'><div class='k'>기관수급</div><div class='v' style='color:{"#22c55e" if sc.inst_score>0 else "#ef4444" if sc.inst_score<0 else "#5a7299"};'>{sc.inst_signal}</div></div>
<div class='kv'><div class='k'>돌파품질</div><div class='v' style='color:{"#ef4444" if sc.fake_breakout else "#22c55e" if sc.close_above_pivot else "#5a7299"};'>{"⚠️Fake BK" if sc.fake_breakout else "✅종가안착" if sc.close_above_pivot else "—"}</div></div>
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
        "🔥 AI·반도체 (30종)":  "NVDA,AMD,AVGO,QCOM,ARM,MRVL,TSM,AMAT,LRCX,KLAC,MU,SMCI,ASML,INTC,MCHP,MPWR,ON,WOLF,AMBA,CRUS,SWKS,NXPI,ENTG,ONTO,ACLS,MKSI,ICHR,KLIC,COHU,FORM",
        "💻 빅테크 (25종)":     "AAPL,MSFT,GOOG,AMZN,META,NFLX,CRM,ORCL,ADBE,IBM,NOW,INTU,WDAY,TEAM,ZM,DOCU,OKTA,HUBS,TWLO,DDOG,MDB,ESTC,CFLT,GTLB,VEEV",
        "⚡ 에너지·원전 (25종)": "SMR,NNE,CEG,VST,ETR,NRG,IREN,FSLR,ENPH,CLNE,NEE,AES,PCG,EXC,DTE,PPL,SO,D,XEL,OKE,LNG,AR,EQT,CTRA,FANG",
        "🚀 고성장주 (30종)":    "PLTR,AXON,CRWD,ZS,DDOG,SNOW,NET,ABNB,UBER,MSTR,AFRM,BILL,TTD,RBRK,IOT,SAMSARA,GTLB,GLBE,MNDY,ASAN,APP,APPLOVIN,IBOTTA,DUOL,CELH,WING,ELF,BROS,CAVA,RXRX",
        "💊 헬스케어 (25종)":    "LLY,NVO,ABBV,JNJ,UNH,MRNA,PFE,AMGN,GILD,REGN,ISRG,BSX,DXCM,INSP,NTRA,RARE,EXAS,PGNY,ACAD,IONS,ARWR,ALNY,BMRN,SRPT,PTGX",
        "🏦 금융·핀테크 (25종)": "JPM,BAC,GS,MS,BLK,V,MA,PYPL,SQ,COIN,HOOD,SOFI,NU,AFRM,UPST,LC,MARA,RIOT,CORZ,HUT,CLSK,IREN,WBTC,SEZL,DAVE",
        "🛡️ 사이버보안 (20종)":  "CRWD,ZS,PANW,FTNT,S,OKTA,CYBR,QLYS,TENB,RPD,VRNS,DDOG,ESTC,SIEM,SUMO,CYBE,HACK,CFIX,SFHG,EVTL",
        "🤖 AI 소프트웨어 (20종)":"PLTR,AI,PATH,BBAI,SOUN,GTLB,MNDY,ASAN,TTD,CELH,HIMS,IONQ,QUBT,RGTI,QBTS,ARQQ,BFLY,OUST,LIDR,AEVA",
        "🎮 소비·엔터 (25종)":   "TSLA,DIS,ABNB,BKNG,CMG,MCD,SBUX,NKE,LULU,RBLX,SPOT,SNAP,PINS,MTCH,MGM,LVS,WYNN,PENN,DKNG,FLUT,EXPE,TRIP,LYFT,DASH,CART",
        "📦 전체 통합 (100종)":  "NVDA,AMD,AVGO,QCOM,ARM,MRVL,AMAT,LRCX,KLAC,MU,AAPL,MSFT,GOOG,AMZN,META,NFLX,CRM,ADBE,NOW,INTU,PLTR,AXON,CRWD,ZS,DDOG,SNOW,NET,UBER,ABNB,MSTR,LLY,NVO,ABBV,UNH,ISRG,REGN,GILD,JPM,BAC,GS,V,MA,COIN,PYPL,SMR,NNE,CEG,VST,IREN,FSLR,ENPH,PANW,FTNT,OKTA,CYBR,TTWO,DIS,SPOT,RBLX,CELH,WING,CAVA,APP,TTD,BILL,AFRM,MNDY,GTLB,GLBE,IOT,RXRX,TSLA,MSFT,WDAY,MDB,ESTC,HUBS,BSX,DXCM,INSP,BLK,MS,SOFI,NU,MARA,LNG,EQT,AR,MRNA,PFE,AMGN,NXPI,MCHP,ON,INTC,SMCI,ASML,WBTC,SOUN,BBAI,PATH,AI,IONQ,QUBT",
        "⭐ 내 위시리스트":       "NVDA,TSLA,MSFT,AAPL,AMZN,META,GOOG,PLTR,SMR,IREN,AXON,CRWD,MSTR,SNOW,NET,AMD,ARM,AVGO,LLY,V",
    }
    st.markdown("""
<div style='margin-bottom:14px;'>
<div style='font-size:15px;font-weight:800;color:#d8e8ff;margin-bottom:4px;'>종목 스캔 → 등급 + v2 조건 충족률 + 매수 추천</div>
<div style='font-size:11px;color:#5a7299;'>최대 150종목 · 전체 통합 100종목 포함 · X/9 조건 충족 + 추천등급 + 손절/익절/트레일링</div>
</div>""", unsafe_allow_html=True)
    r1c1,r1c2,r1c3=st.columns([2,1.2,1.2])
    with r1c1: sc_preset=st.selectbox("📂 프리셋",list(PRESETS.keys()),key="sc_preset")
    with r1c2: sc_tf=st.selectbox("타임프레임",list(TF_OPTIONS.keys()),key="sc_tf")
    with r1c3: sc_style=st.selectbox("스타일",["스윙","단타"],key="sc_style")
    sc_custom=st.text_input("✏️ 직접 입력",value="",placeholder="AAPL,NVDA,TSLA…",key="sc_custom")
    sc_tickers=(
        [t.strip().upper() for t in sc_custom.replace("\n",",").split(",") if t.strip()]
        if sc_custom.strip()
        else [t.strip().upper() for t in PRESETS[sc_preset].split(",") if t.strip()]
    )
    sc_tickers=list(dict.fromkeys(sc_tickers))[:150]
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
                rc,rl,rd,pc,tc=get_recommendation(s.entry_conds)
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
                    "cond_passed":pc,"cond_total":tc,"rec_color":rc,"rec_label":rl,"rec_desc":rd,
                    "rvol":s.sc.rvol,"inst_signal":s.sc.inst_signal,"inst_score":s.sc.inst_score,
                    "fake_breakout":s.sc.fake_breakout,"close_above_pivot":s.sc.close_above_pivot,
                    "is_perfect_vcp":s.sc.is_perfect_vcp,"vcp_score":s.sc.vcp_score,
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
<div style='background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.4);border-radius:12px;padding:10px 16px;text-align:center;'><div style='font-size:11px;color:#5a7299;font-weight:700;'>🔥 강력추천</div><div style='font-size:24px;font-weight:900;color:#06b6d4;'>{rec_cnt["cyan"]}</div></div>
<div style='background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.4);border-radius:12px;padding:10px 16px;text-align:center;'><div style='font-size:11px;color:#5a7299;font-weight:700;'>✅ 적극추천</div><div style='font-size:24px;font-weight:900;color:#22c55e;'>{rec_cnt["green"]}</div></div>
<div style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.4);border-radius:12px;padding:10px 16px;text-align:center;'><div style='font-size:11px;color:#5a7299;font-weight:700;'>🟡 조건부</div><div style='font-size:24px;font-weight:900;color:#f59e0b;'>{rec_cnt["yellow"]}</div></div>
<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);border-radius:12px;padding:10px 16px;text-align:center;'><div style='font-size:11px;color:#5a7299;font-weight:700;'>🔴 보류</div><div style='font-size:24px;font-weight:900;color:#ef4444;'>{rec_cnt["red"]}</div></div>
<div style='margin-left:auto;background:rgba(90,114,153,0.07);border:1px solid #1a284040;border-radius:12px;padding:10px 16px;text-align:center;'><div style='font-size:11px;color:#5a7299;font-weight:700;'>총 분석</div><div style='font-size:24px;font-weight:900;color:#d8e8ff;'>{len(results)}</div></div>
</div>""", unsafe_allow_html=True)
            ICONS={"SSS":"⭐","SS":"🌟","S":"✅","A":"📈","B":"➡️","C":"⚠️","D":"📉"}
            tabs_grades=st.tabs([f"{ICONS.get(g,'')} {g} ({grade_cnt.get(g,0)})" for g in GRADE_ORDER]+["📋 전체"])
            def render_cards(items):
                if not items:
                    st.markdown("<div style='padding:24px;text-align:center;color:#5a7299;'>해당 등급 없음</div>",unsafe_allow_html=True)
                    return
                cols3=st.columns(3)
                for idx,r in enumerate(items):
                    g=r["grade"]; gc_=GRADE_COLOR.get(g,"#5a7299"); gbg=GRADE_BG.get(g,"rgba(90,114,153,0.07)")
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
<div style='display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-bottom:6px;'>
<div style='font-size:11px;color:#5a7299;'>TT <span style='color:#d8e8ff;font-weight:700;'>{r["tt"]}/8</span></div>
<div style='font-size:11px;color:#5a7299;'>RS <span style='color:#d8e8ff;font-weight:700;'>{r["rs"]}</span></div>
<div style='font-size:11px;color:#5a7299;'>RSI <span style='color:#d8e8ff;font-weight:700;'>{r["rsi"]:.0f}</span></div>
<div style='font-size:11px;color:#5a7299;'>ADX <span style='color:#d8e8ff;font-weight:700;'>{r["adx"]:.0f}</span></div>
<div style='font-size:11px;color:#5a7299;'>RVOL <span style='color:{"#22c55e" if r.get("rvol",1)>=1.5 else "#f59e0b" if r.get("rvol",1)>=1.2 else "#d8e8ff"};font-weight:700;'>{r.get("rvol",1.0):.2f}x</span></div>
<div style='font-size:11px;color:{wc};'>{ws_} <span style='font-weight:700;'>{abs(r["weekly"]):.1f}%</span></div>
</div>
<div style='font-size:10px;margin-bottom:6px;color:#5a7299;'>
기관: <span style='color:{"#22c55e" if r.get("inst_score",0)>0 else "#ef4444" if r.get("inst_score",0)<0 else "#5a7299"};font-weight:700;'>{r.get("inst_signal","—")}</span>
{"&nbsp;<span style='color:#ef4444;'>⚠️Fake</span>" if r.get("fake_breakout") else ""}{"&nbsp;<span style='color:#06b6d4;font-weight:700;'>🔥VCP완전체</span>" if r.get("is_perfect_vcp") else ""}{"&nbsp;<span style='color:#22c55e;'>✅안착</span>" if r.get("close_above_pivot") else ""}
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
<span style='font-size:10px;color:#5a7299;'>🔄 잔량 트레일링</span>
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
                df_t=pd.DataFrame([{"티커":r["ticker"],"등급":r["grade"],"점수":r["score"],"추천":r["rec_label"],"v2조건":f"{r['cond_passed']}/{r['cond_total']}","Stage":r["stage"],"TT":f"{r['tt']}/8","RS":r["rs"],"진입":f"${r['entry']:.2f}","손절":f"${r['stop']:.2f}({r['stop_pct']:.1f}%)","+18%청산":f"${r['tp1']:.2f}({r['tp1_pct']:+.1f}%)","트레일":f"-{r['trail_pct']*100:.0f}%","R:R":f"{r['weighted_rr']:.1f}"} for r in results])
                st.dataframe(df_t,use_container_width=True,hide_index=True,column_config={"점수":st.column_config.ProgressColumn("점수",min_value=0,max_value=100,format="%d")})
                leaders=[r for r in results if r["stage"]==2 and r["rec_color"] in ("cyan","green")]
                if leaders:
                    st.success(f"🌟 Stage2 + v2 통과: {', '.join(r['ticker'] for r in leaders[:10])}")


# ══════════════════════════════════════════════════════════════════
# 탭 3 — 만점 티커 찾기
# ══════════════════════════════════════════════════════════════════
with _tabs[2]:
    st.markdown("""
<div style='margin-bottom:16px;'>
<div style='font-size:16px;font-weight:900;color:#06b6d4;margin-bottom:4px;'>🎯 만점 티커 찾기 — 9/9 + TT 완전 충족</div>
<div style='font-size:12px;color:#5a7299;'>9/9 조건 + TT 기준 이상 + Stage2 동시 충족 종목만 추출합니다</div>
</div>""", unsafe_allow_html=True)
    with st.spinner("SPY 국면 확인 중…"):
        try:
            spy_res=build_signal("SPY","스윙","스윙 (1D)")
            spy_bias=spy_res[0].bias if spy_res else "알수없음"
        except Exception: spy_bias="알수없음"
    spy_c={"상승장":"#22c55e","횡보장":"#f59e0b","하락장":"#ef4444"}.get(spy_bias,"#5a7299")
    spy_warn=spy_bias!="상승장"

    # ── 백테스트 기반 국면별 매수 전략 ───────────────────────────
    _spy_guide = {
        "상승장": {
            "icon": "🟢", "title": "SPY 상승장 — 최적 매수 환경",
            "A": ("🔥 A등급 우선 진입", "#06b6d4",
                  "강세장 A등급: 승률 51.2% / PF 2.75 — v2 최고 성과 구간"),
            "S": ("✅ S등급 함께 진입", "#22c55e",
                  "강세장 S등급: 승률 46.2% / PF 1.88 — 9/9 조건 충족 시 적극 진입"),
            "action": "buy",
        },
        "횡보장": {
            "icon": "🟡", "title": "SPY 횡보장 — 선별 매수",
            "A": ("⚠️ A등급 진입 보류", "#ef4444",
                  "중립장 A등급: PF 0.93 (손실 구간) — 진입 금지"),
            "S": ("🔍 S등급 기준 강화", "#f59e0b",
                  "중립장 S등급: 승률 38.6% / PF 1.15 — 9/9 조건 + TT 7/8+ 시만 소량"),
            "action": "caution",
        },
        "하락장": {
            "icon": "🔴", "title": "SPY 하락장 — 신규 매수 금지",
            "A": ("🚫 A등급 진입 금지", "#ef4444", "하락장 전 등급 진입 금지 (v2 백테스트 원칙)"),
            "S": ("🚫 S등급 진입 금지", "#ef4444", "하락장 전 등급 진입 금지 (v2 백테스트 원칙)"),
            "action": "stop",
        },
        "알수없음": {
            "icon": "❓", "title": "SPY 데이터 없음",
            "A": ("❓ 판단 불가", "#5a7299", "SPY 데이터 재확인 필요"),
            "S": ("❓ 판단 불가", "#5a7299", "SPY 데이터 재확인 필요"),
            "action": "unknown",
        },
    }
    _sg = _spy_guide.get(spy_bias, _spy_guide["알수없음"])
    _a_label, _a_color, _a_desc = _sg["A"]
    _s_label, _s_color, _s_desc = _sg["S"]
    _action = _sg["action"]

    st.markdown(f"""
<div style='background:{"rgba(239,68,68,0.08)" if spy_warn else "rgba(34,197,94,0.07)"};
border:1px solid {"rgba(239,68,68,0.3)" if spy_warn else "rgba(34,197,94,0.2)"};
border-radius:14px;padding:14px 16px;margin-bottom:14px;'>
<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
<div style='font-size:24px;'>{_sg["icon"]}</div>
<div style='font-size:14px;font-weight:800;color:{spy_c};'>{_sg["title"]}</div>
</div>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
<div style='background:rgba({("6,182,212" if _action=="buy" else "239,68,68" if _action=="stop" else "245,158,11")},0.1);
border:1px solid rgba({("6,182,212" if _action=="buy" else "239,68,68" if _action=="stop" else "245,158,11")},0.3);
border-radius:10px;padding:10px;'>
<div style='font-size:12px;font-weight:800;color:{_a_color};margin-bottom:3px;'>{_a_label}</div>
<div style='font-size:10px;color:#5a7299;'>{_a_desc}</div>
</div>
<div style='background:rgba({("34,197,94" if _action=="buy" else "239,68,68" if _action=="stop" else "245,158,11")},0.08);
border:1px solid rgba({("34,197,94" if _action=="buy" else "239,68,68" if _action=="stop" else "245,158,11")},0.25);
border-radius:10px;padding:10px;'>
<div style='font-size:12px;font-weight:800;color:{_s_color};margin-bottom:3px;'>{_s_label}</div>
<div style='font-size:10px;color:#5a7299;'>{_s_desc}</div>
</div>
</div>
</div>""", unsafe_allow_html=True)
    PERFECT_PRESETS={
        "🔥 AI·반도체 (30종)":  "NVDA,AMD,AVGO,QCOM,ARM,MRVL,TSM,AMAT,LRCX,KLAC,MU,SMCI,ASML,INTC,MCHP,MPWR,ON,WOLF,AMBA,CRUS,SWKS,NXPI,ENTG,ONTO,ACLS,MKSI,ICHR,KLIC,COHU,FORM",
        "💻 빅테크 (25종)":     "AAPL,MSFT,GOOG,AMZN,META,NFLX,CRM,ORCL,ADBE,IBM,NOW,INTU,WDAY,TEAM,ZM,DOCU,OKTA,HUBS,TWLO,DDOG,MDB,ESTC,CFLT,GTLB,VEEV",
        "⚡ 에너지·원전 (25종)": "SMR,NNE,CEG,VST,ETR,NRG,IREN,FSLR,ENPH,CLNE,NEE,AES,PCG,EXC,DTE,PPL,SO,D,XEL,OKE,LNG,AR,EQT,CTRA,FANG",
        "🚀 고성장주 (30종)":    "PLTR,AXON,CRWD,ZS,DDOG,SNOW,NET,ABNB,UBER,MSTR,AFRM,BILL,TTD,RBRK,IOT,GTLB,GLBE,MNDY,ASAN,APP,DUOL,CELH,WING,ELF,BROS,CAVA,RXRX,HIMS,IBOTTA,SAMSARA",
        "💊 헬스케어 (25종)":    "LLY,NVO,ABBV,JNJ,UNH,MRNA,PFE,AMGN,GILD,REGN,ISRG,BSX,DXCM,INSP,NTRA,RARE,EXAS,PGNY,ACAD,IONS,ARWR,ALNY,BMRN,SRPT,PTGX",
        "🏦 금융·핀테크 (25종)": "JPM,BAC,GS,MS,BLK,V,MA,PYPL,SQ,COIN,HOOD,SOFI,NU,AFRM,UPST,LC,MARA,RIOT,CORZ,HUT,CLSK,IREN,SEZL,DAVE,FLUT",
        "🛡️ 사이버보안 (20종)":  "CRWD,ZS,PANW,FTNT,S,OKTA,CYBR,QLYS,TENB,RPD,VRNS,DDOG,ESTC,NET,SIEM,SUMO,SentinelOne,HACK,CFIX,SFHG",
        "🤖 AI 소프트웨어 (20종)":"PLTR,AI,PATH,BBAI,SOUN,GTLB,MNDY,ASAN,TTD,CELH,HIMS,IONQ,QUBT,RGTI,QBTS,ARQQ,BFLY,OUST,LIDR,AEVA",
        "🎮 소비·엔터 (25종)":   "TSLA,DIS,ABNB,BKNG,CMG,MCD,SBUX,NKE,LULU,RBLX,SPOT,SNAP,PINS,MTCH,MGM,LVS,WYNN,PENN,DKNG,FLUT,EXPE,LYFT,DASH,CART,DUOL",
        "📦 전체 통합 (100종)":  "NVDA,AMD,AVGO,QCOM,ARM,MRVL,AMAT,LRCX,KLAC,MU,AAPL,MSFT,GOOG,AMZN,META,NFLX,CRM,ADBE,NOW,INTU,PLTR,AXON,CRWD,ZS,DDOG,SNOW,NET,UBER,ABNB,MSTR,LLY,NVO,ABBV,UNH,ISRG,REGN,GILD,JPM,BAC,GS,V,MA,COIN,PYPL,SMR,NNE,CEG,VST,IREN,FSLR,ENPH,PANW,FTNT,OKTA,CYBR,DIS,SPOT,RBLX,CELH,WING,CAVA,APP,TTD,BILL,AFRM,MNDY,GTLB,GLBE,IOT,RXRX,TSLA,WDAY,MDB,ESTC,HUBS,BSX,DXCM,INSP,BLK,MS,SOFI,NU,MARA,LNG,EQT,AR,MRNA,PFE,AMGN,NXPI,MCHP,ON,INTC,SMCI,ASML,SOUN,BBAI,PATH,AI,IONQ,QUBT,HIMS,DUOL",
        "⭐ 내 위시리스트":       "NVDA,TSLA,MSFT,AAPL,AMZN,META,GOOG,PLTR,SMR,IREN,AXON,CRWD,MSTR,SNOW,NET,AMD,ARM,AVGO,LLY,V",
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
    pf_tickers=list(dict.fromkeys(pf_tickers))[:150]
    pf_min_tt_num={"6/8 이상":6,"7/8 이상":7,"8/8 만점":8}[pf_min_tt]
    pb1,pb2=st.columns([1,4])
    with pb1: pf_run=st.button("🎯 찾기 시작",use_container_width=True,key="pf_run")
    with pb2: st.markdown(f"<div style='padding:9px 0;font-size:12px;color:#5a7299;'>스캔 <b style='color:#d8e8ff;'>{len(pf_tickers)}종목</b> (전체통합 100종 포함) → <b style='color:#06b6d4;'>9/9 + TT {pf_min_tt} + Stage2 동시 충족만 추출</b></div>",unsafe_allow_html=True)
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
                rc,rl,rd,pc,tc=get_recommendation(s.entry_conds)
                tt_passed=s.sc.trend_template["passed"]; stage=s.sc.weinstein["stage"]
                row={"ticker":tk,"score":s.sc.total,"grade":s.sc.grade,"cond_passed":pc,"cond_total":tc,"tt_passed":tt_passed,"stage":stage,"tt_conds":s.sc.trend_template["conditions"],"perf12m":s.sc.trend_template["perf12m"],"rs":s.sc.rs["rs_score"],"rsi":s.sc.rsi_val,"adx":s.sc.adx_val,"vol_r":s.sc.vol_ratio,"vcp":s.sc.vcp["breakout_ready"],"vcp_det":s.sc.vcp["detected"],"golden":s.sc.golden_cross,"squeeze":s.sc.bb_squeeze,"bias":s.bias,"weekly":s.weekly_perf,"entry":tp2.entry_price,"stop":tp2.stop_price,"stop_pct":tp2.stop_pct,"tp1":tp2.tp1_price,"tp1_pct":tp2.tp1_pct,"trail_pct":tp2.trail_pct,"be_trigger":tp2.be_trigger,"be_stop":tp2.be_stop,"weighted_rr":tp2.weighted_rr,"rec_label":rl}
                if pc==tc and tt_passed>=pf_min_tt_num and stage==2: perfect.append(row)
                elif pc>=tc-1 and tt_passed>=pf_min_tt_num-1 and stage==2: near.append(row)
            except Exception: continue
        pf_bar.empty(); pf_stat.empty()
        st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(6,182,212,0.1),rgba(34,197,94,0.05));border:1px solid rgba(6,182,212,0.3);border-radius:14px;padding:16px 20px;margin-bottom:16px;display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;text-align:center;'>
<div><div style='font-size:11px;color:#5a7299;'>스캔</div><div style='font-size:28px;font-weight:900;color:#d8e8ff;'>{len(pf_tickers)}</div></div>
<div style='border-left:1px solid #1e3a5f;'><div style='font-size:11px;color:#5a7299;'>🔥 만점</div><div style='font-size:28px;font-weight:900;color:#06b6d4;'>{len(perfect)}</div></div>
<div style='border-left:1px solid #1e3a5f;'><div style='font-size:11px;color:#5a7299;'>⚡ 아깝게</div><div style='font-size:28px;font-weight:900;color:#f59e0b;'>{len(near)}</div></div>
<div style='border-left:1px solid #1e3a5f;'><div style='font-size:11px;color:#5a7299;'>TT 기준</div><div style='font-size:20px;font-weight:900;color:#a855f7;'>{pf_min_tt}</div></div>
</div>""", unsafe_allow_html=True)
        if not perfect and not near:
            st.warning("조건 충족 종목이 없습니다. TT 기준을 낮추거나 다른 프리셋을 시도해보세요.")
        else:
            if perfect:
                perfect.sort(key=lambda x:(x["tt_passed"],x["score"]),reverse=True)
                st.markdown(f"""
<div style='background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.3);border-radius:12px;padding:12px 16px;margin-bottom:12px;'>
<div style='font-size:13px;font-weight:800;color:#06b6d4;'>🔥 만점 {len(perfect)}종목 — 9/9 + TT {pf_min_tt} + Stage2 전부 충족</div>
<div style='font-size:11px;color:#5a7299;margin-top:3px;'>{"⚠️ SPY "+spy_bias+" — 신중 접근" if spy_warn else "✅ SPY 상승장 + 만점 = v2 최고 성과 구간"}</div>
</div>""", unsafe_allow_html=True)
                pf_cols=st.columns(min(3,len(perfect)))
                for i,r in enumerate(perfect):
                    with pf_cols[i%3]:
                        wc="#22c55e" if r["weekly"]>=0 else "#ef4444"; ws_="▲" if r["weekly"]>=0 else "▼"
                        wrrc="#22c55e" if r["weighted_rr"]>=2 else "#f59e0b"
                        tt_html="".join([f"<div style='font-size:10px;color:{'#22c55e' if ok else '#ef4444'};padding:1px 0;'>{'✓' if ok else '✗'} {lbl}</div>" for lbl,ok in r["tt_conds"]])
                        badges=" ".join(b for b in["🚀VCP" if r["vcp"] else("⚡" if r["vcp_det"] else ""),"🌟골든" if r["golden"] else "","⚡수축" if r["squeeze"] else ""] if b)
                        st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(6,182,212,0.12),rgba(34,197,94,0.08));border:2px solid rgba(6,182,212,0.5);border-radius:16px;padding:16px;margin-bottom:14px;'>
<div style='display:flex;justify-content:space-between;margin-bottom:10px;'>
<div>
<div style='font-size:24px;font-weight:900;color:#06b6d4;'>{r["ticker"]}</div>
<div style='font-size:11px;color:#22c55e;font-weight:700;'>Stage2🚀 · {r["grade"]}등급 {r["score"]}점</div>
{f"<div style='font-size:10px;color:#5a7299;'>{badges}</div>" if badges else ""}
</div>
<div style='text-align:center;background:rgba(6,182,212,0.15);border:1px solid rgba(6,182,212,0.4);border-radius:10px;padding:6px 12px;'>
<div style='font-size:10px;color:#5a7299;'>v2 조건</div>
<div style='font-size:20px;font-weight:900;color:#06b6d4;'>9/9</div>
</div>
</div>
<div style='background:rgba({"6,182,212" if _action=="buy" else "239,68,68" if _action=="stop" else "245,158,11"},0.1);
border:1px solid rgba({"6,182,212" if _action=="buy" else "239,68,68" if _action=="stop" else "245,158,11"},0.3);
border-radius:8px;padding:6px;text-align:center;margin-bottom:10px;'>
<div style='font-size:14px;font-weight:900;color:{"#06b6d4" if _action=="buy" else "#ef4444" if _action=="stop" else "#f59e0b"};'>
{"🔥 A등급 우선 매수 추천" if r["grade"] in ["A"] and _action=="buy" else "✅ S등급 매수 추천" if _action=="buy" else "🚫 현재 매수 금지" if _action=="stop" else "⚠️ 조건 강화 후 소량만"}
</div>
<div style='font-size:10px;color:#5a7299;margin-top:1px;'>SPY {spy_bias} 기준</div>
</div>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:10px;margin-bottom:10px;'>
<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
<div style='font-size:20px;font-weight:900;color:#06b6d4;'>TT {r["tt_passed"]}/8</div>
<div><div style='font-size:11px;font-weight:700;color:#22c55e;'>✅ Stage 2 확인</div><div style='font-size:10px;color:#5a7299;'>12M: {r["perf12m"]:+.1f}%</div></div>
</div>
{tt_html}
</div>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:8px;'>
<div style='background:rgba(255,255,255,0.03);border-radius:7px;padding:5px;text-align:center;'><div style='font-size:10px;color:#5a7299;'>RS</div><div style='font-size:14px;font-weight:700;color:{"#22c55e" if r["rs"]>=80 else "#f59e0b"};'>{r["rs"]}</div></div>
<div style='background:rgba(255,255,255,0.03);border-radius:7px;padding:5px;text-align:center;'><div style='font-size:10px;color:#5a7299;'>RSI</div><div style='font-size:14px;font-weight:700;color:#d8e8ff;'>{r["rsi"]:.0f}</div></div>
<div style='background:rgba(255,255,255,0.03);border-radius:7px;padding:5px;text-align:center;'><div style='font-size:10px;color:#5a7299;'>ADX</div><div style='font-size:14px;font-weight:700;color:{"#22c55e" if r["adx"]>=25 else "#f59e0b"};'>{r["adx"]:.0f}</div></div>
</div>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;'>
<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:7px;padding:6px;text-align:center;'><div style='font-size:9px;color:#5a7299;'>🛑 손절</div><div style='font-size:12px;font-weight:700;color:#ef4444;'>${r["stop"]:.2f}</div><div style='font-size:9px;color:#ef4444;'>{r["stop_pct"]:.1f}%</div></div>
<div style='background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);border-radius:7px;padding:6px;text-align:center;'><div style='font-size:9px;color:#5a7299;'>✅ +18% 50%</div><div style='font-size:12px;font-weight:700;color:#22c55e;'>${r["tp1"]:.2f}</div><div style='font-size:9px;color:#22c55e;'>{r["tp1_pct"]:+.1f}%</div></div>
<div style='background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.25);border-radius:7px;padding:6px;text-align:center;'><div style='font-size:9px;color:#5a7299;'>🔄 트레일</div><div style='font-size:12px;font-weight:700;color:#06b6d4;'>-{r["trail_pct"]*100:.0f}%</div><div style='font-size:9px;color:#06b6d4;'>고점추적</div></div>
</div>
<div style='display:flex;justify-content:space-between;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06);'>
<div style='font-size:10px;color:#5a7299;'>진입 <b style='color:#06b6d4;'>${r["entry"]:.2f}</b></div>
<div style='font-size:10px;color:#5a7299;'>R:R <b style='color:{wrrc};'>{r["weighted_rr"]:.1f}:1</b></div>
<div style='font-size:10px;color:{wc};'>{ws_} {abs(r["weekly"]):.1f}%</div>
</div>
</div>""", unsafe_allow_html=True)
            if near:
                near.sort(key=lambda x:(x["cond_passed"],x["tt_passed"],x["score"]),reverse=True)
                with st.expander(f"⚡ 아깝게 탈락 {len(near)}종목"):
                    nr_cols=st.columns(3)
                    for i,r in enumerate(near):
                        with nr_cols[i%3]:
                            st.markdown(f"""
<div style='background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.3);border-radius:12px;padding:12px;margin-bottom:10px;'>
<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
<div style='font-size:18px;font-weight:900;color:#f59e0b;'>{r["ticker"]}</div>
<div style='text-align:right;'><div style='font-size:11px;font-weight:700;color:#f59e0b;'>⚡ 거의 완벽</div><div style='font-size:10px;color:#5a7299;'>TT {r["tt_passed"]}/8 · {r["cond_passed"]}/{r["cond_total"]}</div></div>
</div>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px;'>
<div style='text-align:center;font-size:10px;'><div style='color:#5a7299;'>손절</div><div style='color:#ef4444;font-weight:700;'>${r["stop"]:.2f}</div></div>
<div style='text-align:center;font-size:10px;'><div style='color:#5a7299;'>+18%50%</div><div style='color:#22c55e;font-weight:700;'>{r["tp1_pct"]:+.1f}%</div></div>
<div style='text-align:center;font-size:10px;'><div style='color:#5a7299;'>트레일</div><div style='color:#06b6d4;font-weight:700;'>-{r["trail_pct"]*100:.0f}%</div></div>
</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 탭 4 — 실전 매매 가이드
# ══════════════════════════════════════════════════════════════════
with _tabs[3]:
    st.markdown("""
<div style='margin-bottom:20px;'>
<div style='font-size:18px;font-weight:900;color:#06b6d4;margin-bottom:6px;'>📖 실전 매매 가이드 — 처음부터 끝까지</div>
<div style='font-size:12px;color:#5a7299;'>v2 백테스트 (PF 1.85, 승률 46.1%) 전략을 실전에서 그대로 따라하는 방법</div>
</div>""", unsafe_allow_html=True)

    # ── PHASE 1: 매일 아침 루틴 ──────────────────────────────────
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(6,182,212,0.08),rgba(0,0,0,0));border:1px solid rgba(6,182,212,0.3);border-radius:14px;padding:18px;margin-bottom:14px;'>
<div style='font-size:13px;font-weight:800;color:#06b6d4;margin-bottom:12px;'>☀️ PHASE 1 — 매일 아침 (장 시작 전 10분)</div>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:12px;'>
<div style='font-size:11px;font-weight:700;color:#f59e0b;margin-bottom:8px;'>① SPY 먼저 분석 (필수)</div>
<div style='font-size:11px;color:#d8e8ff;line-height:1.8;'>
1. 차트 분석 탭에서 <b>SPY</b> 입력 → 분석<br>
2. Weinstein Stage 확인<br>
&nbsp;&nbsp;&nbsp;• <b style='color:#22c55e;'>Stage 2</b> → 오늘 매수 가능<br>
&nbsp;&nbsp;&nbsp;• <b style='color:#f59e0b;'>Stage 1</b> → 신중하게 선별 매수<br>
&nbsp;&nbsp;&nbsp;• <b style='color:#ef4444;'>Stage 3/4</b> → 신규 매수 중단<br>
3. MA200 위에 있는지 확인<br>
&nbsp;&nbsp;&nbsp;• MA200 하회 = 매수 금지
</div>
</div>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:12px;'>
<div style='font-size:11px;font-weight:700;color:#f59e0b;margin-bottom:8px;'>② 보유 종목 손절가 관리</div>
<div style='font-size:11px;color:#d8e8ff;line-height:1.8;'>
보유 중인 종목마다:<br>
1. 어제 고점 확인<br>
2. <b>새 트레일링 손절가 계산</b><br>
&nbsp;&nbsp;&nbsp;S등급: 고점 × 0.90<br>
&nbsp;&nbsp;&nbsp;A등급: 고점 × 0.88<br>
3. 어제 손절가보다 높으면<br>
&nbsp;&nbsp;&nbsp;<b style='color:#06b6d4;'>증권사 앱에서 즉시 변경</b><br>
4. 절대 낮추지 않는다
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ── PHASE 2: 종목 발굴 ───────────────────────────────────────
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(34,197,94,0.06),rgba(0,0,0,0));border:1px solid rgba(34,197,94,0.25);border-radius:14px;padding:18px;margin-bottom:14px;'>
<div style='font-size:13px;font-weight:800;color:#22c55e;margin-bottom:12px;'>🔍 PHASE 2 — 종목 발굴 (주 1~2회)</div>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;'>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:12px;'>
<div style='font-size:11px;font-weight:700;color:#22c55e;margin-bottom:6px;'>방법 1: 빠른 스캔 탭</div>
<div style='font-size:11px;color:#5a7299;line-height:1.7;'>
관심 섹터 프리셋 선택<br>
→ 스캔 시작<br>
→ <b style='color:#06b6d4;'>🔥강력추천</b> 먼저 확인<br>
→ v2 조건 9/9 종목 우선<br>
→ Stage2 + S등급 이상만
</div>
</div>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:12px;'>
<div style='font-size:11px;font-weight:700;color:#22c55e;margin-bottom:6px;'>방법 2: 만점 티커 찾기 탭</div>
<div style='font-size:11px;color:#5a7299;line-height:1.7;'>
TT 6/8 이상 선택<br>
→ 찾기 시작<br>
→ 만점 카드에서 TT 세부<br>
&nbsp;&nbsp;&nbsp;조건 전체 확인<br>
→ 9/9 + TT 6+ = 최우선 후보
</div>
</div>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:12px;'>
<div style='font-size:11px;font-weight:700;color:#22c55e;margin-bottom:6px;'>진입 후보 체크리스트</div>
<div style='font-size:11px;color:#5a7299;line-height:1.7;'>
✅ v2 조건 7개 이상<br>
✅ Weinstein Stage 2<br>
✅ TT 6/8 이상<br>
✅ RS 65 이상<br>
✅ ADX 18 이상<br>
✅ SPY 상승장
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ── PHASE 3: 진입 실행 ───────────────────────────────────────
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(59,130,246,0.06),rgba(0,0,0,0));border:1px solid rgba(59,130,246,0.25);border-radius:14px;padding:18px;margin-bottom:14px;'>
<div style='font-size:13px;font-weight:800;color:#3b82f6;margin-bottom:12px;'>🎯 PHASE 3 — 진입 실행</div>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:12px;'>
<div style='font-size:11px;font-weight:700;color:#3b82f6;margin-bottom:8px;'>진입 방법</div>
<div style='font-size:11px;color:#d8e8ff;line-height:1.8;'>
<b style='color:#06b6d4;'>VCP 돌파 진입 (최선)</b><br>
→ 저항선 직전 지정가 설정<br>
→ 거래량 1.4배 이상 확인 후 진입<br><br>
<b style='color:#22c55e;'>눌림목 진입</b><br>
→ MA20 근처 지지 확인 후<br>
→ 분할 매수 (50% → 50%)<br><br>
<b style='color:#5a7299;'>현재가 진입 (일반)</b><br>
→ 조건 충족 시 바로 진입
</div>
</div>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:12px;'>
<div style='font-size:11px;font-weight:700;color:#3b82f6;margin-bottom:8px;'>포지션 사이즈 계산</div>
<div style='font-size:11px;color:#d8e8ff;line-height:1.9;'>
<b>계좌의 1.5%를 리스크로 설정</b><br><br>
예: 계좌 $10,000<br>
허용 손실 = $10,000 × 1.5% = <b style='color:#ef4444;'>$150</b><br><br>
진입가 $100, 손절가 $95<br>
1주당 손실 = $5<br>
매수 수량 = $150 ÷ $5 = <b style='color:#06b6d4;'>30주</b><br><br>
<span style='color:#5a7299;font-size:10px;'>한 종목 최대 계좌의 10% 초과 금지</span>
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)


    # 🔢 인터랙티브 포지션 사이즈 계산기
    st.markdown(
        "<div style='font-size:13px;font-weight:800;color:#3b82f6;margin:14px 0 8px;'>"
        "🔢 포지션 사이즈 계산기 — 바로 입력해서 매수 수량 확인</div>",
        unsafe_allow_html=True
    )
    _pc1, _pc2, _pc3, _pc4 = st.columns(4)
    with _pc1: _acc   = st.number_input("💰 계좌 ($)", value=10000, step=1000, min_value=100, key="calc_acc")
    with _pc2: _risk  = st.number_input("리스크 (%)", value=1.5, step=0.5, min_value=0.5, max_value=5.0, key="calc_risk")
    with _pc3: _entry = st.number_input("진입가 ($)", value=100.0, step=1.0, min_value=0.01, key="calc_entry")
    with _pc4: _stop  = st.number_input("손절가 ($)", value=92.0, step=1.0, min_value=0.01, key="calc_stop")

    if _entry > _stop > 0 and _acc > 0:
        _risk_amt     = _acc * (_risk / 100)
        _loss_per     = _entry - _stop
        _shares_calc  = int(_risk_amt / _loss_per) if _loss_per > 0 else 0
        _max_shares   = int(_acc * 0.10 / _entry)
        _shares_final = min(_shares_calc, _max_shares)
        _total_final  = _shares_final * _entry
        _max_loss     = _shares_final * _loss_per
        _stop_pct_val = (_stop / _entry - 1) * 100
        _tp1_price_v  = round(_entry * 1.18, 2)
        _tp1_profit   = _shares_final // 2 * (_tp1_price_v - _entry)
        _acct_pct     = _total_final / _acc * 100
        _capped       = _shares_final < _shares_calc

        _ca, _cb, _cc, _cd = st.columns(4)
        with _ca: st.markdown(
            f"<div style='background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.3);border-radius:10px;padding:12px;text-align:center;'>"
            f"<div style='font-size:10px;color:#5a7299;'>매수 수량</div>"
            f"<div style='font-size:24px;font-weight:900;color:#06b6d4;'>{_shares_final}주</div>"
            f"<div style='font-size:10px;color:#5a7299;'>{'⚠️ 10%한도 적용' if _capped else '✅ 계산값'}</div></div>",
            unsafe_allow_html=True)
        with _cb: st.markdown(
            f"<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-radius:10px;padding:12px;text-align:center;'>"
            f"<div style='font-size:10px;color:#5a7299;'>최대 손실</div>"
            f"<div style='font-size:20px;font-weight:900;color:#ef4444;'>${_max_loss:,.0f}</div>"
            f"<div style='font-size:10px;color:#ef4444;'>손절 {_stop_pct_val:.1f}%</div></div>",
            unsafe_allow_html=True)
        with _cc: st.markdown(
            f"<div style='background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);border-radius:10px;padding:12px;text-align:center;'>"
            f"<div style='font-size:10px;color:#5a7299;'>1차익절 +18%</div>"
            f"<div style='font-size:20px;font-weight:900;color:#22c55e;'>${_tp1_price_v:.2f}</div>"
            f"<div style='font-size:10px;color:#22c55e;'>수익 +${_tp1_profit:,.0f}</div></div>",
            unsafe_allow_html=True)
        with _cd: st.markdown(
            f"<div style='background:rgba(255,255,255,0.03);border:1px solid #1e3a5f;border-radius:10px;padding:12px;text-align:center;'>"
            f"<div style='font-size:10px;color:#5a7299;'>총 투자금</div>"
            f"<div style='font-size:20px;font-weight:900;color:#d8e8ff;'>${_total_final:,.0f}</div>"
            f"<div style='font-size:10px;color:#5a7299;'>계좌의 {_acct_pct:.1f}%</div></div>",
            unsafe_allow_html=True)
    else:
        st.info("진입가 > 손절가 조건을 확인하세요.")

    # ── PHASE 4: 손절 관리 ───────────────────────────────────────
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(239,68,68,0.07),rgba(0,0,0,0));border:1px solid rgba(239,68,68,0.3);border-radius:14px;padding:18px;margin-bottom:14px;'>
<div style='font-size:13px;font-weight:800;color:#ef4444;margin-bottom:12px;'>🛑 PHASE 4 — 손절 관리 (가장 중요)</div>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:12px;'>
<div style='font-size:11px;font-weight:700;color:#ef4444;margin-bottom:8px;'>초기 손절 설정</div>
<div style='font-size:11px;color:#d8e8ff;line-height:1.8;'>
진입 즉시 증권사 앱에서<br>
<b>조건부 주문 / 스톱로스</b> 설정<br><br>
손절가 = 프로그램 표시값<br>
(ATR×1.2 vs -8% 타이트한 쪽)<br><br>
<b style='color:#ef4444;'>⚠️ 반드시 즉시 설정!</b><br>
"나중에 설정하려다" 큰 손실 남
</div>
</div>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:12px;'>
<div style='font-size:11px;font-weight:700;color:#ef4444;margin-bottom:8px;'>브레이크이븐 전환 (+10%)</div>
<div style='font-size:11px;color:#d8e8ff;line-height:1.8;'>
주가가 <b>+10% 달성</b>하면:<br>
→ 손절가를 <b style='color:#f59e0b;'>진입가+1%</b>로 올림<br>
→ 이제 이 종목은 무조건 수익<br><br>
예: 진입 $100<br>
+10% = $110 달성 시<br>
손절가 → <b style='color:#f59e0b;'>$101</b>로 변경
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ── PHASE 5: 익절 전략 ───────────────────────────────────────
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(34,197,94,0.07),rgba(0,0,0,0));border:1px solid rgba(34,197,94,0.3);border-radius:14px;padding:18px;margin-bottom:14px;'>
<div style='font-size:13px;font-weight:800;color:#22c55e;margin-bottom:12px;'>✅ PHASE 5 — 익절 전략</div>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px;'>
<div style='background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);border-radius:10px;padding:12px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:4px;'>1단계</div>
<div style='font-size:14px;font-weight:800;color:#22c55e;'>+18% 달성</div>
<div style='background:#1e3a5f;border-radius:99px;height:5px;margin:8px 0;overflow:hidden;'><div style='width:50%;height:100%;background:#22c55e;border-radius:99px;'></div></div>
<div style='font-size:11px;color:#d8e8ff;font-weight:700;'>보유량 50% 매도</div>
<div style='font-size:10px;color:#5a7299;margin-top:4px;'>손절가 → 진입가+1%로 올림<br>나머지 50%는 트레일링</div>
</div>
<div style='background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.3);border-radius:10px;padding:12px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:4px;'>2단계 (이후)</div>
<div style='font-size:14px;font-weight:800;color:#06b6d4;'>트레일링 스톱</div>
<div style='background:#1e3a5f;border-radius:99px;height:5px;margin:8px 0;overflow:hidden;'><div style='width:100%;height:100%;background:#06b6d4;border-radius:99px;'></div></div>
<div style='font-size:11px;color:#d8e8ff;font-weight:700;'>잔량 50% 추세 추종</div>
<div style='font-size:10px;color:#5a7299;margin-top:4px;'>고점 대비 7~12% 추적<br>고점 갱신될수록 손절 상향</div>
</div>
<div style='background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.25);border-radius:10px;padding:12px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;margin-bottom:4px;'>최종 청산</div>
<div style='font-size:14px;font-weight:800;color:#a855f7;'>트레일 스톱 도달</div>
<div style='background:#1e3a5f;border-radius:99px;height:5px;margin:8px 0;overflow:hidden;'><div style='width:100%;height:100%;background:#a855f7;border-radius:99px;'></div></div>
<div style='font-size:11px;color:#d8e8ff;font-weight:700;'>잔량 전량 매도</div>
<div style='font-size:10px;color:#5a7299;margin-top:4px;'>고점 대비 -7~12% 하락 시<br>추세 종료 판단</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ── PHASE 6: 트레일링 스톱 실전 ─────────────────────────────
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(6,182,212,0.08),rgba(0,0,0,0));border:1px solid rgba(6,182,212,0.3);border-radius:14px;padding:18px;margin-bottom:14px;'>
<div style='font-size:13px;font-weight:800;color:#06b6d4;margin-bottom:12px;'>🔄 PHASE 6 — 트레일링 스톱 실전 방법 (매일 반복)</div>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:14px;'>
<div style='font-size:11px;font-weight:700;color:#06b6d4;margin-bottom:10px;'>📋 매일 저녁 루틴 (5분)</div>
<div style='font-size:11px;color:#d8e8ff;line-height:2.0;'>
<span style='background:rgba(6,182,212,0.2);border-radius:4px;padding:1px 7px;font-weight:700;color:#06b6d4;'>①</span>&nbsp; 증권사 앱에서 오늘 <b>고가</b> 확인<br>
<span style='background:rgba(6,182,212,0.2);border-radius:4px;padding:1px 7px;font-weight:700;color:#06b6d4;'>②</span>&nbsp; 새 손절가 계산 (아래 표 참고)<br>
<span style='background:rgba(6,182,212,0.2);border-radius:4px;padding:1px 7px;font-weight:700;color:#06b6d4;'>③</span>&nbsp; <b>어제 손절가보다 높을 때만</b> 변경<br>
<span style='background:rgba(6,182,212,0.2);border-radius:4px;padding:1px 7px;font-weight:700;color:#06b6d4;'>④</span>&nbsp; 절대로 낮추지 않는다<br>
<span style='background:rgba(239,68,68,0.2);border-radius:4px;padding:1px 7px;font-weight:700;color:#ef4444;'>⑤</span>&nbsp; 주가가 손절가 이하 → <b>즉시 전량 매도</b>
</div>
</div>
<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:14px;'>
<div style='font-size:11px;font-weight:700;color:#06b6d4;margin-bottom:10px;'>🧮 등급별 손절가 계산표</div>
<div style='font-size:11px;color:#d8e8ff;line-height:2.0;'>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;text-align:center;margin-bottom:8px;'>
<div style='background:rgba(6,182,212,0.15);border-radius:6px;padding:5px;font-weight:700;color:#06b6d4;'>등급</div>
<div style='background:rgba(6,182,212,0.15);border-radius:6px;padding:5px;font-weight:700;color:#06b6d4;'>비율</div>
<div style='background:rgba(6,182,212,0.15);border-radius:6px;padding:5px;font-weight:700;color:#06b6d4;'>계산식</div>
</div>
<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;text-align:center;'>
<div style='background:rgba(6,182,212,0.08);border-radius:6px;padding:5px;'>SS/SSS</div><div style='background:rgba(6,182,212,0.08);border-radius:6px;padding:5px;'>-8%</div><div style='background:rgba(6,182,212,0.08);border-radius:6px;padding:5px;'>고점×0.92</div>
<div style='background:rgba(34,197,94,0.08);border-radius:6px;padding:5px;'>S</div><div style='background:rgba(34,197,94,0.08);border-radius:6px;padding:5px;'>-10%</div><div style='background:rgba(34,197,94,0.08);border-radius:6px;padding:5px;'>고점×0.90</div>
<div style='background:rgba(74,222,128,0.07);border-radius:6px;padding:5px;'>A</div><div style='background:rgba(74,222,128,0.07);border-radius:6px;padding:5px;'>-12%</div><div style='background:rgba(74,222,128,0.07);border-radius:6px;padding:5px;'>고점×0.88</div>
</div>
<div style='margin-top:8px;font-size:10px;color:#5a7299;'>✳️ 1차 익절(+18%) 후에는 -7%로 타이트 적용</div>
</div>
</div>
</div>
<div style='margin-top:12px;background:rgba(6,182,212,0.06);border-radius:10px;padding:12px;'>
<div style='font-size:11px;font-weight:700;color:#06b6d4;margin-bottom:8px;'>📝 실전 예시 (NVDA, S등급, 진입 $100)</div>
<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:6px;text-align:center;font-size:10px;'>
<div style='background:rgba(0,0,0,0.2);border-radius:7px;padding:7px;'><div style='color:#5a7299;margin-bottom:3px;'>진입일</div><div style='color:#06b6d4;font-weight:700;'>$100</div><div style='color:#ef4444;margin-top:3px;'>손절 $92</div></div>
<div style='background:rgba(0,0,0,0.2);border-radius:7px;padding:7px;'><div style='color:#5a7299;margin-bottom:3px;'>3일 후</div><div style='color:#22c55e;font-weight:700;'>고점 $110</div><div style='color:#ef4444;margin-top:3px;'>손절 $99▲</div></div>
<div style='background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);border-radius:7px;padding:7px;'><div style='color:#f59e0b;margin-bottom:3px;'>+10% 달성</div><div style='color:#f59e0b;font-weight:700;'>$110</div><div style='color:#f59e0b;margin-top:3px;'>손절→$101</div></div>
<div style='background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);border-radius:7px;padding:7px;'><div style='color:#22c55e;margin-bottom:3px;'>+18% 달성</div><div style='color:#22c55e;font-weight:700;'>$118</div><div style='color:#22c55e;margin-top:3px;'>50% 매도!</div></div>
<div style='background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.3);border-radius:7px;padding:7px;'><div style='color:#06b6d4;margin-bottom:3px;'>고점 $140</div><div style='color:#06b6d4;font-weight:700;'>손절$126</div><div style='color:#06b6d4;margin-top:3px;'>추세 추종</div></div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ── 절대 원칙 + 현실적 기대치 ─────────────────────────────
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
<div style='background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.25);border-radius:14px;padding:18px;'>
<div style='font-size:13px;font-weight:800;color:#ef4444;margin-bottom:12px;'>⚠️ 절대 원칙 — 이것만 지켜도 손실 통제</div>
<div style='font-size:11px;line-height:2.1;color:#d8e8ff;'>
<div style='margin-bottom:4px;'><span style='background:rgba(239,68,68,0.3);border-radius:4px;padding:2px 7px;font-weight:800;color:#ef4444;'>①</span> 손절가 = 이유 없이 즉시 매도<br>
<span style='color:#5a7299;font-size:10px;margin-left:22px;'>"조금만 더 기다리면 오르겠지" = 파멸의 시작</span></div>
<div style='margin-bottom:4px;'><span style='background:rgba(239,68,68,0.3);border-radius:4px;padding:2px 7px;font-weight:800;color:#ef4444;'>②</span> 한 종목 = 계좌 최대 10%만<br>
<span style='color:#5a7299;font-size:10px;margin-left:22px;'>확신이 강해도 절대 초과 금지</span></div>
<div style='margin-bottom:4px;'><span style='background:rgba(239,68,68,0.3);border-radius:4px;padding:2px 7px;font-weight:800;color:#ef4444;'>③</span> SPY MA200 하회 = 신규 매수 전면 중단<br>
<span style='color:#5a7299;font-size:10px;margin-left:22px;'>하락장에서 개별주는 더 빠진다</span></div>
<div style='margin-bottom:4px;'><span style='background:rgba(239,68,68,0.3);border-radius:4px;padding:2px 7px;font-weight:800;color:#ef4444;'>④</span> 3연속 손절 = 1주일 완전 휴식<br>
<span style='color:#5a7299;font-size:10px;margin-left:22px;'>감정적 복수 매매 = 자산 파괴</span></div>
<div style='margin-bottom:4px;'><span style='background:rgba(239,68,68,0.3);border-radius:4px;padding:2px 7px;font-weight:800;color:#ef4444;'>⑤</span> 동시 보유 = 최대 5~6종목<br>
<span style='color:#5a7299;font-size:10px;margin-left:22px;'>그 이상은 관리 불가</span></div>
</div>
</div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""
<div style='background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.2);border-radius:14px;padding:18px;'>
<div style='font-size:13px;font-weight:800;color:#06b6d4;margin-bottom:12px;'>📊 현실적 기대치</div>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;'>
<div style='background:rgba(0,0,0,0.2);border-radius:8px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>백테스트 승률</div>
<div style='font-size:22px;font-weight:900;color:#22c55e;'>46.1%</div>
<div style='font-size:10px;color:#5a7299;'>10번 중 4~5번 수익</div>
</div>
<div style='background:rgba(0,0,0,0.2);border-radius:8px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>Profit Factor</div>
<div style='font-size:22px;font-weight:900;color:#06b6d4;'>1.85</div>
<div style='font-size:10px;color:#5a7299;'>$100 잃을 때 $185 수익</div>
</div>
</div>
<div style='font-size:11px;color:#d8e8ff;line-height:1.9;'>
<b style='color:#f59e0b;'>✳️ 승률 46%는 낮아 보이지만 정상입니다.</b><br>
수익이 나는 이유는 손절은 작게(-5~8%),<br>
수익은 크게(+18% 이상) 가져가는 구조 때문.<br><br>
<b style='color:#22c55e;'>실전 기대 성과:</b><br>
• 승률: 42~48% (데이터 지연 감안)<br>
• PF: 1.4~1.6 (슬리피지 감안)<br>
• 처음 3개월: 소액으로 연습 필수
</div>
</div>""", unsafe_allow_html=True)


st.markdown("""
<div style='text-align:center;padding:20px 0 8px;color:#5a7299;font-size:11px;'>
v2 백테스트 검증 전략 · 모든 투자 결정은 본인 책임<br>
Minervini(SEPA/VCP) · Weinstein(Stage) · O'Neil(CANSLIM) · P.T.Jones(200MA) · Livermore(Pivot)
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 탭 5 — 내 보유 종목 관리
# ══════════════════════════════════════════════════════════════════
with _tabs[4]:

    st.markdown("""
<div style='margin-bottom:16px;'>
<div style='font-size:16px;font-weight:900;color:#06b6d4;margin-bottom:4px;'>
💼 내 보유 종목 관리
</div>
<div style='font-size:12px;color:#5a7299;line-height:1.6;'>
보유 종목을 입력하면 → 전날 고점 자동 조회 → 오늘 손절가 자동 계산<br>
1차 익절(+18%) · 브레이크이븐 · 트레일링 스톱을 실시간으로 추적합니다
</div>
</div>""", unsafe_allow_html=True)

    # ── session_state로 보유 종목 목록 관리 ────────────────────────
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = [
            {"ticker": "", "shares": 10, "entry": 0.0}
        ]

    # + 버튼 / 전체 삭제
    _col_add, _col_clear, _col_spacer = st.columns([1, 1, 4])
    with _col_add:
        if st.button("➕ 종목 추가", use_container_width=True, key="port_add"):
            st.session_state["portfolio"].append(
                {"ticker": "", "shares": 10, "entry": 0.0}
            )
            st.rerun()
    with _col_clear:
        if st.button("🗑️ 전체 초기화", use_container_width=True, key="port_clear"):
            st.session_state["portfolio"] = [{"ticker": "", "shares": 10, "entry": 0.0}]
            st.rerun()

    st.markdown("---")

    # ── 종목별 입력 + 요약 카드 ────────────────────────────────────
    _to_delete = []
    for _pi, _pos in enumerate(st.session_state["portfolio"]):
        _pc1, _pc2, _pc3, _pc4, _pc5 = st.columns([1.8, 1, 1.2, 0.5, 0.5])
        with _pc1:
            _tk_input = st.text_input(
                f"종목 #{_pi+1}", value=_pos["ticker"],
                placeholder="AAPL", key=f"port_tk_{_pi}",
                label_visibility="collapsed"
            ).strip().upper()
        with _pc2:
            _sh_input = st.number_input(
                "수량", value=int(_pos["shares"]),
                min_value=1, step=1, key=f"port_sh_{_pi}",
                label_visibility="collapsed"
            )
        with _pc3:
            _en_input = st.number_input(
                "매입가", value=float(_pos["entry"]) if _pos["entry"] else 0.0,
                min_value=0.0, step=1.0, format="%.2f", key=f"port_en_{_pi}",
                label_visibility="collapsed"
            )
        with _pc4:
            _grade_sel = st.selectbox(
                "등급", ["S(10%)", "SS(8%)", "A(12%)", "익절후(7%)"],
                key=f"port_gr_{_pi}", label_visibility="collapsed"
            )
        with _pc5:
            if st.button("❌", key=f"port_del_{_pi}", help="이 종목 삭제"):
                _to_delete.append(_pi)

        # session_state 업데이트 (ticker는 반드시 대문자로)
        _tk_upper = _tk_input.strip().upper()
        st.session_state["portfolio"][_pi]["ticker"] = _tk_upper
        st.session_state["portfolio"][_pi]["shares"]  = _sh_input
        st.session_state["portfolio"][_pi]["entry"]   = _en_input

        # 종목과 매입가가 입력된 경우만 카드 표시
        if _tk_upper and _en_input > 0:
            _tk_input = _tk_upper  # 이후 코드에서도 대문자 사용
            _trail_map = {"S(10%)":0.10,"SS(8%)":0.08,"A(12%)":0.12,"익절후(7%)":0.07}
            _trail_pct = _trail_map.get(_grade_sel, 0.10)

            # 실시간 현재가 + 전날 고점 자동 조회
            _rt = fetch_realtime_price(_tk_input)
            _curr_px  = _rt.get("price", _en_input) if _rt else _en_input
            _day_h    = _rt.get("day_high", _curr_px) if _rt else _curr_px
            _day_h    = max(_day_h or _curr_px, _curr_px)

            # 핵심 가격 계산 (v2 백테스트 동일)
            _auto_stop  = round(_day_h * (1 - _trail_pct), 2)
            _be_trig    = round(_en_input * 1.10, 2)
            _be_stop_px = round(_en_input * 1.01, 2)
            _tp1_px     = round(_en_input * 1.18, 2)
            _gain_now   = (_curr_px / _en_input - 1) * 100
            _pnl_now    = (_curr_px - _en_input) * _sh_input
            _pnl_stop   = (_auto_stop - _en_input) * _sh_input

            # 단계 판정
            if _curr_px <= _auto_stop:
                _ph_c, _ph_t = "#ef4444", "🛑 손절가 도달 — 즉시 매도"
            elif _curr_px >= _tp1_px:
                _ph_c, _ph_t = "#22c55e", "🎯 +18% 달성 — 50% 매도 시점"
            elif _curr_px >= _be_trig:
                _ph_c, _ph_t = "#06b6d4", "✅ 브레이크이븐 — 손절가 올리기"
            elif _gain_now >= 0:
                _ph_c, _ph_t = "#22c55e", "📈 수익 보유 중"
            else:
                _ph_c, _ph_t = "#f59e0b", "📉 매수가 이하"

            with st.expander(
                f"📊 {_tk_input}  매입 ${_en_input:.2f}  |  현재가 ${_curr_px:.2f}  "
                f"({_gain_now:+.1f}%)  |  {_ph_t}",
                expanded=False
            ):
                # 5개 핵심 카드
                _ca, _cb, _cc, _cd, _ce = st.columns(5)
                with _ca:
                    st.markdown(f"""
<div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>🛑 오늘 손절가</div>
<div style='font-size:20px;font-weight:900;color:#ef4444;'>${_auto_stop:.2f}</div>
<div style='font-size:10px;color:#ef4444;'>고점 ${_day_h:.2f} × {1-_trail_pct:.2f}</div>
<div style='font-size:9px;color:#5a7299;margin-top:2px;'>자동 갱신됨 ⚡</div>
</div>""", unsafe_allow_html=True)
                with _cb:
                    st.markdown(f"""
<div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>⚡ 브레이크이븐</div>
<div style='font-size:20px;font-weight:900;color:{"#06b6d4" if _curr_px>=_be_trig else "#f59e0b"};'>${_be_trig:.2f}</div>
<div style='font-size:10px;color:#f59e0b;'>{"✅ 달성!" if _curr_px>=_be_trig else "+10% 목표"}</div>
<div style='font-size:9px;color:#5a7299;margin-top:2px;'>→ 손절 ${_be_stop_px:.2f}</div>
</div>""", unsafe_allow_html=True)
                with _cc:
                    st.markdown(f"""
<div style='background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>✅ 1차 익절</div>
<div style='font-size:20px;font-weight:900;color:{"#06b6d4" if _curr_px>=_tp1_px else "#22c55e"};'>${_tp1_px:.2f}</div>
<div style='font-size:10px;color:#22c55e;'>{"✅ 달성!" if _curr_px>=_tp1_px else "+18% 50%청산"}</div>
<div style='font-size:9px;color:#5a7299;margin-top:2px;'>{_sh_input//2}주 매도</div>
</div>""", unsafe_allow_html=True)
                with _cd:
                    st.markdown(f"""
<div style='background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.25);border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>📊 현재 손익</div>
<div style='font-size:20px;font-weight:900;color:{"#22c55e" if _gain_now>=0 else "#ef4444"};'>{_gain_now:+.1f}%</div>
<div style='font-size:10px;color:#5a7299;'>${_curr_px:.2f}</div>
<div style='font-size:9px;color:{"#22c55e" if _pnl_now>=0 else "#ef4444"};margin-top:2px;'>{"+" if _pnl_now>=0 else ""}${abs(_pnl_now):,.0f}</div>
</div>""", unsafe_allow_html=True)
                with _ce:
                    st.markdown(f"""
<div style='background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.25);border-radius:10px;padding:10px;text-align:center;'>
<div style='font-size:10px;color:#5a7299;'>💰 손절 청산시</div>
<div style='font-size:20px;font-weight:900;color:{"#22c55e" if _pnl_stop>=0 else "#ef4444"};'>{"+" if _pnl_stop>=0 else ""}${abs(_pnl_stop):,.0f}</div>
<div style='font-size:10px;color:#5a7299;'>{_sh_input}주 기준</div>
<div style='font-size:9px;color:#5a7299;margin-top:2px;'>${_auto_stop:.2f}에서</div>
</div>""", unsafe_allow_html=True)

                # 단계별 행동 지침
                if _curr_px <= _auto_stop:
                    st.error(f"🚨 손절가 ${_auto_stop:.2f} 도달! 지금 즉시 {_tk_input} 전량 시장가 매도하세요. 이유 불문.")
                elif _curr_px >= _tp1_px:
                    st.success(f"🎯 +18% 달성! {_sh_input//2}주 지금 바로 매도 → 손절가를 ${_be_stop_px:.2f}(진입가+1%)로 올리세요 → 잔량 {_sh_input-_sh_input//2}주는 트레일링 {int(_trail_pct*100)}% 계속 적용")
                elif _curr_px >= _be_trig:
                    st.info(f"✅ +10% 달성! 지금 즉시 손절 주문을 ${_be_stop_px:.2f}(진입가+1%)로 변경하세요. 이제 최소 +1% 수익 확정.")
                else:
                    st.markdown(f"""
<div style='background:rgba(255,255,255,0.02);border:1px solid #1e3a5f;border-radius:10px;padding:10px 14px;font-size:11px;color:#d8e8ff;line-height:1.9;'>
<b style='color:#f59e0b;'>①</b> 손절가 <b style='color:#ef4444;'>${_auto_stop:.2f}</b> 이하 도달 시 즉시 전량 매도<br>
<b style='color:#f59e0b;'>②</b> <b style='color:#f59e0b;'>${_be_trig:.2f}</b>(+10%) 달성 시 → 손절가를 ${_be_stop_px:.2f}로 올리기<br>
<b style='color:#22c55e;'>③</b> <b style='color:#22c55e;'>${_tp1_px:.2f}</b>(+18%) 달성 시 → {_sh_input//2}주 매도 + 트레일링 7% 전환<br>
<b style='color:#5a7299;'>④</b> 손절가는 오늘 고점 ${_day_h:.2f} 기준 자동 계산 (매일 자동 갱신)
</div>""", unsafe_allow_html=True)

                st.caption(f"⚡ 데이터 기준: {_rt.get('time','—') if _rt else '—'} | 손절가 = 고점 ${_day_h:.2f} × {1-_trail_pct:.2f} = ${_auto_stop:.2f}")

        st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)

    # 삭제 처리
    if _to_delete:
        for _di in sorted(_to_delete, reverse=True):
            if len(st.session_state["portfolio"]) > 1:
                st.session_state["portfolio"].pop(_di)
        st.rerun()

    # ── 포트폴리오 요약 ─────────────────────────────────────────
    _valid_pos = [
        p for p in st.session_state["portfolio"]
        if p["ticker"] and p["entry"] > 0
    ]
    if len(_valid_pos) >= 2:
        st.markdown("---")
        st.markdown("<div style='font-size:13px;font-weight:700;color:#d8e8ff;margin-bottom:10px;'>📋 포트폴리오 요약</div>", unsafe_allow_html=True)
        _total_invested = 0
        _total_pnl = 0
        _summary_cols = st.columns(min(4, len(_valid_pos)))
        for _si, _sp in enumerate(_valid_pos):
            _s_rt = fetch_realtime_price(_sp["ticker"])
            _s_curr = _s_rt.get("price", _sp["entry"]) if _s_rt else _sp["entry"]
            _s_gain = (_s_curr / _sp["entry"] - 1) * 100 if _sp["entry"] > 0 else 0
            _s_pnl  = (_s_curr - _sp["entry"]) * _sp["shares"]
            _total_invested += _sp["entry"] * _sp["shares"]
            _total_pnl += _s_pnl
            with _summary_cols[_si % 4]:
                _sc_v = "#22c55e" if _s_gain >= 0 else "#ef4444"
                st.markdown(f"""
<div style='background:rgba(255,255,255,0.03);border:1px solid #1e3a5f;border-radius:10px;padding:10px;text-align:center;margin-bottom:8px;'>
<div style='font-size:13px;font-weight:800;color:#06b6d4;'>{_sp["ticker"]}</div>
<div style='font-size:16px;font-weight:700;color:{_sc_v};'>{_s_gain:+.1f}%</div>
<div style='font-size:10px;color:#5a7299;'>{_sp["shares"]}주 · ${_s_curr:.2f}</div>
<div style='font-size:10px;color:{_sc_v};'>{"+" if _s_pnl>=0 else ""}${abs(_s_pnl):,.0f}</div>
</div>""", unsafe_allow_html=True)

        _total_return = _total_pnl / _total_invested * 100 if _total_invested > 0 else 0
        st.markdown(f"""
<div style='background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.25);border-radius:12px;padding:12px 16px;display:flex;gap:24px;'>
<div><span style='font-size:11px;color:#5a7299;'>총 투자금</span>
<span style='font-size:14px;font-weight:700;color:#d8e8ff;margin-left:8px;'>${_total_invested:,.0f}</span></div>
<div><span style='font-size:11px;color:#5a7299;'>총 손익</span>
<span style='font-size:14px;font-weight:700;color:{"#22c55e" if _total_pnl>=0 else "#ef4444"};margin-left:8px;'>{"+" if _total_pnl>=0 else ""}${abs(_total_pnl):,.0f}</span></div>
<div><span style='font-size:11px;color:#5a7299;'>수익률</span>
<span style='font-size:14px;font-weight:700;color:{"#22c55e" if _total_return>=0 else "#ef4444"};margin-left:8px;'>{_total_return:+.1f}%</span></div>
<div style='margin-left:auto;font-size:10px;color:#5a7299;'>⚡ 실시간 현재가 기준</div>
</div>""", unsafe_allow_html=True)
