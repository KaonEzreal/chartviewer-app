import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import FREE_SCAN_LIMIT, FREE_TICKER_LIMIT, TF_OPTIONS
from core.indicators import sma
from core.signal import build_signal, final_action_line

st.set_page_config(page_title="한눈에 보는 차트 분석", page_icon="📈", layout="centered")

CUSTOM_CSS = """
<style>
:root{
  --bg:#0b0f16;
  --card:#0f1724;
  --muted:#93a4bf;
  --line:#1c2a42;
  --good:#22c55e;
  --warn:#f59e0b;
  --bad:#ef4444;
  --pink:#ff4fd8;
  --blue:#40e0ff;
}
html, body, [class*="css"]  { background-color: var(--bg) !important; color: #e7eefc !important; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 580px; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0)); border:1px solid var(--line); border-radius:18px; padding:14px; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
.kv { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.k { color: var(--muted); font-size: 12px; }
.v { font-weight:700; }
.small { font-size:12px; color: var(--muted); }
.bigscore { font-size:74px; font-weight:900; line-height:1; letter-spacing:-0.04em; }
.good { color: var(--good); }
.warn { color: var(--warn); }
.bad { color: var(--bad); }
.pink { color: var(--pink); }
.blue { color: var(--blue); }
.tag { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:4px 10px; font-size:12px; margin-right:6px; }
.notice { border:1px solid #2f466d; background:#11203a; border-radius:14px; padding:12px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def money(x: float) -> str:
    return f"${x:,.2f}"



def score_class(score: int) -> str:
    if score >= 88:
        return "pink"
    if score >= 72:
        return "good"
    if score >= 55:
        return "warn"
    return "bad"



def sparkline_figure(df: pd.DataFrame, title: str):
    d = df.tail(160)
    close = d["Close"]
    ma = sma(close, 10)
    vol = d["Volume"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=close, mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=d.index, y=ma, mode="lines", name="MA10"))
    fig.add_trace(go.Bar(x=d.index, y=vol, name="Volume", opacity=0.3, yaxis="y2"))
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, x=0.02, y=0.95, font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, zeroline=False, showline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, showticklabels=False),
    )
    return fig


query = st.query_params
plan = str(query.get("plan", "FREE")).upper()
if plan not in {"FREE", "PRO", "ELITE"}:
    plan = "FREE"

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='small'>한눈에 보는 차트 분석 · 모바일 최적화 출시형 빌드</div>", unsafe_allow_html=True)
st.markdown(f"<div style='margin-top:8px'><span class='tag'>{plan} 플랜</span><span class='tag'>패키지명 com.signalscore.chartviewer</span></div>", unsafe_allow_html=True)

c1, c2 = st.columns([1.05, 0.95], vertical_alignment="center")
with c1:
    ticker = st.text_input("티커", value="AAPL", help="예: AAPL, NVDA, TSLA")
with c2:
    style = st.selectbox("스타일", ["단타", "스윙"], index=0)

tf_default = "스윙 (1D)" if style == "스윙" else "단타 (1H)"
tf_choice = st.selectbox("타임프레임", list(TF_OPTIONS.keys()), index=list(TF_OPTIONS.keys()).index(tf_default))
st.markdown("</div>", unsafe_allow_html=True)

ticker = ticker.strip().upper()
if not ticker:
    st.stop()

result = build_signal(ticker, style, tf_choice)
if result is None:
    st.error("데이터가 부족하거나 티커를 불러오지 못했어요. 다른 타임프레임을 시도해보세요.")
    st.stop()

sig, df = result
last_price = float(df["Close"].iloc[-1])
ma20_ui = float(sma(df["Close"], 20).iloc[-1])

if sig.vix_text:
    cls = "good"
    if "경고" in sig.vix_text:
        cls = "bad"
    elif "주의" in sig.vix_text:
        cls = "warn"
    st.markdown(f"<div class='card'><div class='{cls}'>{sig.vix_text}</div><div class='small' style='margin-top:4px'>TF: {tf_choice} · 스타일: {style}</div></div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center; font-size:38px; font-weight:900;' class='blue'>{ticker}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center; font-size:20px; font-weight:700;'>{money(last_price)}</div>", unsafe_allow_html=True)
st.plotly_chart(sparkline_figure(df, f"Price · {tf_choice} · MA10 · Volume"), use_container_width=True)

cls = score_class(sig.score)
st.markdown(f"<div style='text-align:center'><div class='small'>AI 추천 점수</div><div class='bigscore {cls}'>{sig.score}</div><div class='small'>등급 [{sig.grade}]</div></div>", unsafe_allow_html=True)

action = final_action_line(sig.score, sig.bias, sig.rsi, sig.vix, last_price, ma20_ui, tf_choice)
st.markdown(f"<div style='text-align:center; margin-top:10px; font-weight:800;' class='blue'>{action}</div>", unsafe_allow_html=True)

reasons = sig.score_reasons if plan in {"PRO", "ELITE"} else sig.score_reasons[:2]
st.markdown("<div style='margin-top:12px'>" + "".join([f"<div class='notice' style='margin-bottom:8px'>{i+1}. {r}</div>" for i, r in enumerate(reasons)]) + "</div>", unsafe_allow_html=True)
if plan == "FREE":
    st.markdown("<div class='small' style='margin-top:8px'>PRO에서 점수 이유 전체, 관심종목, 알림, 분석 히스토리를 열 수 있어요.</div>", unsafe_allow_html=True)

rows = []
rows.append(("출력 시간", sig.asof))
rows.append(("추세", sig.bias))
rows.append(("주간 성과", f"{sig.weekly_perf:+.2f}%"))
rows.append(("파동", sig.wave))
rows.append(("에너지", sig.energy))
if sig.obv_ratio is not None and not math.isnan(sig.obv_ratio):
    rows.append(("OBV 잔존율", f"{sig.obv_ratio:.2f}x"))
rows.append(("패턴", sig.pattern))
rows.append(("신호", f"RSI {sig.rsi:.0f} / MFI {sig.mfi:.0f}"))
rows.append(("MA20", money(ma20_ui)))
rows.append(("볼륨 확인", f"{sig.breakdown.vol_ratio:.2f}x"))
for k, v in rows:
    st.markdown(f"<div class='kv'><div class='k'>{k}</div><div class='v'>{v}</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
up_pct = (sig.target / last_price - 1) * 100
dn_pct = (sig.stop / last_price - 1) * 100
st.markdown(f"<div style='display:flex; gap:12px'><div style='flex:1; border:1px solid #1c2a42; border-radius:14px; padding:12px'><div class='k'>목표가</div><div class='v good' style='font-size:22px'>{money(sig.target)} ({up_pct:+.1f}%)</div></div><div style='flex:1; border:1px solid #1c2a42; border-radius:14px; padding:12px'><div class='k'>손절가</div><div class='v bad' style='font-size:22px'>{money(sig.stop)} ({dn_pct:+.1f}%)</div></div></div>", unsafe_allow_html=True)
rr = abs(up_pct / dn_pct) if dn_pct != 0 else float("inf")
st.markdown(f"<div class='small' style='margin-top:10px'>리스크/리워드 ≈ {rr:.2f}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("여러 티커 빠른 스캔"):
    tickers_raw = st.text_area("티커 목록 (쉼표/줄바꿈)", value="NVDA,TSLA,SMR,IREN,PGY,GOOG")
    tickers = [t.strip().upper() for t in tickers_raw.replace("\n", ",").split(",") if t.strip()]
    max_scan = 30 if plan == "ELITE" else (15 if plan == "PRO" else FREE_SCAN_LIMIT)
    if len(tickers) > max_scan:
        st.warning(f"현재 플랜에서는 최대 {max_scan}개 티커까지 스캔할 수 있어요.")
        tickers = tickers[:max_scan]
    if st.button("스캔 실행"):
        rows = []
        for t in tickers:
            res = build_signal(t, style, tf_choice)
            if not res:
                continue
            s, d = res
            last = float(d["Close"].iloc[-1])
            rows.append([t, last, s.score, s.grade, (s.target / last - 1) * 100, (s.stop / last - 1) * 100])
        if rows:
            out = pd.DataFrame(rows, columns=["Ticker", "Last", "Score", "Grade", "Target%", "Stop%"])
            out = out.sort_values("Score", ascending=False)
            st.dataframe(out, use_container_width=True, hide_index=True)
        else:
            st.info("스캔 결과가 없어요.")

st.markdown("<div class='small' style='margin-top:8px'>주의: 이 앱은 투자 조언이 아니며 개인 참고용 도구입니다.</div>", unsafe_allow_html=True)
