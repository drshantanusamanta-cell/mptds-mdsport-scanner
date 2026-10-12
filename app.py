"""
Wealthcon Investment Scorer — MDSPORT & MPTDS
Designed & developed by Dr. Shantanu Samanta (Kolkata).

UI layer only. Scoring rules live in engine.py, data fetching in data_sources.py.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_sources import DataError, fetch_company, fetch_fy_prices
from engine import (
    DE_LIMIT, MDSPORT_YEARS, MPTDS_MCAP_MIN, MPTDS_PARAMS, SENSITIVITY, SPORT_LABELS,
    VALID_ABOVE, WATCHLIST_FROM, evaluate_mdsport, evaluate_mptds, period_end,
)

# ----------------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------------
APP_NAME = "Wealthcon Investment Scorer"
AUTHOR = "Dr. Shantanu Samanta"
AUTHOR_TITLE = "Physician · Quantitative Market Analyst & Trading-Systems Developer"
AUTHOR_PLACE = "Kolkata, India"
YEAR = date.today().year

st.set_page_config(
    page_title=f"{APP_NAME} · {AUTHOR}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, collapsed on phones
    menu_items={"About": f"**{APP_NAME}**\n\nDesigned & developed by **{AUTHOR}**, {AUTHOR_PLACE}."},
)

# ----------------------------------------------------------------------------
# Palette & styles (responsive: fluid grids + clamp() type + breakpoints)
# ----------------------------------------------------------------------------
C = {
    "indigo": "#4338CA", "violet": "#7C3AED", "pink": "#DB2777", "teal": "#0D9488",
    "sky": "#0284C7", "amber": "#F59E0B", "green": "#16A34A", "red": "#DC2626",
}
VERDICT_STYLE = {
    "VALID": ("linear-gradient(135deg,#059669,#22C55E)", "✅"),
    "WATCHLIST": ("linear-gradient(135deg,#D97706,#FBBF24)", "👀"),
    "REJECTED": ("linear-gradient(135deg,#B91C1C,#F43F5E)", "⛔"),
    "INVALID": ("linear-gradient(135deg,#B91C1C,#F43F5E)", "⛔"),
}
SERIES_COLORS = {"Price": C["violet"], "Tax": C["teal"], "Dividend": C["amber"], "Sales": C["sky"]}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {font-family:'Inter',system-ui,sans-serif;}
.block-container {padding-top:3.4rem; padding-bottom:2rem; max-width:1320px;}

/* ---------- hero ---------- */
.hero {background:linear-gradient(120deg,#312E81 0%,#6D28D9 45%,#DB2777 100%);
  color:#fff; border-radius:22px; padding:clamp(18px,3.2vw,34px) clamp(18px,3.5vw,40px);
  box-shadow:0 14px 40px -18px rgba(109,40,217,.65); position:relative; overflow:hidden;}
.hero:after {content:""; position:absolute; right:-60px; top:-60px; width:240px; height:240px;
  border-radius:50%; background:radial-gradient(circle,rgba(255,255,255,.22),transparent 70%);}
.hero h1 {font-family:'Poppins',sans-serif; font-size:clamp(1.45rem,3.6vw,2.45rem);
  margin:0 0 .25rem 0; color:#fff; line-height:1.15; padding:0;}
.hero p {margin:.15rem 0 0 0; opacity:.93; font-size:clamp(.88rem,1.6vw,1.02rem);}
.byline {display:inline-flex; align-items:center; gap:.55rem; flex-wrap:wrap; margin-top:.9rem;
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28); border-radius:999px;
  padding:.38rem .9rem .38rem .38rem; backdrop-filter:blur(4px); font-size:clamp(.78rem,1.4vw,.92rem);}
.avatar {width:30px; height:30px; border-radius:50%; display:grid; place-items:center;
  background:linear-gradient(135deg,#FBBF24,#F97316); color:#3B0764; font-weight:700; font-size:.8rem;}
.byline b {font-weight:600;}

/* ---------- grids ---------- */
.grid {display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr)); margin:14px 0;}
.grid-2 {display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr)); margin:14px 0;}

/* ---------- cards ---------- */
.stat {border-radius:16px; padding:14px 16px; color:#fff; min-height:92px;
  box-shadow:0 8px 22px -14px rgba(0,0,0,.55);}
.stat .k {font-size:.74rem; text-transform:uppercase; letter-spacing:.07em; opacity:.88;}
.stat .v {font-family:'Poppins',sans-serif; font-size:clamp(1.15rem,2.3vw,1.55rem); font-weight:600; margin-top:2px;}
.stat .s {font-size:.8rem; opacity:.9;}
.verdict {border-radius:20px; padding:clamp(16px,2.4vw,24px); color:#fff;
  box-shadow:0 14px 34px -18px rgba(0,0,0,.6);}
.verdict .t {font-size:.8rem; letter-spacing:.09em; text-transform:uppercase; opacity:.9;}
.verdict .big {font-family:'Poppins',sans-serif; font-size:clamp(1.7rem,4.4vw,2.6rem); font-weight:700; line-height:1.1; margin:.2rem 0;}
.verdict .r {font-size:.92rem; opacity:.95;}
.pill {display:inline-block; padding:.18rem .62rem; border-radius:999px; font-size:.76rem; font-weight:600;
  margin:.4rem .3rem 0 0; background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.35);}
.gate {border-radius:16px; padding:14px 16px; border:1px solid rgba(127,127,127,.22);
  background:rgba(127,127,127,.06); border-top:5px solid var(--c);}
.gate h4 {margin:0 0 .25rem 0; font-family:'Poppins',sans-serif; font-size:1.02rem; padding:0;}
.gate .row {display:flex; justify-content:space-between; gap:8px; font-size:.86rem; padding:2px 0;}
.gate .muted {opacity:.7; font-size:.78rem; margin-top:4px;}
.legend {display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 10px;}
.legend span {padding:.22rem .6rem; border-radius:8px; color:#fff; font-size:.78rem; font-weight:600;}
.note {border-left:4px solid #F59E0B; background:rgba(245,158,11,.10); border-radius:10px;
  padding:.55rem .8rem; font-size:.86rem; margin:.35rem 0;}
.section-title {font-family:'Poppins',sans-serif; font-size:clamp(1.05rem,2vw,1.3rem); font-weight:600; margin:.6rem 0 .2rem;}

/* ---------- author card ---------- */
.author {border-radius:18px; padding:16px; color:#fff;
  background:linear-gradient(150deg,#0F766E 0%,#0369A1 55%,#4338CA 100%);
  box-shadow:0 12px 30px -18px rgba(3,105,161,.8);}
.author .name {font-family:'Poppins',sans-serif; font-size:1.08rem; font-weight:600; margin-top:.5rem;}
.author .role {font-size:.82rem; opacity:.95; margin-top:.15rem; line-height:1.4;}
.author .tags {margin-top:.55rem;}
.author .tags span {display:inline-block; font-size:.7rem; padding:.15rem .5rem; margin:.2rem .2rem 0 0;
  border-radius:999px; background:rgba(255,255,255,.18);}
.author .big-avatar {width:46px; height:46px; border-radius:14px; display:grid; place-items:center;
  background:linear-gradient(135deg,#FBBF24,#F97316); color:#3B0764; font-weight:700; font-family:'Poppins',sans-serif;}

/* ---------- footer ---------- */
.footer {margin-top:2rem; border-radius:20px; padding:clamp(16px,2.6vw,26px);
  background:linear-gradient(120deg,rgba(67,56,202,.10),rgba(219,39,119,.10));
  border:1px solid rgba(127,127,127,.2); font-size:.84rem;}
.footer .who {font-family:'Poppins',sans-serif; font-size:1rem; font-weight:600;}
.footer .disc {opacity:.78; margin-top:.6rem; line-height:1.5;}

/* ---------- widgets ---------- */
div.stButton > button[kind="primary"], div.stFormSubmitButton > button {
  background:linear-gradient(120deg,#6D28D9,#DB2777); border:0; color:#fff; font-weight:600;
  border-radius:12px; min-height:2.75rem;}
div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button:hover {filter:brightness(1.08); color:#fff;}
[data-testid="stForm"] {border-radius:18px; border:1px solid rgba(127,127,127,.22);
  background:rgba(127,127,127,.04);}
.stTabs [data-baseweb="tab-list"] {gap:6px; flex-wrap:wrap;}
.stTabs [data-baseweb="tab"] {border-radius:10px 10px 0 0; padding:8px 14px;}

/* ---------- tablet ---------- */
@media (max-width: 1024px) {
  .block-container {padding-left:1.2rem; padding-right:1.2rem;}
}
/* ---------- phone ---------- */
@media (max-width: 640px) {
  .block-container {padding:3.2rem 0.7rem 1.5rem;}
  .byline {border-radius:16px;}
  .hero {border-radius:16px;}
  .hero:after {display:none;}
  .stat {min-height:auto; padding:12px 14px;}
  .verdict {border-radius:16px;}
  .stTabs [data-baseweb="tab"] {padding:6px 10px; font-size:.85rem;}
  .footer {border-radius:16px;}
}
</style>
"""


def html(markup: str) -> None:
    """Render HTML. Lines are stripped so Markdown never mistakes indentation for code."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def fmt_num(v: float, digits: int = 1, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{digits}f}{suffix}"


def fy_short(label: str) -> str:
    """'Mar 2025' -> 'FY25'; other year-ends keep their month: 'Dec 2024' -> "Dec '24"."""
    mon, yr = label.split()
    return f"FY{yr[-2:]}" if mon == "Mar" else f"{mon} '{yr[-2:]}"


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def style_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height, autosize=True, margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(127,127,127,.18)", zeroline=False)
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, theme="streamlit")


# ----------------------------------------------------------------------------
# Cached data access
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_company(symbol: str, basis: str):
    return fetch_company(symbol, basis)


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(symbol: str, labels: tuple[str, ...]):
    return fetch_fy_prices(symbol, list(labels))


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def initials(name: str) -> str:
    parts = [p for p in name.replace("Dr.", "").split() if p]
    return "".join(p[0] for p in parts[:2]).upper()


def render_hero() -> None:
    html(f"""
    <div class="hero">
      <h1>📈 {APP_NAME}</h1>
      <p><b>MDSPORT</b> quality-growth scoring &nbsp;·&nbsp; <b>MPTDS</b> trend-sustainability test &nbsp;·&nbsp; NSE / BSE equities</p>
      <div class="byline"><span class="avatar">{initials(AUTHOR)}</span>
        <span>Designed &amp; developed by <b>{AUTHOR}</b> · {AUTHOR_PLACE}</span></div>
    </div>
    """)


def author_card() -> str:
    return f"""
    <div class="author">
      <div class="big-avatar">{initials(AUTHOR)}</div>
      <div class="name">{AUTHOR}</div>
      <div class="role">{AUTHOR_TITLE}<br>{AUTHOR_PLACE}</div>
      <div class="tags"><span>Clinical medicine</span><span>Quant analysis</span>
        <span>NSE · NIFTY systems</span><span>Python · Streamlit</span></div>
    </div>
    """


def render_sidebar() -> None:
    with st.sidebar:
        html(author_card())
        st.markdown("")
        st.markdown("##### How to read the verdicts")
        html(f"""
        <div class="legend">
          <span style="background:{C['green']}">VALID</span>
          <span style="background:{C['amber']}">WATCHLIST</span>
          <span style="background:{C['red']}">REJECTED / INVALID</span>
        </div>
        """)
        st.caption(
            f"**MDSPORT**: D/E < {DE_LIMIT:.1f} gate, then 5 SPORT parameters × {MDSPORT_YEARS} years × 0–4 pts. "
            f">{VALID_ABOVE} valid · {WATCHLIST_FROM}–{VALID_ABOVE} watchlist.\n\n"
            f"**MPTDS**: Market cap ≥ ₹{MPTDS_MCAP_MIN:,.0f} Cr and rising Price, Tax, Dividend and Sales trends."
        )


def render_form() -> dict | None:
    with st.form("query_form"):
        c1, c2 = st.columns([1.6, 1])  # two rows of two: comfortable on tablets, stacks on phones
        c3, c4 = st.columns([1.6, 1])
        symbol = c1.text_input("NSE symbol / BSE code", value=st.session_state.get("last_symbol", "TCS"),
                               placeholder="e.g. TCS, INFY, ASIANPAINT").strip().upper()
        basis = c2.selectbox("Statements", ["Auto", "Consolidated", "Standalone"],
                             help="Auto = consolidated, falling back to standalone when consolidated is empty.")
        sens = c3.selectbox("MPTDS sensitivity", list(SENSITIVITY),
                            format_func=lambda k: f"{k} (≥{SENSITIVITY[k]['min_growth']:.0f}%, R²≥{SENSITIVITY[k]['min_r2']})")
        years = c4.slider("MPTDS years", 5, 10, 10, help="Trend window for MPTDS (5–10 years). MDSPORT always uses 5.")

        with st.expander("Advanced options"):
            a1, a2, a3 = st.columns(3)
            use_de = a1.checkbox("Enter D/E manually", help="Use when you have the exact Non-Current Liabilities / Total Equity figure.")
            de_val = a1.number_input("D/E (NCL ÷ Equity)", min_value=0.0, max_value=20.0, value=0.0, step=0.01)
            large_min = a2.number_input("Large-cap cut-off (₹ Cr)", min_value=0.0, value=float(MPTDS_MCAP_MIN), step=500.0)
            mid_min = a3.number_input("Mid-cap cut-off (₹ Cr)", min_value=0.0, value=5000.0, step=500.0,
                                      help="The framework only fixes the ₹10,000 Cr line; set mid/small to your own convention.")

        go_btn = st.form_submit_button("🚀  Analyse", type="primary", width="stretch")

    if go_btn:
        st.session_state["last_symbol"] = symbol
        st.session_state["query"] = dict(
            symbol=symbol, basis=basis, sens=sens, years=years,
            de_override=de_val if use_de else None, large_min=large_min, mid_min=mid_min,
        )
    return st.session_state.get("query")


def stat(label: str, value: str, sub: str, grad: str) -> str:
    return f'<div class="stat" style="background:{grad}"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def render_company_strip(data, years_used: int) -> None:
    html(f"""
    <div class="grid">
      {stat("Company", data.name[:38], f"{data.symbol} · {data.basis}", "linear-gradient(135deg,#4338CA,#7C3AED)")}
      {stat("Market cap", "₹" + fmt_num(data.market_cap, 0) + " Cr", "Screener.in", "linear-gradient(135deg,#0369A1,#0EA5E9)")}
      {stat("Current price", "₹" + fmt_num(data.current_price, 2), "Screener.in", "linear-gradient(135deg,#0F766E,#14B8A6)")}
      {stat("Annual data", f"{years_used} yrs", f"{data.pnl.index[0]} → {data.pnl.index[-1]}", "linear-gradient(135deg,#BE185D,#F472B6)")}
    </div>
    """)


def verdict_card(title: str, verdict: str, big: str, reason: str, pills: list[str]) -> str:
    grad, icon = VERDICT_STYLE[verdict]
    pills_html = "".join(f'<span class="pill">{p}</span>' for p in pills)
    return f"""<div class="verdict" style="background:{grad}">
      <div class="t">{title}</div><div class="big">{icon} {big}</div>
      <div class="r">{reason}</div><div>{pills_html}</div></div>"""


# ----------------------------------------------------------------------------
# MDSPORT tab
# ----------------------------------------------------------------------------
POINT_COLORS = ["#DC2626", "#F97316", "#FACC15", "#4ADE80", "#16A34A"]


def mdsport_gauge(score: int) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": " / 100", "font": {"size": 30}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#4338CA", "thickness": 0.28},
            "steps": [
                {"range": [0, WATCHLIST_FROM], "color": "rgba(220,38,38,.35)"},
                {"range": [WATCHLIST_FROM, VALID_ABOVE], "color": "rgba(245,158,11,.40)"},
                {"range": [VALID_ABOVE, 100], "color": "rgba(22,163,74,.40)"},
            ],
            "threshold": {"line": {"color": "#DB2777", "width": 4}, "value": score},
        },
        title={"text": "MDSPORT score"},
    ))
    fig = style_fig(fig, height=270)
    fig.update_layout(margin=dict(l=34, r=34, t=40, b=8))
    return fig


def mdsport_heatmap(res) -> go.Figure:
    pts = res.points
    vals = res.values
    names = {"S": "Sales", "P": "PAT", "O": "OPM", "R": "ROCE", "T": "Tax"}
    short = [f"{i[0]} · {names[i[0]]}" for i in pts.index]
    text = [[f"{fmt_num(vals.iat[r, c], 1, '%')}<br><b>{int(pts.iat[r, c])} pt</b>"
             for c in range(pts.shape[1])] for r in range(pts.shape[0])]
    scale = [[0.0, POINT_COLORS[0]], [0.25, POINT_COLORS[1]], [0.5, POINT_COLORS[2]],
             [0.75, POINT_COLORS[3]], [1.0, POINT_COLORS[4]]]
    fig = go.Figure(go.Heatmap(
        z=pts.to_numpy(), x=[fy_short(c) for c in pts.columns], y=short, zmin=0, zmax=4, colorscale=scale,
        text=text, texttemplate="%{text}", textfont={"size": 11, "color": "#111827"},
        xgap=4, ygap=4, showscale=False,
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig = style_fig(fig, height=360)
    fig.update_layout(title=dict(text="Year-by-year points (0–4 each)", x=0, font=dict(size=14)))
    return fig


def render_mdsport(res) -> None:
    show_chart(mdsport_gauge(res.score))
    with st.container():
        de_color = C["green"] if res.de_pass else (C["red"] if not math.isnan(res.de_ratio) else C["amber"])
        de_txt = "PASS" if res.de_pass else ("FAIL — auto reject" if not math.isnan(res.de_ratio) else "Not available")
        html(f"""
        <div class="grid-2">
        <div class="gate" style="--c:{de_color}">
          <h4>D · Debt-to-Equity gate</h4>
          <div class="row"><span>D/E ratio</span><b>{fmt_num(res.de_ratio, 2)}</b></div>
          <div class="row"><span>Limit</span><b>&lt; {DE_LIMIT:.2f}</b></div>
          <div class="row"><span>Result</span><b style="color:{de_color}">{de_txt}</b></div>
          <div class="muted">Period: {res.de_period}. Proxy = Borrowings ÷ (Equity capital + Reserves); framework definition is NCL ÷ Total equity.</div>
        </div>
        <div class="gate" style="--c:{C['sky']}">
          <h4>M · Market capitalisation</h4>
          <div class="row"><span>Market cap</span><b>₹{fmt_num(res.mcap, 0)} Cr</b></div>
          <div class="row"><span>Class</span><b>{res.mcap_class}</b></div>
          <div class="muted">Classification only — does not add points.</div>
        </div>
        </div>
        """)

    html('<div class="section-title">SPORT scorecard</div>')
    html("""<div class="legend">""" + "".join(
        f'<span style="background:{POINT_COLORS[i]};color:{"#111827" if i in (2, 3) else "#fff"}">{lab}</span>'
        for i, lab in enumerate(["&lt;5% → 0", "5–10% → 1", "10–15% → 2", "15–20% → 3", "&gt;20% → 4"])
    ) + "</div>")
    show_chart(mdsport_heatmap(res))

    totals = res.param_totals
    fig = go.Figure(go.Bar(
        x=totals.to_numpy(), y=[i.split(" · ")[1] for i in totals.index], orientation="h",
        marker=dict(color=[C["sky"], C["violet"], C["teal"], C["amber"], C["pink"]]),
        text=[f"{int(v)}/20" for v in totals], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}/20<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 24])
    fig.update_yaxes(autorange="reversed")
    fig = style_fig(fig, height=260)
    fig.update_layout(title=dict(text="Points per parameter (max 20)", x=0, font=dict(size=14)))
    show_chart(fig)

    st.caption(f"Data coverage: {res.coverage:.0%} of the 25 scoring cells.")
    for n in res.notes:
        html(f'<div class="note">⚠️ {n}</div>')


# ----------------------------------------------------------------------------
# MPTDS tab
# ----------------------------------------------------------------------------
def render_mptds(res) -> None:
    mcol = C["green"] if res.mcap_pass else C["red"]
    cards = [f"""<div class="gate" style="--c:{mcol}">
      <h4>{'✅' if res.mcap_pass else '❌'} M · Market cap</h4>
      <div class="row"><span>Value</span><b>₹{fmt_num(res.mcap, 0)} Cr</b></div>
      <div class="row"><span>Needs</span><b>≥ ₹{MPTDS_MCAP_MIN:,.0f} Cr</b></div></div>"""]
    for p in MPTDS_PARAMS:
        m = res.metrics[p]
        col = C["green"] if m["passes"] else C["red"]
        note = f'<div class="muted">{m["note"]}</div>' if m["note"] else ""
        cards.append(f"""<div class="gate" style="--c:{col}">
          <h4>{'✅' if m['passes'] else '❌'} {p[0]} · {p}</h4>
          <div class="row"><span>Avg YoY growth</span><b>{fmt_num(m['avg_growth'], 1, '%')}</b></div>
          <div class="row"><span>R² (trend fit)</span><b>{fmt_num(m['r2'], 2)}</b></div>
          <div class="row"><span>CAGR</span><b>{fmt_num(m['cagr'], 1, '%')}</b></div>
          <div class="row"><span>Years used</span><b>{m['n_years']}</b></div>{note}</div>""")
    html('<div class="grid">' + "".join(cards) + "</div>")
    st.caption(f"Pass rule: average YoY growth ≥ {res.min_growth:.0f}%, R² ≥ {res.min_r2} and a rising regression line.")

    html('<div class="section-title">Trend lines</div>')
    units = {"Price": "₹", "Tax": "₹ Cr", "Dividend": "₹ Cr", "Sales": "₹ Cr"}
    for i in range(0, len(MPTDS_PARAMS), 2):
        cols = st.columns(2)  # side-by-side on desktop/tablet, stacked on phones
        for col, p in zip(cols, MPTDS_PARAMS[i:i + 2]):
            s = res.series[p].dropna()
            m = res.metrics[p]
            fig = go.Figure()
            xl = [fy_short(l) for l in s.index]
            fig.add_trace(go.Bar(x=xl, y=s.to_numpy(), name=p,
                                 marker=dict(color=SERIES_COLORS[p], opacity=0.85),
                                 hovertemplate="%{x}: %{y:,.2f}<extra></extra>"))
            if not math.isnan(m["slope"]) and len(s) >= 3:
                xs = np.array([period_end(l).year + period_end(l).month / 12.0 for l in s.index])
                fig.add_trace(go.Scatter(x=xl, y=m["intercept"] + m["slope"] * xs, mode="lines",
                                         name="Regression", line=dict(color="#DB2777", width=3, dash="dot")))
            badge = "✅" if m["passes"] else "❌"
            fig = style_fig(fig, height=290)
            fig.update_layout(title=dict(text=f"{badge} {p} ({units[p]}) · R² {fmt_num(m['r2'], 2)}", x=0, font=dict(size=14)),
                              showlegend=False)
            with col:
                show_chart(fig)

    norm = res.series.apply(lambda c: c / c[c > 0].dropna().iloc[0] * 100 if (c > 0).any() else c * np.nan)
    fig = go.Figure()
    for p in MPTDS_PARAMS:
        fig.add_trace(go.Scatter(x=[fy_short(l) for l in norm.index], y=norm[p], mode="lines+markers", name=p,
                                 line=dict(color=SERIES_COLORS[p], width=3), connectgaps=False))
    fig = style_fig(fig, height=340)
    fig.update_layout(title=dict(text=f"All four rebased to 100 ({len(norm)}-year window)", x=0, font=dict(size=14)),
                      hovermode="x unified")
    show_chart(fig)


# ----------------------------------------------------------------------------
# Data & methodology tabs
# ----------------------------------------------------------------------------
def render_data(data, mds, mpt) -> None:
    st.markdown("**MDSPORT inputs (%, last 5 years)**")
    st.dataframe(mds.values.style.format("{:.1f}", na_rep="—"), width="stretch")
    st.markdown("**MPTDS series**")
    st.dataframe(mpt.series.style.format("{:,.2f}", na_rep="—"), width="stretch")
    with st.expander("Full Screener P&L"):
        st.dataframe(data.pnl, width="stretch")
    with st.expander("Balance sheet & ratios"):
        st.dataframe(data.balance_sheet, width="stretch")
        st.dataframe(data.ratios, width="stretch")
    csv = pd.concat({"MPTDS": mpt.series.T, "MDSPORT": mds.values}).to_csv().encode()
    st.download_button("⬇️  Download inputs (CSV)", csv, file_name=f"{data.symbol}_wealthcon_scores.csv",
                       mime="text/csv", width="stretch")
    st.caption(f"Source: [{data.url}]({data.url}) and Yahoo Finance (split-adjusted financial-year-end close).")


def render_method() -> None:
    st.markdown(f"""
#### MDSPORT — quality-growth score
**Screening filters (MD)**
- **M · Market cap** — size class (Large / Mid / Small). Classification only.
- **D · Debt-to-Equity** — Non-Current Liabilities ÷ Total Equity. **D/E ≥ {DE_LIMIT:.2f} is an automatic REJECT**, whatever the score.

**Scoring parameters (SPORT)** — each graded **0–4 points for every one of the last {MDSPORT_YEARS} years**
| Parameter | Measured as |
|---|---|
| **S** Sales growth % | YoY growth in sales |
| **P** PAT growth % | YoY growth in net profit |
| **O** OPM % | Operating profit margin level |
| **R** ROCE % | Return on capital employed level |
| **T** Tax growth % | YoY growth in tax paid (PBT − PAT) |

Slabs: &lt;5% → 0 · 5–10% → 1 · 10–15% → 2 · 15–20% → 3 · &gt;20% → 4.
Maximum = 5 × 5 × 4 = **100**. **Valid > {VALID_ABOVE}**, **Watchlist {WATCHLIST_FROM}–{VALID_ABOVE}**, **Rejected < {WATCHLIST_FROM}**.

#### MPTDS — trend-sustainability test
- **M** Market cap ≥ ₹{MPTDS_MCAP_MIN:,.0f} Cr
- **P** Price — financial-year-end (March) close
- **T** Tax paid — PBT − PAT
- **D** Dividend paid — PAT × Dividend payout %
- **S** Sales — revenue from operations

Each of P, T, D, S must be in a rising trend over 5–10 years: **Moderate** = average growth ≥ 5% and R² ≥ 0.25; **Strict** = ≥ 10% and R² ≥ 0.55. **VALID** only when M and all four trends pass.

#### Implementation choices
- Growth off a zero or negative base is treated as undefined (scores 0 / skipped) instead of producing infinite values.
- A value of exactly 20% scores 3 points ("> 20%" is needed for 4).
- D/E uses Screener's Borrowings ÷ (Equity capital + Reserves) as a proxy for NCL ÷ Equity; override it under *Advanced options* when you have the exact figure.
- Prices are never back-filled: a year with no trade near its year-end is left blank.
- Banks and NBFCs don't report OPM/ROCE in the same way, so MDSPORT under-scores them by design.
""")


def render_footer() -> None:
    html(f"""
    <div class="footer">
      <div class="who">🩺📊 {AUTHOR}</div>
      <div>{AUTHOR_TITLE} · {AUTHOR_PLACE}</div>
      <div class="disc"><b>Disclaimer:</b> This tool is for education and research only and is not investment advice
      or a recommendation to buy or sell any security. Data is pulled from third-party sources (Screener.in, Yahoo Finance)
      and may be delayed, incomplete or revised. Verify figures independently and consult a SEBI-registered investment
      adviser before investing.</div>
      <div class="disc">© {YEAR} {AUTHOR}. All rights reserved.</div>
    </div>
    """)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
def main() -> None:
    html(CSS)
    render_sidebar()
    render_hero()
    st.markdown("")
    q = render_form()

    if not q:
        html(f"""
        <div class="grid-2">
          {verdict_card("Step 1", "WATCHLIST", "Enter a symbol", "Type an NSE symbol (e.g. TCS) or a BSE code and press Analyse.", ["Screener.in", "Yahoo Finance"])}
          {verdict_card("Step 2", "VALID", "Read the verdicts", "MDSPORT scores quality growth out of 100; MPTDS checks four long-term uptrends.", ["MDSPORT", "MPTDS"])}
        </div>""")
        render_footer()
        return

    if not q["symbol"]:
        st.warning("Please enter a symbol.")
        render_footer()
        return

    try:
        with st.spinner(f"Fetching filings for {q['symbol']} from Screener.in…"):
            data = load_company(q["symbol"], q["basis"])
    except DataError as e:
        st.error(str(e))
        render_footer()
        return

    pnl_m = data.pnl.tail(q["years"])
    with st.spinner("Fetching year-end prices from Yahoo Finance…"):
        prices = load_prices(data.symbol, tuple(pnl_m.index))

    cfg = SENSITIVITY[q["sens"]]
    mds = evaluate_mdsport(data.pnl, data.ratios, data.balance_sheet, data.market_cap,
                           de_override=q["de_override"], large_min=q["large_min"], mid_min=q["mid_min"])
    mpt = evaluate_mptds(data.market_cap, prices, pnl_m, cfg["min_growth"], cfg["min_r2"])

    render_company_strip(data, len(data.pnl))
    passed = sum(m["passes"] for m in mpt.metrics.values()) + int(mpt.mcap_pass)
    mds_card = verdict_card(
        "MDSPORT · quality growth", mds.verdict, f"{mds.score}/100 · {mds.verdict}", mds.reason,
        [f"D/E {fmt_num(mds.de_ratio, 2)}", mds.mcap_class, f"Coverage {mds.coverage:.0%}"],
    )
    gate_pills = [("✓ " if mpt.mcap_pass else "✗ ") + "Mcap"] + [
        ("✓ " if mpt.metrics[p]["passes"] else "✗ ") + p for p in MPTDS_PARAMS
    ]
    mpt_card = verdict_card(
        "MPTDS · trend sustainability", mpt.verdict, mpt.verdict,
        f"{passed} of 5 gates passed · {q['sens']} sensitivity", gate_pills,
    )
    html(f'<div class="grid-2">{mds_card}{mpt_card}</div>')
    if prices.isna().all():
        html('<div class="note">⚠️ Yahoo Finance returned no prices for this symbol, so the MPTDS Price trend cannot pass.</div>')

    t1, t2, t3, t4 = st.tabs(["🎯 MDSPORT", "📈 MPTDS", "📋 Data", "📘 Method"])
    with t1:
        render_mdsport(mds)
    with t2:
        render_mptds(mpt)
    with t3:
        render_data(data, mds, mpt)
    with t4:
        render_method()

    render_footer()


main()
