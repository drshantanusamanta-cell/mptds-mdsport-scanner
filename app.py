import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from scipy.stats import linregress
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Wealthcon Investment Scorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SCRAPING UTILITIES ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener_data(symbol: str):
    """Fetches P&L, balance sheet, and market cap metrics from screener.in."""
    clean_sym = symbol.strip().upper().replace(".NS", "")
    url = f"https://www.screener.in/company/{clean_sym}/consolidated/"
    res = requests.get(url, headers=HEADERS, timeout=12)
    
    if res.status_code != 200:
        url = f"https://www.screener.in/company/{clean_sym}/"
        res = requests.get(url, headers=HEADERS, timeout=12)
        if res.status_code != 200:
            return None

    soup = BeautifulSoup(res.text, "html.parser")
    
    # 1. Extract Market Cap
    market_cap = 0.0
    for li in soup.select("ul#top-ratios li"):
        name = li.select_one(".name")
        val = li.select_one(".nowrap .number")
        if name and "Market Cap" in name.get_text():
            if val:
                market_cap = float(val.get_text().replace(",", "").strip())
            break
            
    # 2. Extract Profit & Loss Statement Table
    pnl_section = soup.find("section", {"id": "profit-loss"})
    if not pnl_section:
        return None
        
    table = pnl_section.find("table")
    if not table:
        return None
        
    headers = [th.get_text().strip() for th in table.find_all("th") if th.get_text().strip()]
    years = [h for h in headers if any(char.isdigit() for char in h) and "TTM" not in h]
    
    data_dict = {}
    for row in table.find("tbody").find_all("tr"):
        row_name_col = row.find("td", class_="name")
        if not row_name_col:
            continue
        row_name = row_name_col.get_text().strip()
        cols = [td.get_text().replace(",", "").replace("%", "").strip() for td in row.find_all("td")[1:]]
        
        num_years = len(years)
        vals = []
        for c in cols[:num_years]:
            try:
                vals.append(float(c))
            except ValueError:
                vals.append(np.nan)
        data_dict[row_name] = vals

    df_pnl = pd.DataFrame(data_dict, index=years)
    return {"market_cap": market_cap, "pnl": df_pnl, "symbol": clean_sym}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_annual_prices(symbol: str, years: list):
    """Fetches yearly closing prices from Yahoo Finance matching financial year ends."""
    ticker = f"{symbol.strip().upper()}.NS"
    hist = yf.Ticker(ticker).history(period="12y")
    if hist.empty:
        return pd.Series(dtype=float)
        
    hist.index = pd.to_datetime(hist.index)
    annual = hist['Close'].resample('YE-MAR').last()
    
    price_map = {}
    for dt, close_val in annual.items():
        label = dt.strftime("%b %Y")
        price_map[label] = round(float(close_val), 2)
        
    return pd.Series(price_map)

# --- QUANTITATIVE ENGINE (MPTDS & MDSPORT) ---
def compute_trend_metrics(series: pd.Series):
    """Calculates linear regression slope, R², and compound/average annual percentage growth."""
    series = series.dropna()
    n = len(series)
    if n < 3:
        return {"growth_pct": 0.0, "r2": 0.0, "valid": False}
        
    y = series.values
    x = np.arange(n)
    
    slope, intercept, r_val, p_val, std_err = linregress(x, y)
    r2 = r_val ** 2
    
    pct_changes = pd.Series(y).pct_change().dropna() * 100
    avg_growth = float(pct_changes.mean())
    
    return {
        "slope": float(slope),
        "r2": float(r2),
        "growth_pct": avg_growth,
        "n_years": n
    }

def evaluate_mptds(mkt_cap, price_series, pnl_df, min_growth=5.0, min_r2=0.25):
    """Computes parameters for MPTDS: Market Cap, Price, Tax, Dividend, Sales."""
    m_valid = mkt_cap >= 10000.0
    
    sales = pnl_df.get("Sales", pd.Series(dtype=float))
    pat = pnl_df.get("Net Profit", pd.Series(dtype=float))
    pbt = pnl_df.get("Profit before tax", pd.Series(dtype=float))
    payout = pnl_df.get("Dividend Payout %", pd.Series(dtype=float))
    
    tax = pbt - pat
    tax = tax.apply(lambda x: max(x, 0.0))
    dividend = (pat * payout / 100.0).apply(lambda x: max(x, 0.0))
    
    metrics = {
        "Price": compute_trend_metrics(price_series),
        "Tax": compute_trend_metrics(tax),
        "Dividend": compute_trend_metrics(dividend),
        "Sales": compute_trend_metrics(sales),
    }
    
    for k, v in metrics.items():
        v["meets_criteria"] = (v["growth_pct"] >= min_growth) and (v["r2"] >= min_r2)
        
    all_trends_valid = all(v["meets_criteria"] for v in metrics.values())
    overall_verdict = "VALID" if (m_valid and all_trends_valid) else "INVALID"
    
    raw_table = pd.DataFrame({
        "Price (₹)": price_series,
        "Tax Paid (Cr)": tax,
        "Dividend (Cr)": dividend,
        "Sales (Cr)": sales
    }).dropna(how='all')
    
    return {
        "m_valid": m_valid,
        "overall_verdict": overall_verdict,
        "metrics": metrics,
        "raw_table": raw_table
    }

def evaluate_mdsport(mkt_cap, pnl_df):
    """Calculates 0-100 score across 5 fundamental momentum factors over last 5 years."""
    df5 = pnl_df.tail(5)
    score_breakdown = {}
    
    sales = df5.get("Sales", pd.Series(dtype=float)).pct_change().dropna()
    score_sales = 20 if (sales.mean() > 0.12 and (sales > 0).all()) else 12 if sales.mean() > 0.08 else 5
    score_breakdown["Sales Momentum"] = (score_sales, 20)
    
    pat = df5.get("Net Profit", pd.Series(dtype=float)).pct_change().dropna()
    score_pat = 25 if (pat.mean() > 0.15 and (pat > 0).all()) else 15 if pat.mean() > 0.08 else 5
    score_breakdown["Profit After Tax (PAT)"] = (score_pat, 25)
    
    opm = df5.get("OPM %", pd.Series(dtype=float))
    score_opm = 20 if (opm.mean() >= 18.0 and opm.std() < 4.0) else 12 if opm.mean() >= 12.0 else 5
    score_breakdown["Operating Margin Stability"] = (score_opm, 20)
    
    pbt = df5.get("Profit before tax", pd.Series(dtype=float))
    tax_paid = (pbt - df5.get("Net Profit", pd.Series(dtype=float))).pct_change().dropna()
    score_tax = 20 if tax_paid.mean() > 0.08 else 10 if tax_paid.mean() > 0.0 else 0
    score_breakdown["Tax Payment Momentum"] = (score_tax, 20)
    
    score_size = 15 if mkt_cap >= 25000 else 10 if mkt_cap >= 10000 else 5
    score_breakdown["Market Cap & Stability"] = (score_size, 15)
    
    total_score = sum(val[0] for val in score_breakdown.values())
    return total_score, score_breakdown

# --- USER INTERFACE ---
st.title("📈 Wealthcon Investment Scorer")
st.markdown("Quantitative Screener for **MDSPORT** Momentum and **MPTDS** Trend Sustainability.")

with st.sidebar:
    st.header("⚙️ Configuration")
    symbol = st.text_input("NSE Ticker / Company", value="RELIANCE", placeholder="e.g. TCS, INFY, HDFCBANK").upper().strip()
    
    sensitivity = st.selectbox(
        "Trend Sensitivity",
        ["Moderate — Growth ≥5%, R²≥0.25", "Strict — Growth ≥10%, R²≥0.55"]
    )
    min_growth = 10.0 if "Strict" in sensitivity else 5.0
    min_r2 = 0.55 if "Strict" in sensitivity else 0.25
    
    data_period = st.selectbox("Data Period", ["Max Available (up to 10 yr)", "Last 5 Years Only"])
    run_btn = st.button("Calculate", type="primary", use_container_width=True)

if run_btn or symbol:
    with st.spinner(f"Fetching financial filings & price history for {symbol}..."):
        raw_data = fetch_screener_data(symbol)
        
    if not raw_data:
        st.error(f"Could not locate financial filings for ticker `{symbol}`. Please verify the NSE symbol.")
    else:
        mkt_cap = raw_data["market_cap"]
        pnl = raw_data["pnl"]
        
        if data_period == "Last 5 Years Only":
            pnl = pnl.tail(5)
            
        prices = fetch_annual_prices(raw_data["symbol"], list(pnl.index))
        aligned_prices = pnl.index.map(lambda yr: prices.get(yr, np.nan))
        price_series = pd.Series(aligned_prices, index=pnl.index).ffill().bfill()
        
        mptds_res = evaluate_mptds(mkt_cap, price_series, pnl, min_growth, min_r2)
        mdsport_score, mdsport_breakdown = evaluate_mdsport(mkt_cap, pnl)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Cap", f"₹{mkt_cap:,.0f} Cr", delta="Large Cap" if mkt_cap >= 10000 else "Mid/Small Cap")
        with col2:
            st.metric("MDSPORT Score", f"{mdsport_score} / 100")
        with col3:
            verdict_color = "normal" if mptds_res['overall_verdict'] == "VALID" else "inverse"
            st.metric("MPTDS Status", mptds_res["overall_verdict"])

        st.divider()

        tab1, tab2, tab3 = st.tabs(["📊 MPTDS Trend Analysis", "🎯 MDSPORT Breakdown", "📋 Raw Financials"])
        
        with tab1:
            st.subheader("MPTDS Parameter Linear Regression")
            
            p_cols = st.columns(4)
            param_names = ["Price", "Tax", "Dividend", "Sales"]
            
            for idx, p_name in enumerate(param_names):
                p_data = mptds_res["metrics"][p_name]
                status_icon = "✅" if p_data["meets_criteria"] else "❌"
                with p_cols[idx]:
                    st.markdown(f"### {status_icon} {p_name}")
                    st.write(f"**Avg Growth:** {p_data['growth_pct']:.2f}%")
                    st.write(f"**R² Fit:** {p_data['r2']:.2f}")
                    st.caption(f"Threshold: Growth ≥{min_growth}%, R² ≥{min_r2}")
            
            chart_df = mptds_res["raw_table"].copy()
            norm_df = (chart_df / chart_df.iloc[0]) * 100
            
            fig = go.Figure()
            for col in norm_df.columns:
                fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode='lines+markers', name=col))
                
            fig.update_layout(
                title="Normalized 10-Year Growth Comparison (Base = 100)",
                xaxis_title="Financial Year",
                yaxis_title="Normalized Value",
                hovermode="x unified",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("MDSPORT Factor Scores")
            for factor, (pts, max_pts) in mdsport_breakdown.items():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{factor}**")
                    st.progress(pts / max_pts)
                with col_b:
                    st.write(f"**{pts} / {max_pts}**")

        with tab3:
            st.subheader("Extracted Statements & Series")
            st.dataframe(mptds_res["raw_table"], use_container_width=True)