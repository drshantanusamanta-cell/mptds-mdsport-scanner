# 📈 Wealthcon Investment Scorer

**Designed & developed by Dr. Shantanu Samanta** — Physician · Quantitative Market Analyst & Trading-Systems Developer · Kolkata, India

A Streamlit app that screens NSE/BSE companies with two complementary frameworks, using fundamentals from Screener.in and prices from Yahoo Finance. The layout adapts to desktop, tablet/iPad and phone screens.

| Framework | Question it answers | Output |
|---|---|---|
| **MDSPORT** | Is this a high-quality, growing, low-debt business? | Score /100 → VALID · WATCHLIST · REJECTED |
| **MPTDS** | Are price, tax, dividend and sales in a sustained uptrend? | VALID · INVALID |

## Methodology

### MDSPORT
- **M · Market cap** – size class (Large / Mid / Small). Classification only.
- **D · Debt-to-Equity** – Non-Current Liabilities ÷ Total Equity. **D/E ≥ 0.30 is an automatic REJECT.**
- **S · P · O · R · T** – Sales growth %, PAT growth %, OPM %, ROCE %, Tax growth %. Each is graded **0–4 points for each of the last 5 financial years**:

  | Value | < 5% | 5–10% | 10–15% | 15–20% | > 20% |
  |---|---|---|---|---|---|
  | Points | 0 | 1 | 2 | 3 | 4 |

- Maximum 5 × 5 × 4 = **100**. **> 75 VALID**, **50–75 WATCHLIST**, **< 50 REJECTED**.

### MPTDS
- **M** Market cap ≥ ₹10,000 Cr · **P** FY-end (March) close · **T** Tax = PBT − PAT · **D** Dividend = PAT × Payout % · **S** Sales.
- P, T, D and S must each show a rising trend over 5–10 years: **Moderate** = average YoY growth ≥ 5% and R² ≥ 0.25; **Strict** = ≥ 10% and R² ≥ 0.55.
- **VALID** only when M and all four trends pass.

### Implementation notes
- Growth off a zero or negative base is undefined: it scores 0 (MDSPORT) or is skipped (MPTDS average) rather than producing infinite values.
- Exactly 20% scores 3 points; more than 20% is needed for 4.
- Screener's summary balance sheet doesn't separate non-current liabilities, so D/E uses **Borrowings ÷ (Equity capital + Reserves)** as a proxy. Enter the exact NCL-based figure under *Advanced options* when you have it.
- Prices are split-adjusted closes on the last trading day on or before each year-end. Missing years stay blank — they are never back- or forward-filled.
- Banks/NBFCs don't report OPM/ROCE the same way, so MDSPORT under-scores them by design.

## Project layout
```
app.py             Streamlit UI (responsive layout, charts, author credits)
engine.py          MDSPORT / MPTDS rules — pure functions, no network
data_sources.py    Screener.in parser + Yahoo Finance price fetch
tests/             Offline unit tests + headless UI smoke tests
.streamlit/        Theme
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Test
```bash
pip install pytest
pytest -q
```
Tests run fully offline (network calls are mocked).

## Data sources & fair use
Fundamentals are read from public Screener.in company pages and prices from Yahoo Finance via `yfinance`. Results are cached for an hour to keep requests light. Please respect both sites' terms of use.

## Disclaimer
For education and research only — not investment advice or a recommendation to buy or sell any security. Data from third-party sources may be delayed, incomplete or revised. Verify independently and consult a SEBI-registered investment adviser before investing.

© 2026 Dr. Shantanu Samanta. All rights reserved.
