# Wealthcon Investment Scorer 📈

A custom financial screening web application built using Python and Streamlit. This app pulls fundamental and price data from Screener.in and Yahoo Finance to run quantitative trend sustainability analysis (MPTDS) and growth momentum scoring (MDSPORT).

## Features
- Scrapes live financial data from Screener.in
- Fetches historical price data via Yahoo Finance
- Calculates linear regression $R^2$ and annual growth percentages
- Interactive visualizations using Plotly

## Local Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`