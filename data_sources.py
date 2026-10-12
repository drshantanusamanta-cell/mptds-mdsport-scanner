"""
Data layer: Screener.in (fundamentals) + Yahoo Finance (prices).

Network code is kept separate from the HTML/price *parsers* so the parsers can
be tested offline with saved HTML or synthetic frames.

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from engine import is_period_label, period_end, pick

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}
TIMEOUT = 15


class DataError(Exception):
    """Raised with a user-readable message when data cannot be obtained."""


class NotFoundError(DataError):
    """The symbol has no Screener.in page (so it may be a company name instead)."""


@dataclass
class CompanyData:
    symbol: str
    name: str
    basis: str              # "Consolidated" / "Standalone"
    market_cap: float       # Rs crore
    current_price: float
    pnl: pd.DataFrame       # index = FY labels ("Mar 2025"), columns = row names
    balance_sheet: pd.DataFrame
    ratios: pd.DataFrame
    url: str


# ----------------------------------------------------------------------------
# Parsing (offline-testable)
# ----------------------------------------------------------------------------
def clean_label(text: str) -> str:
    """'Sales\xa0+' -> 'Sales' ; collapses whitespace."""
    t = text.replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"\s*\+$", "", t).strip()


def parse_number(text: str) -> float:
    t = text.replace(",", "").replace("%", "").replace("₹", "").replace("Cr.", "").strip()
    if t in ("", "-", "—", "--"):
        return np.nan
    try:
        return float(t)
    except ValueError:
        return np.nan


def parse_section(soup: BeautifulSoup, section_id: str) -> pd.DataFrame:
    """Parse a Screener data table (profit-loss / balance-sheet / ratios).

    Column alignment is by position against the full header row (the first,
    blank header cell lines up with the row-label cell), and only real period
    columns such as 'Mar 2025' are kept — 'TTM' and anything else is dropped.
    The row label is taken from the first cell regardless of its CSS class.
    """
    section = soup.find("section", id=section_id)
    table = section.find("table") if section else None
    if table is None:
        return pd.DataFrame()

    header_row = table.find("thead").find("tr") if table.find("thead") else table.find("tr")
    headers = [clean_label(th.get_text()) for th in header_row.find_all("th")]
    keep = [(i, h) for i, h in enumerate(headers) if i > 0 and is_period_label(h)]
    if not keep:
        return pd.DataFrame()

    body = table.find("tbody") or table
    data: dict[str, list[float]] = {}
    for tr in body.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        label = clean_label(cells[0].get_text())
        if not label or label in data:
            continue
        data[label] = [parse_number(cells[i].get_text()) if i < len(cells) else np.nan for i, _ in keep]

    df = pd.DataFrame(data, index=[h for _, h in keep])
    # Chronological order, de-duplicated periods
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[sorted(df.index, key=period_end)]


def parse_top_ratio(soup: BeautifulSoup, name: str) -> float:
    for li in soup.select("#top-ratios li"):
        label = li.select_one(".name")
        if label and name.lower() in clean_label(label.get_text()).lower():
            num = li.select_one(".number")
            return parse_number(num.get_text()) if num else np.nan
    return np.nan


def parse_company_page(html: str, symbol: str, basis: str, url: str) -> CompanyData:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    return CompanyData(
        symbol=symbol,
        name=clean_label(h1.get_text()) if h1 else symbol,
        basis=basis,
        market_cap=parse_top_ratio(soup, "Market Cap"),
        current_price=parse_top_ratio(soup, "Current Price"),
        pnl=parse_section(soup, "profit-loss"),
        balance_sheet=parse_section(soup, "balance-sheet"),
        ratios=parse_section(soup, "ratios"),
        url=url,
    )


def has_usable_pnl(data: CompanyData) -> bool:
    return not data.pnl.empty and pick(data.pnl, "sales").notna().sum() >= 3


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
def _get(url: str) -> requests.Response | None:
    try:
        res = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise DataError(f"Could not reach Screener.in ({exc.__class__.__name__}). Check your connection.") from exc
    if res.status_code in (403, 429):
        raise DataError("Screener.in is rate-limiting requests right now. Wait a minute and try again.")
    return res if res.status_code == 200 else None


def fetch_company(symbol: str, basis: str = "Auto") -> CompanyData:
    """basis: 'Auto' (consolidated, falling back to standalone), 'Consolidated' or 'Standalone'."""
    sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
    if not re.fullmatch(r"[A-Z0-9&\-]+", sym):
        raise NotFoundError(f"'{symbol}' doesn't look like an NSE symbol or BSE code.")

    attempts = {
        "Auto": [("Consolidated", "consolidated/"), ("Standalone", "")],
        "Consolidated": [("Consolidated", "consolidated/")],
        "Standalone": [("Standalone", "")],
    }[basis]

    last = None
    for label, suffix in attempts:
        url = f"https://www.screener.in/company/{sym}/{suffix}"
        res = _get(url)
        if res is None:
            continue
        data = parse_company_page(res.text, sym, label, url)
        last = data
        if has_usable_pnl(data):
            return data
    if last is not None:
        raise DataError(f"Screener.in has no usable annual P&L for `{sym}` ({basis.lower()} basis).")
    raise NotFoundError(f"No Screener.in page found for `{sym}`. Check the NSE symbol (e.g. TCS, HDFCBANK).")


def yahoo_ticker(symbol: str) -> str:
    return f"{symbol}.BO" if symbol.isdigit() else f"{symbol}.NS"


def fy_end_prices(close: pd.Series, labels: list[str], tolerance_days: int = 15) -> pd.Series:
    """Last close on/before each period end (e.g. 31 Mar). NaN when there is no
    trade within `tolerance_days` — gaps are left as gaps, never back-filled."""
    close = close.dropna()
    if close.empty:
        return pd.Series(np.nan, index=labels, dtype=float)
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close = pd.Series(close.to_numpy(), index=idx).sort_index()

    out = {}
    for lab in labels:
        end = period_end(lab)
        window = close.loc[end - pd.Timedelta(days=tolerance_days): end]
        out[lab] = round(float(window.iloc[-1]), 2) if len(window) else np.nan
    return pd.Series(out, dtype=float)


def fetch_fy_prices(symbol: str, labels: list[str]) -> pd.Series:
    """Split-adjusted (not dividend-adjusted) financial-year-end closes."""
    import yfinance as yf

    if not labels:
        return pd.Series(dtype=float)
    start = (period_end(labels[0]) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    try:
        hist = yf.Ticker(yahoo_ticker(symbol)).history(start=start, auto_adjust=False)
    except Exception:  # yfinance raises a variety of errors on bad tickers / throttling
        hist = pd.DataFrame()
    if hist is None or hist.empty or "Close" not in hist:
        return pd.Series(np.nan, index=labels, dtype=float)
    return fy_end_prices(hist["Close"], labels)


# ----------------------------------------------------------------------------
# Company name -> NSE symbol
# ----------------------------------------------------------------------------
@dataclass
class Match:
    symbol: str      # Screener / NSE code, e.g. "SHRIRAMFIN" (numeric for BSE-only)
    name: str        # e.g. "Shriram Finance Ltd"
    source: str      # "alias" / "Screener.in" / "Yahoo Finance"


# Common short-forms that a text search can miss. Long-standing, stable symbols only.
ALIASES = {
    "SBI": "SBIN", "STATE BANK": "SBIN", "STATE BANK OF INDIA": "SBIN",
    "L&T": "LT", "L & T": "LT", "LARSEN": "LT", "LARSEN AND TOUBRO": "LT", "LARSEN & TOUBRO": "LT",
    "HUL": "HINDUNILVR", "HINDUSTAN UNILEVER": "HINDUNILVR",
    "RIL": "RELIANCE", "RELIANCE INDUSTRIES": "RELIANCE",
    "M&M": "M&M", "MAHINDRA": "M&M", "MAHINDRA AND MAHINDRA": "M&M", "MAHINDRA & MAHINDRA": "M&M",
    "AIRTEL": "BHARTIARTL", "BHARTI AIRTEL": "BHARTIARTL",
    "HDFC BANK": "HDFCBANK", "ICICI BANK": "ICICIBANK", "KOTAK": "KOTAKBANK", "KOTAK BANK": "KOTAKBANK",
    "KOTAK MAHINDRA BANK": "KOTAKBANK", "AXIS BANK": "AXISBANK", "INFOSYS": "INFY",
    "BAJAJ FINANCE": "BAJFINANCE", "BAJAJ FINSERV": "BAJAJFINSV", "SHRIRAM FINANCE": "SHRIRAMFIN",
    "ASIAN PAINTS": "ASIANPAINT", "SUN PHARMA": "SUNPHARMA", "DR REDDY": "DRREDDY", "DR REDDYS": "DRREDDY",
    "ULTRATECH": "ULTRACEMCO", "ULTRATECH CEMENT": "ULTRACEMCO", "NESTLE": "NESTLEIND",
    "POWER GRID": "POWERGRID", "COAL INDIA": "COALINDIA", "HERO MOTOCORP": "HEROMOTOCO",
    "EICHER": "EICHERMOT", "EICHER MOTORS": "EICHERMOT", "TECH MAHINDRA": "TECHM",
    "HCL": "HCLTECH", "HCL TECH": "HCLTECH", "BRITANNIA": "BRITANNIA", "PIDILITE": "PIDILITIND",
}

_SUFFIX_RE = re.compile(r"\b(LTD|LIMITED|CO|COMPANY|CORP|CORPORATION|INC|THE)\b")


def normalize_name(text: str) -> str:
    t = text.upper().replace(".", " ").replace(",", " ").replace("'", "")
    t = _SUFFIX_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def looks_like_symbol(text: str) -> bool:
    """'TCS', 'm&m', '532978' -> True; 'shriram finance' -> False."""
    t = text.strip()
    return bool(t) and " " not in t and bool(re.fullmatch(r"[A-Za-z0-9&\-]{1,20}", t))


def parse_screener_search(items) -> list[Match]:
    """Screener's search returns [{"name": ..., "url": "/company/CODE/consolidated/"}, ...].
    Non-company rows (e.g. 'search everywhere') have no /company/ URL and are skipped."""
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        m = re.search(r"/company/([^/]+)/", str(it.get("url", "")))
        if m and it.get("name"):
            out.append(Match(m.group(1).upper(), clean_label(str(it["name"])), "Screener.in"))
    return out


def parse_yahoo_quotes(quotes) -> list[Match]:
    """Keep Indian equities only (NSE 'NSI' / 'BSE'), NSE listings first."""
    nse, bse = [], []
    for q in quotes or []:
        if q.get("quoteType") != "EQUITY":
            continue
        sym = str(q.get("symbol", ""))
        name = q.get("longname") or q.get("shortname") or sym
        if sym.endswith(".NS"):
            nse.append(Match(sym[:-3].upper(), name, "Yahoo Finance"))
        elif sym.endswith(".BO"):
            bse.append(Match(sym[:-3].upper(), name, "Yahoo Finance"))
    return nse + bse


def rank_matches(query: str, matches: list[Match]) -> list[Match]:
    """Best match first: exact symbol, then name similarity; duplicates removed."""
    from difflib import SequenceMatcher

    q = normalize_name(query)

    def score(m: Match) -> float:
        n = normalize_name(m.name)
        s = SequenceMatcher(None, q, n).ratio()
        if m.symbol == q.replace(" ", ""):
            s += 1.0
        if n.startswith(q):
            s += 0.5
        elif all(w in n.split() for w in q.split()):
            s += 0.3
        return s

    seen, out = set(), []
    for m in sorted(matches, key=score, reverse=True):
        if m.symbol not in seen:
            seen.add(m.symbol)
            out.append(m)
    return out


def _search_screener(query: str) -> list[Match]:
    try:
        res = requests.get("https://www.screener.in/api/company/search/",
                           params={"q": query, "v": 3, "fts": 1}, headers=HEADERS, timeout=TIMEOUT)
        if res.status_code != 200:
            return []
        return parse_screener_search(res.json())
    except (requests.RequestException, ValueError):
        return []


def _search_yahoo(query: str) -> list[Match]:
    try:
        import yfinance as yf
        found = yf.Search(query, max_results=10, news_count=0, lists_count=0,
                          include_cb=False, raise_errors=False)
        return parse_yahoo_quotes(found.quotes)
    except Exception:
        return []


def resolve_company(query: str, limit: int = 8) -> list[Match]:
    """Turn free text ("Shriram Finance", "sbi", "asian paints ltd") into ranked
    candidate symbols. Order: built-in aliases -> Screener.in search -> Yahoo search."""
    q = normalize_name(query)
    if not q:
        return []
    found: list[Match] = []
    if q in ALIASES:
        found.append(Match(ALIASES[q], query.strip().title(), "alias"))
    results = _search_screener(query.strip())
    if not results:
        results = _search_yahoo(query.strip())
    ranked = rank_matches(query, results)
    if found:  # alias stays first; drop its duplicate from search results
        ranked = [m for m in ranked if m.symbol != found[0].symbol]
        # Prefer the real company name from search if it found the same symbol
        same = [m for m in results if m.symbol == found[0].symbol]
        if same:
            found[0] = Match(found[0].symbol, same[0].name, "alias")
    return (found + ranked)[:limit]
