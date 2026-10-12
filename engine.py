"""
Scoring engine for the Wealthcon Investment Scorer.

Pure functions only (no network, no Streamlit) so every rule can be unit-tested.

MDSPORT
    M  Market cap  -> size class only (does not add points)
    D  Debt/Equity -> hard gate: D/E >= 0.30 is an automatic REJECT
    S  Sales growth %      \
    P  PAT growth %         |  each graded 0-4 points per year,
    O  OPM %                |  over the last 5 financial years
    R  ROCE %               |  5 params x 5 years x 4 pts = 100
    T  Tax growth %        /
    Verdict: > 75 VALID, 50-75 WATCHLIST, < 50 REJECTED (or D/E gate)

MPTDS
    M  Market cap >= Rs 10,000 Cr
    P  Price (financial-year-end close)   \
    T  Tax paid  = PBT - PAT               |  each must be in an increasing
    D  Dividend  = PAT x Payout %          |  trend: average YoY growth and
    S  Sales                              /   regression R^2 above threshold
    Verdict: VALID only if M passes and all four trends pass.

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ----------------------------------------------------------------------------
# Constants from the methodology
# ----------------------------------------------------------------------------
DE_LIMIT = 0.30
MDSPORT_YEARS = 5
POINTS_PER_YEAR = 4
VALID_ABOVE = 75
WATCHLIST_FROM = 50
MPTDS_MCAP_MIN = 10_000.0  # Rs crore

SENSITIVITY = {
    "Moderate": {"min_growth": 5.0, "min_r2": 0.25},
    "Strict": {"min_growth": 10.0, "min_r2": 0.55},
}

# Screener row-name aliases (first match wins). Banks/NBFCs use different labels.
ALIASES = {
    "sales": ["Sales", "Revenue", "Revenue from Operations"],
    "net_profit": ["Net Profit", "Profit after tax"],
    "pbt": ["Profit before tax"],
    "payout": ["Dividend Payout %"],
    "opm": ["OPM %", "Financing Margin %"],
    "roce": ["ROCE %"],
    "equity_capital": ["Equity Capital"],
    "reserves": ["Reserves"],
    "borrowings": ["Borrowings", "Borrowing"],
}

_PERIOD_RE = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})$")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def is_period_label(label: str) -> bool:
    return bool(_PERIOD_RE.match(str(label).strip()))


def period_end(label: str) -> pd.Timestamp:
    """'Mar 2025' -> 2025-03-31 (month end)."""
    return pd.Timestamp(pd.to_datetime(label, format="%b %Y")) + pd.offsets.MonthEnd(0)


def pick(df: pd.DataFrame | None, key: str) -> pd.Series:
    """Return the first column of df matching an alias of `key`, else an empty series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in ALIASES.get(key, [key]):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def yoy_growth(series: pd.Series) -> pd.Series:
    """Year-on-year growth in %. Undefined (NaN) when the previous value is <= 0,
    because a % change off a zero or negative base is meaningless (and was the
    source of +inf scores in the original code)."""
    s = pd.to_numeric(series, errors="coerce")
    prev = s.shift(1)
    g = (s - prev) / prev * 100.0
    g[(prev <= 0) | prev.isna() | s.isna()] = np.nan
    return g.iloc[1:]


def slab_points(value: float) -> int:
    """< 5 -> 0 | 5-10 -> 1 | 10-15 -> 2 | 15-20 -> 3 | > 20 -> 4.
    Boundaries: lower bound inclusive; exactly 20 scores 3 ("> 20%" needed for 4).
    Missing / undefined values score 0."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    if value < 5:
        return 0
    if value < 10:
        return 1
    if value < 15:
        return 2
    if value <= 20:
        return 3
    return 4


def market_cap_class(mcap: float, large_min: float = MPTDS_MCAP_MIN, mid_min: float = 5_000.0) -> str:
    if mcap is None or mcap <= 0 or math.isnan(mcap):
        return "Unknown"
    if mcap >= large_min:
        return "Large Cap"
    if mcap >= mid_min:
        return "Mid Cap"
    return "Small Cap"


def debt_to_equity(bs: pd.DataFrame | None) -> tuple[float, str]:
    """Latest Borrowings / (Equity Capital + Reserves).

    The methodology defines D/E as Non-Current Liabilities / Total Equity.
    Screener's summary balance sheet does not split current vs non-current
    liabilities, so total borrowings is used as the closest available proxy
    (it reproduces the 0.45 Reliance example closely). Use the manual override
    in the app when you have the exact NCL figure."""
    if bs is None or bs.empty:
        return np.nan, "—"
    eq = pick(bs, "equity_capital").fillna(0) + pick(bs, "reserves").fillna(0)
    debt = pick(bs, "borrowings")
    frame = pd.DataFrame({"debt": debt, "eq": eq}).dropna()
    frame = frame[frame["eq"] > 0]
    if frame.empty:
        return np.nan, "—"
    last = frame.index[-1]
    return float(frame.loc[last, "debt"] / frame.loc[last, "eq"]), str(last)


# ----------------------------------------------------------------------------
# MDSPORT
# ----------------------------------------------------------------------------
@dataclass
class MdsportResult:
    score: int
    verdict: str                 # VALID / WATCHLIST / REJECTED
    reason: str
    de_ratio: float
    de_period: str
    de_pass: bool
    mcap: float
    mcap_class: str
    values: pd.DataFrame         # parameter x year, raw % values
    points: pd.DataFrame         # parameter x year, 0-4
    param_totals: pd.Series      # parameter -> points out of 20
    coverage: float              # share of the 25 cells that had data
    notes: list[str] = field(default_factory=list)


SPORT_LABELS = {
    "S": "Sales Growth %",
    "P": "PAT Growth %",
    "O": "OPM %",
    "R": "ROCE %",
    "T": "Tax Growth %",
}


def evaluate_mdsport(
    pnl: pd.DataFrame,
    ratios: pd.DataFrame | None,
    bs: pd.DataFrame | None,
    mcap: float,
    de_override: float | None = None,
    large_min: float = MPTDS_MCAP_MIN,
    mid_min: float = 5_000.0,
) -> MdsportResult:
    notes: list[str] = []
    n = MDSPORT_YEARS

    sales = pick(pnl, "sales")
    pat = pick(pnl, "net_profit")
    pbt = pick(pnl, "pbt")
    tax = pbt - pat

    # Growth needs n+1 annual points to give n growth years
    rows = {
        "S": yoy_growth(sales).tail(n),
        "P": yoy_growth(pat).tail(n),
        "O": pick(pnl, "opm").tail(n),
        "R": pick(ratios, "roce").reindex(pnl.index).tail(n) if ratios is not None else pd.Series(dtype=float),
        "T": yoy_growth(tax).tail(n),
    }
    years = list(pnl.index[-n:])
    values = pd.DataFrame({k: v.reindex(years) for k, v in rows.items()}).T
    values.index = [f"{k} · {SPORT_LABELS[k]}" for k in values.index]
    points = values.apply(lambda col: col.map(slab_points))

    coverage = float(values.notna().to_numpy().mean()) if values.size else 0.0
    if pick(pnl, "opm").isna().all():
        notes.append("OPM % not reported (typical for banks/NBFCs) — O scores 0. "
                     "MDSPORT is designed for non-financial companies.")
    if ratios is None or pick(ratios, "roce").isna().all():
        notes.append("ROCE % not available — R scores 0.")
    if (pat.tail(n + 1) <= 0).any() or (tax.tail(n + 1) <= 0).any():
        notes.append("A loss or zero-tax year makes the following year's growth undefined; it scores 0.")
    if len(pnl.index) < n + 1:
        notes.append(f"Only {len(pnl.index)} annual columns available; growth needs {n + 1}.")

    score = int(points.to_numpy().sum())

    if de_override is not None:
        de, de_period = float(de_override), "manual"
    else:
        de, de_period = debt_to_equity(bs)
    de_known = not math.isnan(de)
    de_pass = de_known and de < DE_LIMIT
    if not de_known:
        notes.append("D/E could not be computed — enter it manually to complete the debt gate.")

    if de_known and not de_pass:
        verdict, reason = "REJECTED", f"Debt gate failed: D/E {de:.2f} ≥ {DE_LIMIT:.2f}"
    elif score > VALID_ABOVE:
        verdict, reason = "VALID", f"Score {score} > {VALID_ABOVE}"
    elif score >= WATCHLIST_FROM:
        verdict, reason = "WATCHLIST", f"Score {score} in {WATCHLIST_FROM}–{VALID_ABOVE}"
    else:
        verdict, reason = "REJECTED", f"Score {score} < {WATCHLIST_FROM}"
    if not de_known and verdict != "REJECTED":
        reason += " (debt gate unverified)"

    return MdsportResult(
        score=score,
        verdict=verdict,
        reason=reason,
        de_ratio=de,
        de_period=de_period,
        de_pass=de_pass,
        mcap=mcap,
        mcap_class=market_cap_class(mcap, large_min, mid_min),
        values=values,
        points=points,
        param_totals=points.sum(axis=1),
        coverage=coverage,
        notes=notes,
    )


# ----------------------------------------------------------------------------
# MPTDS
# ----------------------------------------------------------------------------
def trend_test(series: pd.Series, min_growth: float, min_r2: float) -> dict:
    """Average YoY growth (%) + linear-regression R^2 on the actual year axis."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    out = {
        "n_years": len(s), "avg_growth": np.nan, "cagr": np.nan, "r2": np.nan,
        "slope": np.nan, "intercept": np.nan, "passes": False, "note": "",
    }
    if len(s) < 3:
        out["note"] = "Fewer than 3 data points"
        return out

    x = np.array([period_end(i).year + (period_end(i).month / 12.0) for i in s.index]) \
        if all(is_period_label(i) for i in s.index) else np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)

    if np.ptp(y) == 0:
        out.update(r2=0.0, slope=0.0, avg_growth=0.0, note="Flat series")
        return out

    fit = linregress(x, y)
    g = yoy_growth(s).dropna()
    out.update(slope=float(fit.slope), intercept=float(fit.intercept), r2=float(fit.rvalue ** 2))
    out["avg_growth"] = float(g.mean()) if len(g) else np.nan
    if y[0] > 0 and y[-1] > 0:
        span = x[-1] - x[0]
        out["cagr"] = float(((y[-1] / y[0]) ** (1 / span) - 1) * 100) if span > 0 else np.nan
    skipped = (len(s) - 1) - len(g)
    if skipped:
        out["note"] = f"{skipped} year(s) skipped in growth (zero/negative base)"

    out["passes"] = bool(
        not math.isnan(out["avg_growth"])
        and out["avg_growth"] >= min_growth
        and out["r2"] >= min_r2
        and out["slope"] > 0
    )
    return out


@dataclass
class MptdsResult:
    verdict: str            # VALID / INVALID
    mcap: float
    mcap_pass: bool
    metrics: dict           # name -> trend_test dict
    series: pd.DataFrame    # Price, Tax, Dividend, Sales by year
    min_growth: float
    min_r2: float


MPTDS_PARAMS = ["Price", "Tax", "Dividend", "Sales"]


def evaluate_mptds(
    mcap: float,
    price: pd.Series,
    pnl: pd.DataFrame,
    min_growth: float = 5.0,
    min_r2: float = 0.25,
) -> MptdsResult:
    pat = pick(pnl, "net_profit")
    tax = pick(pnl, "pbt") - pat
    dividend = pat * pick(pnl, "payout") / 100.0
    series = pd.DataFrame(
        {
            "Price": pd.to_numeric(price.reindex(pnl.index), errors="coerce"),
            "Tax": tax,
            "Dividend": dividend,
            "Sales": pick(pnl, "sales"),
        },
        index=pnl.index,
    )
    metrics = {p: trend_test(series[p], min_growth, min_r2) for p in MPTDS_PARAMS}
    mcap_pass = bool(mcap and mcap >= MPTDS_MCAP_MIN)
    verdict = "VALID" if mcap_pass and all(m["passes"] for m in metrics.values()) else "INVALID"
    return MptdsResult(verdict, mcap, mcap_pass, metrics, series, min_growth, min_r2)
