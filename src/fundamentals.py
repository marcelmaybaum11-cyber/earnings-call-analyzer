"""Key fundamentals and simple fair-value estimates via yfinance.

Two fair-value perspectives, both deliberately simple and transparent:

1. Analyst consensus - the low / mean / high 12-month price targets that
   sell-side analysts publish (Yahoo aggregates them).
2. Peter Lynch rule of thumb - a stock is fairly valued when its P/E
   equals its earnings growth rate (PEG = 1). Fair value = EPS x growth%.
   The growth rate is capped at 50% because the rule breaks down for
   hypergrowth; a negative-growth or negative-EPS stock gets no estimate.

Every field can be missing on Yahoo's side, so all values are Optional
and the app must degrade gracefully.
"""

import yfinance as yf

LYNCH_GROWTH_CAP = 0.50  # the PEG=1 rule is meaningless past ~50% growth

METRIC_FIELDS = {
    "currentPrice": "Price",
    "marketCap": "Market cap",
    "trailingPE": "P/E (trailing)",
    "forwardPE": "P/E (forward)",
    "trailingEps": "EPS (ttm)",
    "revenueGrowth": "Revenue growth (yoy)",
    "earningsGrowth": "Earnings growth (yoy)",
    "profitMargins": "Profit margin",
    "returnOnEquity": "Return on equity",
    "freeCashflow": "Free cash flow",
    "totalDebt": "Total debt",
    "beta": "Beta",
}


def get_fundamentals(ticker: str) -> dict:
    """Return {'metrics': {...}, 'fair_value': {...}} with None for gaps."""
    info = yf.Ticker(ticker).info or {}

    metrics = {field: info.get(field) for field in METRIC_FIELDS}

    eps = info.get("trailingEps")
    growth = info.get("earningsGrowth")
    lynch = None
    if eps and growth and eps > 0 and growth > 0:
        lynch = eps * min(growth, LYNCH_GROWTH_CAP) * 100

    fair_value = {
        "current": info.get("currentPrice"),
        "analyst_low": info.get("targetLowPrice"),
        "analyst_mean": info.get("targetMeanPrice"),
        "analyst_high": info.get("targetHighPrice"),
        "analyst_count": info.get("numberOfAnalystOpinions"),
        "recommendation": info.get("recommendationKey"),
        "lynch": lynch,
    }
    return {"metrics": metrics, "fair_value": fair_value}


if __name__ == "__main__":
    import json

    result = get_fundamentals("NVDA")
    print(json.dumps(result, indent=2, default=str))
