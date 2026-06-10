"""Fetch post-earnings-call stock returns via yfinance.

Earnings calls typically happen after market close, so the market's first
reaction shows up on the NEXT trading day. We therefore use the close on
(or immediately before) the call date as the baseline price, and measure
returns 1 and 5 trading days after it.

The download and the return calculation are separate functions so that
batch jobs (build_dataset.py) can download each ticker's history once and
compute many returns from it, instead of hitting Yahoo once per call date.
"""

import pandas as pd
import yfinance as yf

# Padding around the call date so weekends/holidays never leave us short
# of trading days on either side.
LOOKBACK_DAYS = 10
LOOKAHEAD_DAYS = 15


def download_close_prices(ticker: str, start, end) -> pd.Series:
    """Download daily closing prices (dividend/split adjusted) as a Series."""
    prices = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=True
    )["Close"]
    if isinstance(prices, pd.DataFrame):  # newer yfinance returns a DataFrame
        prices = prices[ticker]
    if prices.empty:
        raise ValueError(f"No price data for {ticker} between {start} and {end}")
    return prices


def compute_post_call_returns(
    prices: pd.Series, call_date, horizons: tuple[int, ...] = (1, 5)
) -> dict[str, float | None]:
    """Return {'return_1d': ..., 'return_5d': ...} as decimal fractions.

    Baseline price = last close on or before the call date. A horizon's
    value is None when not enough trading days exist after the call.
    """
    call_ts = pd.Timestamp(call_date)

    before = prices.loc[:call_ts]
    if before.empty:
        raise ValueError(f"No price data on or before {call_date}")
    base_idx = len(before) - 1
    base_price = prices.iloc[base_idx]

    results: dict[str, float | None] = {}
    for h in horizons:
        target_idx = base_idx + h
        if target_idx < len(prices):
            results[f"return_{h}d"] = float(prices.iloc[target_idx] / base_price - 1)
        else:
            results[f"return_{h}d"] = None
    return results


def get_post_call_returns(
    ticker: str, call_date: str, horizons: tuple[int, ...] = (1, 5)
) -> dict[str, float | None]:
    """Convenience wrapper: download a small window and compute returns."""
    call_ts = pd.Timestamp(call_date)
    prices = download_close_prices(
        ticker,
        start=call_ts - pd.Timedelta(days=LOOKBACK_DAYS),
        end=call_ts + pd.Timedelta(days=LOOKAHEAD_DAYS),
    )
    return compute_post_call_returns(prices, call_ts, horizons)


if __name__ == "__main__":
    # Microsoft's FY24 Q2 earnings call was Jan 30, 2024 (after close).
    returns = get_post_call_returns("MSFT", "2024-01-30")
    print("MSFT earnings call 2024-01-30:")
    for horizon, ret in returns.items():
        print(f"  {horizon}: {ret:+.2%}" if ret is not None else f"  {horizon}: n/a")
