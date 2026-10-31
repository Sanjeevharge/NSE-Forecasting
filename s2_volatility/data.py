"""Download NSE daily prices (yfinance) and run validation checks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
           "SBIN.NS", "ITC.NS", "LT.NS", "BHARTIARTL.NS", "HINDUNILVR.NS",
           "AXISBANK.NS", "KOTAKBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "WIPRO.NS"]
PRICES = Path(__file__).parent / "data" / "prices.csv"
RESULTS = Path(__file__).parent / "data" / "results.csv"


def download(start: str = "2015-01-01") -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(TICKERS, start=start, auto_adjust=True, group_by="ticker", progress=False)
    frames = []
    for t in TICKERS:
        if t not in raw.columns.get_level_values(0):
            print(f"WARN: no data for {t}")
            continue
        d = raw[t].dropna(how="all").reset_index()
        d.columns = [str(c).lower() for c in d.columns]
        d["ticker"] = t
        frames.append(d[["date", "ticker", "close", "volume"]])
    return pd.concat(frames, ignore_index=True)


def validate(df: pd.DataFrame, max_gap_days: int = 7, max_abs_ret: float = 0.4) -> list:
    """Return a list of issue strings; raise on hard errors."""
    issues = []
    if df.duplicated(["ticker", "date"]).any():
        raise ValueError("duplicate (ticker, date) rows")
    if (df["close"] <= 0).any():
        raise ValueError("non-positive close prices")
    for t, g in df.sort_values("date").groupby("ticker"):
        gaps = pd.to_datetime(g["date"]).diff().dt.days
        if (gaps > max_gap_days).any():
            issues.append(f"{t}: {int((gaps > max_gap_days).sum())} gaps > {max_gap_days} days")
        ret = np.log(g["close"]).diff().abs()
        if (ret > max_abs_ret).any():
            issues.append(f"{t}: {int((ret > max_abs_ret).sum())} daily moves > {max_abs_ret:.0%} (unadjusted split?)")
        if len(g) < 1000:
            issues.append(f"{t}: only {len(g)} rows")
    return issues


if __name__ == "__main__":
    df = download()
    issues = validate(df)
    PRICES.parent.mkdir(exist_ok=True)
    df.to_csv(PRICES, index=False)
    print(f"{df['ticker'].nunique()} tickers, {len(df)} rows -> {PRICES}")
    print("\n".join(issues) if issues else "validation: no issues")
    sys.exit(0)
