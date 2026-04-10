from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from src.utils import ensure_dir

RAW_DIR = Path("data/raw")

def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/start/end.")
    df = df.reset_index()

    # Standard columns (ignore Adj Close if present)
    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data missing columns: {missing}")

    df = df[keep].copy()
    return df

def load_csv(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Either run: python -m src.data --ticker {ticker} --start YYYY-MM-DD --end YYYY-MM-DD "
            f"or place a CSV at that path."
        )
    df = pd.read_csv(path)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {sorted(required)}")
    return df

def save_csv(df: pd.DataFrame, ticker: str) -> Path:
    ensure_dir(RAW_DIR)
    path = RAW_DIR / f"{ticker}.csv"
    df.to_csv(path, index=False)
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="e.g., AAPL")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    df = fetch_yfinance(args.ticker, args.start, args.end)
    path = save_csv(df, args.ticker)
    print(f"Saved: {path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
