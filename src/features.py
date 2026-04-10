from __future__ import annotations
import pandas as pd
import numpy as np

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Clean & type-fix ----
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Force numeric on price/volume columns (handles commas, blanks, etc.)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with bad Date or missing essential numeric data
    df = df.dropna(subset=["Date", "Close", "Volume"]).copy()

    df = df.sort_values("Date").reset_index(drop=True)
    # --------------------------

    # Returns
    df["ret_1"] = df["Close"].pct_change()
    df["log_ret_1"] = np.log(df["Close"]).diff()

    # Trend indicators
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["sma_ratio"] = df["Close"] / (df["sma_10"] + 1e-9)


    # Volatility
    df["vol_5"] = df["ret_1"].rolling(5).std()

    # Momentum
    df["mom_5"] = df["Close"] - df["Close"].shift(5)

    # Volume change
    df["volchg_1"] = df["Volume"].pct_change()

    # RSI
    df["rsi_14"] = rsi(df["Close"], 14)

    # Label: next-day direction
    df["y_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop NaNs from rolling windows + last row (no next-day label)
    df = df.dropna().reset_index(drop=True)
    return df

def feature_columns() -> list[str]:
    return [
        "ret_1", "log_ret_1",
        "sma_5", "sma_10", "ema_10",
        "vol_5", "mom_5",
        "volchg_1",
        "rsi_14",
        "sma_ratio"
    ]
