from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config
from src.data import load_csv
from src.features import make_features
from src.train_model import time_split
from src.agent import TradingAgent, AgentState
from src.utils import ensure_dir, set_seed

MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")

def max_drawdown(values: list[float]) -> float:
    peak = -1e18
    mdd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / (peak + 1e-9)
        mdd = max(mdd, dd)
    return mdd

def backtest_agent(test: pd.DataFrame, model, cols: list[str], cfg: Config):
    agent = TradingAgent(cfg)
    state = AgentState()

    cash = cfg.initial_cash
    shares = 0
    values: list[float] = []
    trades = 0

    proba = model.predict_proba(test[cols])[:, 1]  # P(up)

    for i, row in test.reset_index(drop=True).iterrows():
        price = float(row["Close"])
        sma10 = float(row["sma_10"])
        sma5 = float(row["sma_5"])
        p_up = float(proba[i])

        action = agent.decide(p_up, price, sma10, sma5, state)


        if action == "BUY" and shares < cfg.max_shares:
            cost = price * (1.0 + cfg.fee_rate)
            if cash >= cost:
                cash -= cost
                shares += 1
                state.holding = True
                state.entry_price = price
                trades += 1

        elif action == "SELL" and shares > 0:
            revenue = price * (1.0 - cfg.fee_rate)
            cash += revenue
            shares -= 1
            if shares == 0:
                state.holding = False
                state.entry_price = None
            trades += 1

        values.append(cash + shares * price)

    total_return = (values[-1] - cfg.initial_cash) / cfg.initial_cash
    mdd = max_drawdown(values)
    return total_return, mdd, trades, values

def backtest_buy_hold(test: pd.DataFrame, cfg: Config):
    prices = test["Close"].astype(float).tolist()
    first = prices[0]
    cash = cfg.initial_cash
    shares = 0
    values: list[float] = []

    cost = first * (1.0 + cfg.fee_rate)
    if cash >= cost:
        cash -= cost
        shares = 1

    for p in prices:
        values.append(cash + shares * p)

    total_return = (values[-1] - cfg.initial_cash) / cfg.initial_cash
    mdd = max_drawdown(values)
    trades = 1 if shares == 1 else 0
    return total_return, mdd, trades, values

def backtest_random(test: pd.DataFrame, cfg: Config):
    set_seed(cfg.seed)
    prices = test["Close"].astype(float).tolist()
    cash = cfg.initial_cash
    shares = 0
    values: list[float] = []
    trades = 0

    for p in prices:
        action = np.random.choice(["BUY", "SELL", "HOLD"], p=[0.2, 0.2, 0.6])

        if action == "BUY" and shares < cfg.max_shares:
            cost = p * (1.0 + cfg.fee_rate)
            if cash >= cost:
                cash -= cost
                shares += 1
                trades += 1

        elif action == "SELL" and shares > 0:
            revenue = p * (1.0 - cfg.fee_rate)
            cash += revenue
            shares -= 1
            trades += 1

        values.append(cash + shares * p)

    total_return = (values[-1] - cfg.initial_cash) / cfg.initial_cash
    mdd = max_drawdown(values)
    return total_return, mdd, trades, values

def plot_curves(dates, curves, labels, out_path: Path):
    ensure_dir(out_path.parent)
    plt.figure()
    for vals, lab in zip(curves, labels):
        plt.plot(dates, vals, label=lab)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    cfg = Config()

    artifact_path = MODEL_DIR / f"{args.ticker}_rf.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model not found: {artifact_path}. Run train_model first.")
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    cols = artifact["feature_cols"]

    raw = load_csv(args.ticker)
    df = make_features(raw)
    _, _, test = time_split(df, cfg)

    agent_ret, agent_mdd, agent_trades, agent_vals = backtest_agent(test, model, cols, cfg)
    bh_ret, bh_mdd, bh_trades, bh_vals = backtest_buy_hold(test, cfg)
    rnd_ret, rnd_mdd, rnd_trades, rnd_vals = backtest_random(test, cfg)

    print("=== Trading Backtest (Test Period) ===")
    print(f"Agent:     return={agent_ret*100:7.2f}% | maxDD={agent_mdd*100:6.2f}% | trades={agent_trades}")
    print(f"Buy&Hold:  return={bh_ret*100:7.2f}% | maxDD={bh_mdd*100:6.2f}% | trades={bh_trades}")
    print(f"Random:    return={rnd_ret*100:7.2f}% | maxDD={rnd_mdd*100:6.2f}% | trades={rnd_trades}")

    out_path = OUT_DIR / f"{args.ticker}_portfolio_comparison.png"
    dates = test["Date"].tolist()
    plot_curves(dates, [agent_vals, bh_vals, rnd_vals], ["Agent", "Buy&Hold", "Random"], out_path)
    print(f"Saved plot: {out_path}")

if __name__ == "__main__":
    main()
