from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Data split (time-based)
    train_ratio: float = 0.70
    val_ratio: float = 0.15  # test = remainder

    # Trading simulation
    initial_cash: float = 10_000.0
    max_shares: int = 1          # keep it simple: 0 or 1 share
    fee_rate: float = 0.001      # 0.1% per trade

    # Agent policy thresholds
    buy_threshold: float = 0.42  # P(up) >= this and trend ok => buy
    sell_threshold: float = 0.40 # P(up) <= this or trend breaks => sell
    stop_loss: float = 0.08      # -3% from entry => sell
    take_profit: float = 0.25   # +5% from entry => sell

    seed: int = 42
