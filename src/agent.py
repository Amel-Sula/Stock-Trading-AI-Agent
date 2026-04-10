from __future__ import annotations
from dataclasses import dataclass
from src.config import Config

@dataclass
class AgentState:
    holding: bool = False
    entry_price: float | None = None

class TradingAgent:
    """
    Rule-based rational decision module (Technique #2).
    Trend-following policy with ML confirmation:
    - Uses SMA(5) vs SMA(10) + price vs SMA(10) to detect trend.
    - Uses model probability P(up) as confirmation for entry/exit.
    - Uses stop-loss and take-profit as risk controls.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def decide(
        self,
        p_up: float,
        price: float,
        sma_10: float,
        sma_5: float,
        state: AgentState
    ) -> str:
        # Risk controls if holding
        if state.holding and state.entry_price is not None:
            change = (price - state.entry_price) / state.entry_price
            if change <= -self.cfg.stop_loss:
                return "SELL"
            if change >= self.cfg.take_profit:
                return "SELL"

        # Trend definitions
        uptrend = (sma_5 >= sma_10) and (price >= sma_10)
        downtrend = (sma_5 < sma_10) and (price < sma_10)

        if not state.holding:
            # Enter only if trend is up AND model is at least slightly confident
            if uptrend and (p_up >= self.cfg.buy_threshold):
                return "BUY"
            return "HOLD"
        else:
            # Exit if trend breaks OR model strongly disagrees
            if downtrend:
                return "SELL"
            return "HOLD"
