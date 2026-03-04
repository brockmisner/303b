"""
Synthetic Exit Engine — Box Spread Optimization for Binary Markets.

UP + DOWN = 1.00, so owning both guarantees $1 at settlement.
If 1.0 - opp_ask > direct_bid (after fees), buying the opposite token
is mathematically superior to selling the held token.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SyntheticExitConfig:
    min_improvement: float = 0.015   # require >= 1.5c improvement over direct sell
    min_notional: float = 5.0        # Polymarket $5 USDC minimum
    fee_rate: float = 0.0625         # 625 bps — p*(1-p)*rate


class SyntheticExitEngine:
    def __init__(self, cfg: SyntheticExitConfig = None):
        self.cfg = cfg or SyntheticExitConfig()
        self.total_synthetic_exits: int = 0
        self.total_improvement_usd: float = 0.0

    @staticmethod
    def _fee(price: float, rate: float) -> float:
        return price * (1.0 - price) * rate

    def evaluate(self, *, side: str, bid: float, opp_ask: float,
                 size: float, offset: float = 0.0) -> dict:
        """Evaluate whether synthetic exit (box spread) beats direct sell.

        Args:
            side: Position side being exited ("UP" or "DOWN")
            bid: Current best bid for the held token
            opp_ask: Current best ask for the opposite token
            size: Number of shares to exit
            offset: SIS urgency offset (added to opp_ask for limit price)

        Returns dict:
            use_synthetic: bool - True if box spread is better
            synthetic_price: float - limit price for BUY opposite
            implied_bid: float - 1.0 - opp_ask (value locked per share)
            improvement: float - implied_bid - bid (per-share gain)
            improvement_net: float - improvement after fee differential
            notional: float - cost of buying opposite (synthetic_price * size)
        """
        result = {
            "use_synthetic": False,
            "synthetic_price": 0.0,
            "implied_bid": 0.0,
            "improvement": 0.0,
            "improvement_net": 0.0,
            "notional": 0.0,
        }

        if opp_ask <= 0.01 or opp_ask >= 0.99:
            return result

        implied_bid = 1.0 - opp_ask
        improvement = implied_bid - bid

        # Fee comparison: selling at bid vs buying opposite at opp_ask
        fee_sell = self._fee(bid, self.cfg.fee_rate)
        fee_buy = self._fee(opp_ask, self.cfg.fee_rate)
        improvement_net = improvement - (fee_buy - fee_sell)

        syn_price = round(min(0.99, opp_ask + offset), 2)
        notional = syn_price * size

        use = (
            improvement_net >= self.cfg.min_improvement
            and notional >= self.cfg.min_notional
        )

        if use:
            self.total_synthetic_exits += 1
            self.total_improvement_usd += improvement_net * size

        result.update({
            "use_synthetic": use,
            "synthetic_price": syn_price,
            "implied_bid": round(implied_bid, 4),
            "improvement": round(improvement, 4),
            "improvement_net": round(improvement_net, 4),
            "notional": round(notional, 2),
        })
        return result
