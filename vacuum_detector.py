from dataclasses import dataclass
from collections import deque
import math

@dataclass
class VacuumConfig:
    # windows (in ticks)
    window: int = 40

    # depth collapse thresholds
    depth_drop_ratio: float = 0.35   # depth_now < 0.35 * median_depth

    # spread jump thresholds
    spread_jump: float = 0.03        # absolute jump in one tick
    spread_high: float = 0.06        # absolute spread considered wide

    # quote churn thresholds (events/sec)
    churn_high: float = 8.0

    # decision thresholds
    vacuum_score_trigger: int = 2    # 2 out of 3 signals -> vacuum

class VacuumDetector:
    def __init__(self, cfg: VacuumConfig = None):
        self.cfg = cfg or VacuumConfig()
        self.spreads = deque(maxlen=self.cfg.window)
        self.bid_depths = deque(maxlen=self.cfg.window)
        self.ask_depths = deque(maxlen=self.cfg.window)

        # churn: count best-price changes per second
        self._best_changes = deque(maxlen=200)  # timestamps (ms)

    def _median(self, xs):
        if not xs:
            return 0.0
        s = sorted(xs)
        return float(s[len(s)//2])

    def update(
        self,
        *,
        now_ms: int,
        best_bid: float,
        best_ask: float,
        bid_depth_1: float,
        ask_depth_1: float,
        best_price_changed: bool,
    ) -> dict:
        spread = max(0.0, float(best_ask) - float(best_bid))
        self.spreads.append(spread)
        self.bid_depths.append(max(0.0, float(bid_depth_1)))
        self.ask_depths.append(max(0.0, float(ask_depth_1)))

        if best_price_changed:
            self._best_changes.append(now_ms)

        # churn per second over last 1000 ms
        cutoff = now_ms - 1000
        while self._best_changes and self._best_changes[0] < cutoff:
            self._best_changes.popleft()
        churn = float(len(self._best_changes))

        spread_med = self._median(self.spreads)
        bid_med = self._median(self.bid_depths)
        ask_med = self._median(self.ask_depths)

        # signal 1: spread jump / wide
        spread_jump = False
        if len(self.spreads) >= 2:
            spread_jump = (self.spreads[-1] - self.spreads[-2]) >= self.cfg.spread_jump
        spread_wide = spread >= self.cfg.spread_high

        # signal 2: depth collapse
        bid_collapse = (bid_med > 0 and bid_depth_1 < self.cfg.depth_drop_ratio * bid_med)
        ask_collapse = (ask_med > 0 and ask_depth_1 < self.cfg.depth_drop_ratio * ask_med)
        depth_collapse = bid_collapse or ask_collapse

        # signal 3: churn high
        churn_high = churn >= self.cfg.churn_high

        score = int(spread_jump or spread_wide) + int(depth_collapse) + int(churn_high)
        vacuum = score >= self.cfg.vacuum_score_trigger

        return {
            "vacuum": bool(vacuum),
            "vacuum_score": int(score),
            "spread": round(spread, 4),
            "churn_1s": round(churn, 2),
            "depth_collapse": bool(depth_collapse),
        }