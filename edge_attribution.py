# edge_attribution.py
# Tracks which signals actually make money

import json
import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AttributionConfig:
    log_dir: str = "logs"
    min_tick_interval_ms: int = 100


class EdgeAttribution:

    def __init__(self, config: AttributionConfig):

        self.cfg = config

        os.makedirs(config.log_dir, exist_ok=True)

        self.tick_file = os.path.join(config.log_dir, "edge_ticks.jsonl")
        self.exec_file = os.path.join(config.log_dir, "edge_exec.jsonl")

        self.last_tick = 0

    # ─────────────────────────────────────────
    # Tick snapshot
    # ─────────────────────────────────────────

    def log_tick(self, now_ms: int, features: Dict[str, Any], decision):

        if now_ms - self.last_tick < self.cfg.min_tick_interval_ms:
            return

        self.last_tick = now_ms

        rec = {
            "ts": now_ms,
            "features": features,
            "decision": decision
        }

        with open(self.tick_file, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ─────────────────────────────────────────
    # Execution event
    # ─────────────────────────────────────────

    def log_exec(
        self,
        ts,
        intent_id,
        side,
        action,
        mode,
        price,
        size_req,
        state,
        order_id,
        filled_qty,
        avg_price,
        realized_edge,
        realized_pnl,
        detail="",
        tags=None
    ):

        rec = {
            "ts": ts,
            "intent_id": intent_id,
            "side": side,
            "action": action,
            "mode": mode,
            "price": price,
            "size_req": size_req,
            "state": state,
            "order_id": order_id,
            "filled_qty": filled_qty,
            "avg_price": avg_price,
            "realized_edge": realized_edge,
            "realized_pnl": realized_pnl,
            "detail": detail,
            "tags": tags or {}
        }

        with open(self.exec_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
