# execution_fsm.py
# Order lifecycle controller for PolyBot

from dataclasses import dataclass
from typing import Optional, Dict, Any
import uuid


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class ExecFSMConfig:
    min_dispatch_interval_ms: int = 250
    maker_ttl_ms: int = 1200
    miss_cooldown_ms: int = 350
    stale_order_ms: int = 5000


# ─────────────────────────────────────────────
# Intent State
# ─────────────────────────────────────────────

class _Intent:

    def __init__(self, decision, now_ms: int):
        self.intent_id = str(uuid.uuid4())
        self.side = decision.side
        self.token_id = decision.token_id
        self.price = decision.price
        self.size = decision.size
        self.mode = decision.mode
        self.action = decision.action
        self.reason = decision.reason

        self.created_ms = now_ms
        self.last_update_ms = now_ms

        self.state = "REQUESTED"
        self.order_id = None


# ─────────────────────────────────────────────
# FSM Engine
# ─────────────────────────────────────────────

class ExecutionFSM:

    def __init__(self, config: ExecFSMConfig):
        self.cfg = config

        self.active_intent: Dict[str, Optional[_Intent]] = {
            "UP": None,
            "DOWN": None
        }

        self.last_dispatch_ms: Dict[str, int] = {
            "UP": 0,
            "DOWN": 0
        }

        self.cooldown_until: Dict[str, int] = {
            "UP": 0,
            "DOWN": 0
        }

    # ─────────────────────────────────────────
    # Request new execution
    # ─────────────────────────────────────────

    def request(self, decision, now_ms: int):

        side = decision.side

        if now_ms < self.cooldown_until[side]:
            return None

        if now_ms - self.last_dispatch_ms[side] < self.cfg.min_dispatch_interval_ms:
            return None

        existing = self.active_intent.get(side)

        if existing and existing.state not in ("FILLED", "REJECT", "ERROR", "MISS", "CONFIRMED_MISS"):
            return None

        intent = _Intent(decision, now_ms)

        self.active_intent[side] = intent
        self.last_dispatch_ms[side] = now_ms

        return {
            "action": "ORDER",
            "intent_id": intent.intent_id,
            "side": intent.side,
            "token_id": intent.token_id,
            "price": intent.price,
            "size": intent.size,
            "order_side": intent.action,
            "mode": intent.mode,
            "exit_reason": intent.reason
        }

    # ─────────────────────────────────────────
    # Order submitted
    # ─────────────────────────────────────────

    def on_submitted(self, side, intent_id, order_id, now_ms):

        intent = self.active_intent.get(side)

        if not intent or intent.intent_id != intent_id:
            return

        intent.order_id = order_id
        intent.state = "SUBMITTED"
        intent.last_update_ms = now_ms

    # ─────────────────────────────────────────
    # Order result
    # ─────────────────────────────────────────

    def on_result(
        self,
        side,
        intent_id,
        now_ms,
        state,
        order_id=None,
        filled_qty=0,
        avg_price=0,
        detail=""
    ):

        intent = self.active_intent.get(side)

        if not intent or intent.intent_id != intent_id:
            return

        intent.state = state
        intent.last_update_ms = now_ms
        if order_id is not None:
            intent.order_id = order_id

        if state in ("MISS", "CONFIRMED_MISS"):
            self.cooldown_until[side] = now_ms + self.cfg.miss_cooldown_ms

    # ─────────────────────────────────────────
    # Cancel event
    # ─────────────────────────────────────────

    def on_cancelled(self, side, intent_id, now_ms, detail=""):

        intent = self.active_intent.get(side)

        if not intent or intent.intent_id != intent_id:
            return

        intent.state = "CANCELLED"
        intent.last_update_ms = now_ms

    # ─────────────────────────────────────────
    # Maintenance tick
    # ─────────────────────────────────────────

    def maybe_tick(self, now_ms):
        """Return list of cancel actions for all stale/expired sides."""
        cancels = []

        for side, intent in self.active_intent.items():

            if not intent:
                continue

            if intent.state != "SUBMITTED":
                continue

            age = now_ms - intent.last_update_ms

            if intent.mode.startswith("maker") and age > self.cfg.maker_ttl_ms:

                cancels.append({
                    "action": "CANCEL",
                    "order_id": intent.order_id,
                    "side": side,
                    "intent_id": intent.intent_id
                })
                continue

            if age > self.cfg.stale_order_ms:

                cancels.append({
                    "action": "CANCEL",
                    "order_id": intent.order_id,
                    "side": side,
                    "intent_id": intent.intent_id
                })

        return cancels if cancels else None
