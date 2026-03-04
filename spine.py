# spine.py — Decision Spine for PolyBot
# Centralizes: features → veto ordering → decision (one per side per tick)
#
# Plugs into your existing modules:
#   - gamma_engine.GammaEngine.should_block_exit(T, p_side, z, bid, regime, thesis_broken)
#   - vacuum_detector.VacuumDetector.update(now_ms, best_bid, best_ask, bid_depth_1, ask_depth_1, best_price_changed)
#   - synthetic_exit.SyntheticExitEngine.evaluate(side, bid, opp_ask, size, offset)
#
# Guarantees:
#   * one decision per side per tick (debounce)
#   * veto layers respected (fixed order)
#   * no duplicate exits (exit locks)
#   * deterministic decision surface (reason/mode + debug)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


# ─────────────────────────────────────────────
# Feature Container (single source of truth)
# ─────────────────────────────────────────────
@dataclass
class Features:
    now_ms: int

    # prices
    price_binance: float
    price_coinbase: float
    price_rtds: float
    rtds_age_ms: int
    T_sec: float                 # STATE.sec_remaining — used for gamma veto

    # book (for THIS side)
    bid: float
    ask: float
    spread: float
    depth: float                 # (bid_depth_1 + ask_depth_1) or your preferred proxy
    bid_depth_1: float
    ask_depth_1: float
    quote_age_ms: float
    quote_updates_per_s: float
    best_price_changed: bool     # you can feed this from your existing churn tracker

    # model
    z: float
    p_cone: float                # P(UP wins)
    sigma_1m: float
    regime: str

    # oracle prediction / posture
    p_update: float
    oracle_gamma: float
    oracle_posture: str          # "MAKER" / "MAKER_AGGR" / "CROSS"

    # flow
    flow: float                  # centered [-1,1]
    flow_accel: float

    # inventory (for THIS side)
    side: str                    # "UP" or "DOWN"
    token_id: str
    inventory: float
    available_qty: float
    in_flight: float

    # exits / unwind state
    thesis_broken: bool
    book_sane: bool
    unwind_state: str            # from UnwindControllerV2.update(...)
    sis_urgency: float           # optional: PositionMonitor exit urgency
    sis_frac: float              # optional: PositionMonitor exit fraction
    sis_offset: float            # optional: PositionMonitor exit offset
    exit_reason: str             # optional: PositionMonitor reason (prob_stop_loss, sis_exit, ...)

    # opposite book (for synthetic exit / parity)
    opp_best_ask: float
    opp_token_id: str


# ─────────────────────────────────────────────
# Decision object
# ─────────────────────────────────────────────
@dataclass
class Decision:
    action: str                  # BUY / SELL / HOLD
    side: str                    # "UP" / "DOWN" (which token this order targets)
    token_id: str
    size: float
    price: float
    mode: str                    # maker / taker / vacuum_cross / stale_quote / synthetic_exit / force_exit / thesis_exit
    reason: str
    debug: Dict[str, Any]


# ─────────────────────────────────────────────
# Decision Spine
# ─────────────────────────────────────────────
class DecisionSpine:
    """
    Deterministic policy spine.
    Call decide(f) once per side per tick.

    Exit ordering:
      1) hard safety
      2) inventory sanity
      3) thesis broken
      4) gamma veto
      5) synthetic exit (optional improvement)
      6) SIS/PM exits (if you pass them in)
      7) default hold

    Entry ordering:
      1) book sanity / budget sanity (caller can gate too)
      2) vacuum cross
      3) oracle posture cross
      4) stale quote snipe
      5) default hold
    """

    def __init__(
        self,
        *,
        gamma_engine,
        vacuum_detector,
        synthetic_exit,
        decision_cooldown_ms: int = 40,   # per-side debounce
        exit_lock_ms: int = 450,          # prevents duplicate exits while in-flight churn happens
    ):
        self.gamma = gamma_engine
        self.vacuum = vacuum_detector
        self.synthetic = synthetic_exit

        self._decision_cooldown_ms = int(decision_cooldown_ms)
        self._exit_lock_ms = int(exit_lock_ms)

        self._last_decision_ms: Dict[str, int] = {"UP": 0, "DOWN": 0}
        self._exit_locked_until_ms: Dict[str, int] = {"UP": 0, "DOWN": 0}

    # ─────────────────────────────────────────
    # public API
    # ─────────────────────────────────────────
    def decide(self, f: Features) -> Optional[Decision]:
        side = str(f.side).upper()
        now = int(f.now_ms)
        available_qty = float(max(0.0, f.available_qty))

        # 0) One decision per side per tick (debounce)
        if now - self._last_decision_ms.get(side, 0) < self._decision_cooldown_ms:
            return None
        self._last_decision_ms[side] = now

        # 1) HARD SAFETY (force exit regardless of gamma unless book is dead)
        #    Note: your main.py also has hard guards; this is extra belt+suspenders.
        if f.spread >= 0.60 or f.bid <= 0.05:
            if f.inventory > 0 and f.bid > 0.01:
                self._exit_locked_until_ms[side] = now + self._exit_lock_ms
                return Decision(
                    action="SELL",
                    side=side,
                    token_id=f.token_id,
                    size=float(max(0.0, f.inventory)),
                    price=float(f.bid),
                    mode="force_exit",
                    reason="force_exit_threshold",
                    debug={"spread": f.spread, "bid": f.bid, "ask": f.ask},
                )
            return None

        # 2) INVENTORY SANITY
        #    - If there is nothing available to act with, hold.
        #    - If an exit is already locked / in-flight, do not duplicate.
        if f.available_qty <= 0 and f.inventory <= 0:
            return None

        if now < self._exit_locked_until_ms.get(side, 0):
            # Exit already issued very recently; avoid double-sells.
            return None

        # ─────────────────────────────────────────
        # EXIT PATH (if we have inventory)
        # ─────────────────────────────────────────
        if f.inventory > 0 and f.bid > 0.01:
            # Never route an exit when all inventory is already in-flight.
            if available_qty <= 0.0:
                return None

            # 3) THESIS BROKEN — immediate thesis exit (overrides gamma)
            if bool(f.thesis_broken):
                self._exit_locked_until_ms[side] = now + self._exit_lock_ms
                return Decision(
                    action="SELL",
                    side=side,
                    token_id=f.token_id,
                    size=available_qty,
                    price=float(f.bid),
                    mode="thesis_exit",
                    reason="thesis_broken",
                    debug={"unwind_state": f.unwind_state, "book_sane": f.book_sane},
                )

            # 4) GAMMA VETO (blocks EV exits / SIS trims unless thesis broken)
            p_side = float(f.p_cone) if side == "UP" else float(1.0 - f.p_cone)
            gamma_block, gamma_why = self.gamma.should_block_exit(
                T=float(max(0.0, f.T_sec)),
                p_side=float(p_side),
                z=float(f.z),
                bid=float(f.bid),
                regime=str(f.regime),
                thesis_broken=bool(f.thesis_broken),
            )
            if gamma_block:
                return None

            # 5) SYNTHETIC EXIT (box spread) — if it strictly dominates direct sell
            #    This returns a BUY on the opposite token (locks $1 at settlement).
            #    Caller should mark BOTH legs in-flight like your current synthetic block does.
            if self.synthetic is not None and f.opp_best_ask > 0.01 and f.opp_best_ask < 0.99:
                syn = self.synthetic.evaluate(
                    side=side,
                    bid=float(f.bid),
                    opp_ask=float(f.opp_best_ask),
                    size=available_qty,
                    offset=float(f.sis_offset or 0.0),
                )
                if syn.get("use_synthetic"):
                    self._exit_locked_until_ms[side] = now + self._exit_lock_ms
                    return Decision(
                        action="BUY",
                        side=("DOWN" if side == "UP" else "UP"),
                        token_id=str(f.opp_token_id),
                        size=available_qty,
                        price=float(syn["synthetic_price"]),
                        mode="synthetic_exit",
                        reason="synthetic_dominates_direct_sell",
                        debug={
                            "exit_side": side,
                            "exit_token_id": f.token_id,
                            "direct_bid": f.bid,
                            "opp_ask": f.opp_best_ask,
                            "implied_bid": syn.get("implied_bid"),
                            "improvement_net": syn.get("improvement_net"),
                        },
                    )

            # 6) POSITION MONITOR / SIS EXIT (you pass in sis_* + exit_reason)
            #    If you don't pass SIS signals, this simply won't trigger.
            if f.exit_reason:
                # Use SIS fraction if present; otherwise full exit
                frac = float(f.sis_frac) if f.sis_frac and f.sis_frac > 0 else 1.0
                size = float(max(0.0, available_qty * frac))

                if size > 0:
                    self._exit_locked_until_ms[side] = now + self._exit_lock_ms
                    return Decision(
                        action="SELL",
                        side=side,
                        token_id=f.token_id,
                        size=size,
                        price=float(max(0.01, f.bid - float(f.sis_offset or 0.0))),
                        mode="exit",
                        reason=str(f.exit_reason),
                        debug={
                            "sis_u": f.sis_urgency,
                            "sis_frac": f.sis_frac,
                            "sis_offset": f.sis_offset,
                            "unwind_state": f.unwind_state,
                            "gamma": gamma_why,
                        },
                    )

            # Default exit HOLD
            return None

        # ─────────────────────────────────────────
        # ENTRY PATH (if flat on this side)
        # ─────────────────────────────────────────

        # Basic sanity (caller already has deeper gates)
        if not f.book_sane:
            return None
        if f.in_flight and f.in_flight > 0:
            return None
        if f.ask <= 0.01 or f.ask >= 0.99:
            return None

        # A) VACUUM / MICROSTRUCTURE CROSS
        vac = self.vacuum.update(
            now_ms=int(f.now_ms),
            best_bid=float(f.bid),
            best_ask=float(f.ask),
            bid_depth_1=float(f.bid_depth_1),
            ask_depth_1=float(f.ask_depth_1),
            best_price_changed=bool(f.best_price_changed),
        )
        vacuum_on = bool(vac.get("vacuum", False))
        if vacuum_on and float(f.p_update) > 0.50:
            return Decision(
                action="BUY",
                side=side,
                token_id=f.token_id,
                size=float(f.available_qty),
                price=float(f.ask),
                mode="vacuum_cross",
                reason="liquidity_vacuum",
                debug={"vac": vac, "p_update": f.p_update},
            )

        # B) ORACLE GAMMA POSTURE CROSS
        if str(f.oracle_posture).upper() == "CROSS":
            return Decision(
                action="BUY",
                side=side,
                token_id=f.token_id,
                size=float(f.available_qty),
                price=float(f.ask),
                mode="oracle_cross",
                reason="oracle_gamma",
                debug={"oracle_gamma": f.oracle_gamma, "p_update": f.p_update},
            )

        # C) STALE QUOTE SNIPE
        if f.quote_age_ms > 600 and f.spread < 0.03 and f.depth > 50:
            if float(f.p_update) > 0.55:
                return Decision(
                    action="BUY",
                    side=side,
                    token_id=f.token_id,
                    size=float(f.available_qty),
                    price=float(f.ask),
                    mode="stale_quote_snipe",
                    reason="stale_liquidity",
                    debug={"quote_age_ms": f.quote_age_ms, "depth": f.depth, "p_update": f.p_update},
                )

        return None
