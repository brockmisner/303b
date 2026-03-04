from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ── Convexity-aware exit model ─────────────────────────────────────

def convexity_weight(T: float) -> float:
    """Binary options have Γ ~ 1/√T — more convex near expiry.
    Early: need stronger reason to exit.
    Late: easier to exit."""
    if T > 120:
        return 0.5
    elif T > 60:
        return 0.75
    elif T > 30:
        return 1.0
    else:
        return 1.5


def required_ev_gap(T: float, base: float = 0.02) -> float:
    """Convexity-adjusted EV gap required to justify exit."""
    return base * convexity_weight(T)


@dataclass
class UnwindConfig:
    # State transition thresholds
    build_window_s: float = 12.0                # BUILDING lasts this long after entry
    unwind_urgency_threshold: float = 0.45      # urgency alone cannot trigger UNWINDING from HOLDING
    unwind_time_s: float = 30.0                 # time remaining below this => UNWINDING (if z weak)
    ev_deterioration_eps: float = 0.01          # EV must deteriorate by this much to confirm unwind

    profit_lock_threshold: float = 0.05         # profit/share above this + EV sell>hold => LOCK_PROFIT

    force_exit_spread: float = 0.60             # spread above this => FORCE_EXIT
    force_exit_bid_floor: float = 0.05          # bid below this => FORCE_EXIT
    force_exit_time_s: float = 15.0             # time remaining below this => FORCE_EXIT

    # Catastrophic collapse thresholds (routes to UNWINDING, NOT FORCE_EXIT)
    collapse_spread: float = 0.40               # spread above this => catastrophic
    collapse_bid_floor: float = 0.10            # bid below this => catastrophic
    collapse_unwind_T: float = 30.0             # catastrophic + T <= this => UNWINDING even without thesis broken

    # Time-critical z threshold: strong z near expiry stays HOLDING, not UNWINDING
    time_critical_z: float = 0.8                # |z| >= this at T < 30s → stay HOLDING (convex mode)

    # Thesis broken (institutional) ──────────────────────────────────
    # Note: widened to leave breathing room for the Gamma Engine's 0.45 pin band
    thesis_p_break_early: float = 0.35          # UP thesis broken if p_side < this (T > 120s)
    thesis_p_break_mid: float = 0.40            # mid-window (60 < T <= 120s)
    thesis_p_break_late: float = 0.44           # late (T <= 60s) - carefully placed just below pin band (0.45)

    thesis_z_invert: float = 0.40               # z inversion threshold (direction flipped)
    flow_against_threshold: float = 0.30        # centered flow threshold for "against"
    flow_persist_ticks: int = 3                 # require N consecutive ticks of adverse flow

    # Book-sanity guards
    max_sane_spread: float = 0.20               # spread above this => book insane
    min_sane_bid: float = 0.05                  # bid below this => book insane
    max_sane_ask: float = 0.95                  # ask above this => book insane

    # Dust / micro-trim controls
    dust_threshold: float = 0.05                # inventory below this => treat as FLAT (suppress actions)
    micro_disable_threshold: float = 1.0        # inventory below this => no fractional trims; full-exit sizing only


class UnwindControllerV2:
    """
    Exposure lifecycle controller (state machine).

    States:
      FLAT -> BUILDING -> HOLDING -> UNWINDING -> LOCK_PROFIT -> FORCE_EXIT -> FLAT

    (Note: Convexity/Gamma ride logic is now handled strictly by the external GammaEngine layer
    to prevent competing authorities).
    """

    def __init__(self, cfg: Optional[UnwindConfig] = None):
        self.cfg = cfg or UnwindConfig()
        self.state: str = "FLAT"
        self._entry_ts_ms: Optional[int] = None
        self._last_inventory: float = 0.0
        self._book_sane: bool = True
        self._thesis_broken: bool = False
        self._thesis_broken_raw: bool = False
        self._thesis_broken_streak: int = 0
        self._flow_against_streak: int = 0

    def reset(self) -> None:
        self.state = "FLAT"
        self._entry_ts_ms = None
        self._last_inventory = 0.0
        self._book_sane = True
        self._thesis_broken = False
        self._thesis_broken_raw = False
        self._thesis_broken_streak = 0
        self._flow_against_streak = 0

    def _entered_position(self, inventory: float) -> bool:
        return (abs(self._last_inventory) < self.cfg.dust_threshold) and (abs(inventory) >= self.cfg.dust_threshold)

    def _exited_position(self, inventory: float) -> bool:
        return (abs(self._last_inventory) >= self.cfg.dust_threshold) and (abs(inventory) < self.cfg.dust_threshold)

    def _is_catastrophic(self, spread: float, best_bid: float) -> bool:
        """Structural collapse — softer than FORCE_EXIT."""
        return float(spread) > self.cfg.collapse_spread or float(best_bid) < self.cfg.collapse_bid_floor

    def _p_break_cutoff(self, T: float) -> float:
        """Probability threshold for declaring thesis broken."""
        if T > 120:
            return self.cfg.thesis_p_break_early
        elif T > 60:
            return self.cfg.thesis_p_break_mid
        else:
            return self.cfg.thesis_p_break_late

    def _update_flow_streak(self, *, side: str, flow_centered: float) -> None:
        """Track persistence of flow against our position."""
        s = str(side).upper()
        f = float(flow_centered)

        against = False
        if s == "UP":
            against = (f <= -self.cfg.flow_against_threshold)
        elif s == "DOWN":
            against = (f >= self.cfg.flow_against_threshold)

        if against:
            self._flow_against_streak = min(20, self._flow_against_streak + 1)
        else:
            self._flow_against_streak = 0

    def _thesis_broken_institutional(
        self,
        *,
        side: str,
        p_side: float,
        z: float,
        T: float,
        flow_centered: float,
    ) -> bool:
        """
        Institutional thesis-broken test. (Acts as the override for GammaEngine).
        """
        s = str(side).upper()
        p = float(p_side)
        zz = float(z)
        TT = float(T)

        # 1) Z inversion: strongest signal that direction flipped
        if s == "UP" and zz <= -self.cfg.thesis_z_invert:
            return True
        if s == "DOWN" and zz >= self.cfg.thesis_z_invert:
            return True

        # 2) Probability collapse: time-adjusted threshold
        p_cut = self._p_break_cutoff(TT)
        if s == "UP" and p <= p_cut:
            return True
        if s == "DOWN" and p >= (1.0 - p_cut):
            return True

        # 3) Persistent adverse flow + weak z
        self._update_flow_streak(side=s, flow_centered=flow_centered)
        if self._flow_against_streak >= int(self.cfg.flow_persist_ticks):
            if abs(zz) < 0.8:   # z not strong enough to ignore flow
                return True

        return False

    def update(
        self,
        *,
        now_ms: int,
        inventory: float,
        urgency: float,
        profit_per_share: float,
        ev_sell: float,
        ev_hold: float,
        spread: float,
        best_bid: float,
        time_remaining: float,
        z: float = 0.0,
        regime: str = "NORMAL",
        p_side: float = 0.5,
        side: str = "UP",
        flow_centered: float = 0.0,
        best_ask: float = 1.0,
    ) -> str:
        """
        Update state given current conditions. Returns state string.

        Priority cascade:
          1. FLAT (dust)
          2. FORCE_EXIT (hard thresholds ONLY: spread > 0.60, bid < 0.05, T <= 15s)
          3. Catastrophic (0.40/0.10): UNWINDING if thesis broken or late, else HOLDING
          4. LOCK_PROFIT (profitable + EV gap, convexity-adjusted)
          5. BUILDING (first 12s post-entry)
          6. HOLDING ↔ UNWINDING (two-stage gating)
        """
        inv = float(abs(inventory))
        _z = float(z)
        _T = float(time_remaining)
        _p = float(p_side)
        _spread = float(spread)
        _bid = float(best_bid)
        _ask = float(best_ask)
        _flow = float(flow_centered)
        _pps = float(profit_per_share)

        self._book_sane = (
            _spread <= self.cfg.max_sane_spread
            and _bid >= self.cfg.min_sane_bid
            and _ask <= self.cfg.max_sane_ask
        )

        _raw = self._thesis_broken_institutional(
            side=side, p_side=_p, z=_z, T=_T, flow_centered=_flow,
        )
        self._thesis_broken_raw = _raw

        # Hysteresis: require 2 consecutive broken ticks, cool down by 1
        if _raw:
            self._thesis_broken_streak += 1
        else:
            self._thesis_broken_streak = max(0, self._thesis_broken_streak - 1)
        self._thesis_broken = (self._thesis_broken_streak >= 2)

        if self._entered_position(inv):
            self._entry_ts_ms = int(now_ms)
            self.state = "BUILDING"

        if self._exited_position(inv):
            self.state = "FLAT"
            self._entry_ts_ms = None
            self._flow_against_streak = 0
            self._thesis_broken_streak = 0
            self._thesis_broken = False

        # ── 1. FLAT (dust) ──
        if inv < self.cfg.dust_threshold:
            self.state = "FLAT"
            self._entry_ts_ms = None
            self._last_inventory = inv
            return self.state

        # ── 2. FORCE_EXIT (hard thresholds ONLY) ──
        hard_force_exit = (
            _spread > self.cfg.force_exit_spread
            or _bid < self.cfg.force_exit_bid_floor
            or _T <= self.cfg.force_exit_time_s
        )
        if hard_force_exit:
            self.state = "FORCE_EXIT"
            self._last_inventory = inv
            return self.state

        # ── 3. Catastrophic (0.40/0.10): UNWINDING if thesis broken or late ──
        if self._is_catastrophic(_spread, _bid):
            if self._thesis_broken or _T <= self.cfg.collapse_unwind_T:
                self.state = "UNWINDING"
            else:
                self.state = "HOLDING"
            self._last_inventory = inv
            return self.state

        # ── 4. LOCK_PROFIT (convexity-adjusted) ──
        _req_gap = required_ev_gap(_T)
        _profit_ok = _pps >= self.cfg.profit_lock_threshold
        _ev_gap_ok = (float(ev_sell) - float(ev_hold)) > _req_gap

        if _profit_ok and _ev_gap_ok:
            self.state = "LOCK_PROFIT"
            self._last_inventory = inv
            return self.state

        # ── 5. BUILDING: first N seconds post-entry ──
        if self.state == "BUILDING":
            _ev_negative = float(ev_sell) > float(ev_hold) and _pps < 0
            _time_critical = _T <= self.cfg.unwind_time_s and abs(_z) < self.cfg.time_critical_z
            if _ev_negative or _time_critical:
                self.state = "UNWINDING"
            else:
                if self._entry_ts_ms is None:
                    self._entry_ts_ms = int(now_ms)
                age_s = (int(now_ms) - int(self._entry_ts_ms)) / 1000.0
                if age_s >= self.cfg.build_window_s:
                    self.state = "HOLDING"

            self._last_inventory = inv
            return self.state

        # ── 6. HOLDING ↔ UNWINDING: two-stage gating ──
        _ev_deteriorating = float(ev_sell) > (float(ev_hold) + self.cfg.ev_deterioration_eps)
        _urgency_extreme = float(urgency) > self.cfg.unwind_urgency_threshold
        _time_critical = _T <= self.cfg.unwind_time_s and abs(_z) < self.cfg.time_critical_z

        if _time_critical:
            self.state = "UNWINDING"
        elif _urgency_extreme and _ev_deteriorating:
            self.state = "UNWINDING"
        else:
            self.state = "HOLDING"

        self._last_inventory = inv
        return self.state

    # ── Policy helpers ─────────────────────────────────────────────

    def allow_survival(self) -> bool:
        """Only allow survival hedges in HOLDING."""
        return self.state == "HOLDING"

    def allow_fractional_trim(self, inventory: float,
                               z: float = 0.0, time_remaining: float = 0.0,
                               regime: str = "NORMAL") -> bool:
        """
        Fractional trim allowed only in UNWINDING, and only when inventory is not tiny.
        (Gamma shield is now handled by GammaEngine veto).
        """
        inv = float(abs(inventory))
        if inv < self.cfg.micro_disable_threshold:
            return False
        if self.state != "UNWINDING":
            return False
        if not self._book_sane:
            return False
        return True

    def force_full_exit(self, inventory: float) -> bool:
        """
        If LOCK_PROFIT or FORCE_EXIT, always full exit.
        Also full-exit when inventory is tiny (avoid dust spam).
        Thesis broken => full exit allowed regardless of book sanity.
        """
        inv = float(abs(inventory))
        if inv < self.cfg.micro_disable_threshold:
            return True
        if self._thesis_broken:
            return True
        return self.state in ("LOCK_PROFIT", "FORCE_EXIT")

    def suppress_actions(self, inventory: float) -> bool:
        """No actions in FLAT or dust."""
        if float(abs(inventory)) < self.cfg.dust_threshold:
            return True
        if self.state == "FLAT":
            return True
        return False
