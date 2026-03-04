# position_monitor.py — Unified EV-driven position exit engine + Signal Invalidation Surface
"""
Subsumes ExitManager + EndgameManager + SIS into one p/EV-based decision system.

All exit decisions are probabilistic — no fixed price thresholds.
  hold_EV = p_side (probability this token settles at $1.00)
  sell_EV = bid - fee(bid)
  fee     = bid * (1-bid) * market_rate   (passed via fee_fn)

Decision priority (checked in order):
  1. PROB STOP-LOSS: p collapsed AND sell_EV >= hold_EV - penalty
  2. EV EXIT: profitable AND market overpaying vs model
  2.5. SIS EXIT: signal invalidation surface urgency >= threshold
  3. GAMMA DANGER: in uncertainty band AND profitable AND sell_EV adequate
  4. TRAILING STOP: in gamma band AND trail activated AND bid < HWM
  5. ENDGAME (T < 25s): strong loser → SELL; gamma+profitable → SELL

Signal Invalidation Surface (SIS):
  Continuous urgency u ∈ [0,1] from three logistic sub-scores:
    S_edge:   edge retention ratio decay + negative edge spike
    S_mom:    z_ema level + dz/dt slope (direction-aligned)
    S_struct: flow flip + time pressure + p_adv + regime multiplier
  Combined via product form, smoothed with EMA + stickiness.

Usage:
    PM = PositionMonitor()
    decision = PM.evaluate("UP", entry_price=0.15, bid=0.24,
                           p_cone=0.26, T_sec=180, sigma_1m=0.002,
                           regime="CALM", now_ms=ms_now(),
                           edge_entry=0.04, edge_now=0.02,
                           z_ema_now=0.8, z_ema_prev=0.9,
                           flow_centered=0.1)
    if decision.action == "SELL":
        # queue exit order with decision.exit_urgency / decision.exit_frac
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import math
import logging
import time

logger = logging.getLogger("position_monitor")

Side = Literal["UP", "DOWN"]


# ─────────────────────────────────────────────────────────────────────────────
# Signal Invalidation Surface (SIS) — continuous exit urgency
# ─────────────────────────────────────────────────────────────────────────────

def _sig(x: float) -> float:
    """Safe logistic squash → [0, 1]."""
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _dir_sign(side: str) -> float:
    """+1 for UP position (z>0 is good), -1 for DOWN (z<0 is good)."""
    return 1.0 if side == "UP" else -1.0


def edge_now_for_position(side: str, p_cone: float,
                          best_bid_up: float, best_bid_dn: float) -> float:
    """Current edge for a held position (sell-side EV vs model probability).
    Simple form without fees — we care about relative decay, not exact cents."""
    p_up = float(p_cone)
    p_dn = 1.0 - p_up
    if side == "UP":
        return p_up - float(best_bid_up)
    else:
        return p_dn - float(best_bid_dn)


@dataclass
class SISConfig:
    """All tunable constants for the Signal Invalidation Surface."""
    # ── Edge decay ──
    r0: float = 0.55           # start caring once ~45% of edge lost
    k1: float = 8.0            # steepness of edge retention sigmoid
    e_scale: float = 0.01      # 1 cent edge scale for negative-edge sigmoid
    k2: float = 6.0            # steepness of negative-edge sigmoid

    # ── Momentum slope ──
    z_floor: float = 0.10      # below this z, you're not winning
    k3: float = 4.0            # z level sigmoid steepness
    dz_scale: float = 0.02     # z units per tick (normalize slope)
    k4: float = 10.0           # slope sigmoid steepness (was 6.0, steeper for faster reversal detection)

    # ── Structural: flow ──
    flow_floor: float = 0.0    # below 0 = pressure against you
    k5: float = 5.0            # flow sigmoid steepness

    # ── Structural: adverse selection ──
    adv0: float = 0.25         # p_adv baseline
    adv_scale: float = 0.10    # p_adv normalization
    k8: float = 3.0            # adv sigmoid steepness

    # ── Structural: time pressure ──
    T_crit: float = 60.0       # inside 60s → urgency ramps
    T_scale: float = 20.0      # sigmoid width
    k7: float = 3.0            # time sigmoid steepness

    # ── Structural: sigma ──
    sigma_baseline: float = 0.0010   # ~10bp baseline vol
    k_sigma: float = 2.0              # sigma sigmoid steepness

    # ── Regime multipliers ──
    regime_mult: dict = field(default_factory=lambda: {
        "CALM": 0.7,
        "NORMAL": 1.0,
        "TRANSITION": 1.2,
        "HIGH_VOL": 1.5,
        "VOL_EVENT": 1.9,
    })

    # ── Combination weights ──
    # NOTE: product-form max u_raw = 1-(1-wE)(1-wM)(1-wS); weights must allow u_raw > u_aggressive
    wE: float = 0.50    # edge decay (tuned so max u_raw ≈ 0.83, above u_aggressive=0.80)
    wM: float = 0.40    # momentum
    wS: float = 0.35    # structure

    # ── Smoothing ──
    eta: float = 0.15           # EMA step size
    decay_floor: float = 0.92  # urgency decays slowly (stickiness)

    # ── Exit policy thresholds ──
    u_hold: float = 0.25      # below this → HOLD
    u_maker: float = 0.55     # above → maker with concession
    u_aggressive: float = 0.80  # above → aggressive exit (if book sane)

    # ── Min hold guard ──
    min_hold_ms: int = 5000


class SignalInvalidationSurface:
    """
    Continuous Signal Invalidation Surface.

    Outputs smooth exit_urgency ∈ [0,1] that monotonically increases
    when the trade thesis deteriorates.

    u ~ 0.0 → hold
    u ~ 0.3 → passive maker exit (tight, re-quote)
    u ~ 0.6 → maker exit with concession / priority
    u ~ 0.8+ → aggressive exit (FAK/FOK if book real, else GTC max concession)
    """

    def __init__(self, cfg: SISConfig = None):
        self.cfg = cfg or SISConfig()
        # Per-side smoothed urgency state
        self._u: dict[str, float] = {"UP": 0.0, "DOWN": 0.0}
        # Per-side z_ema prev (for slope computation)
        self._z_ema_prev: dict[str, float] = {"UP": 0.0, "DOWN": 0.0}
        # Per-side last log threshold crossed (prevent log spam)
        self._last_log_band: dict[str, str] = {"UP": "hold", "DOWN": "hold"}

    def clear(self, side: str = None) -> None:
        """Clear on window rollover or position close."""
        if side:
            self._u[side] = 0.0
            self._z_ema_prev[side] = 0.0
            self._last_log_band[side] = "hold"
        else:
            self._u = {"UP": 0.0, "DOWN": 0.0}
            self._z_ema_prev = {"UP": 0.0, "DOWN": 0.0}
            self._last_log_band = {"UP": "hold", "DOWN": "hold"}

    def update_z_ema_prev(self, side: str, z_ema: float) -> None:
        """Call every tick to maintain z_ema history for slope."""
        self._z_ema_prev[side] = float(z_ema)

    def compute(
        self,
        side: str,                # "UP" or "DOWN" (position held)
        edge_entry: float,        # edge at entry
        edge_now: float,          # current edge for same side
        z_ema_now: float,         # current z_ema
        flow_centered: float,     # [-1, 1] flow imbalance
        sigma_1m: float,          # STATE.sigma_1m
        T_sec: float,             # seconds remaining
        regime: str,              # regime label
        entry_ts_ms: int = 0,     # entry timestamp
        now_ms: int = 0,          # current time
    ) -> tuple[float, dict]:
        """
        Compute exit urgency u ∈ [0,1].

        Returns (u, diagnostics_dict) with all sub-scores for logging.
        """
        cfg = self.cfg
        if now_ms <= 0:
            now_ms = int(time.time() * 1000)

        # Min hold guard — still update z_ema_prev to avoid stale slope spike on expiry
        if entry_ts_ms > 0 and (now_ms - entry_ts_ms) < cfg.min_hold_ms:
            self._z_ema_prev[side] = float(z_ema_now)
            return self._u.get(side, 0.0), {"reason": "min_hold",
                                              "hold_ms": now_ms - entry_ts_ms}

        sgn = _dir_sign(side)
        u_prev = self._u.get(side, 0.0)
        z_ema_prev = self._z_ema_prev.get(side, z_ema_now)

        # ═══════════════════════════════════════════════════════════════
        # S_edge: Edge Decay Score
        # ═══════════════════════════════════════════════════════════════
        e0 = max(abs(edge_entry), 1e-4)   # floor to 1bp
        r = edge_now / e0                  # retention ratio (can go negative)
        S_edge_decay = _sig(cfg.k1 * (cfg.r0 - r))
        S_edge_flip = _sig(cfg.k2 * (-edge_now / cfg.e_scale))
        S_edge = max(S_edge_decay, S_edge_flip)

        # ═══════════════════════════════════════════════════════════════
        # S_mom: Momentum Slope Score
        # ═══════════════════════════════════════════════════════════════
        z_dir = sgn * float(z_ema_now)
        dz_dir = sgn * float(z_ema_now - z_ema_prev)
        S_mom_level = _sig(cfg.k3 * (cfg.z_floor - z_dir))
        S_mom_slope = _sig(cfg.k4 * (-dz_dir / cfg.dz_scale))
        S_mom = max(S_mom_level, S_mom_slope)

        # ═══════════════════════════════════════════════════════════════
        # S_struct: Structural Flip Score
        # ═══════════════════════════════════════════════════════════════

        # Flow aligned to position
        flow_dir = sgn * float(flow_centered)
        S_flow = _sig(cfg.k5 * (cfg.flow_floor - flow_dir))

        # Adverse selection proxy (time-based)
        p_adv = _clamp(1.0 - float(T_sec) / 300.0, 0.0, 1.0)
        S_adv = _sig(cfg.k8 * ((p_adv - cfg.adv0) / cfg.adv_scale))

        # Time pressure
        time_press = _sig(cfg.k7 * ((cfg.T_crit - float(T_sec)) / cfg.T_scale))

        # Sigma pressure
        S_sig = _sig(cfg.k_sigma * ((float(sigma_1m) - cfg.sigma_baseline) / cfg.sigma_baseline))

        # Regime multiplier
        reg_mult = cfg.regime_mult.get(str(regime).upper(), 1.0)

        # Combine structural (any red flag raises urgency)
        S_struct = 1.0 - (1.0 - S_flow) * (1.0 - S_adv) * (1.0 - time_press) * (1.0 - 0.35 * S_sig)
        S_struct = _clamp(S_struct * min(2.0, reg_mult), 0.0, 1.0)

        # ═══════════════════════════════════════════════════════════════
        # Combine (product form — prevents >1 blowups)
        # ═══════════════════════════════════════════════════════════════
        wE, wM, wS = cfg.wE, cfg.wM, cfg.wS
        u_raw = 1.0 - (1.0 - wE * S_edge) * (1.0 - wM * S_mom) * (1.0 - wS * S_struct)
        u_raw = _clamp(u_raw, 0.0, 1.0)

        # ═══════════════════════════════════════════════════════════════
        # Smooth + stickiness
        # ═══════════════════════════════════════════════════════════════
        u = u_prev + cfg.eta * (u_raw - u_prev)
        u = max(u, u_prev * cfg.decay_floor)  # don't drop instantly
        u = _clamp(u, 0.0, 1.0)

        # Store state
        self._u[side] = u
        self._z_ema_prev[side] = float(z_ema_now)

        # ── Threshold crossing log ──
        if u >= cfg.u_aggressive:
            band = "aggressive"
        elif u >= cfg.u_maker:
            band = "maker_concession"
        elif u >= cfg.u_hold:
            band = "maker_tight"
        else:
            band = "hold"

        prev_band = self._last_log_band.get(side, "hold")
        if band != prev_band:
            self._last_log_band[side] = band
            logger.info(
                f"SIS({side}): u={u:.3f} band={band} "
                f"S_edge={S_edge:.3f}(r={r:.2f}) "
                f"S_mom={S_mom:.3f}(z_dir={z_dir:.3f} dz={dz_dir:.4f}) "
                f"S_struct={S_struct:.3f}(flow={flow_dir:.2f} "
                f"p_adv={p_adv:.2f} time_p={time_press:.2f}) "
                f"regime={regime} T={T_sec:.0f}s "
                f"edge_entry={edge_entry:.4f} edge_now={edge_now:.4f}"
            )

        diag = {
            "u": round(u, 4),
            "u_raw": round(u_raw, 4),
            "S_edge": round(S_edge, 4),
            "S_mom": round(S_mom, 4),
            "S_struct": round(S_struct, 4),
            "r": round(r, 3),
            "z_dir": round(z_dir, 3),
            "dz_dir": round(dz_dir, 4),
            "flow_dir": round(flow_dir, 3),
            "p_adv": round(p_adv, 3),
            "time_press": round(time_press, 3),
            "band": band,
            "regime": regime,
        }
        return u, diag

    def exit_frac(self, u: float) -> float:
        """Continuous exit fraction: maps u → fraction of inventory to exit.
        u < 0.25 → 0%, u = 1.0 → 100%."""
        return _clamp((u - self.cfg.u_hold) / (1.0 - self.cfg.u_hold), 0.0, 1.0)

    def exit_offset(self, u: float, base_offset: float = 0.01) -> float:
        """Continuous maker exit offset: widens with urgency.
        base_offset + 0.06*u, clamped to [0.01, 0.10]."""
        return _clamp(base_offset + 0.06 * u, 0.01, 0.10)


# ─────────────────────────────────────────────────────────────────────────────
# Position Monitor Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionMonitorConfig:
    # ── EV exit: market overpaying vs hold EV ──
    fee_buffer: float = 0.005
    min_profit_for_ev_exit: float = 0.02  # 2c minimum locked profit

    # ── Probabilistic stop-loss (p collapsed, not just low bid) ──
    p_cut_loss_early: float = 0.15   # at T > 60s
    p_cut_loss_late: float = 0.20    # at T < 30s
    stop_penalty_early: float = 0.04  # max EV haircut for stop
    stop_penalty_late: float = 0.02

    # ── Gamma danger (profitable in uncertainty band) ──
    gamma_band_mult: float = 2.0
    gamma_band_min: float = 0.20
    gamma_buffer: float = 0.01

    # ── Trailing stop (ONLY inside gamma band) ──
    trail_offset_early: float = 0.08   # wide trail at T > 60s
    trail_offset_late: float = 0.03    # tight trail at T < 30s
    trail_activation_profit: float = 0.03  # minimum profit to activate trailing

    # ── Trailing stop flicker guard ──
    trail_persistence_ms: int = 300    # bid must stay below trail for this long
    trail_min_spread: float = 0.10     # max spread allowed for trailing stop fire

    # ── Endgame (subsumes EndgameManager) ──
    endgame_T: float = 25.0
    endgame_strong_win: float = 0.85
    endgame_strong_loss: float = 0.15

    # ── Variance penalty by regime (base multipliers) ──
    variance_base: float = 0.02
    variance_gamma_k: float = 0.06
    regime_variance_mult: dict = field(default_factory=lambda: {
        "CALM": 0.8,
        "NORMAL": 1.0,
        "TRANSITION": 1.2,
        "VOL_EVENT": 1.5,
        "HIGH_VOL": 1.8,
        "ADVERSARIAL": 2.0,
    })

    # ── Loss-exit guard: don't EV-exit at a loss unless thesis is truly broken ──
    loss_exit_p_cutoff_early: float = 0.42  # T > 60s: p_side must be below this to allow loss EV exit
    loss_exit_p_cutoff_late: float = 0.46   # T < 30s: stricter near expiry

    # ── Pin-band hold: near p=0.50, model is least reliable, convexity is highest ──
    pin_band_low: float = 0.44
    pin_band_high: float = 0.56
    pin_band_min_T: float = 45.0            # only hold pin-band if T > this

    # ── Book sanity gate: don't trust EV from degenerate books ──
    book_sane_max_spread: float = 0.20
    book_sane_min_bid: float = 0.05
    book_sane_max_ask: float = 0.95

    # ── Controls ──
    min_eval_interval_ms: int = 100
    reentry_cooldown_ms: int = 5000
    min_bid: float = 0.02


@dataclass
class ExitDecision:
    action: str        # "SELL" or "HOLD"
    reason: str        # machine-readable reason code
    hold_ev: float     # p_side
    sell_ev: float     # bid - fee
    profit: float      # sell_ev - entry_price
    penalty: float     # variance/gamma penalty applied
    ev_gap: float      # hold_ev - sell_ev
    # SIS fields (populated when SIS fires or holds)
    exit_urgency: float = 0.0
    exit_frac: float = 0.0
    exit_offset: float = 0.0


class PositionMonitor:
    """
    Unified exit engine evaluating every tick whether to hold or sell.
    Includes Signal Invalidation Surface for continuous momentum decay exits.
    """

    def __init__(self, cfg: PositionMonitorConfig = None):
        self.cfg = cfg or PositionMonitorConfig()
        self._last_eval_ms: int = 0

        # Per-side trailing stop state
        self._trail_hwm: dict[str, float] = {"UP": 0.0, "DOWN": 0.0}
        self._trail_activated: dict[str, bool] = {"UP": False, "DOWN": False}
        self._trail_breach_ts: dict[str, int] = {"UP": 0, "DOWN": 0}
        self._reentry_blocked_until: dict[str, int] = {"UP": 0, "DOWN": 0}

        # Signal Invalidation Surface
        self.sis = SignalInvalidationSurface()

    def reset_window(self, now_ms: int = 0):
        """Call on window rollover to clear trailing state."""
        self._trail_hwm = {"UP": 0.0, "DOWN": 0.0}
        self._trail_activated = {"UP": False, "DOWN": False}
        self._trail_breach_ts = {"UP": 0, "DOWN": 0}
        self._reentry_blocked_until = {"UP": 0, "DOWN": 0}
        self._last_eval_ms = 0
        self.sis.clear()

    def is_reentry_blocked(self, side: str, now_ms: int) -> bool:
        """After an exit, block re-entry for cooldown period."""
        return now_ms < self._reentry_blocked_until.get(side, 0)

    def _gamma_band(self, sigma_1m: float, T_sec: float) -> tuple[float, float]:
        """Dynamic uncertainty band centered at 0.5."""
        sigma_T = sigma_1m * math.sqrt(max(0.01, T_sec) / 60.0)
        band = max(self.cfg.gamma_band_min,
                   self.cfg.gamma_band_mult * sigma_T)
        lower = max(0.0, 0.5 - band)
        upper = min(1.0, 0.5 + band)
        return lower, upper

    def _interpolate_by_time(self, T_sec: float,
                              early_val: float, late_val: float,
                              T_early: float = 120.0,
                              T_late: float = 30.0) -> float:
        """Linear interpolation between early/late values based on time remaining."""
        if T_sec >= T_early:
            return early_val
        if T_sec <= T_late:
            return late_val
        frac = (T_sec - T_late) / (T_early - T_late)
        return late_val + frac * (early_val - late_val)

    def _variance_penalty(self, p_side: float, T_sec: float,
                           sigma_1m: float, regime: str) -> float:
        """Variance penalty — EV haircut to avoid binary settlement risk."""
        cfg = self.cfg
        regime_mult = cfg.regime_variance_mult.get(regime, 1.0)
        penalty = cfg.variance_base * regime_mult
        uncertainty = 4.0 * p_side * (1.0 - p_side)
        if T_sec < 45.0:
            time_urgency = 1.0 + (45.0 - T_sec) / 45.0
            penalty += cfg.variance_gamma_k * uncertainty * time_urgency * regime_mult
        if T_sec < 20.0:
            penalty += cfg.variance_gamma_k * 0.5 * regime_mult
        if not (0.12 < p_side < 0.88):
            penalty *= 0.25
        return penalty

    def evaluate(
        self,
        side: str,           # "UP" or "DOWN"
        entry_price: float,  # avg cost basis
        bid: float,          # live bid for this token
        p_cone: float,       # model probability of UP winning
        T_sec: float,        # seconds remaining
        sigma_1m: float,     # 1-min realized vol
        regime: str = "NORMAL",
        spread: float = 0.10,
        now_ms: int = 0,
        # ── SIS signals ──
        edge_entry: float = 0.0,
        edge_now: float = 0.0,
        z_ema_now: float = None,
        flow_centered: float = 0.0,
        entry_ts_ms: int = 0,
        fee_fn=None,
    ) -> ExitDecision:
        """
        Core evaluation: should we hold or sell this position?
        Returns ExitDecision with action, reason, EV metadata, and SIS urgency.
        """
        cfg = self.cfg
        if now_ms <= 0:
            now_ms = int(time.time() * 1000)

        # Throttle check
        if now_ms - self._last_eval_ms < cfg.min_eval_interval_ms:
            return ExitDecision("HOLD", "throttled", 0.0, 0.0, 0.0, 0.0, 0.0)
        self._last_eval_ms = now_ms

        # ── Core EV math ──
        p_side = p_cone if side == "UP" else (1.0 - p_cone)
        hold_ev = p_side
        if bid < cfg.min_bid:
            return ExitDecision("HOLD", "bid_too_low", hold_ev, 0.0, 0.0, 0.0, hold_ev)

        price = min(0.99, max(0.01, bid))
        _f = fee_fn(price) if fee_fn is not None else (price * (1.0 - price) * 0.0625)
        sell_ev = price - _f
        profit = sell_ev - entry_price
        ev_gap = hold_ev - sell_ev
        penalty = self._variance_penalty(p_side, T_sec, sigma_1m, regime)

        # ── Update trailing stop HWM ──
        if sell_ev > self._trail_hwm.get(side, 0.0):
            self._trail_hwm[side] = sell_ev
            self._trail_breach_ts[side] = 0
        if profit >= cfg.trail_activation_profit:
            self._trail_activated[side] = True

        gamma_lo, gamma_hi = self._gamma_band(sigma_1m, T_sec)
        in_gamma = gamma_lo <= p_side <= gamma_hi

        # ── Compute SIS urgency (always, for status reporting) ──
        _sis_u = 0.0
        _sis_diag = {}
        _sis_frac = 0.0
        _sis_offset = 0.0
        if z_ema_now is not None and edge_entry > 0:
            _sis_u, _sis_diag = self.sis.compute(
                side=side,
                edge_entry=edge_entry,
                edge_now=edge_now,
                z_ema_now=z_ema_now,
                flow_centered=flow_centered,
                sigma_1m=sigma_1m,
                T_sec=T_sec,
                regime=regime,
                entry_ts_ms=entry_ts_ms,
                now_ms=now_ms,
            )
            _sis_frac = self.sis.exit_frac(_sis_u)
            _sis_offset = self.sis.exit_offset(_sis_u)
            # Always update z_ema_prev for next tick's slope
            self.sis.update_z_ema_prev(side, z_ema_now)

        # ═══════════════════════════════════════════════════════════
        # BOOK SANITY GATE — don't trust EV from degenerate books
        # ═══════════════════════════════════════════════════════════
        _book_sane = (
            spread < cfg.book_sane_max_spread
            and bid >= cfg.book_sane_min_bid
            and (bid + spread) <= cfg.book_sane_max_ask
        )

        # ── Loss-exit threshold (interpolated by time) ──
        _loss_p_cutoff = self._interpolate_by_time(
            T_sec, cfg.loss_exit_p_cutoff_early, cfg.loss_exit_p_cutoff_late,
            T_early=60.0, T_late=30.0)

        # ── Pin-band detection: p near 0.50 = max uncertainty, max convexity ──
        _in_pin_band = (
            cfg.pin_band_low <= p_side <= cfg.pin_band_high
            and T_sec > cfg.pin_band_min_T
        )

        # ═══════════════════════════════════════════════════════════
        # DECISION CASCADE (checked in priority order)
        # ═══════════════════════════════════════════════════════════

        # ── 1. PROBABILISTIC STOP-LOSS ──
        # (book sanity not required here — p_side is model-driven, not book-driven)
        p_cut = self._interpolate_by_time(
            T_sec, cfg.p_cut_loss_early, cfg.p_cut_loss_late,
            T_early=120.0, T_late=30.0)
        stop_pen = self._interpolate_by_time(
            T_sec, cfg.stop_penalty_early, cfg.stop_penalty_late,
            T_early=120.0, T_late=30.0)

        if p_side < p_cut and sell_ev >= (hold_ev - stop_pen) and _book_sane:
            self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
            return ExitDecision("SELL", "prob_stop_loss",
                                hold_ev, sell_ev, profit, stop_pen, ev_gap,
                                _sis_u, _sis_frac, _sis_offset)

        # ── 2. EV EXIT ── (requires book sanity — don't trust degenerate quotes)
        if profit >= cfg.min_profit_for_ev_exit and _book_sane:
            if sell_ev >= (hold_ev - cfg.fee_buffer):
                self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                return ExitDecision("SELL", "ev_exit_lock_profit",
                                    hold_ev, sell_ev, profit, cfg.fee_buffer, ev_gap,
                                    _sis_u, _sis_frac, _sis_offset)

        # ── 2.5 SIS EXIT: Signal Invalidation Surface ──
        # Guards: book must be sane, not in pin-band at a loss, thesis must be broken for loss exits
        _sis_loss_blocked = False
        if _sis_u >= self.sis.cfg.u_hold and _sis_frac > 0:
            if not _book_sane:
                _sis_loss_blocked = True
                logger.info(f"SIS_BLOCKED({side}): book_insane spread={spread:.2f} bid={bid:.2f}")
            elif profit < 0 and _in_pin_band:
                _sis_loss_blocked = True
                logger.info(f"SIS_BLOCKED({side}): pin_band p={p_side:.3f} profit={profit:.4f}")
            elif profit < 0 and p_side > _loss_p_cutoff:
                _sis_loss_blocked = True
                logger.info(f"SIS_BLOCKED({side}): thesis_not_broken p={p_side:.3f} cutoff={_loss_p_cutoff:.3f}")

            if not _sis_loss_blocked:
                logger.info(
                    f"SIS_EXIT({side}): u={_sis_u:.3f} frac={_sis_frac:.2f} "
                    f"offset={_sis_offset:.3f} profit={profit:.4f} "
                    f"band={_sis_diag.get('band', 'N/A')} "
                    f"S_edge={_sis_diag.get('S_edge', 0):.3f} "
                    f"S_mom={_sis_diag.get('S_mom', 0):.3f} "
                    f"S_struct={_sis_diag.get('S_struct', 0):.3f}"
                )
                self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                return ExitDecision("SELL", "sis_exit",
                                    hold_ev, sell_ev, profit, penalty, ev_gap,
                                    _sis_u, _sis_frac, _sis_offset)

        # ── 3. GAMMA DANGER ──
        if in_gamma and profit > 0 and sell_ev >= (hold_ev - cfg.gamma_buffer):
            self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
            return ExitDecision("SELL", "gamma_danger",
                                hold_ev, sell_ev, profit, cfg.gamma_buffer, ev_gap,
                                _sis_u, _sis_frac, _sis_offset)

        # ── 4. TRAILING STOP (only in gamma band) ──
        if in_gamma and self._trail_activated.get(side, False):
            trail_offset = self._interpolate_by_time(
                T_sec, cfg.trail_offset_early, cfg.trail_offset_late,
                T_early=120.0, T_late=30.0)
            hwm = self._trail_hwm.get(side, 0.0)
            trail_level = hwm - trail_offset

            if sell_ev < trail_level:
                breach_ts = self._trail_breach_ts.get(side, 0)
                if breach_ts == 0:
                    self._trail_breach_ts[side] = now_ms
                elif (now_ms - breach_ts >= cfg.trail_persistence_ms
                      and spread <= cfg.trail_min_spread):
                    self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                    return ExitDecision("SELL", "trailing_stop",
                                        hold_ev, sell_ev, profit, trail_offset, ev_gap,
                                        _sis_u, _sis_frac, _sis_offset)
            else:
                self._trail_breach_ts[side] = 0

        # ── 5. ENDGAME (T < 25s) ──
        if T_sec < cfg.endgame_T:
            if p_side >= cfg.endgame_strong_win:
                return ExitDecision("HOLD", "endgame_strong_win",
                                    hold_ev, sell_ev, profit, penalty, ev_gap,
                                    _sis_u, _sis_frac, _sis_offset)
            if p_side <= cfg.endgame_strong_loss:
                self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                return ExitDecision("SELL", "endgame_strong_loss",
                                    hold_ev, sell_ev, profit, penalty, ev_gap,
                                    _sis_u, _sis_frac, _sis_offset)
            if in_gamma and profit > 0:
                self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                return ExitDecision("SELL", "endgame_gamma_exit",
                                    hold_ev, sell_ev, profit, penalty, ev_gap,
                                    _sis_u, _sis_frac, _sis_offset)
            if ev_gap < cfg.fee_buffer and profit >= 0 and _book_sane:
                self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                return ExitDecision("SELL", "endgame_ev_exit",
                                    hold_ev, sell_ev, profit, penalty, ev_gap,
                                    _sis_u, _sis_frac, _sis_offset)

        # ── DEFAULT: HOLD ──
        return ExitDecision("HOLD", "hold_ev_superior",
                            hold_ev, sell_ev, profit, penalty, ev_gap,
                            _sis_u, _sis_frac, _sis_offset)

    def status_dict(self, side: str) -> dict:
        """Return current monitoring state for debug logging."""
        return {
            "pm_trail_hwm": round(self._trail_hwm.get(side, 0.0), 4),
            "pm_trail_active": self._trail_activated.get(side, False),
            "pm_reentry_blocked": self._reentry_blocked_until.get(side, 0),
            "sis_u": round(self.sis._u.get(side, 0.0), 4),
        }
