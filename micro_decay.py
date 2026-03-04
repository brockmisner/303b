"""
Microstructure-aware edge decay model.

Predicts how much edge will decay between signal-time and execution-time
based on latency, book depth, spread, adverse selection probability,
and volatility.  Returns a single "required buffer" that the sniper
and execution loop can use to avoid firing non-survivable trades.
"""

import math
from dataclasses import dataclass


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class MicroDecayParams:
    # --- latency / time decay ---
    latency_k: float = 0.0016       # edge decays with lag (per ms)
    retry_kappa: float = 0.60       # additional decay per second of waiting/stepping

    # --- slippage model ---
    slip_spread_w: float = 0.55     # fraction of spread you typically "pay"
    slip_thin_w: float = 0.02       # extra slippage when top is thin

    # --- adverse selection ---
    adv_a_sigma: float = 30.0
    adv_b_flow: float = 0.40
    adv_c_vel: float = 0.03
    adv_d_fasttail: float = 0.20
    adverse_impact: float = 0.015   # price units hit when adverse happens

    # --- floors/caps ---
    min_buffer: float = 0.003       # 0.3c (reduced: SIS trims earlier)
    max_buffer: float = 0.032       # 3.2c (reduced: partial exits reduce risk)


_DEFAULT_PARAMS = MicroDecayParams()


def micro_edge_buffer(
    *,
    spread: float,
    bid_sz: float,
    ask_sz: float,
    sigma: float,
    lag_p50_ms: float,
    lag_p10_ms: float,
    trade_vel: float,
    flow_bias: float,           # [-1, +1]
    retries_expected: float = 0.0,  # seconds of additional delay if stepping
    params: MicroDecayParams = _DEFAULT_PARAMS,
) -> dict:
    """
    Compute the dynamic edge buffer required at signal-time to survive
    through to execution.

    Returns:
        {
            "buffer": float,      # required extra edge at signal time
            "slip": float,        # estimated slippage cost
            "p_adv": float,       # adverse selection probability
            "adv_cost": float,    # p_adv * adverse_impact
            "time_decay": float,  # edge shrink factor (0..1)
            "why": str,           # diagnostic string
        }
    """
    sp = max(0.0, float(spread))
    bid_sz = max(0.0, float(bid_sz))
    ask_sz = max(0.0, float(ask_sz))
    top_depth = bid_sz + ask_sz

    # --- slippage: pay some fraction of spread + thin top penalty ---
    thin_pen = params.slip_thin_w * clamp(1.0 - (top_depth / 80.0), 0.0, 1.0)
    slip = params.slip_spread_w * sp + thin_pen

    # --- adverse selection probability (bounded) ---
    fast_tail = 1.0 if lag_p10_ms < 220.0 else 0.0
    p_adv = (
        sigma * params.adv_a_sigma
        + abs(flow_bias) * params.adv_b_flow
        + trade_vel * params.adv_c_vel
        + fast_tail * params.adv_d_fasttail
    )
    p_adv = clamp(p_adv, 0.0, 0.65)
    adv_cost = p_adv * params.adverse_impact

    # --- time/latency decay: shrink "survivable edge" as lag rises ---
    lag_term = math.exp(-params.latency_k * max(0.0, lag_p50_ms))
    retry_term = math.exp(-params.retry_kappa * max(0.0, retries_expected))
    time_decay = clamp(lag_term * retry_term, 0.25, 1.0)  # never assume >75% loss

    # Convert decay into an additive buffer:
    # If time_decay=0.70 -> buffer adds ~30% of (slip+adv_cost)
    # plus a small fixed floor.
    buffer = params.min_buffer + (slip + adv_cost) * (1.0 + (1.0 - time_decay))

    # SIS-regime scaling: continuous exits reduce risk, so buffer can tighten
    buffer *= 0.80

    buffer = clamp(buffer, params.min_buffer, params.max_buffer)

    why = (
        f"sp={sp:.3f} thin={thin_pen:.3f} p_adv={p_adv:.2f} "
        f"lag50={lag_p50_ms:.0f}ms td={time_decay:.2f}"
    )
    return {
        "buffer": float(buffer),
        "slip": float(slip),
        "p_adv": float(p_adv),
        "adv_cost": float(adv_cost),
        "time_decay": float(time_decay),
        "why": why,
    }
