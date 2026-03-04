from dataclasses import dataclass
from typing import Tuple


@dataclass
class GammaConfig:
    # Pin band protection
    pin_low: float = 0.45
    pin_high: float = 0.55
    pin_min_T: float = 60.0

    # Gamma ride window
    ride_min_T: float = 20.0
    ride_max_T: float = 75.0
    ride_min_abs_z: float = 0.80

    # Convexity floor: don't sell cheap gamma
    # This is a time-based floor on bid before we allow profit-taking in ride mode.
    bid_floor_T75: float = 0.70
    bid_floor_T45: float = 0.78
    bid_floor_T30: float = 0.83
    bid_floor_T20: float = 0.86

    # Monetize rules
    monetize_T: float = 30.0
    monetize_p: float = 0.70
    monetize_bid: float = 0.85

    # Regimes where we do NOT ride gamma
    violent_regimes: Tuple[str, ...] = ("HIGH_VOL", "VOL_EVENT", "ADVERSARIAL")


def _interp_floor(T: float, cfg: GammaConfig) -> float:
    # piecewise linear between anchors
    # T>=75 => 0.70, 45 => 0.78, 30 => 0.83, 20 => 0.86
    if T >= 75: return cfg.bid_floor_T75
    if T >= 45:
        # 75->45
        return cfg.bid_floor_T45 + (cfg.bid_floor_T75 - cfg.bid_floor_T45) * ((T - 45) / 30.0)
    if T >= 30:
        # 45->30
        return cfg.bid_floor_T30 + (cfg.bid_floor_T45 - cfg.bid_floor_T30) * ((T - 30) / 15.0)
    if T >= 20:
        # 30->20
        return cfg.bid_floor_T20 + (cfg.bid_floor_T30 - cfg.bid_floor_T20) * ((T - 20) / 10.0)
    return cfg.bid_floor_T20


class GammaEngine:
    def __init__(self, cfg: GammaConfig = None):
        self.cfg = cfg or GammaConfig()

    def should_block_exit(
        self,
        *,
        T: float,
        p_side: float,
        z: float,
        bid: float,
        regime: str,
        thesis_broken: bool,
    ) -> Tuple[bool, str]:
        """
        Returns (block_exit, reason).
        If block_exit=True -> do not allow SIS trims / EV exits.
        FORCE_EXIT and thesis_broken always override (block_exit=False).
        """
        cfg = self.cfg
        reg = str(regime).upper()

        if thesis_broken:
            return False, "THESIS_BROKEN"

        # Pin band hold (highest protection)
        if T >= cfg.pin_min_T and cfg.pin_low <= p_side <= cfg.pin_high:
            return True, "PIN_BAND_HOLD"

        # Don't ride in violent regimes
        if reg in cfg.violent_regimes:
            return False, "VIOLENT_REGIME"

        # Monetize window: allow exits
        if T <= cfg.monetize_T and (p_side >= cfg.monetize_p or bid >= cfg.monetize_bid):
            return False, "MONETIZE"

        # Gamma ride: block selling below convexity floor
        if cfg.ride_min_T <= T <= cfg.ride_max_T and abs(z) >= cfg.ride_min_abs_z:
            floor_bid = _interp_floor(T, cfg)
            if bid < floor_bid:
                return True, f"GAMMA_RIDE_bid<{floor_bid:.2f}"
            return False, "GAMMA_RIDE_floor_met"

        return False, "NO_BLOCK"
