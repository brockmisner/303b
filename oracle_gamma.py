import math
from dataclasses import dataclass

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

@dataclass
class OracleGammaConfig:
    base_dev_frac: float = 0.00035
    dt_ms_min: float = 100.0
    dt_ms_max: float = 1300.0

    gamma_maker_aggr: float = 0.10
    gamma_cross: float = 0.25

    max_snap_sigmas: float = 2.0

class OracleGammaEngine:
    def __init__(self, cfg: OracleGammaConfig = None):
        self.cfg = cfg or OracleGammaConfig()

    def expected_dt_s(self, p_update: float) -> float:
        # p_update high => sooner
        dt_ms = self.cfg.dt_ms_max * (1.0 - p_update) + self.cfg.dt_ms_min
        dt_ms = clamp(dt_ms, self.cfg.dt_ms_min, self.cfg.dt_ms_max)
        return dt_ms / 1000.0

    def compute(
        self,
        *,
        p_update: float,
        p_bin: float,
        p_rtds: float,
        sigma_1m: float,
        p_cone_now: float,
        p_cone_pred: float,
    ) -> dict:
        dt_s = self.expected_dt_s(p_update)
        dp = abs(float(p_cone_pred) - float(p_cone_now))
        og = (float(p_update) * dp) / max(1e-3, dt_s)

        return {
            "oracle_gamma": round(og, 4),
            "oracle_dt_s": round(dt_s, 4),
            "dp_cone": round(dp, 4),
        }

    def posture(self, oracle_gamma: float) -> str:
        if oracle_gamma >= self.cfg.gamma_cross:
            return "CROSS"
        if oracle_gamma >= self.cfg.gamma_maker_aggr:
            return "MAKER_AGGR"
        return "MAKER"
