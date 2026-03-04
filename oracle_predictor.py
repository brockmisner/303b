import math
from dataclasses import dataclass
from typing import Dict

@dataclass
class PredictorConfig:
    # Tuned specifically for a ~740ms median oracle lag
    beta_0: float = -4.2   # Intercept (Keep base probability near 0)
    beta_1: float = 4.0    # Displacement weight (Heavy weight on price stretch)
    beta_2: float = 0.8    # Time weight (Aggressive ramp as tau > 0.7s)
    beta_3: float = 2.0    # Flow acceleration (Confirmation weight)

    # Base threshold: ~3.5 bps is the sweet spot to filter out the noise floor
    # observed in the sigma column (mean sigma was ~6.8 bps)
    base_dev_frac: float = 0.00035

    # Regime multipliers
    regime_k: dict = None

    # Safety clamps
    max_D: float = 10.0
    max_tau_s: float = 10.0
    disagree_penalty: float = 1.0
    disagree_thresh: float = -0.05  # flow_align below this penalizes

class RTDSPredictor:
    def __init__(self, cfg: PredictorConfig = None):
        self.cfg = cfg or PredictorConfig()
        if self.cfg.regime_k is None:
            self.cfg.regime_k = {
                "CALM": 0.7,
                "NORMAL": 1.0,
                "TRANSITION": 1.2,
                "HIGH_VOL": 1.5,
                "VOL_EVENT": 2.0,
                "ADVERSARIAL": 2.5
            }

        self.p_update: float = 0.0
        self.predicted_rtds: float = 0.0
        self.predicted_direction: int = 0

    def update(
        self,
        *,
        p_bin: float,
        p_rtds: float,
        rtds_age_ms: int,
        sigma_1m: float,
        flow_accel: float,
        regime: str,
    ) -> Dict[str, float]:

        if p_rtds <= 0.0 or p_bin <= 0.0:
            return {"p_update": 0.0, "pred_rtds_px": p_rtds, "direction": 0}

        delta_p = float(p_bin) - float(p_rtds)
        abs_delta = abs(delta_p)
        direction = 1 if delta_p > 0 else (-1 if delta_p < 0 else 0)

        k_mult = self.cfg.regime_k.get(str(regime).upper(), 1.0)

        # Effective threshold in absolute price terms (fractional dev * price)
        dev_frac = max(self.cfg.base_dev_frac, float(sigma_1m))
        eff_threshold = float(p_rtds) * dev_frac * float(k_mult)
        eff_threshold = max(1e-6, eff_threshold)

        # Displacement ratio
        D = abs_delta / eff_threshold
        D = min(D, self.cfg.max_D)

        # Time since last RTDS update (seconds), clamped
        tau_s = min(self.cfg.max_tau_s, max(0.0, float(rtds_age_ms) / 1000.0))

        # Flow alignment
        flow_align = float(flow_accel) * float(direction)

        # Logit
        x = (
            self.cfg.beta_0
            + (self.cfg.beta_1 * D)
            + (self.cfg.beta_2 * tau_s)
            + (self.cfg.beta_3 * flow_align)
        )

        # Penalize disagreement (strong flow against delta direction)
        if flow_align < self.cfg.disagree_thresh:
            x -= self.cfg.disagree_penalty

        # Clamp to avoid exp overflow
        x = max(-20.0, min(20.0, x))
        p_update = 1.0 / (1.0 + math.exp(-x))

        self.p_update = float(p_update)
        self.predicted_rtds = float(p_bin)  # proxy: oracle snaps toward Binance
        self.predicted_direction = int(direction)

        return {
            "p_update": round(self.p_update, 4),
            "pred_rtds_px": round(self.predicted_rtds, 2),
            "direction": self.predicted_direction,
            "delta_p": round(delta_p, 2),
            "D_ratio": round(D, 3),
            "tau_s": round(tau_s, 3),
            "flow_accel": round(float(flow_accel), 4),
            "flow_align": round(float(flow_align), 4),
            "k_mult": round(float(k_mult), 3),
        }