"""
Fully Bayesian Execution Posterior Surface — Online leakage model.

Model:
  leak = x^T beta + eps,   eps ~ Normal(0, sigma^2)

Prior (conjugate):
  beta | sigma^2 ~ Normal(beta0, sigma^2 * V0)
  sigma^2 ~ InvGamma(a0, b0)

Posterior predictive for new leakage at features x is Student-t:
  leak | x, data ~ StudentT(df=2a_n, loc=mu(x), scale=s(x))

Outputs per tick:
  - q_alpha(x):  predictive quantile leakage floor (e.g. 90th percentile)
  - p_win:       P(edge_real > 0 | edge_sig, x) = P(leak < edge_sig | x)
  - e_net:       E[edge_real | edge_sig, x] = edge_sig - E[leak | x]
  - decision:    allow if p_win >= π AND edge_sig >= q_alpha(x)

Keys:
  We keep a light hierarchy to avoid sparsity:
    global surface: learns across everything
    keyed surfaces: (regime_bin, route) learn context-specific slopes

If a keyed surface is cold, we fall back to global.
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ── Regime binning ───────────────────────────────────────────────────

def _regime_bin(regime: str) -> str:
    if regime in ("CALM",):
        return "calm"
    elif regime in ("HIGH_VOL", "VOL_EVENT", "ADVERSARIAL"):
        return "hot"
    else:
        return "normal"


SurfaceKey = Tuple[str, str, str]  # (regime_bin, route, side)


# ── Student-t math (exact: CDF + quantile via Lentz continued fraction) ──

def _student_t_cdf_exact(x: float, nu: float) -> float:
    """CDF of standard Student-t via regularized incomplete beta (exact, slow)."""
    if nu <= 0:
        return 0.5
    t2 = x * x
    u = nu / (nu + t2)
    a, b = nu / 2.0, 0.5
    val = _reg_inc_beta(u, a, b)
    if x >= 0:
        return 1.0 - 0.5 * val
    else:
        return 0.5 * val


def _reg_inc_beta(x: float, a: float, b: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Regularized incomplete beta I_x(a,b) via Lentz continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Symmetry transform for stability
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _reg_inc_beta(1.0 - x, b, a, max_iter, eps)

    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x)) / a

    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m

        # even step
        num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= c * d

        # odd step
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < eps:
            break

    return front * f


def _student_t_quantile_exact(p: float, nu: float, mu: float = 0.0, scale: float = 1.0, max_iter: int = 60) -> float:
    """Quantile of location-scale Student-t via bisection on CDF (exact, slow)."""
    if nu <= 0 or scale <= 0:
        return mu
    lo, hi = -15.0, 15.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if _student_t_cdf_exact(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2.0
    return mu + scale * z


# ── Precomputed Student-t quantile LUT (replaces hot-path Lentz loops) ──

_LUT_NU: np.ndarray = np.empty(0)
_LUT_P: np.ndarray = np.empty(0)
_LUT_Q: np.ndarray = np.empty(0)


def _build_student_t_lut() -> None:
    """Build lookup table at module load. One-time ~0.3s cost."""
    global _LUT_NU, _LUT_P, _LUT_Q
    nu_grid = np.unique(np.concatenate([
        np.arange(1, 10, 0.5),
        np.arange(10, 50, 2),
        np.arange(50, 201, 10),
    ]))
    p_grid = np.unique(np.concatenate([
        np.arange(0.01, 0.10, 0.01),
        np.arange(0.10, 0.91, 0.005),
        np.arange(0.91, 0.995, 0.005),
    ]))
    q_table = np.empty((len(nu_grid), len(p_grid)), dtype=np.float64)
    for i, nu in enumerate(nu_grid):
        for j, p in enumerate(p_grid):
            q_table[i, j] = _student_t_quantile_exact(float(p), float(nu))
    _LUT_NU = nu_grid
    _LUT_P = p_grid
    _LUT_Q = q_table


def _student_t_quantile(p: float, nu: float, mu: float = 0.0, scale: float = 1.0) -> float:
    """Fast quantile via LUT bilinear interpolation. O(1)."""
    if len(_LUT_Q) == 0 or nu <= 0 or scale <= 0:
        return mu
    p = max(0.01, min(0.99, p))
    nu = max(1.0, min(200.0, nu))
    i = int(np.searchsorted(_LUT_NU, nu, side='right')) - 1
    i = max(0, min(i, len(_LUT_NU) - 2))
    j = int(np.searchsorted(_LUT_P, p, side='right')) - 1
    j = max(0, min(j, len(_LUT_P) - 2))
    nu_lo, nu_hi = _LUT_NU[i], _LUT_NU[i + 1]
    p_lo, p_hi = _LUT_P[j], _LUT_P[j + 1]
    t_nu = (nu - nu_lo) / max(1e-15, nu_hi - nu_lo)
    t_p = (p - p_lo) / max(1e-15, p_hi - p_lo)
    q00, q10 = _LUT_Q[i, j], _LUT_Q[i + 1, j]
    q01, q11 = _LUT_Q[i, j + 1], _LUT_Q[i + 1, j + 1]
    z = (1 - t_nu) * ((1 - t_p) * q00 + t_p * q01) + t_nu * ((1 - t_p) * q10 + t_p * q11)
    return mu + scale * float(z)


def _student_t_cdf(x: float, nu: float) -> float:
    """Fast CDF via inverse LUT interpolation. O(1)."""
    if len(_LUT_Q) == 0 or nu <= 0:
        return 0.5
    nu = max(1.0, min(200.0, nu))
    i = int(np.searchsorted(_LUT_NU, nu, side='right')) - 1
    i = max(0, min(i, len(_LUT_NU) - 2))
    nu_lo, nu_hi = _LUT_NU[i], _LUT_NU[i + 1]
    t_nu = (nu - nu_lo) / max(1e-15, nu_hi - nu_lo)
    q_row = (1.0 - t_nu) * _LUT_Q[i, :] + t_nu * _LUT_Q[i + 1, :]
    j = int(np.searchsorted(q_row, x))
    if j <= 0:
        return float(_LUT_P[0])
    if j >= len(_LUT_P):
        return float(_LUT_P[-1])
    q_lo, q_hi = q_row[j - 1], q_row[j]
    p_lo, p_hi = _LUT_P[j - 1], _LUT_P[j]
    t = (x - q_lo) / max(1e-15, q_hi - q_lo)
    return float(p_lo + t * (p_hi - p_lo))


_build_student_t_lut()


# ── Bayesian linear regression bucket (NIΓ) ──────────────────────────

@dataclass
class RegressionPrior:
    """
    Conjugate prior for Bayesian linear regression.
      beta | sigma^2 ~ N(beta0, sigma^2 V0)
      sigma^2 ~ InvGamma(a0, b0)

    V0 is represented as diagonal for simplicity/stability online.
    """
    beta0: np.ndarray          # shape (d,)
    v0_diag: np.ndarray        # diagonal of V0, shape (d,)
    a0: float = 6.0
    b0: float = 0.0005


class BayesianLinearBucket:
    """
    Online sufficient-stat update for Bayesian linear regression with NIΓ prior.
    Maintains:
      XtX, Xty, yTy, n

    Posterior:
      Vn = (V0^{-1} + XtX)^{-1}
      betan = Vn (V0^{-1} beta0 + Xty)
      an = a0 + n/2
      bn = b0 + 0.5*(yTy + beta0^T V0^{-1} beta0 - betan^T (V0^{-1}+XtX) betan)

    Predictive:
      leak | x ~ StudentT(df=2an, loc=x^T betan, scale=sqrt((bn/an) * (1 + x^T Vn x)))
    """

    def __init__(self, prior: RegressionPrior):
        self.prior = prior
        self.d = int(prior.beta0.shape[0])

        self.n = 0
        self.XtX = np.zeros((self.d, self.d), dtype=float)
        self.Xty = np.zeros((self.d,), dtype=float)
        self.yTy = 0.0

        self._obs = deque(maxlen=120)
        self._last_ts_ms: Optional[int] = int(time.time() * 1000)  # init to now so first ts_ms call can compute decay

        # cached posterior
        self._dirty = True
        self._last_ridge = None
        self._Vn = None
        self._betan = None
        self._an = None
        self._bn = None

    def update(self, x: np.ndarray, y: float, weight: float = 1.0,
               ts_ms: Optional[int] = None, half_life_s: float = 0.0,
               min_factor: float = 0.25) -> None:
        """
        One observation update.
        weight is supported as a simple multiplicative weight on sufficient stats.
        ts_ms / half_life_s enable exponential time-decay on the sufficient stats
        before incorporating the new observation.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.d:
            return

        w = float(weight)
        if w <= 0:
            return

        # Apply time decay before adding the new observation
        if ts_ms is not None and half_life_s > 0:
            self.apply_time_decay(ts_ms, half_life_s, min_factor)

        self._obs.append(float(y))
        self.n += 1

        # sufficient stats
        self.XtX += w * np.outer(x, x)
        self.Xty += w * x * y
        self.yTy += w * (y * y)

        self._dirty = True

    def apply_time_decay(self, now_ts_ms: int, half_life_s: float,
                         min_factor: float = 0.25) -> None:
        """
        Exponentially decay sufficient statistics based on elapsed time.
        decay_factor = max(min_factor, 2^(-dt_s / half_life_s))
        """
        if self._last_ts_ms is None or half_life_s <= 0:
            self._last_ts_ms = now_ts_ms
            return

        dt_s = max(0.0, (now_ts_ms - self._last_ts_ms) / 1000.0)
        if dt_s <= 0:
            return

        decay = max(float(min_factor), 2.0 ** (-dt_s / float(half_life_s)))

        self.XtX *= decay
        self.Xty *= decay
        self.yTy *= decay
        self.n = round(self.n * decay)  # soft effective-n

        self._last_ts_ms = now_ts_ms
        self._dirty = True

    def _ensure_posterior(self, ridge_lambda: float = 0.0) -> None:
        if not self._dirty and self._last_ridge == ridge_lambda:
            return

        beta0 = self.prior.beta0
        v0_diag = self.prior.v0_diag
        a0 = self.prior.a0
        b0 = self.prior.b0

        # V0^{-1} for diagonal V0
        V0_inv = np.diag(1.0 / np.maximum(v0_diag, 1e-12))

        Lambda = V0_inv + self.XtX  # precision
        if ridge_lambda > 0:
            Lambda = Lambda + ridge_lambda * np.eye(self.d)
        # solve instead of invert where possible
        try:
            Vn = np.linalg.inv(Lambda)
        except np.linalg.LinAlgError:
            # small ridge for numerical stability
            Vn = np.linalg.inv(Lambda + 1e-8 * np.eye(self.d))

        rhs = V0_inv @ beta0 + self.Xty
        betan = Vn @ rhs

        an = a0 + 0.5 * self.n

        quad0 = float(beta0.T @ (V0_inv @ beta0))
        quadn = float(betan.T @ (Lambda @ betan))
        bn = b0 + 0.5 * (self.yTy + quad0 - quadn)

        # guardrails
        bn = max(bn, 1e-12)
        an = max(an, 1e-6)

        self._Vn = Vn
        self._betan = betan
        self._an = an
        self._bn = bn
        self._last_ridge = ridge_lambda
        self._dirty = False

    def df(self, ridge_lambda: float = 0.0) -> float:
        self._ensure_posterior(ridge_lambda)
        return 2.0 * float(self._an)

    def predictive_params(self, x: np.ndarray, ridge_lambda: float = 0.0) -> Tuple[float, float, float]:
        """
        Returns (mu, scale, df) for predictive Student-t at x.
        """
        self._ensure_posterior(ridge_lambda)
        x = np.asarray(x, dtype=float).reshape(-1)

        mu = float(x.T @ self._betan)

        # scale^2 = (bn/an) * (1 + x^T Vn x)
        xtVx = float(x.T @ (self._Vn @ x))
        s2 = (float(self._bn) / float(self._an)) * (1.0 + max(xtVx, 0.0))
        scale = math.sqrt(max(s2, 1e-12))

        return mu, scale, self.df(ridge_lambda)

    def quantile_floor(self, x: np.ndarray, alpha: float = 0.90, ridge_lambda: float = 0.0) -> float:
        if self.n < 10:
            return 0.0
        mu, scale, df = self.predictive_params(x, ridge_lambda)
        q = _student_t_quantile(alpha, df, mu, scale)
        return q

    def p_win(self, edge_sig: float, x: np.ndarray, ridge_lambda: float = 0.0) -> float:
        """
        P(edge_real > 0) = P(leak < edge_sig) under predictive.
        """
        if self.n < 10:
            return 0.5
        mu, scale, df = self.predictive_params(x, ridge_lambda)
        z = (edge_sig - mu) / max(scale, 1e-12)
        return _student_t_cdf(z, df)

    def expected_net_edge(self, edge_sig: float, x: np.ndarray, ridge_lambda: float = 0.0) -> float:
        """
        E[edge_real] = edge_sig - E[leak] where E[leak]=mu(x).
        """
        mu, _, _ = self.predictive_params(x, ridge_lambda)
        return edge_sig - mu

    def diagnostics(self) -> Dict:
        self._ensure_posterior()
        # quick dispersion proxy from residual scale at intercept-only x
        return {
            "n": int(self.n),
            "df": round(self.df(), 2),
            "a": round(float(self._an), 5),
            "b": round(float(self._bn), 6),
            "beta": [round(float(v), 6) for v in self._betan.tolist()],
        }


# ── Main model ──────────────────────────────────────────────────────

@dataclass
class LeakageModelConfig:
    # Feature surface settings
    # x = [1, spread, p_adv, flow, lag_s, sigma_k]
    # We normalize lag and sigma to keep coefficients in sane ranges.
    feature_dim: int = 8

    # Priors (in leakage units)
    prior_mu: float = 0.035          # intercept prior (baseline leakage)
    prior_sigma_beta: float = 0.08   # prior std for non-intercept betas
    prior_sigma0: float = 0.012      # prior noise std (leakage)

    prior_a0: float = 6.0
    prior_b0: float = 0.0005

    # Decision controls
    q_alpha: float = 0.90
    p_win_threshold: float = 0.55

    # Operating limits / warmup
    edge_floor: float = 0.025       # conservative fallback before warmup (was 0.040)
    warmup_fills: int = 30

    # Leakage clipping
    max_leak_clip: float = 0.25
    min_leak_clip: float = -0.05

    # Surface clamp (how big a min-edge you'll ever demand)
    q_floor_clamp_lo: float = 0.02
    q_floor_clamp_hi: float = 0.12

    # Cold bucket threshold
    min_bucket_n: int = 15

    # Ridge regularization on XtX precision matrix
    ridge_lambda: float = 1e-4

    # Time-decay weighting (exponential discounting of sufficient stats)
    decay_half_life_s: float = 900.0    # 15 min half-life
    decay_min_factor: float = 0.25      # floor for exponential decay on large gaps

    # Dynamic prior shift on regime flip
    enable_regime_prior_shift: bool = True
    prior_shift_var_mult: float = 4.0       # inflate prior variance after flip
    prior_shift_copy_global_beta: bool = True  # seed new prior mean from global posterior


class LeakageModel:
    """
    Fully Bayesian posterior surface.

    - global_bucket learns leakage surface across all fills
    - surface buckets keyed by (regime_bin, route) learn context-specific slopes
    - cold buckets fall back to global_bucket
    """

    def __init__(self, cfg: Optional[LeakageModelConfig] = None):
        self.cfg = cfg or LeakageModelConfig()

        d = self.cfg.feature_dim
        beta0 = np.zeros((d,), dtype=float)
        beta0[0] = float(self.cfg.prior_mu)

        # diagonal V0 in "beta-space"
        # intercept tighter, others looser
        v0_diag = np.ones((d,), dtype=float) * (self.cfg.prior_sigma_beta ** 2)
        v0_diag[0] = (0.04 ** 2)

        # noise prior: pick b0 consistent with sigma0
        # E[sigma^2] = b0/(a0-1)  => b0 = (a0-1) * sigma0^2
        a0 = float(self.cfg.prior_a0)
        b0 = float(self.cfg.prior_b0)
        if b0 <= 0:
            b0 = max((a0 - 1.0) * (self.cfg.prior_sigma0 ** 2), 1e-12)

        prior = RegressionPrior(beta0=beta0, v0_diag=v0_diag, a0=a0, b0=b0)

        self.global_bucket = BayesianLinearBucket(prior)
        self.buckets: Dict[SurfaceKey, BayesianLinearBucket] = {}

        self.n_fills = 0
        self._warmed_up = False
        self._last_regime_by_context: Dict[Tuple[str, str], str] = {}

    # ── Feature mapping ─────────────────────────────────────────────

    def _x(
        self,
        *,
        spread: float = 0.0,
        p_adv: float = 0.0,
        flow: float = 0.0,
        lag50_ms: float = 0.0,
        sigma: float = 0.0,
    ) -> np.ndarray:
        """
        Feature vector:
          1,
          spread
          p_adv
          spread * p_adv
          abs(flow)
          sign(flow)
          lag_s
          sigma_k
        """
        lag_s = float(lag50_ms) / 1000.0
        sigma_k = float(sigma) * 1000.0

        spread = float(spread)
        p_adv = float(p_adv)
        flow = float(flow)

        flow_abs = abs(flow)
        flow_sign = 1.0 if flow > 0 else (-1.0 if flow < 0 else 0.0)

        return np.array([
            1.0,
            spread,
            p_adv,
            spread * p_adv,
            flow_abs,
            flow_sign,
            lag_s,
            sigma_k,
        ], dtype=float)

    def _get_bucket(self, key: SurfaceKey) -> BayesianLinearBucket:
        if key not in self.buckets:
            # hierarchical init: copy the same prior as global uses
            self.buckets[key] = BayesianLinearBucket(self.global_bucket.prior)
        return self.buckets[key]

    def _effective_bucket(self, key: SurfaceKey) -> BayesianLinearBucket:
        b = self.buckets.get(key)
        if b and b.n >= self.cfg.min_bucket_n:
            return b
        return self.global_bucket

    def _new_bucket_with_shifted_prior(self) -> BayesianLinearBucket:
        """
        Create a fresh bucket whose prior is seeded from the global posterior.
        Used after regime flips so the new bucket starts from learned parameters
        but with inflated variance for fast adaptation.
        """
        self.global_bucket._ensure_posterior(self.cfg.ridge_lambda)

        if (self.cfg.prior_shift_copy_global_beta
                and self.global_bucket._betan is not None):
            beta0 = self.global_bucket._betan.copy()
        else:
            beta0 = self.global_bucket.prior.beta0.copy()

        v0_diag = self.global_bucket.prior.v0_diag.copy() * float(self.cfg.prior_shift_var_mult)

        shifted = RegressionPrior(
            beta0=beta0,
            v0_diag=v0_diag,
            a0=self.global_bucket.prior.a0,
            b0=self.global_bucket.prior.b0,
        )
        return BayesianLinearBucket(shifted)

    # ── Public API ──────────────────────────────────────────────────

    def reset(self) -> None:
        self.buckets.clear()
        self.global_bucket = BayesianLinearBucket(self.global_bucket.prior)
        self.n_fills = 0
        self._warmed_up = False
        self._last_regime_by_context.clear()

    def record(
        self,
        edge_sig: float,
        edge_real: float,
        *,
        lag50_ms: float = 0.0,
        sigma: float = 0.0,
        spread: float = 0.0,
        p_adv: float = 0.0,
        flow: float = 0.0,
        regime: str = "NORMAL",
        route: str = "FAK",
        side: str = "UP",
        weight: float = 1.0,
        ts_ms: Optional[int] = None,
    ) -> Dict:
        """
        Record one fill: updates global + keyed surface posteriors.
        Observed leakage:
          leak = edge_sig - edge_real
        """
        leak_raw = float(edge_sig) - float(edge_real)
        leak = clamp(leak_raw, self.cfg.min_leak_clip, self.cfg.max_leak_clip)

        x = self._x(spread=spread, p_adv=p_adv, flow=flow, lag50_ms=lag50_ms, sigma=sigma)

        _decay_hl = self.cfg.decay_half_life_s
        _decay_min = self.cfg.decay_min_factor

        # global update
        self.global_bucket.update(x, leak, weight=weight,
                                  ts_ms=ts_ms, half_life_s=_decay_hl, min_factor=_decay_min)

        # keyed surface update
        key = (_regime_bin(regime), str(route), str(side))

        # Regime flip detection: reseed keyed bucket with shifted prior
        if self.cfg.enable_regime_prior_shift:
            ctx = (str(route), str(side))
            prev = self._last_regime_by_context.get(ctx)
            if prev is not None and prev != key[0]:
                # Regime flipped — reseed this specific keyed bucket
                self.buckets[key] = self._new_bucket_with_shifted_prior()
            self._last_regime_by_context[ctx] = key[0]

        self._get_bucket(key).update(x, leak, weight=weight,
                                     ts_ms=ts_ms, half_life_s=_decay_hl, min_factor=_decay_min)

        self.n_fills += 1
        if (not self._warmed_up) and self.n_fills >= self.cfg.warmup_fills:
            self._warmed_up = True

        return {
            "leak": round(leak, 6),
            "key": key,
            "global_n": int(self.global_bucket.n),
            "bucket_n": int(self.buckets[key].n),
            "warmed": bool(self._warmed_up),
        }

    def predict(
        self,
        *,
        lag50_ms: float = 0.0,
        sigma: float = 0.0,
        spread: float = 0.0,
        p_adv: float = 0.0,
        flow: float = 0.0,
        regime: str = "NORMAL",
        route: str = "FAK",
        side: str = "UP",
    ) -> float:
        """
        Bayesian quantile floor surface q_alpha(x).
        """
        if not self._warmed_up:
            return self.cfg.edge_floor

        x = self._x(spread=spread, p_adv=p_adv, flow=flow, lag50_ms=lag50_ms, sigma=sigma)
        key = (_regime_bin(regime), str(route), str(side))
        b = self._effective_bucket(key)

        q = b.quantile_floor(x, self.cfg.q_alpha, self.cfg.ridge_lambda)

        if q <= 0:
            return self.cfg.edge_floor

        return clamp(q, self.cfg.q_floor_clamp_lo, self.cfg.q_floor_clamp_hi)

    def min_edge_surface(self, **kwargs) -> float:
        return self.predict(**kwargs)

    def p_win(
        self,
        edge_sig: float,
        *,
        lag50_ms: float = 0.0,
        sigma: float = 0.0,
        spread: float = 0.0,
        p_adv: float = 0.0,
        flow: float = 0.0,
        regime: str = "NORMAL",
        route: str = "FAK",
        side: str = "UP",
    ) -> float:
        """
        P(edge_real > 0 | edge_sig, x) = P(leak < edge_sig | x).
        """
        if not self._warmed_up:
            return 0.5

        x = self._x(spread=spread, p_adv=p_adv, flow=flow, lag50_ms=lag50_ms, sigma=sigma)
        key = (_regime_bin(regime), str(route), str(side))
        b = self._effective_bucket(key)
        return float(b.p_win(float(edge_sig), x, self.cfg.ridge_lambda))

    def should_allow(
        self,
        edge_sig: float,
        *,
        lag50_ms: float = 0.0,
        sigma: float = 0.0,
        spread: float = 0.0,
        p_adv: float = 0.0,
        flow: float = 0.0,
        regime: str = "NORMAL",
        route: str = "FAK",
        side: str = "UP",
    ):
        """
        Full Bayesian decision:
          Gate 1: edge_sig >= q_alpha(x)
          Gate 2: P(win) >= threshold
        """
        if not self._warmed_up:
            return True, {"reason": "warmup", "p_win": 0.5, "q_floor": self.cfg.edge_floor}

        x = self._x(spread=spread, p_adv=p_adv, flow=flow, lag50_ms=lag50_ms, sigma=sigma)
        key = (_regime_bin(regime), str(route), str(side))
        b = self._effective_bucket(key)

        _rl = self.cfg.ridge_lambda
        q_floor = b.quantile_floor(x, self.cfg.q_alpha, _rl)
        q_floor = clamp(q_floor, self.cfg.q_floor_clamp_lo, self.cfg.q_floor_clamp_hi) if q_floor > 0 else self.cfg.edge_floor

        pw = b.p_win(float(edge_sig), x, _rl)
        e_net = b.expected_net_edge(float(edge_sig), x, _rl)

        allow = (float(edge_sig) >= q_floor) and (pw >= self.cfg.p_win_threshold)

        mu, scale, df = b.predictive_params(x, _rl)

        diag = {
            "allow": bool(allow),
            "p_win": round(float(pw), 4),
            "q_floor": round(float(q_floor), 6),
            "e_net": round(float(e_net), 6),
            "mu_leak": round(float(mu), 6),
            "scale": round(float(scale), 6),
            "df": round(float(df), 2),
            "bucket_n": int(b.n),
            "key": key,
        }
        return bool(allow), diag

    def diagnostics(self) -> Dict:
        bsum = {}
        for k, b in self.buckets.items():
            if b.n > 0:
                bsum[str(k)] = b.diagnostics()
        return {
            "n_fills": int(self.n_fills),
            "warmed": bool(self._warmed_up),
            "global": self.global_bucket.diagnostics(),
            "n_buckets": int(len(self.buckets)),
            "buckets": bsum,
        }
