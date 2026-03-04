#!/usr/bin/env python3
"""
PolyBot v2 — Polymarket 5-Min BTC Up/Down Quant Trading Bot
=============================================================

Key innovation: Uses Polymarket's own Chainlink RTDS feed as primary oracle,
eliminating the $20-60 basis error from using Binance/Coinbase prices.

Architecture:
  - chainlink_rtds.py  → Primary oracle (exact resolution feed)
  - oracle_engine.py   → Edge decision logic (cone, Z-gate, persistence)
  - main.py            → Bot orchestration, survival engine, execution

Price hierarchy:
  1. Chainlink RTDS (wss://ws-live-data.polymarket.com)  — PRIMARY
  2. Coinbase WebSocket (real-time trades)                 — sigma + fallback
  3. Binance WebSocket (fallback if Coinbase stale)        — secondary fallback
  4. Chainlink on-chain poll (Polygon)                     — emergency only

Edge pipeline (100 Hz):
  cone_p_and_z → edge calculation → Z gate → persistence → Kelly sizing → IOC/FOK
"""

import asyncio
import logging
import json
import time
import math
import os
import csv
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import sys
import threading

import requests
import websockets
import numpy as np
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs, OrderType, ApiCreds,
    PartialCreateOrderOptions,
    BalanceAllowanceParams, AssetType,
)
from py_clob_client.constants import POLYGON

# Local modules
from oracle_engine import (
    LagAdaptiveZ, PersistState, decide_edge, ms_now,
    fee_per_share, maker_fee, taker_fee, set_market_fee_rate, cone_p_and_z,
    update_sigma_history, Z_TRAJ, reset_sniper_lock,
    confirm_sniper_fire, reset_window_locks,
    BASIS, CB_LEAD,
)
from chainlink_rtds import (
    RTDS, chainlink_rtds_task, STRIKE_CAPTURE,
)
from regime import RegimeClassifier, JumpDetector
from tail_risk import TailRiskGuard
from order_flow import FlowTracker, GoldskyAnalytics
from participation import ParticipationVelocity, PVConfig
from bipower_jump import BipowerJumpFilter, BipowerConfig
from momentum import MomentumEngine, MomentumConfig
from fill_prob import FillProbModel, ExpiryScaler
from sprt import SPRTValidator
from calibration import load_calibration_overrides, get_bucket_adjustment
from vwap_overlay import VWAPTracker
from execution_safety import validate_execution_edge, book_health_score
from micro_decay import micro_edge_buffer
from endgame_manager import EndgameManager, EndgameConfig
from exit_manager import ExitManager, ExitConfig
from position_monitor import PositionMonitor, PositionMonitorConfig, edge_now_for_position
from leakage_model import LeakageModel
from pnl_tracker import PnLTracker
from position import Position, Portfolio, Side
from inventory_layer import InventoryDecisionLayer, InventoryLayerConfig, DecisionType
from telemetry import TelemetryEmitter, TelemetryConfig
from adaptive_executor import L2Tracker, AdaptiveExecutor, MicroSnapshot, ExecutionResult


# ════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

WINDOW_OPEN_MS:     int = 0
FIRST_REAL_BOOK_MS: int = 0

load_dotenv()
os.makedirs("logs", exist_ok=True)

# ── Logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/bot.log", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(sh)


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CANDLE_API_URL      = "https://polymarket.com/api/chainlink-candles"
CRYPTO_PRICE_URL    = "https://polymarket.com/api/crypto/crypto-price"
POLYMARKET_TIMEZONE = ZoneInfo("America/New_York")
SIMULATION_MODE     = os.getenv("SIM_MODE", "true").lower() == "true"

MAX_SPEND_PER_ORDER_USD    = float(os.getenv("MAX_SPEND_PER_ORDER_USD", os.getenv("MAX_SPEND_USD", "999999.0")))
WINDOW_SPEND_CAP_USD       = float(os.getenv("WINDOW_SPEND_CAP_USD", "999999.0"))
POLY_BOOK_MAX_AGE_MS       = int(os.getenv("POLY_BOOK_MAX_AGE_MS", "15000"))
POLY_BOOK_RESEED_AHEAD_MS  = int(os.getenv("POLY_BOOK_RESEED_AHEAD_MS", "4000"))

# EV attribution + online calibration knobs
EXEC_LEAKAGE_WARN = float(os.getenv("EXEC_LEAKAGE_WARN", "0.30"))
CAL_EDGE_BASE_SHIFT = float(os.getenv("CAL_EDGE_BASE_SHIFT", "0.0"))
CAL_MIN_P_BASE_SHIFT = float(os.getenv("CAL_MIN_P_BASE_SHIFT", "0.0"))
CAL_MAX_PAY_BASE_SHIFT = float(os.getenv("CAL_MAX_PAY_BASE_SHIFT", "0.0"))
CAL_DOWN_EDGE_BONUS = float(os.getenv("CAL_DOWN_EDGE_BONUS", "0.003"))
CAL_DOWN_MIN_P_BONUS = float(os.getenv("CAL_DOWN_MIN_P_BONUS", "0.008"))
CALIBRATION_LOG_FILE = os.getenv("CALIBRATION_LOG_FILE", "logs/ev_calibration.csv")
IMBALANCE_DECAY_K = float(os.getenv("IMBALANCE_DECAY_K", "0.90"))
CALIBRATION_OUTPUT_FILE = os.getenv("CALIBRATION_OUTPUT_FILE", "logs/calibration_output.json")
_CAL_OVERRIDES: dict = load_calibration_overrides(CALIBRATION_OUTPUT_FILE)

# Oracle staleness thresholds
RTDS_FRESH_MS        = 45_000   
COINBASE_STALE_MS    = 5_000    
BTC_FEED_MAX_AGE_MS  = 3_000    

# Vol regime gates 
SIGMA_MIN_TRADE = 0.0005   
SIGMA_MAX_TRADE = 0.0060   

EXIT_ALPHA_BASE = float(os.getenv("EXIT_ALPHA_BASE", "0.55"))
EXIT_MIN_BUF    = float(os.getenv("EXIT_MIN_BUF", "0.01"))    
EXIT_HARD_FLOOR_T = float(os.getenv("EXIT_HARD_FLOOR_T", "15"))  

# ── Session PnL & Circuit Breaker ──────────────────────────────────────────
SESSION_PNL:              float = 0.0
CONSEC_LOSSES:            int   = 0
LAST_WINDOW_PNL:          float = 0.0
SESSION_LOSS_LIMIT:       float = float(os.getenv("SESSION_LOSS_LIMIT", "-50.0"))
MAX_CONSEC_LOSSES:        int   = int(os.getenv("MAX_CONSEC_LOSSES", "4"))
CIRCUIT_BREAKER_ACTIVE:   bool  = False
CIRCUIT_BREAKER_UNTIL:    float = 0.0
CIRCUIT_BREAKER_PAUSE_S:  int   = int(os.getenv("CIRCUIT_BREAKER_PAUSE_S", "1800"))

LOSS_STREAK_KELLY_PENALTY: float = 1.0   
LOSS_STREAK_TRADES_LEFT:   int   = 0     

RISK_A_BASE = float(os.getenv("RISK_A_BASE", "0.35"))
RISK_B_BASE = float(os.getenv("RISK_B_BASE", "0.50"))
RISK_A_MIN  = float(os.getenv("RISK_A_MIN",  "0.15"))
RISK_A_MAX  = float(os.getenv("RISK_A_MAX",  "0.55"))
RISK_B_MIN  = float(os.getenv("RISK_B_MIN",  "0.20"))
RISK_B_MAX  = float(os.getenv("RISK_B_MAX",  "0.80"))

EQUITY_PEAK: float = 0.0
EQUITY_LAST: float = 0.0
DRAWDOWN:    float = 0.0
DD_START = float(os.getenv("DD_START", "0.03"))   
DD_MAX   = float(os.getenv("DD_MAX",   "0.12"))   
DD_FLOOR = float(os.getenv("DD_FLOOR", "0.35"))   

ROLL_SHARPE_N  = int(os.getenv("ROLL_SHARPE_N", "20"))
SHARPE_SOFT    = float(os.getenv("SHARPE_SOFT", "-0.15"))
SHARPE_HARD    = float(os.getenv("SHARPE_HARD", "-0.40"))
SHARPE_MULT_MIN = float(os.getenv("SHARPE_MULT_MIN", "0.25"))
SHARPE_MULT_MAX = float(os.getenv("SHARPE_MULT_MAX", "1.25"))

WINDOW_RETURNS: deque  = deque(maxlen=200)
WIN_REALIZED_START     = None
ROLL_SHARPE: float     = 0.0
SHARPE_MULT: float     = 1.0
SHARPE_PAUSED: bool    = False

DAY_LOSS_LIMIT = float(os.getenv("DAY_LOSS_LIMIT", "-150.0"))
DAY_DD_MAX     = float(os.getenv("DAY_DD_MAX", "0.10"))
DAY_PAUSE_SEC  = int(os.getenv("DAY_PAUSE_SEC", "86400"))
DAY_KILL_ACTIVE:    bool  = False
DAY_KILL_UNTIL_MS:  int   = 0
DAY_KEY:            str   = None
DAY_EQUITY_START:   float = 0.0
DAY_EQUITY_PEAK:    float = 0.0


def record_pnl(proceeds: float, cost: float) -> None:
    global SESSION_PNL, CONSEC_LOSSES, LAST_WINDOW_PNL
    global CIRCUIT_BREAKER_ACTIVE, CIRCUIT_BREAKER_UNTIL
    global LOSS_STREAK_KELLY_PENALTY, LOSS_STREAK_TRADES_LEFT

    pnl = proceeds - cost
    SESSION_PNL += pnl
    LAST_WINDOW_PNL = pnl

    if pnl < -0.01:
        CONSEC_LOSSES += 1
    elif pnl > 0.01:
        CONSEC_LOSSES = 0
        if LOSS_STREAK_TRADES_LEFT > 0:
            LOSS_STREAK_TRADES_LEFT -= 1
            if LOSS_STREAK_TRADES_LEFT == 0:
                LOSS_STREAK_KELLY_PENALTY = 1.0
                logger.info("LOSS_STREAK penalty expired — Kelly restored to 1.0×")

    logger.info(
        f"PNL: trade={pnl:+.2f} session={SESSION_PNL:+.2f} streak={CONSEC_LOSSES} "
        f"kelly_pen={LOSS_STREAK_KELLY_PENALTY:.2f}({LOSS_STREAK_TRADES_LEFT} left)"
    )

    if CONSEC_LOSSES >= 3 and LOSS_STREAK_TRADES_LEFT == 0:
        LOSS_STREAK_KELLY_PENALTY = 0.5
        LOSS_STREAK_TRADES_LEFT = 5
        logger.warning(f"LOSS_STREAK: {CONSEC_LOSSES} consecutive losses — Kelly halved for next 5 trades")

    if SESSION_PNL < SESSION_LOSS_LIMIT or CONSEC_LOSSES >= MAX_CONSEC_LOSSES:
        CIRCUIT_BREAKER_ACTIVE = True
        CIRCUIT_BREAKER_UNTIL  = ms_now() + CIRCUIT_BREAKER_PAUSE_S * 1000
        logger.critical(f"CIRCUIT BREAKER: PnL={SESSION_PNL:.2f} consec={CONSEC_LOSSES} pausing {CIRCUIT_BREAKER_PAUSE_S}s")


EMERGENCY_EXIT_MODE: bool = False
EMERGENCY_EXIT_UNTIL_MS: int = 0

def enter_emergency_exit(ttl_ms: int = 30_000) -> None:
    global EMERGENCY_EXIT_MODE, EMERGENCY_EXIT_UNTIL_MS
    EMERGENCY_EXIT_MODE = True
    EMERGENCY_EXIT_UNTIL_MS = ms_now() + ttl_ms
    logger.warning(f"EMERGENCY_EXIT_MODE: activated for {ttl_ms}ms — entries blocked")

def check_emergency_exit() -> bool:
    global EMERGENCY_EXIT_MODE
    if EMERGENCY_EXIT_MODE and ms_now() >= EMERGENCY_EXIT_UNTIL_MS:
        EMERGENCY_EXIT_MODE = False
        logger.info("EMERGENCY_EXIT_MODE: expired, resuming normal operation")
    return EMERGENCY_EXIT_MODE

def get_available_usdc_balance() -> float:
    try:
        if client is None:
            return 0.0
        resp = client.get_balance_allowance(params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        balance   = float(resp.get("balance",   0) or 0) / 1e6
        allowance = float(resp.get("allowance", 0) or 0) / 1e6
        logger.info(f"BALANCE_CHECK: balance=${balance:.2f} allowance=${allowance:.2f}")
        if balance > 0 and allowance == 0:
            return balance
        return max(0.0, min(balance, allowance))
    except Exception as e:
        logger.warning(f"Balance fetch failed: {e}")
        return 0.0

def get_exchange_conditional_balance(token_id: str) -> float:
    try:
        if client is None: return 0.0
        resp = client.get_balance_allowance(params=BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=str(token_id)))
        return float(resp.get("balance", 0) or 0.0) / 1e6
    except Exception as e:
        return 0.0

def get_reserved_conditional(token_id: str) -> float:
    try:
        if client is None: return 0.0
        orders = client.get_orders()
        reserved = 0.0
        for o in (orders or []):
            if (o.get("asset_id") == token_id or o.get("token_id") == token_id) and str(o.get("side", "")).upper() == "SELL":
                sz = float(o.get("original_size", 0) or o.get("size", 0) or 0)
                matched = float(o.get("size_matched", 0) or 0)
                reserved += max(sz - matched, 0.0)
        return reserved
    except Exception:
        return 0.0

def compute_exit_urgency(T_sec: float, T_crit: float = 60.0) -> float:
    return max(0.0, min(1.0, (T_crit - T_sec) / T_crit))

def exit_ttl_ms(urgency: float) -> int:
    return int(600 - 450 * urgency)

def book_is_sane(bid: float, ask: float) -> bool:
    return (bid > 0.02 and ask < 0.98 and 0.0 < (ask - bid) < 0.20)

def floor3(x: float) -> float:
    return math.floor(float(x) * 1000.0) / 1000.0

EXIT_BAL_EPS = 0.001

def portfolio_liquidation_value(up_bid: float, dn_bid: float) -> float:
    v_up = float(POS_UP.inventory) * max(0.0, float(up_bid) - fee_per_share(float(up_bid)))
    v_dn = float(POS_DOWN.inventory) * max(0.0, float(dn_bid) - fee_per_share(float(dn_bid)))
    return float(v_up + v_dn)

def portfolio_gross_shares() -> float:
    return float(abs(POS_UP.inventory) + abs(POS_DOWN.inventory))

def portfolio_net_shares() -> float:
    return float(POS_UP.inventory - POS_DOWN.inventory)

def net_sell_after_fee(bid: float) -> float:
    return max(0.0, float(bid) - fee_per_share(float(bid)))

_FLIP_BREACH_TS: dict = {"UP": 0, "DOWN": 0}
_FLIP_P_THRESHOLD = 0.48
_FLIP_PERSIST_MS  = 200

def _mono_ms() -> int:
    return int(time.monotonic() * 1000)

def early_sell_profit_gate(sec_remaining: float, bid: float, entry_price: float, p_cone: float = 0.5, side: str = "UP", T_entry: float = 0.0):
    net_bid = net_sell_after_fee(float(bid))
    entry = max(0.01, float(entry_price))
    p_side = p_cone if side == "UP" else (1.0 - p_cone)
    edge = max(0.0, p_side - entry)
    T = float(sec_remaining)
    T_e = float(T_entry) if T_entry and T_entry > 0 else 230.0
    alpha_time = EXIT_ALPHA_BASE * max(0.0, min(1.0, T / T_e))
    exit_target = max(entry + EXIT_MIN_BUF, entry + alpha_time * edge)

    if T <= EXIT_HARD_FLOOR_T and net_bid > entry + EXIT_MIN_BUF:
        return True, net_bid, exit_target

    _now_mono = _mono_ms()
    if p_side < _FLIP_P_THRESHOLD and net_bid > entry + EXIT_MIN_BUF:
        if _FLIP_BREACH_TS[side] == 0:
            _FLIP_BREACH_TS[side] = _now_mono
        elif _now_mono - _FLIP_BREACH_TS[side] >= _FLIP_PERSIST_MS:
            _FLIP_BREACH_TS[side] = 0
            return True, net_bid, exit_target
    else:
        _FLIP_BREACH_TS[side] = 0

    return (net_bid >= exit_target), net_bid, exit_target


def log_risk_state(collateral, liq, risk_budget, gross_before, net_before, new_spend, gross_after, capped_size, limit_price, side):
    logger.info(
        "RISK_STATE | "
        f"collat=${collateral:.2f} liq=${liq:.2f} budget=${risk_budget:.2f} | "
        f"gross_before={gross_before:.2f} net_before={net_before:.2f} | "
        f"new_spend=${new_spend:.2f} gross_after=${gross_after:.2f} | "
        f"size={capped_size}@{limit_price:.2f} side={side}"
    )

def settle_window():
    up_qty = float(POS_UP.inventory)
    dn_qty = float(POS_DOWN.inventory)

    if up_qty <= 0 and dn_qty <= 0:
        return 0.0

    strike = float(STATE.open_price) if np.isfinite(STATE.open_price) else None
    if strike is None:
        up_cost = up_qty * float(POS_UP.avg_entry_price) if up_qty > 0 else 0.0
        dn_cost = dn_qty * float(POS_DOWN.avg_entry_price) if dn_qty > 0 else 0.0
        total_cost = up_cost + dn_cost
        if total_cost > 0: record_pnl(0.0, total_cost)
        return -total_cost

    close_price = None
    try:
        _prev_start_unix = get_window_start_unix(offset=-1)
        _prev_start = datetime.fromtimestamp(_prev_start_unix, tz=POLYMARKET_TIMEZONE)
        _prev_end = _prev_start + timedelta(minutes=5)
        _settle_data = fetch_crypto_price(_prev_start, _prev_end)
        if _settle_data and _settle_data.get("completed", False) and "closePrice" in _settle_data:
            close_price = float(_settle_data["closePrice"])
            if "openPrice" in _settle_data and STATE.strike_type != "OFFICIAL":
                strike = float(_settle_data["openPrice"])
    except Exception:
        pass

    if close_price is None:
        if RTDS.price and RTDS.age_ms() < 30_000: close_price = float(RTDS.price)
    if close_price is None:
        if STATE.btc_ts_ms > 0 and not np.isnan(STATE.btc_price) and STATE.btc_price > 0: close_price = float(STATE.btc_price)

    if close_price is None or close_price <= 0:
        up_cost = up_qty * float(POS_UP.avg_entry_price) if up_qty > 0 else 0.0
        dn_cost = dn_qty * float(POS_DOWN.avg_entry_price) if dn_qty > 0 else 0.0
        total_cost = up_cost + dn_cost
        if total_cost > 0: record_pnl(0.0, total_cost)
        return -total_cost

    up_wins = close_price >= strike
    up_entry_cost = up_qty * float(POS_UP.avg_entry_price) if up_qty > 0 else 0.0
    dn_entry_cost = dn_qty * float(POS_DOWN.avg_entry_price) if dn_qty > 0 else 0.0

    if up_wins:
        up_proceeds, dn_proceeds = up_qty * 1.0, 0.0
    else:
        up_proceeds, dn_proceeds = 0.0, dn_qty * 1.0

    total_proceeds = up_proceeds + dn_proceeds
    total_cost = up_entry_cost + dn_entry_cost
    settlement_pnl = total_proceeds - total_cost

    if total_cost > 0 or total_proceeds > 0:
        record_pnl(total_proceeds, total_cost)

    if total_cost > 0:
        for _sprt_side, _sprt_won in [("UP", up_wins), ("DOWN", not up_wins)]:
            _sp_model = ENTRY_P_CONE.get(_sprt_side)
            _sp_market = ENTRY_P_MARKET.get(_sprt_side)
            if _sp_model is not None and _sp_market is not None:
                SPRT.record(p_model=max(0.0, min(1.0, float(_sp_model))), p_market=max(0.0, min(1.0, float(_sp_market))), outcome=_sprt_won)

    return settlement_pnl

def regime_risk_multipliers():
    label = getattr(REGIME, "label", "NORMAL")
    flip = float(getattr(REGIME, "flip_rate", 0.0) or 0.0)
    intensity = float(getattr(REGIME.vol_detector, "intensity", 0.0) or 0.0) if hasattr(REGIME, "vol_detector") else 0.0
    a_mult, b_mult = 1.0, 1.0
    why = [label]

    if flip >= 0.10: a_mult *= 0.65; b_mult *= 0.70; why.append(f"flip>=0.10({flip:.2f})")
    elif flip >= 0.06: a_mult *= 0.80; b_mult *= 0.85; why.append(f"flip>=0.06({flip:.2f})")
    elif flip <= 0.02: a_mult *= 1.10; b_mult *= 1.05; why.append(f"flip<=0.02({flip:.2f})")

    if label in ("HIGH_VOL", "VOL_EVENT"): a_mult *= 0.75; b_mult *= 0.80; why.append("vol_regime")
    elif label == "TRANSITION": a_mult *= 0.85; b_mult *= 0.90; why.append("transition")
    elif label == "CALM": a_mult *= 1.10; b_mult *= 1.05; why.append("calm")

    if intensity >= 1.0: a_mult *= 0.85; b_mult *= 0.90; why.append(f"intensity>=1.0({intensity:.2f})")
    return a_mult, b_mult, "|".join(why)

def drawdown_risk_multiplier(equity: float) -> float:
    global EQUITY_PEAK, EQUITY_LAST, DRAWDOWN
    EQUITY_LAST = float(equity)
    if EQUITY_PEAK <= 0: EQUITY_PEAK = float(equity)
    if equity > EQUITY_PEAK: EQUITY_PEAK = float(equity)
    dd = max(0.0, (EQUITY_PEAK - equity) / EQUITY_PEAK) if EQUITY_PEAK > 0 else 0.0
    DRAWDOWN = dd
    if dd <= DD_START: return 1.0
    if dd >= DD_MAX: return float(DD_FLOOR)
    t = (dd - DD_START) / max(1e-9, (DD_MAX - DD_START))
    return float(1.0 - t * (1.0 - DD_FLOOR))

def _compute_roll_sharpe(rets) -> float:
    if len(rets) < 5: return 0.0
    m = sum(rets) / len(rets)
    var = sum((x - m) ** 2 for x in rets) / max(1, (len(rets) - 1))
    sd = math.sqrt(var)
    if sd < 1e-12: return 0.0
    return m / sd

def _sharpe_to_mult(sh: float) -> float:
    if sh >= 0.25: return SHARPE_MULT_MAX
    if sh >= 0.0: return 1.0 + (sh / 0.25) * (SHARPE_MULT_MAX - 1.0)
    if sh >= SHARPE_SOFT: return 0.60 + ((sh - SHARPE_SOFT) / max(1e-9, (0.0 - SHARPE_SOFT))) * (1.0 - 0.60)
    if sh <= SHARPE_HARD: return SHARPE_MULT_MIN
    return SHARPE_MULT_MIN + ((sh - SHARPE_HARD) / max(1e-9, (SHARPE_SOFT - SHARPE_HARD))) * (0.60 - SHARPE_MULT_MIN)

def _day_key_now(): return datetime.now(POLYMARKET_TIMEZONE).strftime("%Y-%m-%d")

def update_day_kill_switch(equity: float):
    global DAY_KEY, DAY_EQUITY_START, DAY_EQUITY_PEAK, DAY_KILL_ACTIVE, DAY_KILL_UNTIL_MS
    key = _day_key_now()
    if DAY_KEY != key:
        DAY_KEY = key
        DAY_EQUITY_START = float(equity)
        DAY_EQUITY_PEAK  = float(equity)
        DAY_KILL_ACTIVE = False
        DAY_KILL_UNTIL_MS = 0
    if equity > DAY_EQUITY_PEAK: DAY_EQUITY_PEAK = float(equity)
    day_pnl = equity - DAY_EQUITY_START
    day_dd = max(0.0, (DAY_EQUITY_PEAK - equity) / DAY_EQUITY_PEAK) if DAY_EQUITY_PEAK > 0 else 0.0
    if (day_pnl <= DAY_LOSS_LIMIT) or (day_dd >= DAY_DD_MAX):
        if not DAY_KILL_ACTIVE:
            DAY_KILL_ACTIVE = True
            DAY_KILL_UNTIL_MS = ms_now() + DAY_PAUSE_SEC * 1000
    return day_pnl, day_dd


# ════════════════════════════════════════════════════════════════════════════
# STATE & DATA MODELS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class BookSnapshot:
    token_id:    str
    best_bid:    float = 0.0
    best_ask:    float = 1.0
    spread:      float = 1.0
    imbalance:   float = 0.5
    bid_size:    float = 0.0
    ask_size:    float = 0.0
    source:      str   = "unknown"
    last_update: float = 0.0

class BookCache:
    def __init__(self):
        self._books: dict = {}
        self._lock  = threading.Lock()
    def update(self, token_id: str, snap: BookSnapshot) -> None:
        with self._lock: self._books[token_id] = snap
    def get(self, token_id: str) -> Optional[BookSnapshot]:
        with self._lock: return self._books.get(token_id)
    def delete(self, token_id: str) -> None:
        with self._lock: self._books.pop(token_id, None)
    def is_fresh(self, token_id: str, max_age_ms: float = 1500) -> bool:
        snap = self.get(token_id)
        if not snap: return False
        return (ms_now() - snap.last_update) < max_age_ms

BOOK_CACHE = BookCache()

class MarketState:
    btc_price:     float = np.nan
    btc_ts_ms:     int   = 0
    open_price:    float = np.nan
    sec_remaining: float = np.nan
    sigma_1m:      float = 0.0009
    sigma_slow:    float = 0.0003
    sigma_fast:    float = 0.0003
    sigma_w:       float = 0.5
    strike_type:   Optional[str] = None
    rtds_last_msg_ts:    float = 0.0
    poly_last_msg_ts:    float = 0.0
    trading_suspended:   bool  = False

STATE        = MarketState()
PERSIST      = PersistState()
LAG_ADAPTIVE = LagAdaptiveZ()
REGIME       = RegimeClassifier()
JUMP_DETECTOR = JumpDetector(dt_seconds=1.0)
TAIL_RISK    = TailRiskGuard()
FLOW         = FlowTracker()
PV_TRACKER   = ParticipationVelocity(PVConfig())
BIPOWER      = BipowerJumpFilter(BipowerConfig())
MOMENTUM     = MomentumEngine(MomentumConfig())
POS_MONITOR  = PositionMonitor(PositionMonitorConfig())
ADAPTIVE_EXEC: AdaptiveExecutor = None  
PNL_TRACKER  = PnLTracker()
FILL_PROB    = FillProbModel()
EXPIRY       = ExpiryScaler()
SPRT         = SPRTValidator()
VWAP         = VWAPTracker()
GOLDSKY      = GoldskyAnalytics()
L2_TRACKER   = L2Tracker()
LEAK_MODEL   = LeakageModel()
LATEST_DEBUG: dict = {}

class DirectionalCoverageController:
    def __init__(self, target_up=0.30, target_dn=0.20, window=40):
        self.window = window
        self.target_up = target_up
        self.target_dn = target_dn
        self.up_hist = deque(maxlen=window)
        self.dn_hist = deque(maxlen=window)
    def record_window(self, traded_up: bool, traded_dn: bool):
        self.up_hist.append(1 if traded_up else 0)
        self.dn_hist.append(1 if traded_dn else 0)
    def coverage_up(self): return sum(self.up_hist) / len(self.up_hist) if self.up_hist else 0.0
    def coverage_dn(self): return sum(self.dn_hist) / len(self.dn_hist) if self.dn_hist else 0.0
    def z_bias(self, side: str):
        if side == "UP": coverage, target = self.coverage_up(), self.target_up
        else: coverage, target = self.coverage_dn(), self.target_dn
        return max(-0.10, min(0.15, (coverage - target) * 0.20))

COVERAGE = DirectionalCoverageController(target_up=0.30, target_dn=0.20, window=40)
TRADED_UP_THIS_WINDOW: bool = False
TRADED_DN_THIS_WINDOW: bool = False
TRADES_THIS_WINDOW: int = 0
ENTRY_P_CONE: dict = {"UP": None, "DOWN": None}
ENTRY_P_MARKET: dict = {"UP": None, "DOWN": None}
ENTRY_T_SEC: dict = {"UP": None, "DOWN": None}
ENTRY_EDGE: dict = {"UP": 0.0, "DOWN": 0.0}
ENTRY_Z_EMA: dict = {"UP": 0.0, "DOWN": 0.0}
ENTRY_TS_MS: dict = {"UP": 0, "DOWN": 0}
GTC_EXIT_ATTEMPTS: dict = {}
GTC_EXIT_MAX: int = 8
SIS_LAST_ACTION_U: dict = {"UP": 0.0, "DOWN": 0.0}
SIS_LAST_ACTION_BID: dict = {"UP": 0.0, "DOWN": 0.0}
MAX_TRADES_PER_WINDOW: int = 3
WINDOW_BANKROLL: float = 0.0

MARKET_ID     = ""
UP_TOKEN_ID   = ""
DOWN_TOKEN_ID = ""
POLY_STATE = {}
WINDOW_CRYPTO_PRICE: Optional[dict] = None
LAST_BTC_MOVE_TS: int   = 0
LAST_BTC_PRICE:   float = np.nan
LAST_BTC_DIR:     int   = 0
BOOK_RECONNECT_FLAG:          bool = False
REST_SEED_INFLIGHT:           bool = False
LAST_REST_SEED_MS:            int  = 0
REST_SEED_COOLDOWN_MS:        int  = 5000
REST_SEED_EMPTY_COOLDOWN_MS:  int  = 30000
LAST_REST_SEED_HAD_QUOTES:    bool = False
LAST_STALE_BOOK_LOG_MS:       int  = 0
LAST_NON_WS_BOOK_LOG_MS:      int  = 0
LIQUIDITY_TIMEOUT_THIS_WINDOW: bool = False
_P_CONE_HISTORY: deque = deque(maxlen=20)

import hashlib
RECENT_FILLS: deque = deque(maxlen=2000)
PENDING_LOCKS: Dict[tuple, int] = {}

def order_fingerprint(token_id: str, side: str, price: float, size: float) -> str:
    payload = {"t": token_id, "s": side, "p": round(price, 2), "q": int(size)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

def lock_side(token_id: str, side: str, ttl_ms: int = 1200) -> None:
    PENDING_LOCKS[(token_id, side)] = ms_now() + ttl_ms

def is_locked(token_id: str, side: str) -> bool:
    return ms_now() < PENDING_LOCKS.get((token_id, side), 0)

def find_fill_match(token_id: str, price: float, now_ms: int, window_ms: int = 1200, price_tol: float = 0.02, side: str = None, expected_size: float = 0.0) -> Optional[dict]:
    cutoff = now_ms - window_ms
    for f in reversed(RECENT_FILLS):
        if f["ts_ms"] < cutoff: break
        if f["token_id"] != token_id: continue
        if abs(f["price"] - price) > price_tol: continue
        if side and f.get("side") != side: continue
        if expected_size > 0:
            if abs(float(f.get("size", 0)) - expected_size) / max(1e-9, expected_size) > 0.25: continue
        return f
    return None

def flow_size_scale(flow: float) -> float:
    return max(0.25, min(1.0, 1.0 - abs(flow) * 0.8))

def state_calibration_adjustments(*, side: str, sec_remaining: float, spread: float, lag_p50_ms: float, sigma_ratio: float) -> dict:
    edge_shift = CAL_EDGE_BASE_SHIFT
    min_p_shift = CAL_MIN_P_BASE_SHIFT
    max_pay_shift = CAL_MAX_PAY_BASE_SHIFT

    if sec_remaining <= 60: edge_shift += 0.004
    elif sec_remaining >= 180: edge_shift -= 0.001

    if spread > 0.06: edge_shift += 0.004; max_pay_shift -= 0.010
    elif spread < 0.03: edge_shift -= 0.001; max_pay_shift += 0.005

    if lag_p50_ms > 900: edge_shift += 0.003; min_p_shift += 0.005
    elif lag_p50_ms < 450: edge_shift -= 0.001

    if sigma_ratio > 1.5: edge_shift += 0.002
    elif sigma_ratio < 0.8: min_p_shift += 0.004

    if side == "DOWN": edge_shift += CAL_DOWN_EDGE_BONUS; min_p_shift += CAL_DOWN_MIN_P_BONUS

    if _CAL_OVERRIDES:
        cal_adj = get_bucket_adjustment(_CAL_OVERRIDES, side, sec_remaining, spread, lag_p50_ms, sigma_ratio)
        edge_shift += cal_adj["edge_boost"]
        min_p_shift += cal_adj["min_p_shift"]
        max_pay_shift += cal_adj["max_pay_shift"]

    return {"edge_boost": round(edge_shift, 5), "min_p_shift": round(min_p_shift, 5), "max_pay_shift": round(max_pay_shift, 5)}

_RECENT_EDGES: deque = deque(maxlen=200)

def quality_score(edge: float, spread: float, p_fill: float, lag_p50_ms: float) -> float:
    _RECENT_EDGES.append(edge)
    e = sum(1 for x in _RECENT_EDGES if x <= edge) / len(_RECENT_EDGES) if len(_RECENT_EDGES) > 20 else max(0.0, min(1.0, edge / 0.08))
    sp = max(0.0, min(1.0, 1.0 - spread / 0.08))
    pf = max(0.0, min(1.0, p_fill))
    lag = max(0.0, min(1.0, 1.0 - lag_p50_ms / 1500.0))
    return 0.40 * e + 0.20 * sp + 0.25 * pf + 0.15 * lag

def depth_decayed_imbalance(bids: list, asks: list, mid: float, *, k: float = 0.90, tick_size: float = 0.01) -> float:
    if not bids and not asks: return 0.0
    def _w_sum(levels: list) -> float:
        s = 0.0
        for px, sz in levels:
            try: p, q = float(px), max(0.0, float(sz))
            except Exception: continue
            s += q * math.exp(-k * (abs(p - mid) / max(1e-9, tick_size)))
        return s
    wb, wa = _w_sum(bids), _w_sum(asks)
    tot = wb + wa
    return max(-1.0, min(1.0, (wb - wa) / tot)) if tot > 1e-9 else 0.0

def log_calibration_sample(*, side: str, t_bucket: str, spread: float, lag_ms: float, sigma_ratio: float, edge: float, p_token: float, ask: float, reason: str) -> None:
    try:
        _new = not os.path.exists(CALIBRATION_LOG_FILE)
        with open(CALIBRATION_LOG_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if _new: w.writerow(["ts_ms", "side", "t_bucket", "spread", "lag_ms", "sigma_ratio", "edge", "p_token", "ask", "reason"])
            w.writerow([ms_now(), side, t_bucket, f"{spread:.5f}", f"{lag_ms:.1f}", f"{sigma_ratio:.3f}", f"{edge:.5f}", f"{p_token:.5f}", f"{ask:.4f}", reason])
    except Exception as e:
        logger.debug(f"CALIB_LOG_SKIP: {e}")

EXEC_LOCKED: bool = False
EXEC_LOCK_MS: int = 0
STATE_LOCK = asyncio.Lock()

def exec_lock(ttl_ms: int = 1500):
    global EXEC_LOCKED, EXEC_LOCK_MS
    EXEC_LOCKED = True
    EXEC_LOCK_MS = ms_now() + ttl_ms

def exec_unlock():
    global EXEC_LOCKED
    EXEC_LOCKED = False

def exec_lock_expired() -> bool:
    return ms_now() >= EXEC_LOCK_MS

PENDING_FLIP: dict = {"active": False, "side": None, "token_id": None, "limit": None, "size": 0.0, "expires_ms": 0}
PENDING_FLIP_INTENT: dict = {"active": False, "buy_token_id": None, "buy_side": None, "buy_limit": None, "buy_size": 0.0, "expires_ms": 0}
LAST_FIRE_SNAPSHOT: dict = {}

POS_UP   = Position("UP", lot_method="FIFO")
POS_DOWN = Position("DOWN", lot_method="FIFO")
PORTFOLIO = Portfolio(POS_UP, POS_DOWN)
IADL = InventoryDecisionLayer(InventoryLayerConfig(
    reversal_z=2.2, reversal_conf=0.88, reversal_cooldown_sec=30, late_t_sec=75, late_reversal_z=3.0, late_reversal_conf=0.93,
))
TELEMETRY = TelemetryEmitter(TelemetryConfig(jsonl_path="logs/pnl_telemetry.jsonl"))

@dataclass
class PortfolioPos:
    up: float
    dn: float
    total: float
    dominant_side: Optional[str]
    dominant_token_id: Optional[str]
    avg_cost_up: float
    avg_cost_dn: float

def get_portfolio_pos() -> PortfolioPos:
    up, dn = float(POS_UP.inventory), float(POS_DOWN.inventory)
    total = up + dn
    dom_side = "UP" if up >= dn else "DOWN" if total > 0 else None
    dom_tid = UP_TOKEN_ID if dom_side == "UP" else DOWN_TOKEN_ID if total > 0 else None
    try: acu = float(POS_UP.avg_entry_incl_fee)
    except Exception: acu = 0.5
    try: acd = float(POS_DOWN.avg_entry_incl_fee)
    except Exception: acd = 0.5
    return PortfolioPos(up=up, dn=dn, total=total, dominant_side=dom_side, dominant_token_id=dom_tid, avg_cost_up=acu, avg_cost_dn=acd)

class DualSigmaEstimator:
    def __init__(self, slow_window_s: float = 90.0, fast_window_s: float = 8.0):
        self.slow_window_s = slow_window_s
        self.fast_window_s = fast_window_s
        self.log_prices: deque = deque(maxlen=2000)
        self.timestamps: deque = deque(maxlen=2000)
    def add_price(self, price: float, ts_ms: int):
        if price > 0:
            self.log_prices.append(math.log(price))
            self.timestamps.append(ts_ms)
    def _rv(self, window_s: float) -> float:
        if len(self.log_prices) < 5 or len(self.timestamps) < 5: return 0.0
        cutoff = self.timestamps[-1] - int(window_s * 1000)
        arr = np.array(self.log_prices)
        ts_arr = np.array(self.timestamps)
        lps = arr[ts_arr >= cutoff]
        if len(lps) < 3: return 0.0
        returns = np.diff(lps)
        return float(np.sum(returns ** 2)) if len(returns) >= 2 else 0.0
    def compute_sigma(self):
        if len(self.log_prices) < 5: return 0.0003, 0.0003, 0.0003, 0.5
        rv_slow, rv_fast = self._rv(self.slow_window_s), self._rv(self.fast_window_s)
        sigma_slow = math.sqrt(max(0.0, rv_slow * (60.0 / self.slow_window_s)))
        sigma_fast = math.sqrt(max(0.0, rv_fast * (60.0 / self.fast_window_s)))
        denom = sigma_fast + sigma_slow
        w = sigma_fast / denom if denom > 1e-10 else 0.5
        sigma_eff = max(0.00008, min(0.01, sigma_slow + w * (sigma_fast - sigma_slow)))
        return float(sigma_eff), float(sigma_slow), float(sigma_fast), float(w)

DUAL_SIGMA = DualSigmaEstimator(slow_window_s=90.0, fast_window_s=8.0)
SIGMA_HISTORY: deque = deque(maxlen=300)

MIN_FIRE_GAP_MS_IMPULSE = 700
MIN_FIRE_GAP_MS_DRIFT   = 1500
LAST_FIRE_TS: dict      = {}
ORDER_COOLDOWN: dict    = {}
EXIT_COOLDOWN: dict     = {}
EXEC_ATTEMPTED: set     = set()
EXEC_ATTEMPTED_MAX      = 200

def fire_allowed(token_id: str, order_side: str, mode: str) -> bool:
    now = ms_now()
    key = (token_id, order_side)
    gap = MIN_FIRE_GAP_MS_IMPULSE if mode == "impulse" else MIN_FIRE_GAP_MS_DRIFT
    if now - LAST_FIRE_TS.get(key, 0) < gap: return False
    LAST_FIRE_TS[key] = now
    return True

_LAST_SIGMA_TS:           float = 0.0
_SIGMA_UPDATE_INTERVAL_S: float = 1.0

def _update_sigma(price: float, ts_ms: int) -> None:
    global _LAST_SIGMA_TS
    DUAL_SIGMA.add_price(price, ts_ms)
    now_s = ts_ms / 1000.0
    if now_s - _LAST_SIGMA_TS < _SIGMA_UPDATE_INTERVAL_S: return
    _LAST_SIGMA_TS = now_s
    sigma_eff, sigma_slow, sigma_fast, w = DUAL_SIGMA.compute_sigma()
    SIGMA_HISTORY.append(sigma_eff)
    dyn_floor = float(np.percentile(list(SIGMA_HISTORY), 20)) if len(SIGMA_HISTORY) >= 30 else 0.00015
    _flip_rate = float(getattr(REGIME, "flip_rate", 0.0) or 0.0)
    if _flip_rate > 0.08: dyn_floor *= 1.5
    elif _flip_rate > 0.05: dyn_floor *= 1.25
    sigma_eff = max(sigma_eff, dyn_floor)
    STATE.sigma_1m, STATE.sigma_slow, STATE.sigma_fast, STATE.sigma_w = sigma_eff, sigma_slow, sigma_fast, w
    update_sigma_history(sigma_eff)
    LATEST_DEBUG["sigma_floor"] = dyn_floor

HOST           = "https://clob.polymarket.com"
PRIVATE_KEY    = os.getenv("CLOB_PRIVATE_KEY") or os.getenv("PK")
API_KEY        = os.getenv("CLOB_API_KEY")      or os.getenv("POLY_API_KEY")
API_SECRET     = os.getenv("CLOB_SECRET")       or os.getenv("POLY_SECRET")
API_PASSPHRASE = (os.getenv("CLOB_PASSPHRASE") or os.getenv("CLOB_PASS_PHRASE") or os.getenv("POLY_PASSPHRASE"))
CLOB_FUNDER    = os.getenv("CLOB_FUNDER") or os.getenv("PROXY_FUNDER")

try:
    clob_creds = ApiCreds(api_key=API_KEY, api_secret=API_SECRET, api_passphrase=API_PASSPHRASE) if API_KEY else None
    client = ClobClient(host=HOST, key=PRIVATE_KEY, chain_id=137, signature_type=1, funder=CLOB_FUNDER, creds=clob_creds)
    try:
        refreshed = client.create_or_derive_api_creds()
        if refreshed: client.set_api_creds(refreshed); logger.info("CLOB API creds refreshed.")
    except Exception as e: logger.warning(f"CLOB creds refresh failed: {e}")
except Exception as e:
    logger.error(f"Failed to init ClobClient: {e}")
    client = None

thread_pool:     ThreadPoolExecutor = ThreadPoolExecutor(max_workers=32)
execution_queue: asyncio.Queue      = asyncio.Queue()

if client is not None:
    ADAPTIVE_EXEC = AdaptiveExecutor(client)
    def _toxic_maker_abort(side: str, token_id: str) -> bool:
        try:
            _label = str(getattr(REGIME, "label", "NORMAL") or "NORMAL")
            if _label in ("BURST", "FAST_TAIL", "PANIC", "ADVERSARIAL"): return True
            if float(getattr(REGIME, "flip_rate", 0.0) or 0.0) > 0.12: return True
            if side in ("UP", "DOWN") and LAG_ADAPTIVE.fast_tail_active(side): return True
            if LAST_BTC_MOVE_TS > 0 and (ms_now() - LAST_BTC_MOVE_TS) < 500:
                if side == "UP" and LAST_BTC_DIR == -1: return True
                if side == "DOWN" and LAST_BTC_DIR == 1: return True
        except Exception: return False
        return False
    ADAPTIVE_EXEC.set_toxic_abort_fn(_toxic_maker_abort)
else: logger.warning("AdaptiveExecutor NOT initialized (no client)")

def ensure_logs() -> None:
    if not os.path.exists("logs/trades.csv"):
        with open("logs/trades.csv", "w", newline="") as f:
            csv.writer(f).writerow(["ts_ms", "side", "token_id", "limit_price", "size", "result", "order_id", "exec_ms", "retry_idx", "edge", "p_cone", "z", "sigma_1m", "oracle_source", "oracle_price", "execution_mode", "detail", "regime", "edge_target", "mode", "flow"])

def _prime_tick_size_cache(token_id: str) -> None:
    if client is None: return
    try: client.get_tick_size(token_id); return
    except Exception: pass
    try: client.get_order_book(token_id); return
    except Exception: pass
    try:
        client._ClobClient__tick_sizes[token_id] = "0.01"
        client._ClobClient__tick_size_timestamps[token_id] = time.monotonic()
    except Exception as e: logger.error(f"tick_size inject failed: {e}")

def best_executable_price(signal_side: str, up_snap: BookSnapshot, dn_snap: BookSnapshot, tick_buffer: float = 0.02, allow_implied: bool = False) -> tuple:
    ILLIQUID = 0.50
    own, opp = (up_snap, dn_snap) if signal_side == "UP" else (dn_snap, up_snap)
    own_tok, opp_tok = (UP_TOKEN_ID, DOWN_TOKEN_ID) if signal_side == "UP" else (DOWN_TOKEN_ID, UP_TOKEN_ID)

    if own.best_ask > 0 and own.best_ask < 1.0: direct = (round(min(0.99, own.best_ask + tick_buffer), 2), own_tok, "BUY", False, round(min(0.99, own.best_ask + tick_buffer), 2))
    else: direct = (float("inf"), own_tok, "BUY", False, 0.0)

    if not allow_implied: return direct
    if (opp.best_bid > 0.02 and opp.best_bid < 0.98):
        o_exec = round(max(0.01, opp.best_bid - tick_buffer), 2)
        implied = (round(1.0 - o_exec, 4), opp_tok, "SELL", True, o_exec)
    else: implied = (float("inf"), opp_tok, "SELL", True, 0.0)
    return direct if direct[0] <= implied[0] else implied

RATE_LIMIT_UNTIL_MS: int = 0
RATE_LIMIT_COOLDOWN_MS: int = 2000

def _check_rate_limit_response(resp, error_str: str = "") -> bool:
    global RATE_LIMIT_UNTIL_MS
    is_429 = False
    if isinstance(resp, dict): is_429 = resp.get("status", 0) == 429 or "429" in str(resp.get("errorMsg", ""))
    if "429" in error_str or "Too Many" in error_str: is_429 = True
    if is_429:
        RATE_LIMIT_UNTIL_MS = ms_now() + RATE_LIMIT_COOLDOWN_MS
        logger.error(f"RATE_LIMIT: 429 detected — suspending new orders until {RATE_LIMIT_COOLDOWN_MS}ms cooldown")
    return is_429

def is_rate_limited() -> bool:
    return ms_now() < RATE_LIMIT_UNTIL_MS

def _get_l2_top2(token_id: str):
    try:
        url = f"https://clob.polymarket.com/book?token_id={token_id}"
        r = requests.get(url, timeout=0.35)
        r.raise_for_status()
        data = r.json()
        bids = [(float(x["price"]), float(x.get("size", 0.0))) for x in (data.get("bids") or [])[:2]]
        asks = [(float(x["price"]), float(x.get("size", 0.0))) for x in (data.get("asks") or [])[:2]]
        return bids, asks, True
    except Exception:
        return [], [], False

try: _fok_cap_env = float(os.getenv("FOK_HARD_CAP", "0.95"))
except Exception: _fok_cap_env = 0.86
FOK_HARD_CAP   = min(0.95, max(0.50, _fok_cap_env))
FOK_HARD_FLOOR = 0.01

def dynamic_fok_buffer(spread: float = 0.10, regime_label: str = "NORMAL") -> float:
    base = max(0.01, min(spread, 0.02))
    if regime_label in ("HIGH_VOL", "ADVERSARIAL", "VOL_EVENT"): base = min(0.04, base * 2.0)
    return round(base, 2)

def _execute_fok_aggressive(token_id: str, order_side: str, target_size: float, fair_value: float = 0.50, max_slip: float = 0.06, signal_ask: float = 0.0, signal_bid: float = 0.0, is_exit: bool = False) -> Dict[str, Any]:
    global WINDOW_BANKROLL
    t0 = time.time()
    if client is None: return {"ok": False, "order_id": "", "exec_ms": 0, "used_limit": 0.0, "used_size": 0.0, "error": "No client", "state": "CONFIRMED_MISS"}
    order_side = order_side.upper()
    is_buy = (order_side == "BUY")

    if is_locked(token_id, order_side) and not is_exit:
        return {"ok": False, "order_id": "", "exec_ms": 0, "used_limit": 0.0, "used_size": 0.0, "error": "SKIP_LOCKED", "state": "SKIP_LOCKED"}

    cd_key = (token_id, order_side)
    snap = BOOK_CACHE.get(token_id)
    _used_signal_book = False

    if snap is None or (snap.best_bid == 0.0 and snap.best_ask == 1.0):
        if signal_ask > 0.01 and signal_ask < 0.96:
            bid, ask = float(signal_bid), float(signal_ask)
            _used_signal_book = True
        else: return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000), "used_limit": 0.0, "used_size": 0.0, "error": "NO_BOOK_DATA", "state": "CONFIRMED_MISS"}
    else: bid, ask = float(snap.best_bid), float(snap.best_ask)

    spread = max(0.0, ask - bid)
    _book_broken_buy = (is_buy and ask > 0.95 and spread > 0.90)
    _book_broken_sell = (not is_buy and bid < 0.05 and spread > 0.90)

    if (_book_broken_buy or _book_broken_sell) and not _used_signal_book:
        _sig_spread = signal_ask - signal_bid
        _sig_sane = (signal_ask > 0.01 and signal_ask < 0.96 and signal_bid >= 0.0 and _sig_spread < 0.90)
        if _sig_sane:
            bid, ask = float(signal_bid), float(signal_ask)
            spread = max(0.0, ask - bid)
            _used_signal_book = True
        elif is_exit:
            _sigma = max(0.0005, float(STATE.sigma_1m))
            _spread_est = max(0.0, float(signal_ask) - float(signal_bid))
            if _spread_est < 0.005 or _spread_est > 0.20: _spread_est = min(0.06, 2.0 * _sigma)
            _T = max(1.0, float(STATE.sec_remaining))
            _p_adv = min(0.95, max(0.10, 1.0 - (_T / 300.0)))
            _regime_mult = {"CALM": 0.6, "NORMAL": 1.0, "TRANSITION": 1.2, "HIGH_VOL": 1.5, "VOL_EVENT": 1.8}.get(REGIME.label, 1.0)
            _offset = (0.5 * _spread_est + 15.0 * _sigma + 0.03 * _p_adv) * _regime_mult
            _offset = max(0.01, min(0.10, _offset))
            if not is_buy: _gtc_limit = round(max(FOK_HARD_FLOOR, fair_value - _offset), 2)
            else: _gtc_limit = round(min(FOK_HARD_CAP, fair_value + _offset), 2)
            _gtc_size = round(float(max(0.0001, target_size)), 6)
            if _gtc_size * _gtc_limit < 1.0: _gtc_size = max(_gtc_size, float(math.ceil(1.0 / max(0.01, _gtc_limit))))
            
            _repost_key = (token_id, order_side)
            _repost_count = GTC_EXIT_ATTEMPTS.get(_repost_key, 0)
            if _repost_count >= GTC_EXIT_MAX:
                EXIT_COOLDOWN[_repost_key] = ms_now() + 2000
                return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000), "used_limit": 0.0, "used_size": 0.0, "error": "GTC_BUDGET_EXHAUSTED", "state": "CONFIRMED_MISS"}
            GTC_EXIT_ATTEMPTS[_repost_key] = _repost_count + 1
            try:
                _prime_tick_size_cache(token_id)
                _gtc_side_const = SELL if not is_buy else BUY
                _gtc_args = OrderArgs(price=_gtc_limit, size=_gtc_size, side=_gtc_side_const, token_id=token_id)
                _gtc_signed = client.create_order(_gtc_args, options=PartialCreateOrderOptions(tick_size="0.01", neg_risk=False))
                _gtc_resp = client.post_order(_gtc_signed, OrderType.GTC)
                _gtc_oid = _gtc_resp.get("orderID", "") if isinstance(_gtc_resp, dict) else ""
                if bool(_gtc_resp) and _gtc_resp.get("success", False) and _gtc_oid:
                    _exit_u = compute_exit_urgency(float(STATE.sec_remaining))
                    _ttl = exit_ttl_ms(_exit_u)
                    time.sleep(_ttl / 1000.0)
                    _gtc_info = client.get_order(_gtc_oid)
                    _gtc_matched = math.floor(min(float(_gtc_info.get("size_matched", 0) or 0) if _gtc_info else 0.0, _gtc_size) * 10000) / 10000.0
                    try: client.cancel(_gtc_oid)
                    except Exception: pass
                    if _gtc_matched > 0: return {"ok": True, "order_id": _gtc_oid, "exec_ms": int((time.time() - t0) * 1000), "used_limit": _gtc_limit, "used_size": _gtc_matched, "matched_size": _gtc_matched, "state": "FILL"}
                    else: return {"ok": False, "order_id": _gtc_oid, "exec_ms": int((time.time() - t0) * 1000), "used_limit": _gtc_limit, "used_size": 0.0, "error": "GTC_NO_FILL", "state": "CONFIRMED_MISS"}
                else: return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000), "used_limit": _gtc_limit, "used_size": 0.0, "error": "GTC_POST_FAIL", "state": "CONFIRMED_MISS"}
            except Exception as _gtc_exc: return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000), "used_limit": 0.0, "used_size": 0.0, "error": str(_gtc_exc), "state": "CONFIRMED_MISS"}
        else: return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000), "used_limit": 0.0, "used_size": 0.0, "error": f"BOOK_SANITY ask={ask:.2f}", "state": "CONFIRMED_MISS"}

    size = round(float(max(0.0001, target_size)), 6)
    CLOB_MIN_SHARES = 5.0
    if is_buy:
        _min_shares = max(CLOB_MIN_SHARES, float(math.ceil(1.0 / max(0.01, fair_value))))
        if size < _min_shares: size = _min_shares
    else:
        if size < CLOB_MIN_SHARES: return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000), "used_limit": 0.0, "used_size": 0.0, "error": f"SIZE_BELOW_CLOB_MIN: {size:.2f}<{CLOB_MIN_SHARES}", "state": "CONFIRMED_MISS"}

    _dyn_buf = dynamic_fok_buffer(spread, REGIME.label)
    # FIX: Hard cap dynamic buffer to prevent sweeping thin books
    _dyn_buf = min(_dyn_buf, max(0.01, spread * 1.5)) 

    if is_buy:
        limit = ask + _dyn_buf
        
        # FIX: Volatility-Scaled Slippage Cap
        _regime_lbl = getattr(REGIME, 'label', 'NORMAL')
        _sigma_rt = float(STATE.sigma_1m) if hasattr(STATE, 'sigma_1m') else 0.001
        
        if _regime_lbl == "CALM":
            _hard_slip_cap = 0.015  # 1.5 cents max in calm markets
        elif _regime_lbl in ("HIGH_VOL", "VOL_EVENT", "ADVERSARIAL"):
            _hard_slip_cap = min(0.04, max(0.025, _sigma_rt * 20.0)) # Up to 4 cents if vol is screaming
        else:
            _hard_slip_cap = 0.02   # 2 cents normal
            
        _effective_slip = min(max_slip, _hard_slip_cap)
        fv_cap = fair_value + _effective_slip
        
        limit = min(limit, fv_cap, FOK_HARD_CAP)
        limit = round(min(0.99, max(0.01, limit)), 2)

        if limit < ask and not is_exit: return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000), "used_limit": limit, "used_size": size, "error": f"NON_MARKETABLE limit={limit:.2f}<ask={ask:.2f}", "state": "CONFIRMED_MISS"}
    else:
        limit = bid - _dyn_buf
        fv_floor = fair_value - max_slip
        limit = max(limit, fv_floor, FOK_HARD_FLOOR)
        limit = round(max(0.01, min(0.99, limit)), 2)

        if limit > bid and not is_exit: return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000), "used_limit": limit, "used_size": size, "error": f"NON_MARKETABLE limit={limit:.2f}>bid={bid:.2f}", "state": "CONFIRMED_MISS"}

    if is_buy and (size * limit < 1.0): size = max(size, float(math.ceil(1.0 / max(0.01, limit))))

    try:
        _est_notional = float(size) * float(limit)
        _avail = max(0.0, float(WINDOW_BANKROLL))
        _buffer = 2.0
        if is_buy and _avail <= 0.0:
            ORDER_COOLDOWN[cd_key] = ms_now() + 3000
            return {"ok": False, "order_id": "", "exec_ms": 0, "used_limit": limit, "used_size": float(size), "error": "WINDOW_BUDGET_EXHAUSTED", "state": "CONFIRMED_MISS"}
        if is_buy and _avail > 0 and _est_notional > (_avail - _buffer):
            ORDER_COOLDOWN[cd_key] = ms_now() + 3000
            return {"ok": False, "order_id": "", "exec_ms": 0, "used_limit": limit, "used_size": float(size), "error": "PREFLIGHT_BALANCE", "state": "CONFIRMED_MISS"}
    except Exception: pass

    _prime_tick_size_cache(token_id)

    if is_exit and not is_buy and _used_signal_book and not book_is_sane(bid, ask):
        _exit_u = compute_exit_urgency(float(STATE.sec_remaining))
        _gtc_ttl = exit_ttl_ms(_exit_u)
        try:
            args = OrderArgs(price=limit, size=size, side=SELL, token_id=token_id)
            signed = client.create_order(args, options=PartialCreateOrderOptions(tick_size="0.01", neg_risk=False))
            resp = client.post_order(signed, OrderType.GTC)
            oid = resp.get("orderID", "") if isinstance(resp, dict) else ""
            if bool(resp) and resp.get("success", False) and oid:
                time.sleep(_gtc_ttl / 1000.0)
                _info = client.get_order(oid)
                _matched = math.floor(min(float(_info.get("size_matched", 0) or 0) if _info else 0.0, size) * 10000) / 10000.0
                try: client.cancel(oid)
                except Exception: pass
                if _matched > 0: return {"ok": True, "order_id": oid, "exec_ms": int((time.time() - t0) * 1000), "used_limit": limit, "used_size": _matched, "matched_size": _matched, "state": "FILL"}
                else: return {"ok": False, "order_id": oid, "exec_ms": int((time.time() - t0) * 1000), "used_limit": limit, "used_size": 0.0, "error": "GTC_GATE_NO_FILL", "state": "CONFIRMED_MISS"}
            else: return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000), "used_limit": limit, "used_size": 0.0, "error": "GTC_GATE_POST_FAIL", "state": "CONFIRMED_MISS"}
        except Exception as _gex: return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000), "used_limit": limit, "used_size": 0.0, "error": str(_gex), "state": "CONFIRMED_MISS"}

    try:
        side_const = BUY if is_buy else SELL
        args = OrderArgs(price=limit, size=size, side=side_const, token_id=token_id)
        signed = client.create_order(args, options=PartialCreateOrderOptions(tick_size="0.01", neg_risk=False))
        lock_side(token_id, order_side, ttl_ms=2000)
        resp = client.post_order(signed, OrderType.FAK)

        accepted = bool(resp) and resp.get("success", False)
        exec_ms = int((time.time() - t0) * 1000)
        oid = resp.get("orderID", "") if isinstance(resp, dict) else ""

        matched_size = 0.0
        fill_price = limit
        if accepted and oid:
            try:
                time.sleep(0.15)
                order_info = client.get_order(oid)
                if order_info:
                    matched_size = math.floor(min(float(order_info.get("size_matched", 0) or 0), size) * 10000) / 10000.0
                    _avg_price = order_info.get("associate_trades", [])
                    if _avg_price and isinstance(_avg_price, list) and len(_avg_price) > 0:
                        try:
                            _prices = [float(t.get("price", limit)) for t in _avg_price]
                            _sizes = [float(t.get("size", 1)) for t in _avg_price]
                            _total = sum(_sizes)
                            if _total > 0: fill_price = sum(p * s for p, s in zip(_prices, _sizes)) / _total
                        except Exception: pass
            except Exception as ve: lock_side(token_id, order_side, ttl_ms=3000)

        ok = matched_size > 0
        if ok: return {"ok": True, "order_id": oid, "exec_ms": exec_ms, "used_limit": fill_price, "used_size": matched_size, "error": "", "state": "FILLED"}

        last_err = str(resp.get("errorMsg", resp) if isinstance(resp, dict) else resp)
        return {"ok": False, "order_id": oid, "exec_ms": exec_ms, "used_limit": limit, "used_size": 0.0, "error": last_err or "NO_FILL", "state": "CONFIRMED_MISS"}

    except Exception as e:
        err_str = str(e).lower()
        exec_ms = int((time.time() - t0) * 1000)
        if any(x in err_str for x in ["status_code=400", "not enough balance", "allowance", "status_code=401", "status_code=403", "status_code=422"]):
            lock_side(token_id, order_side, ttl_ms=800)
            return {"ok": False, "order_id": "", "exec_ms": exec_ms, "used_limit": limit, "used_size": 0.0, "error": f"EXCEPTION: {e}", "state": "CONFIRMED_MISS"}
        lock_side(token_id, order_side, ttl_ms=4000)
        return {"ok": False, "order_id": "", "exec_ms": exec_ms, "used_limit": limit, "used_size": 0.0, "error": f"EXCEPTION: {e}", "state": "PENDING_ACK"}


def execute_order_sync(payload: Dict[str, Any]) -> Dict[str, Any]:
    TICK_BUFFER   = 0.01
    t0            = time.time()
    token_id      = payload["token_id"]
    size          = float(payload["size"])
    original_edge = float(payload.get("edge", 0.0))
    retry_idx     = payload.get("retry_idx", 0)
    order_side_str = payload.get("order_side", "BUY")
    is_sell        = (order_side_str == "SELL")

    try:
        if client is None: return {"ok": False, "resp": {"error": "No client"}, "exec_ms": 0, "retry_idx": retry_idx}
        live_snap = BOOK_CACHE.get(token_id)
        if is_sell:
            signal_bid = round(payload["price"] + TICK_BUFFER, 2)
            if live_snap is not None:
                live_bid = live_snap.best_bid
                if signal_bid - live_bid > 0.10: return {"ok": False, "exec_ms": int((time.time()-t0)*1000), "resp": {"error": f"EXIT_SLIPPAGE: {signal_bid - live_bid:.2f}"}, "retry_idx": retry_idx}
                limit_price = round(live_bid - TICK_BUFFER, 2)
            else: limit_price = round(payload["price"], 2)
            limit_price = max(0.01, limit_price)
        else:
            signal_ask = round(payload["price"] - TICK_BUFFER, 2)
            if live_snap is not None:
                live_ask = live_snap.best_ask
                if live_ask - signal_ask > max(0.03, original_edge): return {"ok": False, "exec_ms": int((time.time()-t0)*1000), "resp": {"error": f"EDGE_ERODED: slip={live_ask - signal_ask:.2f}"}, "retry_idx": retry_idx}
                limit_price = round(live_ask + TICK_BUFFER, 2)
            else: limit_price = round(payload["price"], 2)
            limit_price = min(0.99, limit_price)

        _prime_tick_size_cache(token_id)
        side_const = SELL if is_sell else BUY
        args   = OrderArgs(price=limit_price, size=size, side=side_const, token_id=token_id)
        signed = client.create_order(args, options=PartialCreateOrderOptions(tick_size="0.01", neg_risk=False))
        resp = client.post_order(signed, OrderType.FAK)

        exec_ms = int((time.time() - t0) * 1000)
        accepted = bool(resp) and resp.get("success", False)
        oid = resp.get("orderID", "") if isinstance(resp, dict) else ""

        matched_size = 0.0
        if accepted and oid:
            try:
                time.sleep(0.15)
                order_info = client.get_order(oid)
                if order_info: matched_size = min(float(order_info.get("size_matched", 0) or 0), size)
            except Exception: pass

        return {"ok": matched_size > 0, "resp": resp, "exec_ms": exec_ms, "retry_idx": retry_idx, "exec_price": limit_price, "matched_size": matched_size}
    except Exception as e: return {"ok": False, "resp": {"error": str(e)}, "exec_ms": int((time.time()-t0)*1000), "retry_idx": retry_idx}


async def process_order_task(payload: dict) -> None:
    """Concurrent worker task that handles a single order."""
    global TRADED_UP_THIS_WINDOW, TRADED_DN_THIS_WINDOW, TRADES_THIS_WINDOW, WINDOW_BANKROLL
    global ENTRY_P_CONE, ENTRY_P_MARKET, ENTRY_T_SEC

    action = payload.get("action", "ORDER")
    order_side = payload.get("order_side", "BUY").upper()
    token_id = payload["token_id"]
    mode = str(payload.get("mode", "") or "").lower()
    _is_exit = mode in ("exit", "trim", "hedge", "reduce", "close")
    side = payload.get("side", "UP")

    _spread_tier_exec = str(payload.get("spread_tier", "taker_allowed"))
    _lm_route = "MAKER" if _spread_tier_exec in ("maker_only", "maker_preferred") else "TAKER"

    oracle_src = "rtds" if RTDS.is_fresh(RTDS_FRESH_MS) else "coinbase"
    oracle_px  = RTDS.price if RTDS.is_fresh(RTDS_FRESH_MS) else STATE.btc_price

    def pos_for_token(tid: str):
        return POS_UP if tid == UP_TOKEN_ID else POS_DOWN if tid == DOWN_TOKEN_ID else None

    try:
        if action in ("ORDER", "PAIR_BUY", "PAIR_REBALANCE"):
            if client is None: return

            limit_price = round(float(payload["price"]), 2)
            size = float(payload["size"])
            now_ms_exec = ms_now()
            cd_key = (token_id, order_side)
            _exec_method = "UNKNOWN"

            if is_rate_limited(): return

            _fire_id = payload.get("fire_id", "")
            if _fire_id:
                _attempt_key = (_fire_id, token_id, order_side)
                async with STATE_LOCK:
                    if _attempt_key in EXEC_ATTEMPTED: return
                    EXEC_ATTEMPTED.add(_attempt_key)
                    if len(EXEC_ATTEMPTED) > EXEC_ATTEMPTED_MAX:
                        _sorted = sorted(EXEC_ATTEMPTED)
                        EXEC_ATTEMPTED.difference_update(_sorted[:len(_sorted)//2])

            if check_emergency_exit() and not _is_exit:
                exec_unlock()
                return

            if not _is_exit:
                if now_ms_exec < ORDER_COOLDOWN.get(cd_key, 0):
                    exec_unlock()
                    return
            else:
                async with STATE_LOCK:
                    if now_ms_exec < EXIT_COOLDOWN.get(cd_key, 0): return
                    EXIT_COOLDOWN[cd_key] = now_ms_exec + 2000
            
            if size <= 0:
                if not _is_exit: exec_unlock()
                return

            if order_side == "BUY":
                _max_buy_qty = math.floor(MAX_SPEND_PER_ORDER_USD / max(0.01, limit_price))
                if _max_buy_qty < 1:
                    exec_unlock()
                    return
                if size > _max_buy_qty: size = float(_max_buy_qty)
            else:
                # Local inventory fetch logic moved to thread pool section

                size = math.floor(size * 1000) / 1000.0
                if size <= 0.0: return

            loop = asyncio.get_running_loop()
            p_cone = float(payload.get("p_cone", 0.5) or 0.5)
            z_val = float(payload.get("z", 0.0) or 0.0)
            edge_val = float(payload.get("edge", 0.0) or 0.0)
            max_slip = min(0.25, max(0.06, edge_val)) if edge_val >= 0.10 else max(0.03, min(0.10, edge_val))

            _fresh_oracle = RTDS.price if RTDS.is_fresh(RTDS_FRESH_MS) else STATE.btc_price
            if (np.isfinite(_fresh_oracle) and _fresh_oracle > 0 and np.isfinite(STATE.open_price) and STATE.open_price > 0 and STATE.sec_remaining > 0 and STATE.sigma_1m > 0):
                from oracle_engine import cone_p_and_z
                p_cone, _ = cone_p_and_z(_fresh_oracle, STATE.open_price, STATE.sec_remaining, STATE.sigma_1m)
            
            if token_id == UP_TOKEN_ID: token_fair_value, token_prob = p_cone, p_cone
            elif token_id == DOWN_TOKEN_ID: token_fair_value, token_prob = 1.0 - p_cone, 1.0 - p_cone
            else: token_fair_value = token_prob = p_cone if side == "UP" else (1.0 - p_cone)

            if action == "ORDER" and order_side == "SELL" and not _is_exit:
                if not _is_exit: exec_unlock()
                return

            _fok_hard_cap = round(min(0.99, token_fair_value + max_slip), 2)

            if token_prob < 0.03 or token_prob > 0.97:
                if not _is_exit: exec_unlock()
                return

            if action == "ORDER":
                _snap_up_exec = BOOK_CACHE.get(UP_TOKEN_ID)
                _snap_dn_exec = BOOK_CACHE.get(DOWN_TOKEN_ID)
                if _snap_up_exec and _snap_dn_exec:
                    _lag50 = float(LAG_ADAPTIVE.p50_ms()) if hasattr(LAG_ADAPTIVE, 'p50_ms') else 400.0
                    _lag10 = float(LAG_ADAPTIVE.p10_ms(side)) if hasattr(LAG_ADAPTIVE, 'p10_ms') else 200.0
                    _sigma_rt = float(STATE.sigma_1m) if STATE.sigma_1m > 1e-10 else 0.001
                    _exec_side_snap = _snap_up_exec if side == "UP" else _snap_dn_exec
                    _dyn_buf_result = micro_edge_buffer(
                        spread=float(_exec_side_snap.spread), bid_sz=float(getattr(_exec_side_snap, 'bid_size', 50.0)),
                        ask_sz=float(getattr(_exec_side_snap, 'ask_size', 50.0)), sigma=_sigma_rt, lag_p50_ms=_lag50,
                        lag_p10_ms=_lag10, trade_vel=float(len(getattr(FLOW, '_fills', [])) / 30.0) if hasattr(FLOW, '_fills') else 1.0,
                        flow_bias=float(FLOW.imbalance(up_price=0.5, down_price=0.5)) if FLOW else 0.0,
                    )
                    _dyn_buf = _dyn_buf_result["buffer"]
                    _lm_floor = 0.0
                    _lm_allow = True
                    if LEAK_MODEL and hasattr(LEAK_MODEL, "should_allow"):
                        _lm_chk_allow, _lm_chk = LEAK_MODEL.should_allow(
                            edge_val, lag50_ms=_lag50, sigma=_sigma_rt, regime=str(getattr(REGIME, "label", "NORMAL")), route=_lm_route, side=side
                        )
                        _lm_allow, _lm_floor = _lm_chk_allow, _lm_chk.get("q_floor", 0.0)
                    _min_edge_for_check = max(float(_dyn_buf), float(_lm_floor))
                    
                    if (edge_val < _min_edge_for_check or not _lm_allow) and not _is_exit:
                        ORDER_COOLDOWN[cd_key] = ms_now() + 2000
                        with open("logs/trades.csv", "a", newline="") as f:
                            csv.writer(f).writerow([
                                ms_now(), side, token_id, limit_price, size, "EXEC_EDGE_BLOCK", "", 0, 0,
                                payload.get("edge", ""), payload.get("p_cone", ""), payload.get("z", ""), payload.get("sigma_1m", ""),
                                oracle_src, f"{oracle_px:.2f}" if np.isfinite(oracle_px) else "", "FOK_AGGRESSIVE", f"edge_now={edge_val:.4f}",
                            ])
                        exec_unlock()
                        return

                if order_side == "SELL":
                    # FIX: Emergency Exit GTC Clearing
                    _pm_reason = payload.get("exit_reason", "")
                    _STOP_LOSS_REASONS = {"prob_stop_loss", "endgame_strong_loss", "trailing_stop", "gamma_danger", "sis_exit"}
                    if _pm_reason in _STOP_LOSS_REASONS:
                        try:
                            _open_orders = await loop.run_in_executor(thread_pool, client.get_orders)
                            for _o in (_open_orders or []):
                                if _o.get("asset_id") == token_id or _o.get("token_id") == token_id:
                                    await loop.run_in_executor(thread_pool, client.cancel, _o.get("orderID"))
                            await asyncio.sleep(0.3)
                        except Exception as _ce: pass

                    # FIX: Authoritative sizing AFTER async waits
                    _exch_inv = await loop.run_in_executor(thread_pool, get_exchange_conditional_balance, token_id)
                    _reserved = await loop.run_in_executor(thread_pool, get_reserved_conditional, token_id)
                    _pos_local = pos_for_token(token_id)
                    _local_inv = float(_pos_local.inventory) if _pos_local else 0.0
                    _available = max(0.0, _exch_inv - _reserved - EXIT_BAL_EPS)

                    _sis_frac = float(payload.get("sis_frac", 1.0) or 1.0)
                    _desired = _available * _sis_frac
                    size = floor3(min(size, _desired, _local_inv, _available))

                    MIN_EXIT_SHARES = 5
                    if _local_inv < MIN_EXIT_SHARES:
                        _T_now = float(STATE.sec_remaining)
                        if _T_now < 5.0 and _pos_local and _pos_local.inventory > 0:
                            _dust = float(_pos_local.inventory)
                            _pos_local.apply_fill(side=Side.SELL, qty=_dust, price=limit_price, ts_ms=ms_now(), fee_per_share=0.0, meta={"path": "DUST_SETTLE"})
                        EXIT_COOLDOWN[cd_key] = ms_now() + 2000
                        return

                    if size < MIN_EXIT_SHARES:
                        if _available >= MIN_EXIT_SHARES: size = floor3(min(MIN_EXIT_SHARES, _available, _local_inv))
                        elif _reserved > 0.01:
                            EXIT_COOLDOWN[cd_key] = ms_now() + 2000
                            return
                    if size <= 0.0:
                        EXIT_COOLDOWN[cd_key] = ms_now() + 500
                        return

                order_cost = size * limit_price
                avail_bal = WINDOW_BANKROLL if WINDOW_BANKROLL > 0 else 0.0
                if order_side == "BUY" and avail_bal <= 0:
                    exec_unlock()
                    return
                if avail_bal > 0 and order_cost > avail_bal:
                    size = max(1.0, float(math.floor(avail_bal / max(0.01, limit_price))))
                    if size * limit_price > avail_bal or size < 1:
                        exec_unlock()
                        return

                _mode_exec = str(payload.get("mode", "") or "").lower()
                _fok_fv, _fok_slip = token_fair_value, max_slip
                if _mode_exec == "mispricing":
                    sniper_cap = round(float(payload.get("price", 1.0)), 2)
                    _fok_slip = max(0.01, sniper_cap - _fok_fv)

                _exec_regime = str(payload.get("regime", "NORMAL"))
                _exec_snap = BOOK_CACHE.get(token_id)
                _exec_spread = float(_exec_snap.best_ask - _exec_snap.best_bid) if _exec_snap else 0.10
                _exec_T = float(STATE.sec_remaining)
                _spread_tier_exec = str(payload.get("spread_tier", "taker_allowed"))

                # FIX: Depth Guard Routing
                _exec_ask_sz = float(getattr(_exec_snap, "ask_size", 0.0)) if _exec_snap else 0.0
                _exec_bid_sz = float(getattr(_exec_snap, "bid_size", 0.0)) if _exec_snap else 0.0
                _target_depth = _exec_ask_sz if order_side == "BUY" else _exec_bid_sz

                _use_adaptive = (
                    ADAPTIVE_EXEC is not None
                    and order_side == "BUY"
                    and _mode_exec not in ("exit", "trim", "hedge")
                    and _exec_regime in ("CALM", "NORMAL", "HIGH_VOL")
                    and _exec_T > 35.0
                    and _exec_spread <= 0.20
                )

                if _spread_tier_exec == "taker_allowed":
                    if size > (_target_depth * 0.8) and _target_depth > 0:
                        _use_adaptive = True
                    else:
                        _use_adaptive = False
                elif _spread_tier_exec in ("maker_only", "maker_preferred"):
                    pass # Keep _use_adaptive state

                if _use_adaptive:
                    _sig_edge_rt = float(payload.get("edge", 0.0) or 0.0)
                    _sigma_rt = float(STATE.sigma_1m) if STATE.sigma_1m > 1e-10 else 0.001
                    if (_sig_edge_rt / max(0.001, _sigma_rt * 50.0)) < 1.0: _use_adaptive = False
                    if _exec_T < 60 and _exec_spread < 0.06: _use_adaptive = False

                fok_result, _ae_result = None, None

                if _use_adaptive:
                    _ms = MicroSnapshot()
                    if _exec_snap:
                        _ms.best_bid, _ms.best_ask = float(_exec_snap.best_bid), float(_exec_snap.best_ask)
                        _ms.best_bid_sz, _ms.best_ask_sz = float(getattr(_exec_snap, 'bid_size', 50.0)), float(getattr(_exec_snap, 'ask_size', 50.0))
                    _ms.sigma = float(STATE.sigma_1m)
                    _ms.lag_p10_ms = float(LAG_ADAPTIVE.p10_ms(side)) if hasattr(LAG_ADAPTIVE, 'p10_ms') else 500.0
                    _ms.trade_vel = float(len(getattr(FLOW, '_fills', [])) / 30.0) if hasattr(FLOW, '_fills') else 1.0
                    _ps = POLY_STATE.get(token_id, {}) if token_id else {}
                    _mid = 0.5 * (_ms.best_bid + _ms.best_ask)
                    _bids_l2 = list(_ps.get("bids_l2", [])) or [(_ms.best_bid, _ms.best_bid_sz)]
                    _asks_l2 = list(_ps.get("asks_l2", [])) or [(_ms.best_ask, _ms.best_ask_sz)]
                    _ms.imbalance = float(depth_decayed_imbalance(_bids_l2, _asks_l2, _mid, k=IMBALANCE_DECAY_K))
                    _ms.flow_bias = float(FLOW.imbalance(up_price=_ms.best_bid if side == "UP" else 0.5, down_price=_ms.best_bid if side == "DOWN" else 0.5) if FLOW else 0.0)
                    _ms.strike_price = float(STATE.open_price) if np.isfinite(STATE.open_price) else 0.5
                    _ms.current_btc_price = float(STATE.btc_price) if np.isfinite(STATE.btc_price) else 0.0
                    _ms.net_position = float(POS_UP.inventory if side == "UP" else POS_DOWN.inventory)
                    _ms.max_inventory = 200.0

                    try:
                        _ae_result = await loop.run_in_executor(thread_pool, ADAPTIVE_EXEC.execute, side, token_id, float(payload.get("p_cone", 0.5)), float(payload.get("z", 0.0)), size, _ms, 1.0, order_side)
                        fok_result = {"ok": _ae_result.filled, "order_id": _ae_result.order_id, "exec_ms": _ae_result.exec_ms, "used_limit": _ae_result.fill_price, "used_size": _ae_result.fill_size, "error": _ae_result.error, "state": "FILLED" if _ae_result.filled else "CONFIRMED_MISS"}
                        _exec_method = f"ADAPTIVE_{_ae_result.method}"
                    except Exception as e:
                        _use_adaptive = False

                if not _use_adaptive and fok_result is None:
                    _exec_method = "FAK_AGGRESSIVE"
                    _sig_ask, _sig_bid = float(payload.get("signal_ask", 0.0) or 0.0), float(payload.get("signal_bid", 0.0) or 0.0)
                    fok_result = await loop.run_in_executor(thread_pool, _execute_fok_aggressive, token_id, order_side, size, _fok_fv, _fok_slip, _sig_ask, _sig_bid, _is_exit)

                ok = fok_result["ok"]
                trade_state = fok_result.get("state", "FILLED" if ok else "CONFIRMED_MISS")
                
                if not ok:
                    err_text = str(fok_result.get("error", "")).lower()
                    if ("not enough balance" in err_text or "allowance" in err_text) and order_side == "SELL":
                        _exch_inv = get_exchange_conditional_balance(token_id)
                        _pos_local = pos_for_token(token_id)
                        if _pos_local is not None:
                            _ghost = float(_pos_local.inventory) - _exch_inv
                            if _ghost > 0.001: _pos_local.apply_fill(side=Side.SELL, qty=_ghost, price=limit_price, ts_ms=ms_now(), fee_per_share=0.0, meta={"path": "INV_SYNC"})
                        EXIT_COOLDOWN[cd_key] = ms_now() + 500
                    elif _is_exit: EXIT_COOLDOWN[cd_key] = ms_now() + 250
                    elif trade_state in ("EXEC_EDGE_BLOCK", "INSUFFICIENT_BALANCE", "NON_MARKETABLE") or "NON_MARKETABLE" in err_text.upper():
                        ORDER_COOLDOWN[cd_key] = ms_now() + 2000

                if ok:
                    _fill_price_attr = float(fok_result.get("used_limit", limit_price))
                    _fill_size_actual = float(fok_result.get("used_size", size))

                    async with STATE_LOCK:
                        if order_side == "BUY": WINDOW_BANKROLL = max(0.0, WINDOW_BANKROLL - (_fill_size_actual * _fill_price_attr))

                    _sig_edge_attr = float(payload.get("edge", 0.0) or 0.0)
                    _p_side_attr = float(payload.get("p_cone", 0.5)) if side == "UP" else (1.0 - float(payload.get("p_cone", 0.5)))
                    _realized_edge_attr = _p_side_attr - (_fill_price_attr + float(fee_per_share(_fill_price_attr)))
                    _edge_leak_attr = (_sig_edge_attr - _realized_edge_attr) if _sig_edge_attr > 1e-9 else 0.0
                    _edge_leak_ratio_attr = (_edge_leak_attr / _sig_edge_attr) if _sig_edge_attr > 1e-9 else 0.0
                    _ev_cols = [round(_realized_edge_attr, 5), round(_edge_leak_attr * 10000, 1), round(_edge_leak_ratio_attr, 4)]
                    
                    if not _is_exit:
                        _lm_diag = LEAK_MODEL.record(
                            _sig_edge_attr, _realized_edge_attr,
                            lag50_ms=float(payload.get("lag50_ms", 0.0) or 0.0), sigma=float(payload.get("sigma_1m", 0.0) or 0.0),
                            spread=max(0.0, float(payload.get("signal_ask", 0.5)) - float(payload.get("signal_bid", 0.5))), flow=float(payload.get("flow", 0.0) or 0.0),
                            regime=str(payload.get("regime", "NORMAL")), route=_lm_route, side=side, ts_ms=ms_now()
                        )
                else: _ev_cols = ["", "", ""]

                with open("logs/trades.csv", "a", newline="") as f:
                    csv.writer(f).writerow([ms_now(), side, token_id, fok_result.get("used_limit", limit_price), fok_result.get("used_size", size), trade_state, fok_result.get("order_id", ""), fok_result["exec_ms"], 0, payload.get("edge", ""), payload.get("p_cone", ""), payload.get("z", ""), payload.get("sigma_1m", ""), oracle_src, f"{oracle_px:.2f}" if np.isfinite(oracle_px) else "", _exec_method, fok_result.get("error", ""), payload.get("regime", ""), payload.get("edge_target", ""), payload.get("mode", "entry"), payload.get("flow", ""), *_ev_cols])

                if ok:
                    async with STATE_LOCK:
                        if side == "UP": TRADED_UP_THIS_WINDOW = True
                        elif side == "DOWN": TRADED_DN_THIS_WINDOW = True
                        TRADES_THIS_WINDOW += 1

                    if _mode_exec not in ("exit", "trim", "hedge") and side in ("UP", "DOWN"):
                        if ENTRY_P_CONE.get(side) is None:
                            ENTRY_P_CONE[side] = float(payload.get("p_cone", 0.5) or 0.5)
                            ENTRY_T_SEC[side] = float(STATE.sec_remaining)
                            _fill_snap = BOOK_CACHE.get(token_id)
                            ENTRY_P_MARKET[side] = (float(_fill_snap.best_bid) + float(_fill_snap.best_ask)) / 2 if _fill_snap else float(fok_result.get("used_limit", 0.5))
                            ENTRY_EDGE[side] = float(payload.get("edge", 0.0) or 0.0)
                            ENTRY_Z_EMA[side] = float(payload.get("z_ema", 0.0) or 0.0)
                            ENTRY_TS_MS[side] = int(ms_now())

                    _fill_pos = pos_for_token(token_id)
                    if _fill_pos is not None:
                        _actual_qty = _fill_size_actual
                        if order_side == "SELL":
                            _actual_qty = min(_actual_qty, float(_fill_pos.inventory))
                            if _actual_qty > 0:
                                async with STATE_LOCK: _fill_pos.apply_fill(side=Side.SELL, qty=_actual_qty, price=_fill_price_attr, ts_ms=ms_now(), fee_per_share=fee_per_share(_fill_price_attr), meta={"path": _exec_method, "token_id": token_id, "order_side": order_side})
                        else:
                            async with STATE_LOCK: _fill_pos.apply_fill(side=Side.BUY, qty=_actual_qty, price=_fill_price_attr, ts_ms=ms_now(), fee_per_share=fee_per_share(_fill_price_attr), meta={"path": _exec_method, "token_id": token_id, "order_side": order_side})

                    # FIX: Asynchronous Resting Take-Profit (0-Fee EV Capture)
                    if not _is_exit:
                        exec_unlock()
                        
                    if order_side == "BUY" and _fill_size_actual > 0:
                        _tp_target = round(min(0.98, _fill_price_attr + _sig_edge_attr + 0.12), 2)
                        async def post_take_profit(tid, qty, price):
                            try:
                                await asyncio.sleep(0.5)
                                logger.info(f"RESTING_TP: Queuing Maker SELL for {qty} shares at ${price:.2f}")
                                def _sync_post():
                                    _prime_tick_size_cache(tid)
                                    tp_args = OrderArgs(price=price, size=qty, side=SELL, token_id=tid)
                                    tp_signed = client.create_order(tp_args, options=PartialCreateOrderOptions(tick_size="0.01", neg_risk=False))
                                    client.post_order(tp_signed, OrderType.GTC)
                                await loop.run_in_executor(thread_pool, _sync_post)
                            except Exception as e: logger.debug(f"RESTING_TP_FAIL: {e}")
                        asyncio.create_task(post_take_profit(token_id, _fill_size_actual, _tp_target))
                        
                    return

            loop = asyncio.get_running_loop()
            fok_result = await loop.run_in_executor(thread_pool, execute_order_sync, payload)
            ok, resp = fok_result["ok"], fok_result["resp"]
            with open("logs/trades.csv", "a", newline="") as f:
                csv.writer(f).writerow([ms_now(), payload.get("side"), payload["token_id"], payload["price"], payload["size"], "FILLED" if ok else "MISS", resp.get("orderID") if isinstance(resp, dict) else "", fok_result["exec_ms"], fok_result.get("retry_idx", 0), payload.get("edge", ""), payload.get("p_cone", ""), payload.get("z", ""), payload.get("sigma_1m", ""), oracle_src, f"{oracle_px:.2f}" if np.isfinite(oracle_px) else ""])
            if action == "PAIR_BUY" and ok:
                _pair_pos = pos_for_token(payload["token_id"])
                if _pair_pos is not None:
                    async with STATE_LOCK: _pair_pos.apply_fill(side=Side.BUY, qty=float(payload["size"]), price=float(payload["price"]), ts_ms=ms_now(), fee_per_share=fee_per_share(float(payload["price"])), meta={"path": "PAIR_BUY", "token_id": payload["token_id"]})
            elif action == "PAIR_REBALANCE" and ok:
                _pair_pos, _exec_price = pos_for_token(payload["token_id"]), float(fok_result.get("exec_price", payload["price"]))
                _fee = fee_per_share(_exec_price)
                if _pair_pos is not None:
                    async with STATE_LOCK: _pair_pos.apply_fill(side=Side.SELL, qty=float(payload["size"]), price=_exec_price, ts_ms=ms_now(), fee_per_share=_fee, meta={"path": "PAIR_REBALANCE", "token_id": payload["token_id"]})
                _pp = get_portfolio_pos()
                record_pnl(float(payload["size"]) * (_exec_price - _fee), float(payload["size"]) * (_pp.avg_cost_up if payload["side"] == "UP" else _pp.avg_cost_dn))

    except Exception as e:
        logger.error(f"PROCESS_ORDER_TASK_ERROR: {e}", exc_info=True)
    finally:
        if not _is_exit: exec_unlock()
        execution_queue.task_done()


async def execution_loop() -> None:
    ensure_logs()
    while True:
        payload = await execution_queue.get()
        _is_exit = str(payload.get("mode", "") or "").lower() in ("exit", "trim", "hedge", "reduce", "close")
        if not _is_exit: exec_lock()
        asyncio.create_task(process_order_task(payload))


# WEBSOCKET TASKS
# ════════════════════════════════════════════════════════════════════════════

def _process_book_update(msg: dict) -> bool:
    updated = False
    try:
        tid = msg.get("asset_id")
        if not tid:
            changes = msg.get("price_changes", [])
            if changes:
                for change in changes:
                    if _process_book_update(change): updated = True
            return updated

        et = msg.get("event_type")
        if tid not in POLY_STATE:
            POLY_STATE[tid] = {"bid": 0.0, "ask": 1.0, "bid_size": 0.0, "ask_size": 0.0, "last_update": 0, "source": "ws", "bids_l2": [], "asks_l2": []}

        state, now = POLY_STATE[tid], ms_now()

        if et == "best_bid_ask":
            bid, ask = msg.get("bid_price") or msg.get("best_bid"), msg.get("ask_price") or msg.get("best_ask")
            if bid is not None: state["bid"] = float(bid)
            if ask is not None: state["ask"] = float(ask)
            bid_sz, ask_sz = msg.get("bid_size"), msg.get("ask_size")
            if bid_sz is not None: state["bid_size"] = float(bid_sz)
            if ask_sz is not None: state["ask_size"] = float(ask_sz)
            state["last_update"], state["source"], updated = now, "ws", True

        elif et == "orderbook_snapshot" or (msg.get("bids") is not None or msg.get("asks") is not None):
            bids, asks = msg.get("bids", []), msg.get("asks", [])
            try:
                if bids: state["bid"], state["bid_size"] = float(bids[0]["price"]), float(bids[0].get("size", 0))
                if asks: state["ask"], state["ask_size"] = float(asks[0]["price"]), float(asks[0].get("size", 0))
                state["bids_l2"] = [(float(x.get("price", 0.0)), float(x.get("size", 0.0))) for x in bids[:5]]
                state["asks_l2"] = [(float(x.get("price", 1.0)), float(x.get("size", 0.0))) for x in asks[:5]]
            except Exception: pass
            state["last_update"], state["source"], updated = now, "ws", True

        elif et == "price_changes":
            for change in msg.get("price_changes", []):
                side, price, size = change.get("side"), float(change.get("price", 0)), float(change.get("size", 0))
                if side == "BUY" and price == state["bid"]: state["bid_size"], state["bid"] = (size, price) if size > 0 else (0.0, 0.0)
                elif side == "SELL" and price == state["ask"]: state["ask_size"], state["ask"] = (size, price) if size > 0 else (0.0, 1.0)
                
                if side in ("BUY", "SELL"):
                    l2_list = state["bids_l2"] if side == "BUY" else state["asks_l2"]
                    updated_l2 = False
                    for i, (lvl_px, _) in enumerate(l2_list):
                        if lvl_px == price:
                            if size == 0: l2_list.pop(i)
                            else: l2_list[i] = (price, size)
                            updated_l2 = True
                            break
                    if not updated_l2 and size > 0:
                        l2_list.append((price, size))
                        l2_list.sort(key=lambda x: x[0], reverse=(side == "BUY"))

            bb, ba = msg.get("best_bid"), msg.get("best_ask")
            if bb is not None: state["bid"] = float(bb)
            if ba is not None: state["ask"] = float(ba)
            bs, as_ = msg.get("bid_size"), msg.get("ask_size")
            if bs is not None: state["bid_size"] = float(bs)
            if as_ is not None: state["ask_size"] = float(as_)
            state["last_update"], state["source"], updated = now, "ws", True

        elif et == "last_trade_price":
            price = msg.get("price")
            if price is not None:
                side = msg.get("side")
                if side not in ("BUY", "SELL"): return updated
                fill_price, fill_size = float(price), float(msg.get("size", 1.0))
                FLOW.record_fill(token_id=tid, price=fill_price, size=fill_size, ts_ms=now, is_buy=(side == "BUY"))
                RECENT_FILLS.append({"ts_ms": now, "token_id": tid, "side": side, "price": fill_price, "size": fill_size})

        if updated and tid in (UP_TOKEN_ID, DOWN_TOKEN_ID):
            BOOK_CACHE.update(tid, BookSnapshot(token_id=tid, best_bid=state["bid"], best_ask=state["ask"], spread=max(0.0, state["ask"] - state["bid"]), bid_size=state["bid_size"], ask_size=state["ask_size"], source=state["source"], last_update=state["last_update"]))
            L2_TRACKER.update(state["bid_size"], state["ask_size"], 0.0, 0.0)

    except Exception as e: logger.warning(f"WS_BOOK_ERROR: {e}")
    return updated


async def polymarket_book_task(poly_ws_url: str) -> None:
    global BOOK_RECONNECT_FLAG
    retry_delay, last_tokens = 1, (None, None)
    while True:
        try:
            async with websockets.connect(poly_ws_url, ping_interval=None, ping_timeout=None, close_timeout=10, open_timeout=15) as ws:
                logger.info("Poly CLOB WS connected.")
                retry_delay = 1
                while True:
                    if BOOK_RECONNECT_FLAG:
                        us, ds = BOOK_CACHE.get(UP_TOKEN_ID), BOOK_CACHE.get(DOWN_TOKEN_ID)
                        if us and ds and us.source == "ws" and ds.source == "ws" and (ms_now()-us.last_update) < POLY_BOOK_MAX_AGE_MS and (ms_now()-ds.last_update) < POLY_BOOK_MAX_AGE_MS:
                            BOOK_RECONNECT_FLAG = False
                        else:
                            BOOK_RECONNECT_FLAG, last_tokens = False, (None, None)
                            break
                    cur = (UP_TOKEN_ID, DOWN_TOKEN_ID)
                    if cur != last_tokens and cur[0] and cur[1]:
                        await ws.send(json.dumps({"type": "market", "assets_ids": list(cur), "custom_feature_enabled": True}))
                        logger.info(f"Subscribed: UP={cur[0][:12]}… DOWN={cur[1][:12]}…")
                        last_tokens = cur
                    try: raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        up_s, dn_s = BOOK_CACHE.get(UP_TOKEN_ID), BOOK_CACHE.get(DOWN_TOKEN_ID)
                        if (max((ms_now() - up_s.last_update) if up_s and up_s.source == "ws" else 999999, (ms_now() - dn_s.last_update) if dn_s and dn_s.source == "ws" else 999999) if (up_s or dn_s) else 999999) > 60_000:
                            logger.warning("Poly WS: no data for 60s, forcing reconnect")
                            break
                        continue
                    if not raw: continue
                    try: msg_data = json.loads(raw)
                    except ValueError: continue
                    for m in (msg_data if isinstance(msg_data, list) else [msg_data]):
                        if _process_book_update(m):
                            if LAST_BTC_MOVE_TS > 0 and LAST_BTC_DIR != 0:
                                lag = ms_now() - LAST_BTC_MOVE_TS
                                if 10 < lag < 5000: LAG_ADAPTIVE.add_lag(lag, direction=LAST_BTC_DIR)
        except Exception as e:
            logger.error(f"Poly WS error: {e}. Retry in {retry_delay}s")
            await asyncio.sleep(retry_delay)
            retry_delay, last_tokens = min(retry_delay * 2, 60), (None, None)


async def coinbase_trade_task() -> None:
    global LAST_BTC_MOVE_TS, LAST_BTC_PRICE, LAST_BTC_DIR
    url = "wss://ws-feed.exchange.coinbase.com"
    payload = {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker", "matches"]}
    while True:
        try:
            async with websockets.connect(url, ping_interval=30, ping_timeout=30) as ws:
                await ws.send(json.dumps(payload))
                logger.info("Coinbase WS connected (fallback).")
                while True:
                    data = json.loads(await ws.recv())
                    if data.get("type") not in ("ticker", "match", "last_match") or "price" not in data: continue
                    price, ts_ms = float(data["price"]), ms_now()
                    STATE.btc_price, STATE.btc_ts_ms = price, ts_ms
                    if RTDS.recv_ts_ms > 0 and RTDS.age_ms() < 5000: BASIS.update(RTDS.price, price)
                    if not np.isnan(LAST_BTC_PRICE) and abs(price - LAST_BTC_PRICE) >= 8.0:
                        LAST_BTC_MOVE_TS, LAST_BTC_DIR = ts_ms, 1 if price > LAST_BTC_PRICE else -1
                    LAST_BTC_PRICE = price
                    _update_sigma(price, ts_ms)
                    TAIL_RISK.feed_price(price, ts_ms)
                    TAIL_RISK.feed_sigma(STATE.sigma_1m)
        except Exception as e:
            logger.error(f"Coinbase WS error: {e}")
            await asyncio.sleep(1)


# ════════════════════════════════════════════════════════════════════════════
# WINDOW TIMING & MARKET DISCOVERY
# ════════════════════════════════════════════════════════════════════════════

def get_window_start_unix(offset: int = 0) -> int:
    now = datetime.now(POLYMARKET_TIMEZONE)
    ws_local = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0) + timedelta(minutes=5 * offset)
    return int(ws_local.timestamp())

def get_current_window_times():
    ws = datetime.fromtimestamp(get_window_start_unix(0), tz=POLYMARKET_TIMEZONE)
    return ws, ws + timedelta(minutes=5)

def fetch_crypto_price(window_start_dt: datetime, window_end_dt: datetime) -> Optional[dict]:
    params = {"symbol": "BTC", "eventStartTime": window_start_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "variant": "fiveminute", "endDate": window_end_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        resp = requests.get(CRYPTO_PRICE_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data if "openPrice" in data else None
    except Exception as e: return None

def fetch_price_from_candle_api(window_ts: int) -> Optional[float]:
    try:
        resp = requests.get(CANDLE_API_URL, params={"symbol": "BTC", "interval": "5m", "limit": 2, "endTime": (window_ts + 300) * 1000 - 1}, timeout=5)
        resp.raise_for_status()
        for c in resp.json().get("candles", []):
            if c["time"] == window_ts: return float(c["open"])
    except Exception: pass
    return None

def _auto_discover_market_sync() -> Optional[dict]:
    logger.info("Discovering active BTC 5-min market…")
    url, params, now = "https://gamma-api.polymarket.com/events", {"active": "true", "closed": "false", "order": "id", "ascending": "true", "limit": 100, "series_id": 10684}, datetime.now(timezone.utc)
    for offset_idx in range(5):
        if offset_idx > 0: params["offset"] = offset_idx * 100
        try:
            events = requests.get(url, params=params, timeout=15).json()
            if not events: break
        except Exception: break
        for event in events:
            if not event.get("ticker", "").startswith("btc-updown-5m"): continue
            market = event.get("markets", [{}])[0]
            start_str, end_str = market.get("eventStartTime") or event.get("startTime"), market.get("endDate")
            if not start_str or not end_str: continue
            try: st, et = datetime.fromisoformat(start_str.replace("Z", "+00:00")), datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except Exception: continue
            if st > now or et <= now: continue
            clob_raw = market.get("clobTokenIds", "[]")
            clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
            if len(clob_ids) >= 2:
                logger.info(f"Discovered: {event.get('slug')}")
                return {"market_id": event.get("slug"), "up_token": str(clob_ids[0]), "down_token": str(clob_ids[1]), "end_time": et}
    return None

def _seed_book_from_rest() -> None:
    global LAST_REST_SEED_HAD_QUOTES, REST_SEED_INFLIGHT
    got_real = False
    for side, tid in [("UP", UP_TOKEN_ID), ("DOWN", DOWN_TOKEN_ID)]:
        if not tid: continue
        try:
            data = requests.get(f"https://clob.polymarket.com/book?token_id={tid}", timeout=3).json()
            bids, asks = data.get("bids", []), data.get("asks", [])
            bid_r, ask_r = float(bids[0]["price"]) if bids else None, float(asks[0]["price"]) if asks else None
            bid_sz, ask_sz = float(bids[0].get("size", 0)) if bids else 0.0, float(asks[0].get("size", 0)) if asks else 0.0

            prev = BOOK_CACHE.get(tid)
            bb = bid_r if bid_r is not None else (prev.best_bid if prev else None) or POLY_STATE.get(tid, {}).get("bid")
            ba = ask_r if ask_r is not None else (prev.best_ask if prev else None) or POLY_STATE.get(tid, {}).get("ask")
            if bb is None or ba is None: continue
            bb, ba = max(0.0, min(1.0, float(bb))), max(0.0, min(1.0, float(ba)))
            if bb > ba: continue

            existing = BOOK_CACHE.get(tid)
            if existing and existing.source == "ws" and (ms_now()-existing.last_update) < POLY_BOOK_MAX_AGE_MS: continue

            if tid not in POLY_STATE: POLY_STATE[tid] = {"bid": 0.0, "ask": 1.0, "bid_size": 0.0, "ask_size": 0.0, "last_update": 0, "source": "rest"}
            POLY_STATE[tid]["bid"], POLY_STATE[tid]["ask"] = bb, ba
            POLY_STATE[tid]["bid_size"], POLY_STATE[tid]["ask_size"] = bid_sz, ask_sz
            POLY_STATE[tid]["source"], POLY_STATE[tid]["last_update"] = "rest" if (bid_r and ask_r) else "fallback", ms_now()
            BOOK_CACHE.update(tid, BookSnapshot(token_id=tid, best_bid=bb, best_ask=ba, spread=ba-bb, bid_size=bid_sz, ask_size=ask_sz, source=POLY_STATE[tid]["source"], last_update=ms_now()))
            if (ba - bb) < 0.95: got_real = True
        except Exception as e: logger.warning(f"REST SEED {side} failed: {e}")
    LAST_REST_SEED_HAD_QUOTES = got_real

def _seed_book_from_rest_with_reset() -> None:
    global REST_SEED_INFLIGHT
    try: _seed_book_from_rest()
    finally: REST_SEED_INFLIGHT = False


async def market_clock_task(end_time: datetime) -> None:
    global UP_TOKEN_ID, DOWN_TOKEN_ID, BOOK_RECONNECT_FLAG
    global WINDOW_OPEN_MS, FIRST_REAL_BOOK_MS, LIQUIDITY_TIMEOUT_THIS_WINDOW
    global LAST_REST_SEED_HAD_QUOTES, LAST_REST_SEED_MS
    global TRADED_UP_THIS_WINDOW, TRADED_DN_THIS_WINDOW, TRADES_THIS_WINDOW
    global WINDOW_BANKROLL, WINDOW_CRYPTO_PRICE

    last_api_check = 0.0
    last_ws, _ = get_current_window_times()

    while True:
        ws, we = get_current_window_times()
        now_et = datetime.now(POLYMARKET_TIMEZONE)
        STATE.sec_remaining = float((we - now_et).total_seconds())

        if last_ws != ws:
            WINDOW_OPEN_MS, FIRST_REAL_BOOK_MS = ms_now(), 0
            PERSIST.side = PERSIST.start_ms = PERSIST.wait_ms = None
            LIQUIDITY_TIMEOUT_THIS_WINDOW = False
            LAST_REST_SEED_HAD_QUOTES = False
            LAST_REST_SEED_MS = ms_now()
            brain_loop._both_illiquid_since_ms = 0

            _pp_settle, _up_snap_settle, _dn_snap_settle = get_portfolio_pos(), BOOK_CACHE.get(UP_TOKEN_ID), BOOK_CACHE.get(DOWN_TOKEN_ID)
            _model_pnl_this_window = settle_window()

            if client is not None:
                try: PNL_TRACKER.on_window_end(client=client, up_token_id=UP_TOKEN_ID, dn_token_id=DOWN_TOKEN_ID, up_bid=float(_up_snap_settle.best_bid) if _up_snap_settle else 0.0, dn_bid=float(_dn_snap_settle.best_bid) if _dn_snap_settle else 0.0, model_pnl=_model_pnl_this_window, fifo_up_inventory=float(POS_UP.inventory), fifo_dn_inventory=float(POS_DOWN.inventory))
                except Exception as _pnl_err: logger.warning(f"PNL_TRACKER_END_FAIL: {_pnl_err}")

            STRIKE_CAPTURE.reset()
            COVERAGE.record_window(traded_up=TRADED_UP_THIS_WINDOW, traded_dn=TRADED_DN_THIS_WINDOW)
            logger.info(f"COVERAGE UPDATE | UP={COVERAGE.coverage_up():.2%} DN={COVERAGE.coverage_dn():.2%}")
            
            TRADED_UP_THIS_WINDOW, TRADED_DN_THIS_WINDOW, TRADES_THIS_WINDOW = False, False, 0
            for side in ("UP", "DOWN"):
                ENTRY_P_CONE[side], ENTRY_P_MARKET[side], ENTRY_T_SEC[side] = None, None, None
                ENTRY_EDGE[side], ENTRY_Z_EMA[side], ENTRY_TS_MS[side] = 0.0, 0.0, 0
                SIS_LAST_ACTION_U[side], SIS_LAST_ACTION_BID[side], _FLIP_BREACH_TS[side] = 0.0, 0.0, 0
            GTC_EXIT_ATTEMPTS.clear()

            POS_UP.__init__("UP", lot_method="FIFO")
            POS_DOWN.__init__("DOWN", lot_method="FIFO")
            IADL.window_bias, IADL.last_flip_ts_ms = None, None
            POS_MONITOR.reset_window(ms_now())

            STATE.open_price, STATE.strike_type, WINDOW_CRYPTO_PRICE = np.nan, None, None
            for _rst_side in ("UP", "DOWN"):
                if hasattr(STATE, f"_expiry_sell_{_rst_side}"): setattr(STATE, f"_expiry_sell_{_rst_side}", False)

            reset_window_locks()

            FLOW.__init__()                         
            LAG_ADAPTIVE.__init__()                 
            BIPOWER.__init__(BipowerConfig())       
            MOMENTUM.__init__(MomentumConfig())     
            VWAP.__init__()                         
            SPRT.__init__()                         
            TAIL_RISK.__init__()                    
            PV_TRACKER.__init__(PVConfig())         
            JUMP_DETECTOR.__init__(dt_seconds=1.0)  
            EXPIRY.__init__()                       
            FILL_PROB.__init__()                    
            L2_TRACKER.__init__()                   
            Z_TRAJ.__init__()                       
            BASIS.__init__(BASIS.cfg)               
            LATEST_DEBUG.clear()                    
            EXIT_COOLDOWN.clear()                   

            exec_unlock()                           
            PENDING_FLIP["active"], PENDING_FLIP_INTENT["active"] = False, False    
            LAST_FIRE_SNAPSHOT.clear()               
            LAST_FIRE_TS.clear()                     
            ORDER_COOLDOWN.clear()                   
            EXEC_ATTEMPTED.clear()                   
            PENDING_LOCKS.clear()                    
            RECENT_FILLS.clear()                     
            REGIME.__init__()                        
            DUAL_SIGMA.__init__(slow_window_s=90.0, fast_window_s=8.0)  
            SIGMA_HISTORY.clear()                    
            _P_CONE_HISTORY.clear()                  
            _RECENT_EDGES.clear()                    
            LEAK_MODEL.reset()                       

            for _attr in ('_last_fire_ms', '_last_rp', '_last_good_up', '_last_good_dn', '_last_survival_ms', '_last_diff_log_ms', '_last_edge_log_ms', '_last_rej_log', '_zh_window_id', '_zh_last_T', '_last_status_log_ms'):
                if hasattr(brain_loop, _attr): delattr(brain_loop, _attr)

            logger.info("MARKET_RESET_COMPLETE: all microstructure + execution state re-initialized")
            last_ws = ws

            try:
                WINDOW_BANKROLL = min(get_available_usdc_balance(), WINDOW_SPEND_CAP_USD)
                logger.info(f"WINDOW_BANKROLL: ${WINDOW_BANKROLL:.2f} (window_cap=${WINDOW_SPEND_CAP_USD:.2f})")
            except Exception: logger.warning("Failed to snapshot bankroll, using previous")

            # ── Check AI/Edge Monitor Kill Switch ──
            try:
                if os.path.exists("config/edge_health.json"):
                    with open("config/edge_health.json", "r") as f:
                        health_data = json.load(f)
                    if ms_now() - health_data.get("ts_ms", 0) < 300_000:
                        if health_data.get("kill_switch_active", False):
                            global DAY_KILL_ACTIVE, DAY_KILL_UNTIL_MS
                            DAY_KILL_ACTIVE = True
                            DAY_KILL_UNTIL_MS = ms_now() + DAY_PAUSE_SEC * 1000
                            logger.critical(
                                f"EDGE_COLLAPSE_DETECTED: Decay Factor dropped to {health_data.get('decay_factor')}. "
                                f"Triggering DAY_KILL to protect bankroll."
                            )
            except Exception as e:
                logger.debug(f"Edge health check failed: {e}")

            if client is not None:
                try:
                    _u_s, _d_s = BOOK_CACHE.get(UP_TOKEN_ID), BOOK_CACHE.get(DOWN_TOKEN_ID)
                    PNL_TRACKER.on_window_start(client=client, up_token_id=UP_TOKEN_ID, dn_token_id=DOWN_TOKEN_ID, up_bid=float(_u_s.best_bid) if _u_s else 0.0, dn_bid=float(_d_s.best_bid) if _d_s else 0.0, window_id=str(ws))
                except Exception as _pnl_err: logger.warning(f"PNL_TRACKER_START_FAIL: {_pnl_err}")

            try:
                global WIN_REALIZED_START, ROLL_SHARPE, SHARPE_MULT, SHARPE_PAUSED
                cur_realized = float(SESSION_PNL)
                if WIN_REALIZED_START is None: WIN_REALIZED_START = cur_realized
                win_ret = cur_realized - WIN_REALIZED_START
                WIN_REALIZED_START = cur_realized
                WINDOW_RETURNS.append(float(win_ret))
                recent = list(WINDOW_RETURNS)[-ROLL_SHARPE_N:]
                ROLL_SHARPE = _compute_roll_sharpe(recent)
                SHARPE_MULT = max(SHARPE_MULT_MIN, min(SHARPE_MULT_MAX, _sharpe_to_mult(ROLL_SHARPE)))
                SHARPE_PAUSED = (ROLL_SHARPE <= SHARPE_HARD)
                logger.info(f"ROLL_SHARPE | n={len(recent)} ret={win_ret:+.2f} sh={ROLL_SHARPE:+.3f} mult={SHARPE_MULT:.2f} paused={SHARPE_PAUSED}")
            except Exception as e: logger.warning(f"ROLL_SHARPE update failed: {e}")

            loop = asyncio.get_running_loop()
            info = await loop.run_in_executor(None, _auto_discover_market_sync)
            if info:
                new_up, new_dn = info["up_token"], info["down_token"]
                if new_up != UP_TOKEN_ID or new_dn != DOWN_TOKEN_ID:
                    BOOK_RECONNECT_FLAG = True
                    for old in (UP_TOKEN_ID, DOWN_TOKEN_ID):
                        if old: BOOK_CACHE.delete(old)
                    for old_tid in (UP_TOKEN_ID, DOWN_TOKEN_ID):
                        if old_tid and old_tid in POLY_STATE: del POLY_STATE[old_tid]
                    UP_TOKEN_ID, DOWN_TOKEN_ID = new_up, new_dn
                    logger.info(f"Tokens updated: UP={UP_TOKEN_ID[:16]}… DOWN={DOWN_TOKEN_ID[:16]}…")
                    try: set_market_fee_rate(client.get_fee_rate_bps(UP_TOKEN_ID))
                    except Exception as _fee_err: logger.warning(f"FEE_RATE_FETCH_FAIL: {_fee_err} — keeping previous rate")
                    FLOW.set_tokens(UP_TOKEN_ID, DOWN_TOKEN_ID)
                    GOLDSKY.set_market(MARKET_ID, UP_TOKEN_ID, DOWN_TOKEN_ID)
                    await loop.run_in_executor(None, _seed_book_from_rest)
                    await loop.run_in_executor(None, _prime_tick_size_cache, UP_TOKEN_ID)
                    await loop.run_in_executor(None, _prime_tick_size_cache, DOWN_TOKEN_ID)
            else: logger.error("Market rediscovery failed")

        if STATE.strike_type is None:
            unix_ts, now_ts = int(ws.timestamp()), now_et.timestamp()
            should_poll = (280.0 <= STATE.sec_remaining <= 300.0) or (STATE.sec_remaining < 280.0 and now_ts - last_api_check > 5.0)

            if should_poll:
                last_api_check = now_ts
                loop = asyncio.get_running_loop()
                _cp_data = await loop.run_in_executor(None, fetch_crypto_price, ws, we)
                if _cp_data and _cp_data.get("openPrice") is not None:
                    WINDOW_CRYPTO_PRICE = _cp_data
                    STATE.open_price, STATE.strike_type = float(_cp_data["openPrice"]), "OFFICIAL"
                    logger.info(f"STRIKE LOCKED (OFFICIAL): {STATE.open_price:.2f} (completed={_cp_data.get('completed', False)})")

            if STATE.strike_type is None:
                rtds_strike = STRIKE_CAPTURE.try_capture(unix_ts)
                if rtds_strike:
                    STATE.open_price, STATE.strike_type = rtds_strike, "RTDS"
                    logger.info(f"STRIKE LOCKED (RTDS): {STATE.open_price:.2f}")

            if STATE.strike_type is None and should_poll:
                cp = await loop.run_in_executor(None, fetch_price_from_candle_api, unix_ts)
                if cp:
                    STATE.open_price, STATE.strike_type = cp, "CANDLE"
                    logger.info(f"STRIKE LOCKED (CANDLE): {STATE.open_price:.2f}")

        await asyncio.sleep(1.0)

def get_best_oracle_price() -> tuple:
    now = ms_now()
    if RTDS.recv_ts_ms > 0 and (now - RTDS.recv_ts_ms) < RTDS_FRESH_MS: return RTDS.price, "rtds", (now - RTDS.recv_ts_ms)
    if STATE.btc_ts_ms > 0 and (now - STATE.btc_ts_ms) < BTC_FEED_MAX_AGE_MS * 2:
        if BASIS._init and BASIS.is_stable: return BASIS.corrected_price(STATE.btc_price), "coinbase_corrected", (now - STATE.btc_ts_ms)
        return STATE.btc_price, "coinbase", (now - STATE.btc_ts_ms)
    return np.nan, "none", 999_999


# ════════════════════════════════════════════════════════════════════════════
# SURVIVAL ENGINE
# ════════════════════════════════════════════════════════════════════════════

def survival_decide_portfolio(*, p_up: float, z: float, sec_remaining: float, up_snap, dn_snap, pos: PortfolioPos):
    A, B = float(pos.up), float(pos.dn)
    q = A + B
    if q <= 0: return {"action": "HOLD", "reason": "flat", "size": 0, "token_id": "", "limit": 0, "order_side": "SELL"}

    p_up = float(max(0.0, min(1.0, p_up)))
    bid_up, ask_up = float(up_snap.best_bid), float(up_snap.best_ask)
    bid_dn, ask_dn = float(dn_snap.best_bid), float(dn_snap.best_ask)

    EV_exit = A * max(0.0, bid_up - fee_per_share(bid_up)) + B * max(0.0, bid_dn - fee_per_share(bid_dn))
    EV_hold = p_up * A + (1.0 - p_up) * B
    uncertainty = 4.0 * p_up * (1.0 - p_up)
    t_frac = min(1.0, max(0.0, sec_remaining / 300.0))
    theta_pen = q * 0.01 * uncertainty * t_frac
    atmness = math.exp(-0.5 * float(z)**2) if np.isfinite(z) else 0.0
    gamma_pen = q * 0.04 * atmness * (30.0 / max(sec_remaining, 1.0)) ** 1.5
    if sec_remaining < 45: gamma_pen *= 2.0
    elif sec_remaining < 75: gamma_pen *= 1.5

    EV_hold_adj = EV_hold - theta_pen - gamma_pen

    if sec_remaining <= 8:
        _legs = []
        if A > 0 and bid_up > 0.01: _legs.append(("UP", UP_TOKEN_ID, A, bid_up))
        if B > 0 and bid_dn > 0.01: _legs.append(("DOWN", DOWN_TOKEN_ID, B, bid_dn))
        if _legs:
            _legs.sort(key=lambda x: x[2], reverse=True)
            _side, _tid, _qty, _bid = _legs[0]
            return {"action": "EXIT", "reason": "last_8s", "token_id": _tid, "order_side": "SELL", "size": _qty, "limit": round(max(0.01, _bid - 0.01), 2)}
        return {"action": "HOLD", "reason": "last_8s_no_book", "size": 0, "token_id": "", "limit": 0, "order_side": "SELL"}

    EV_EPS = max(0.01, min(0.05, 0.5 * q * uncertainty))
    if (EV_exit - EV_hold_adj) > EV_EPS:
        if A >= B and A > 0 and bid_up > 0.01:
            ok_up, net_up, req_up = early_sell_profit_gate(sec_remaining=sec_remaining, bid=bid_up, entry_price=pos.avg_cost_up, p_cone=p_up, side="UP", T_entry=ENTRY_T_SEC.get("UP") or 230.0)
            if ok_up: return {"action": "TRIM", "reason": "ev_exit_pref", "token_id": UP_TOKEN_ID, "order_side": "SELL", "size": max(1.0, math.floor(A * 0.6)), "limit": round(max(0.01, bid_up - 0.01), 2)}
        if B > 0 and bid_dn > 0.01:
            ok_dn, net_dn, req_dn = early_sell_profit_gate(sec_remaining=sec_remaining, bid=bid_dn, entry_price=pos.avg_cost_dn, p_cone=p_up, side="DOWN", T_entry=ENTRY_T_SEC.get("DOWN") or 230.0)
            if ok_dn: return {"action": "TRIM", "reason": "ev_exit_pref", "token_id": DOWN_TOKEN_ID, "order_side": "SELL", "size": max(1.0, math.floor(B * 0.6)), "limit": round(max(0.01, bid_dn - 0.01), 2)}
        
        _up_te, _dn_te = ENTRY_T_SEC.get("UP") or 230.0, ENTRY_T_SEC.get("DOWN") or 230.0
        _, _u_net, _u_req = early_sell_profit_gate(sec_remaining, bid_up, pos.avg_cost_up, p_up, "UP", _up_te)
        _, _d_net, _d_req = early_sell_profit_gate(sec_remaining, bid_dn, pos.avg_cost_dn, p_up, "DOWN", _dn_te)
        if sec_remaining > EXIT_HARD_FLOOR_T: return {"action": "HOLD", "reason": "alpha_exit_gate", "size": 0, "token_id": "", "limit": 0, "order_side": "SELL", "up_net": round(_u_net, 4), "up_req": round(_u_req, 4), "dn_net": round(_d_net, 4), "dn_req": round(_d_req, 4)}
        return {"action": "HOLD", "reason": "ev_exit_no_book", "size": 0, "token_id": "", "limit": 0, "order_side": "SELL"}

    if sec_remaining < 35 and abs(float(z)) < 0.9 and uncertainty > 0.85:
        if A > B and ask_dn < 0.99 and ask_dn > 0.01: return {"action": "HEDGE", "reason": "pin_risk_balance", "token_id": DOWN_TOKEN_ID, "order_side": "BUY", "size": max(1.0, math.floor((A - B) * 0.5)), "limit": round(min(0.99, ask_dn + 0.01), 2)}
        if B > A and ask_up < 0.99 and ask_up > 0.01: return {"action": "HEDGE", "reason": "pin_risk_balance", "token_id": UP_TOKEN_ID, "order_side": "BUY", "size": max(1.0, math.floor((B - A) * 0.5)), "limit": round(min(0.99, ask_up + 0.01), 2)}

    return {"action": "HOLD", "reason": "ev_hold", "size": 0, "token_id": "", "limit": 0, "order_side": "SELL", "ev_hold": round(EV_hold_adj, 4), "ev_exit": round(EV_exit, 4)}

def maybe_request_flip(ts_ms: int, p_cone: float, z: float, T_sec: float, up_snap, dn_snap):
    total_inv = POS_UP.inventory + POS_DOWN.inventory
    if total_inv <= 0: return False, []

    cur_side = "UP" if POS_UP.inventory >= POS_DOWN.inventory else "DOWN"
    model_side = "UP" if p_cone >= 0.5 else "DOWN"
    if model_side == cur_side: return False, []

    try:
        dec = IADL.gate(ts_ms=ts_ms, portfolio=PORTFOLIO, proposed_side=model_side, z=float(z), p_cone=float(p_cone), T_sec=float(T_sec), up_bid=float(up_snap.best_bid), up_ask=float(up_snap.best_ask), dn_bid=float(dn_snap.best_bid), dn_ask=float(dn_snap.best_ask), debug_in={"reason": "FLIP_CHECK"})
        if dec.kind not in (DecisionType.CLOSE_THEN_REVERSE, DecisionType.UNWIND_STRADDLE): return False, []
    except Exception as e: return False, []

    if cur_side == "UP": sell_tid, sell_qty, sell_bid = UP_TOKEN_ID, float(POS_UP.inventory), float(up_snap.best_bid)
    else: sell_tid, sell_qty, sell_bid = DOWN_TOKEN_ID, float(POS_DOWN.inventory), float(dn_snap.best_bid)

    if sell_qty <= 0 or sell_bid <= 0.01: return False, []

    sell_entry = float(POS_UP.avg_entry_price) if cur_side == "UP" else float(POS_DOWN.avg_entry_price)
    flip_ok, net_bid, req_bid = early_sell_profit_gate(sec_remaining=T_sec, bid=sell_bid, entry_price=sell_entry, p_cone=float(p_cone), side=cur_side, T_entry=max(1.0, 300.0 - T_sec))
    if not flip_ok: return False, []

    sell_limit = round(max(0.01, sell_bid - 0.01), 2)
    if model_side == "UP": buy_tok, buy_ask = UP_TOKEN_ID, float(up_snap.best_ask)
    else: buy_tok, buy_ask = DOWN_TOKEN_ID, float(dn_snap.best_ask)

    buy_limit = round(min(0.99, buy_ask + 0.01), 2)
    PENDING_FLIP_INTENT.update({"active": True, "buy_token_id": buy_tok, "buy_side": model_side, "buy_limit": buy_limit, "buy_size": sell_qty, "expires_ms": ts_ms + 4000})
    return True, [{"token_side": cur_side, "token_id": sell_tid, "qty": sell_qty, "limit_price": sell_limit}]


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

async def dashboard_task() -> None:
    while True:
        _pp = get_portfolio_pos()
        src = LATEST_DEBUG 
        p, z_val, edge, side, reason = src.get("p_cone", np.nan), src.get("z", np.nan), src.get("edge", 0.0), src.get("side", "-"), src.get("reason", "Wait")
        ora_px, ora_src, ora_age = get_best_oracle_price()
        ora_s = f"${ora_px:.2f}({ora_src},{ora_age/1000:.0f}s)" if np.isfinite(ora_px) else "---"
        btc_s = f"${STATE.btc_price:.2f}" if not np.isnan(STATE.btc_price) else "---"
        rtds_s = f"${RTDS.price:.2f}" if RTDS.is_fresh(RTDS_FRESH_MS) else "---"
        strike_s = f"${STATE.open_price:.2f}" if not np.isnan(STATE.open_price) else "---"
        rem_s = f"{STATE.sec_remaining:.0f}s" if not np.isnan(STATE.sec_remaining) else "---"
        pos_str = f"[INV:{_pp.dominant_side} U={_pp.up:.0f} D={_pp.dn:.0f}]" if _pp.total > 0 else ""
        gate = "POSITION_OPEN" if _pp.total > 0 else reason
        cb_str = " [CB:PAUSED]" if CIRCUIT_BREAKER_ACTIVE else ""

        try:
            sys.stdout.write(f"\r\033[K[Live] ORA:{ora_s} | CB:{btc_s} | CL:{rtds_s} | K:{strike_s}({STATE.strike_type}) | T:{rem_s} | sig:{STATE.sigma_1m:.4f}(w={STATE.sigma_w:.2f}) | p:{p:.3f} z:{z_val:.2f} e:{edge:.4f} | PnL:{SESSION_PNL:+.2f}{cb_str} | ${WINDOW_BANKROLL:.0f}({'U' + str(int(_pp.up)) if _pp.up > 0 else ''}{'D' + str(int(_pp.dn)) if _pp.dn > 0 else ''}) => {side} {gate} {pos_str}")
            sys.stdout.flush()
        except UnicodeEncodeError: pass
        await asyncio.sleep(1.0)


# ════════════════════════════════════════════════════════════════════════════
# BRAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

async def brain_loop(eq: asyncio.Queue) -> None:
    global LATEST_DEBUG, LAST_FIRE_SNAPSHOT
    global REST_SEED_INFLIGHT, LAST_REST_SEED_MS
    global LAST_STALE_BOOK_LOG_MS, LAST_NON_WS_BOOK_LOG_MS
    global WINDOW_OPEN_MS, FIRST_REAL_BOOK_MS, LIQUIDITY_TIMEOUT_THIS_WINDOW
    global TRADED_UP_THIS_WINDOW, TRADED_DN_THIS_WINDOW, TRADES_THIS_WINDOW
    global LAST_REST_SEED_HAD_QUOTES, CIRCUIT_BREAKER_ACTIVE
    global DAY_KILL_ACTIVE

    logger.info("Brain loop operational.")
    _last_status_log_ms = 0

    while True:
        await asyncio.sleep(0.01)
        _now = ms_now()

        if _now - _last_status_log_ms >= 10_000:
            _last_status_log_ms = _now
            _up_snap, _dn_snap = BOOK_CACHE.get(UP_TOKEN_ID), BOOK_CACHE.get(DOWN_TOKEN_ID)
            logger.info(
                f"BRAIN: reason={LATEST_DEBUG.get('reason', '?')} | regime={REGIME.label} | "
                f"flow={FLOW.imbalance(up_price=_up_snap.best_bid if _up_snap else 0.5, down_price=_dn_snap.best_bid if _dn_snap else 0.5):.2f}({int(FLOW.trade_velocity(window_s=30.0) * 30)}fills) | "
                f"oracle={LATEST_DEBUG.get('oracle_source', '?')}(age={RTDS.age_ms()}ms,n={RTDS.updates}) | "
                f"btc_age={_now - STATE.btc_ts_ms}ms | strike={STATE.open_price:.2f}({STATE.strike_type}) | "
                f"T={STATE.sec_remaining:.0f}s | sig={STATE.sigma_1m:.5f} | "
                f"UP={f'{_up_snap.best_bid:.2f}/{_up_snap.best_ask:.2f}' if _up_snap else '?/?'}({_up_snap.source if _up_snap else 'none'},{(_now - _up_snap.last_update) if _up_snap else 999999}ms) "
                f"DN={f'{_dn_snap.best_bid:.2f}/{_dn_snap.best_ask:.2f}' if _dn_snap else '?/?'}({_dn_snap.source if _dn_snap else 'none'},{(_now - _dn_snap.last_update) if _dn_snap else 999999}ms) | "
                f"pos={'INV:' + ('UP' if POS_UP.inventory >= POS_DOWN.inventory else 'DOWN') if (POS_UP.inventory + POS_DOWN.inventory) > 0 else 'none'}"
            )

        if CIRCUIT_BREAKER_ACTIVE:
            if ms_now() >= CIRCUIT_BREAKER_UNTIL: CIRCUIT_BREAKER_ACTIVE = False
            else:
                LATEST_DEBUG = {"reason": "CIRCUIT_BREAKER"}
                continue

        oracle_px, oracle_src, oracle_age = get_best_oracle_price()
        LATEST_DEBUG = {"reason": "Wait", "p_cone": np.nan, "z": np.nan, "edge": 0.0, "side": "-", "oracle_source": oracle_src}

        if not hasattr(brain_loop, "_last_diff_log_ms"): brain_loop._last_diff_log_ms = 0
        if (_now - brain_loop._last_diff_log_ms >= 30_000 and RTDS.is_fresh(60_000) and STATE.btc_ts_ms > 0 and not np.isnan(STATE.btc_price)):
            brain_loop._last_diff_log_ms = _now
            logger.info(f"ORACLE_DIFF: rtds=${RTDS.price:.2f} coinbase=${STATE.btc_price:.2f} delta={RTDS.price - STATE.btc_price:+.2f} | using={oracle_src}")

        if RTDS.recv_ts_ms > 0: STRIKE_CAPTURE.feed(RTDS.price, RTDS.recv_ts_ms)

        rp = getattr(brain_loop, '_last_rp', None)
        if (np.isfinite(oracle_px) and not np.isnan(STATE.open_price) and STATE.sec_remaining > 0):
            _prev_oracle = LATEST_DEBUG.get("_prev_oracle_px", oracle_px)
            rp = REGIME.update(sigma_eff=STATE.sigma_1m, delta_sign=1 if oracle_px > _prev_oracle else (-1 if oracle_px < _prev_oracle else 0), z=float(LATEST_DEBUG.get("z", 0.0)), sigma_fast=STATE.sigma_fast if hasattr(STATE, 'sigma_fast') else STATE.sigma_1m, sigma_slow=STATE.sigma_slow if hasattr(STATE, 'sigma_slow') else STATE.sigma_1m, spread=float(LATEST_DEBUG.get("spread_max", 0.01)))
            LATEST_DEBUG["regime"] = REGIME.label

            _up_for_ofi, _dn_for_ofi = BOOK_CACHE.get(UP_TOKEN_ID), BOOK_CACHE.get(DOWN_TOKEN_ID)
            _ofi = FLOW.imbalance(up_price=_up_for_ofi.best_bid, down_price=_dn_for_ofi.best_bid) if (_up_for_ofi and _dn_for_ofi and _up_for_ofi.best_bid > 0 and _dn_for_ofi.best_bid > 0) else 0.0
            LATEST_DEBUG.update(MOMENTUM.update(ts_ms=ms_now(), z=float(LATEST_DEBUG.get("z", 0.0)), ofi=_ofi, price=oracle_px, jump_regime=BIPOWER.jump_regime))
            
            VWAP.feed(oracle_px)
            LATEST_DEBUG.update(VWAP.status_dict())
            LATEST_DEBUG.update(SPRT.status_dict())

            if _prev_oracle > 0 and oracle_px > 0:
                r_t = math.log(oracle_px / _prev_oracle)
                if JUMP_DETECTOR.update(r_t, STATE.sigma_slow if hasattr(STATE, 'sigma_slow') else STATE.sigma_1m): LATEST_DEBUG["jump_detected"] = True
                BIPOWER.feed(r_t)
                LATEST_DEBUG.update(BIPOWER.status_dict())
            LATEST_DEBUG["_prev_oracle_px"] = oracle_px

            p, z = cone_p_and_z(oracle_px, STATE.open_price, STATE.sec_remaining, STATE.sigma_1m, calibration_alpha=rp.calibration_alpha)
            LATEST_DEBUG.update({"p_cone": p, "z": z})
            brain_loop._last_rp = rp

        up_snap, dn_snap = BOOK_CACHE.get(UP_TOKEN_ID), BOOK_CACHE.get(DOWN_TOKEN_ID)

        if not up_snap or not BOOK_CACHE.is_fresh(UP_TOKEN_ID, 800):
            try: await asyncio.get_running_loop().run_in_executor(None, _seed_book_from_rest); up_snap = BOOK_CACHE.get(UP_TOKEN_ID)
            except Exception: pass
        if not dn_snap or not BOOK_CACHE.is_fresh(DOWN_TOKEN_ID, 800):
            try: await asyncio.get_running_loop().run_in_executor(None, _seed_book_from_rest); dn_snap = BOOK_CACHE.get(DOWN_TOKEN_ID)
            except Exception: pass

        if not up_snap or not dn_snap: LATEST_DEBUG["reason"] = "WAITING_FOR_BOOK"; continue
        if STATE.trading_suspended: LATEST_DEBUG["reason"] = "WS_SUSPENDED"; continue
        if is_rate_limited(): LATEST_DEBUG["reason"] = "RATE_LIMITED"; continue
        if np.isnan(oracle_px) or np.isnan(STATE.open_price) or np.isnan(STATE.sec_remaining): LATEST_DEBUG["reason"] = "MISSING_DATA"; continue
        if STATE.strike_type not in ("OFFICIAL", "CANDLE", "RTDS"): LATEST_DEBUG["reason"] = "WAITING_FOR_STRIKE"; continue

        p_preview = LATEST_DEBUG.get("p_cone", np.nan)
        if np.isfinite(p_preview):
            ua, da = up_snap.best_ask, dn_snap.best_ask
            if np.isfinite(ua) and np.isfinite(da):
                eu, ed = p_preview - ua - fee_per_share(ua), (1-p_preview) - da - fee_per_share(da)
                LATEST_DEBUG.update({"side": "UP", "edge": float(eu)} if eu >= ed else {"side": "DOWN", "edge": float(ed)})

        now_ms_val = ms_now()
        eff_max_age = REST_SEED_EMPTY_COOLDOWN_MS + 5000 if not LAST_REST_SEED_HAD_QUOTES else POLY_BOOK_MAX_AGE_MS
        oldest = max(now_ms_val - up_snap.last_update, now_ms_val - dn_snap.last_update)
        if oldest >= max(0, eff_max_age - POLY_BOOK_RESEED_AHEAD_MS) and not REST_SEED_INFLIGHT and (now_ms_val - LAST_REST_SEED_MS >= (REST_SEED_COOLDOWN_MS if LAST_REST_SEED_HAD_QUOTES else REST_SEED_EMPTY_COOLDOWN_MS)):
            REST_SEED_INFLIGHT = True
            LAST_REST_SEED_MS = now_ms_val
            await asyncio.get_running_loop().run_in_executor(None, _seed_book_from_rest_with_reset)

        if oldest >= eff_max_age: LATEST_DEBUG["reason"] = "STALE_POLY_BOOK"; continue

        up_sp, dn_sp = up_snap.best_ask - up_snap.best_bid, dn_snap.best_ask - dn_snap.best_bid
        TAIL_RISK.feed_spread(max(up_sp, dn_sp))

        if not hasattr(brain_loop, "_last_good_up"): brain_loop._last_good_up, brain_loop._last_good_dn = None, None
        if not (up_snap.best_bid <= 0.02 and up_snap.best_ask >= 0.98): brain_loop._last_good_up = up_snap
        elif brain_loop._last_good_up is not None: up_snap, up_sp = brain_loop._last_good_up, brain_loop._last_good_up.best_ask - brain_loop._last_good_up.best_bid
        if not (dn_snap.best_bid <= 0.02 and dn_snap.best_ask >= 0.98): brain_loop._last_good_dn = dn_snap
        elif brain_loop._last_good_dn is not None: dn_snap, dn_sp = brain_loop._last_good_dn, brain_loop._last_good_dn.best_ask - brain_loop._last_good_dn.best_bid

        if (up_snap.best_bid <= 0 and up_snap.best_ask >= 1.0) and (dn_snap.best_bid <= 0 and dn_snap.best_ask >= 1.0):
            LATEST_DEBUG["reason"] = "EMPTY_BOOK"; continue

        up_mid, dn_mid = (up_snap.best_bid + up_snap.best_ask) / 2, (dn_snap.best_bid + dn_snap.best_ask) / 2
        if (up_mid >= 0.93 and dn_mid <= 0.07) or (dn_mid >= 0.93 and up_mid <= 0.07):
            LATEST_DEBUG["reason"] = "MARKET_DECIDED"; continue

        if FIRST_REAL_BOOK_MS == 0 and WINDOW_OPEN_MS > 0: FIRST_REAL_BOOK_MS = now_ms_val
        if oracle_age > BTC_FEED_MAX_AGE_MS * 5: LATEST_DEBUG["reason"] = "ALL_FEEDS_STALE"; continue

        pos = get_portfolio_pos()
        if pos.total > 0:
            _surv_oracle = RTDS.price if RTDS.is_fresh(RTDS_FRESH_MS) else STATE.btc_price
            if (np.isfinite(_surv_oracle) and _surv_oracle > 0 and np.isfinite(STATE.open_price) and STATE.open_price > 0 and STATE.sec_remaining > 0 and STATE.sigma_1m > 0):
                _rp_surv = getattr(brain_loop, '_last_rp', None)
                p_surv, z_surv = cone_p_and_z(_surv_oracle, STATE.open_price, STATE.sec_remaining, STATE.sigma_1m, calibration_alpha=_rp_surv.calibration_alpha if _rp_surv else 1.0)
            else: p_surv, z_surv = float(LATEST_DEBUG.get("p_cone", 0.5) or 0.5), float(LATEST_DEBUG.get("z", 0.0) or 0.0)
            LATEST_DEBUG["p_cone"], LATEST_DEBUG["z"] = p_surv, z_surv

            _u_snap, _d_snap = BOOK_CACHE.get(UP_TOKEN_ID), BOOK_CACHE.get(DOWN_TOKEN_ID)
            _regime_lbl = REGIME.label if REGIME else "NORMAL"

            for _pm_side, _pm_pos, _pm_tid, _pm_snap, _pm_entry in [("UP", POS_UP, UP_TOKEN_ID, _u_snap, pos.avg_cost_up), ("DOWN", POS_DOWN, DOWN_TOKEN_ID, _d_snap, pos.avg_cost_dn)]:
                if _pm_pos.inventory <= 0 or _pm_snap is None or _pm_snap.best_bid <= 0.01: continue
                _pm_bid, _pm_spread = float(_pm_snap.best_bid), max(0.0, float(_pm_snap.best_ask) - float(_pm_snap.best_bid))
                
                _pm_dec = POS_MONITOR.evaluate(
                    side=_pm_side, entry_price=_pm_entry, bid=_pm_bid, p_cone=p_surv, T_sec=STATE.sec_remaining, sigma_1m=STATE.sigma_1m, regime=_regime_lbl, spread=_pm_spread, now_ms=now_ms_val,
                    edge_entry=float(ENTRY_EDGE.get(_pm_side, 0.0)), edge_now=float(edge_now_for_position(_pm_side, p_surv, float(_u_snap.best_bid) if _u_snap else 0.0, float(_d_snap.best_bid) if _d_snap else 0.0)),
                    z_ema_now=float(LATEST_DEBUG.get("z_ema", z_surv) or z_surv), flow_centered=float(LATEST_DEBUG.get("flow_centered", 0.0) or 0.0), entry_ts_ms=int(ENTRY_TS_MS.get(_pm_side, 0))
                )

                _STOP_LOSS_REASONS = {"prob_stop_loss", "endgame_strong_loss", "trailing_stop", "endgame_gamma_exit", "gamma_danger", "sis_exit"}
                _is_stop_loss = _pm_dec.reason in _STOP_LOSS_REASONS
                
                if _pm_dec.action == "SELL" and not SIMULATION_MODE and ((not EXEC_LOCKED) or _is_stop_loss):
                    if _is_stop_loss: _ok_sell, _net, _req = True, net_sell_after_fee(_pm_bid), 0.0
                    else: _ok_sell, _net, _req = early_sell_profit_gate(STATE.sec_remaining, _pm_bid, _pm_entry, p_surv, _pm_side, ENTRY_T_SEC.get(_pm_side) or 230.0)

                    if not _ok_sell:
                        _block_log_key = f"_pm_block_log_{_pm_side}"
                        _mono_now = _mono_ms()
                        if _mono_now - getattr(brain_loop, _block_log_key, 0) >= 500:
                            setattr(brain_loop, _block_log_key, _mono_now)
                            logger.info(f"PM_EXIT_BLOCKED({_pm_side}): {_pm_dec.reason} T={STATE.sec_remaining:.0f}s net={_net:.3f} tgt={_req:.3f} p={p_surv if _pm_side == 'UP' else (1.0 - p_surv):.3f}")
                    else:
                        _sis_u, _sis_frac, _sis_offset = float(_pm_dec.exit_urgency), float(_pm_dec.exit_frac), float(_pm_dec.exit_offset)
                        _exit_inv = float(_pm_pos.inventory)
                        _exit_size = round(max(0.01, _exit_inv * max(_sis_frac, 0.10)), 6)
                        _exit_price = round(max(0.01, _pm_bid - _sis_offset), 2)

                        _last_u, _last_bid = SIS_LAST_ACTION_U.get(_pm_side, 0.0), SIS_LAST_ACTION_BID.get(_pm_side, 0.0)
                        if not (abs(_sis_u - _last_u) >= 0.07) and not (abs(_pm_bid - _last_bid) >= 0.01) and _last_u > 0: continue
                        SIS_LAST_ACTION_U[_pm_side], SIS_LAST_ACTION_BID[_pm_side] = _sis_u, _pm_bid

                        logger.info(f"PM_EXIT({_pm_side}): {_pm_dec.reason} hold_ev={_pm_dec.hold_ev:.3f} sell_ev={_pm_dec.sell_ev:.3f} profit={_pm_dec.profit:.3f}/sh ev_gap={_pm_dec.ev_gap:.3f} penalty={_pm_dec.penalty:.3f} entry={_pm_entry:.2f} bid={_pm_bid:.2f} regime={_regime_lbl} | SIS: u={_sis_u:.3f} frac={_sis_frac:.3f} offset={_sis_offset:.4f} size={_exit_size:.4f}/{_exit_inv:.4f}")
                        await eq.put({"action": "ORDER", "side": _pm_side, "token_id": _pm_tid, "price": _exit_price, "size": _exit_size, "order_side": "SELL", "edge": 0.0, "mode": "exit", "p_cone": p_surv, "z": z_surv, "sigma_1m": STATE.sigma_1m, "regime": _regime_lbl, "exit_reason": _pm_dec.reason, "exit_hold_ev": round(_pm_dec.hold_ev, 4), "exit_sell_ev": round(_pm_dec.sell_ev, 4), "exit_ev_gap": round(_pm_dec.ev_gap, 4), "sis_urgency": round(_sis_u, 4), "sis_frac": round(_sis_frac, 4), "sis_offset": round(_sis_offset, 4), "signal_bid": _pm_bid, "signal_ask": float(_pm_snap.best_ask)})
                        LATEST_DEBUG["reason"] = f"PM_{_pm_dec.reason}_{_pm_side}"
                        brain_loop._last_survival_ms = now_ms_val
                        break

            flow_now = FLOW.imbalance(up_price=_u_snap.best_bid if _u_snap else 0.5, down_price=_d_snap.best_bid if _d_snap else 0.5)
            if ((pos.dominant_side == "UP" and flow_now < -0.70) or (pos.dominant_side == "DOWN" and flow_now > 0.70)) and pos.total > 0:
                logger.warning(f"FLOW REVERSAL: side={pos.dominant_side} but flow={flow_now:.2f} — triggering early exit")
                LATEST_DEBUG["reason"] = "SURV_FLOW_REVERSAL"
                if not SIMULATION_MODE:
                    _fr_side, _fr_tid, _fr_inv, _fr_snap = pos.dominant_side, UP_TOKEN_ID if pos.dominant_side == "UP" else DOWN_TOKEN_ID, pos.up if pos.dominant_side == "UP" else pos.dn, _u_snap if pos.dominant_side == "UP" else _d_snap
                    if _fr_snap and _fr_snap.best_bid > 0.01 and _fr_inv > 0:
                        await eq.put({"action": "ORDER", "side": _fr_side, "token_id": _fr_tid, "price": round(max(0.01, _fr_snap.best_bid - 0.01), 2), "size": float(_fr_inv), "order_side": "SELL", "mode": "exit", "edge": 0.0, "p_cone": p_surv, "z": z_surv, "sigma_1m": STATE.sigma_1m, "signal_bid": float(_fr_snap.best_bid), "signal_ask": float(_fr_snap.best_ask)})
                        brain_loop._last_survival_ms = now_ms_val
                        continue

            if not hasattr(brain_loop, "_last_survival_ms"): brain_loop._last_survival_ms = 0
            if now_ms_val - brain_loop._last_survival_ms < 900: LATEST_DEBUG["reason"] = "POSITION_OPEN"; continue

            flip_ok, flip_sells = maybe_request_flip(ms_now(), p_surv, z_surv, STATE.sec_remaining, up_snap, dn_snap)
            if flip_ok and not SIMULATION_MODE:
                for op in flip_sells: await eq.put({"action": "ORDER", "side": op["token_side"], "token_id": op["token_id"], "price": round(float(op["limit_price"]), 2), "size": float(op["qty"]), "order_side": "SELL", "mode": "close", "edge": 0.0, "p_cone": p_surv, "z": z_surv, "sigma_1m": STATE.sigma_1m})
                LATEST_DEBUG["reason"] = "SURV_FLIP_SELL"
                brain_loop._last_survival_ms = now_ms_val
                continue

            sv = survival_decide_portfolio(p_up=p_surv, z=z_surv, sec_remaining=STATE.sec_remaining, up_snap=up_snap, dn_snap=dn_snap, pos=pos)
            action = sv["action"]
            LATEST_DEBUG.update(sv)
            LATEST_DEBUG["reason"] = f"SURV_{action}"

            if action in ("EXIT", "TRIM", "HEDGE") and not SIMULATION_MODE and sv.get("token_id") and sv.get("size", 0) > 0:
                logger.info(f"SURVIVAL {action}: {sv['reason']} size={sv['size']}")
                await eq.put({"action": "ORDER", "side": ("UP" if sv["token_id"] == UP_TOKEN_ID else "DOWN"), "token_id": sv["token_id"], "price": float(sv["limit"]), "size": float(sv["size"]), "order_side": sv["order_side"], "mode": action.lower(), "edge": 0.0, "p_cone": p_surv, "z": z_surv, "sigma_1m": STATE.sigma_1m})
            brain_loop._last_survival_ms = now_ms_val
            continue

        if PENDING_FLIP_INTENT.get("active", False):
            if ms_now() > PENDING_FLIP_INTENT["expires_ms"]: PENDING_FLIP_INTENT["active"] = False
            elif abs(portfolio_net_shares()) < 1.0:
                if not SIMULATION_MODE: await eq.put({"action": "ORDER", "side": PENDING_FLIP_INTENT["buy_side"], "token_id": PENDING_FLIP_INTENT["buy_token_id"], "price": float(PENDING_FLIP_INTENT["buy_limit"]), "size": float(PENDING_FLIP_INTENT["buy_size"]), "order_side": "BUY", "edge": 0.0, "p_cone": float(LATEST_DEBUG.get("p_cone", 0.5) or 0.5), "z": float(LATEST_DEBUG.get("z", 0.0) or 0.0), "sigma_1m": STATE.sigma_1m})
                PENDING_FLIP_INTENT["active"] = False

        if pos.total == 0 and PENDING_FLIP.get("active", False):
            if ms_now() > PENDING_FLIP["expires_ms"]: PENDING_FLIP["active"] = False
            elif not SIMULATION_MODE and PENDING_FLIP.get("token_id") and PENDING_FLIP.get("size", 0) > 0:
                await eq.put({"action": "ORDER", "side": PENDING_FLIP["side"], "token_id": PENDING_FLIP["token_id"], "price": float(PENDING_FLIP["limit"]), "size": float(PENDING_FLIP["size"]), "order_side": "BUY", "edge": 0.0, "p_cone": LATEST_DEBUG.get("p_cone", np.nan), "z": LATEST_DEBUG.get("z", np.nan), "sigma_1m": STATE.sigma_1m})
                PENDING_FLIP["active"] = False

        btc_for_edge = oracle_px
        eff_up_bid, eff_up_ask, eff_dn_bid, eff_dn_ask = up_snap.best_bid, up_snap.best_ask, dn_snap.best_bid, dn_snap.best_ask
        implied_up, implied_dn = False, False

        if up_sp > 0.50 and dn_sp <= 0.50: eff_up_bid, eff_up_ask, implied_up = round(1.0 - dn_snap.best_ask, 4), round(1.0 - dn_snap.best_bid, 4), True
        elif dn_sp > 0.50 and up_sp <= 0.50: eff_dn_bid, eff_dn_ask, implied_dn = round(1.0 - up_snap.best_ask, 4), round(1.0 - up_snap.best_bid, 4), True

        _pv = PV_TRACKER.update(ts_ms=now_ms_val, z=float(LATEST_DEBUG.get("z", 0.0) or 0.0), T_sec=float(STATE.sec_remaining), up_bid=up_snap.best_bid, up_ask=up_snap.best_ask, up_bid_sz=up_snap.bid_size, up_ask_sz=up_snap.ask_size, dn_bid=dn_snap.best_bid, dn_ask=dn_snap.best_ask, dn_bid_sz=dn_snap.bid_size, dn_ask_sz=dn_snap.ask_size, tv_fast=float(FLOW.trade_velocity(window_s=5.0)), tv_slow=float(FLOW.trade_velocity(window_s=20.0)))
        LATEST_DEBUG.update({"pv": round(_pv["pv"], 3), "pa": round(_pv["pa"], 3), "qc": round(_pv["qc"], 3), "signal_quality": round(_pv["signal_quality"], 3), "pv_confirms": bool(_pv["participation_confirms"]), "ttp_ms": _pv["ttp_ms"], "breakout": bool(_pv["breakout"]), "trap": bool(_pv["trap"])})

        _cb_lead_info = {"anticipation_active": False, "cb_z": 0.0, "cb_z_velocity": 0.0, "direction_agrees": False}
        if (not np.isnan(STATE.btc_price) and STATE.btc_price > 0 and not np.isnan(STATE.open_price) and STATE.open_price > 0 and STATE.sec_remaining > 0 and STATE.sigma_1m > 0):
            _cb_z = CB_LEAD.compute_cb_z(STATE.btc_price, STATE.open_price, STATE.sec_remaining, STATE.sigma_1m)
            _cb_lead_info = CB_LEAD.update(cb_z=_cb_z, rtds_z=float(LATEST_DEBUG.get("z", 0.0) or 0.0), sec_remaining=STATE.sec_remaining, basis_stable=BASIS.is_stable)
            Z_TRAJ.set_anticipation_mode(_cb_lead_info["anticipation_active"] and _cb_lead_info["direction_agrees"], reduced_ticks=1)

        LATEST_DEBUG.update({"cb_z": round(_cb_lead_info.get("cb_z", 0.0), 3), "cb_z_vel": round(_cb_lead_info.get("cb_z_velocity", 0.0), 4), "anticipation": bool(_cb_lead_info.get("anticipation_active", False)), "dir_agrees": bool(_cb_lead_info.get("direction_agrees", False)), "basis": round(BASIS.rolling_basis, 2), "basis_std": round(BASIS.basis_stdev, 2)})

        _sig_spread = max(up_sp, dn_sp)
        _sig_lag_p50 = float(LAG_ADAPTIVE.p50_ms("UP"))
        _sig_sigma_ratio = float(STATE.sigma_fast / STATE.sigma_slow) if STATE.sigma_slow > 1e-10 else 1.0
        
        _pre_md_flow = FLOW.imbalance(up_price=eff_up_bid, down_price=eff_dn_bid)
        _pre_md = micro_edge_buffer(
            spread=_sig_spread, bid_sz=min(up_snap.bid_size + up_snap.ask_size, dn_snap.bid_size + dn_snap.ask_size) * 0.5, ask_sz=min(up_snap.bid_size + up_snap.ask_size, dn_snap.bid_size + dn_snap.ask_size) * 0.5,
            sigma=float(STATE.sigma_1m), lag_p50_ms=_sig_lag_p50, lag_p10_ms=float(LAG_ADAPTIVE.p10_ms("UP")), trade_vel=float(FLOW.trade_velocity(window_s=5.0)), flow_bias=_pre_md_flow
        )

        side, token, limit_price, size, debug = decide_edge(
            btc_price=btc_for_edge, open_price=STATE.open_price, sec_remaining=STATE.sec_remaining, sigma_1m=STATE.sigma_1m,
            up_bid=eff_up_bid, up_ask=eff_up_ask, down_bid=eff_dn_bid, down_ask=eff_dn_ask,
            lag_adaptive=LAG_ADAPTIVE, persist=PERSIST, regime_params=REGIME.current_params, flow_imbalance=_pre_md_flow,
            coverage_bias=COVERAGE.z_bias("UP"), coverage_bias_dn=COVERAGE.z_bias("DOWN"), sigma_fast=STATE.sigma_fast, sigma_slow=STATE.sigma_slow,
            flip_rate=float(getattr(REGIME, 'flip_rate', 0.0) or 0.0), pv_quality=float(_pv["signal_quality"]), pv_confirms=bool(_pv["participation_confirms"]),
            jump_regime=BIPOWER.jump_regime, oracle_age_ms=int(oracle_age), window_key=get_window_start_unix(),
            cb_lead_agrees=bool(_cb_lead_info.get("direction_agrees", False)), cb_anticipation=bool(_cb_lead_info.get("anticipation_active", False)),
            basis_shrinking=bool(BASIS.rolling_basis != 0 and abs(BASIS.rolling_basis) < abs(getattr(BASIS, '_prev_basis', BASIS.rolling_basis))),
            up_edge_boost=float(state_calibration_adjustments(side="UP", sec_remaining=STATE.sec_remaining, spread=_sig_spread, lag_p50_ms=_sig_lag_p50, sigma_ratio=_sig_sigma_ratio)["edge_boost"]),
            down_edge_boost=float(state_calibration_adjustments(side="DOWN", sec_remaining=STATE.sec_remaining, spread=_sig_spread, lag_p50_ms=_sig_lag_p50, sigma_ratio=_sig_sigma_ratio)["edge_boost"]),
            up_min_p_shift=float(state_calibration_adjustments(side="UP", sec_remaining=STATE.sec_remaining, spread=_sig_spread, lag_p50_ms=_sig_lag_p50, sigma_ratio=_sig_sigma_ratio)["min_p_shift"]),
            down_min_p_shift=float(state_calibration_adjustments(side="DOWN", sec_remaining=STATE.sec_remaining, spread=_sig_spread, lag_p50_ms=_sig_lag_p50, sigma_ratio=_sig_sigma_ratio)["min_p_shift"]),
            up_max_pay_shift=float(state_calibration_adjustments(side="UP", sec_remaining=STATE.sec_remaining, spread=_sig_spread, lag_p50_ms=_sig_lag_p50, sigma_ratio=_sig_sigma_ratio)["max_pay_shift"]),
            down_max_pay_shift=float(state_calibration_adjustments(side="DOWN", sec_remaining=STATE.sec_remaining, spread=_sig_spread, lag_p50_ms=_sig_lag_p50, sigma_ratio=_sig_sigma_ratio)["max_pay_shift"]),
            top_depth=float(max(up_snap.bid_size + up_snap.ask_size, dn_snap.bid_size + dn_snap.ask_size)), edge_buffer_dyn=float(_pre_md["buffer"])
        )

        debug["oracle_source"] = oracle_src
        if implied_up or implied_dn: debug["implied_side"] = "UP" if implied_up else "DN"
        LATEST_DEBUG.update(debug)

        _t_bucket = "20_60" if STATE.sec_remaining <= 60 else ("60_120" if STATE.sec_remaining <= 120 else "120_240")
        _dbg_side = str(debug.get("side") or side or "NA")
        log_calibration_sample(side=_dbg_side, t_bucket=_t_bucket, spread=max(up_sp, dn_sp), lag_ms=float(LAG_ADAPTIVE.p50_ms(_dbg_side if _dbg_side in ("UP", "DOWN") else "UP")), sigma_ratio=_sig_sigma_ratio, edge=float(debug.get("edge", 0.0) or 0.0), p_token=float(debug.get("p_token", 0.0) or 0.0), ask=float(limit_price or 0.0), reason=str(debug.get("reason", "")))

        if side is not None and _cb_lead_info.get("anticipation_active", False) and not _cb_lead_info.get("direction_agrees", False):
            debug["reason"], side = "cb_direction_disagree", None
            LATEST_DEBUG.update(debug)

        _FLOW_ADVERSE_THRESHOLD = 0.35  # FIX: Tightened from 0.55
        if side is not None:
            _flow_c = float(debug.get("flow_centered", 0.0) or 0.0)
            if (side == "UP" and _flow_c < -_FLOW_ADVERSE_THRESHOLD) or (side == "DOWN" and _flow_c > _FLOW_ADVERSE_THRESHOLD):
                debug["reason"], side = "flow_adverse_gate", None
                LATEST_DEBUG.update(debug)

        if side is not None:
            _iadl_dec = IADL.gate(ts_ms=now_ms_val, portfolio=PORTFOLIO, proposed_side=side, z=float(debug.get("z", 0.0) or 0.0), p_cone=float(debug.get("p_cone", 0.5) or 0.5), T_sec=STATE.sec_remaining, up_bid=eff_up_bid, up_ask=eff_up_ask, dn_bid=eff_dn_bid, dn_ask=eff_dn_ask, debug_in=debug)
            debug.update({"iadl": _iadl_dec.kind.value, "iadl_reason": _iadl_dec.reason, "iadl_size_mult": _iadl_dec.size_mult})
            LATEST_DEBUG.update(debug)
            
            if _iadl_dec.kind == DecisionType.REJECT: side = None
            elif _iadl_dec.kind in (DecisionType.CLOSE_THEN_REVERSE, DecisionType.UNWIND_STRADDLE):
                if _iadl_dec.plan and not SIMULATION_MODE and not EXEC_LOCKED:
                    for _op in _iadl_dec.plan: await eq.put({"action": "ORDER", "side": _op.token_side, "token_id": UP_TOKEN_ID if _op.token_side == "UP" else DOWN_TOKEN_ID, "price": round(float(_op.limit_price), 2), "size": float(_op.qty), "order_side": _op.action, "edge": 0.0, "p_cone": float(debug.get("p_cone", 0.5) or 0.5), "z": float(debug.get("z", 0.0) or 0.0), "sigma_1m": STATE.sigma_1m})
                side = None
            elif _iadl_dec.kind == DecisionType.FLATTEN:
                if _iadl_dec.plan and not SIMULATION_MODE:
                    for _op in _iadl_dec.plan: await eq.put({"action": "ORDER", "side": _op.token_side, "token_id": UP_TOKEN_ID if _op.token_side == "UP" else DOWN_TOKEN_ID, "price": round(float(_op.limit_price), 2), "size": float(_op.qty), "order_side": _op.action, "edge": 0.0})
                side = None

        if side is None:
            _now_rej = time.time()
            if _now_rej - getattr(brain_loop, '_last_rej_log', 0) > 1.0:
                brain_loop._last_rej_log = _now_rej

        _p = debug.get("p_cone")
        if _p is not None and np.isfinite(_p): _P_CONE_HISTORY.append(float(_p))
        _pcone_slope = _P_CONE_HISTORY[-1] - _P_CONE_HISTORY[-4] if len(_P_CONE_HISTORY) >= 4 else 0.0
        debug["pcone_slope"] = round(_pcone_slope, 5)

        _reason = debug.get("reason", "?")
        if str(_reason).startswith("FIRE") or (now_ms_val - getattr(brain_loop, "_last_edge_log_ms", 0) >= 1000):
            brain_loop._last_edge_log_ms = now_ms_val
            try:
                import json as _json
                with open("logs/edge_stats.jsonl", "a") as _f:
                    _f.write(_json.dumps({"ts": now_ms_val, "reason": _reason, "edge": debug.get("edge"), "z": debug.get("z"), "p_cone": debug.get("p_cone"), "Z_MIN": debug.get("Z_MIN"), "min_ev": debug.get("min_ev"), "side": debug.get("side"), "spread_max": debug.get("spread_max"), "cap": debug.get("cap"), "flow": round(FLOW.imbalance(up_price=eff_up_bid, down_price=eff_dn_bid), 3), "flow_01": debug.get("flow_01"), "flow_boost": debug.get("flow_boost"), "flow_confirms": debug.get("flow_confirms"), "regime": debug.get("regime"), "T": round(STATE.sec_remaining, 1), "sigma": round(STATE.sigma_1m, 6)}) + "\n")
            except Exception: pass

            try:
                import json as _json
                _strike = round(STATE.open_price, 2) if np.isfinite(STATE.open_price) else None
                if not hasattr(brain_loop, "_zh_window_id"): brain_loop._zh_window_id, brain_loop._zh_last_T = None, 0
                _cur_T = STATE.sec_remaining
                if _cur_T > brain_loop._zh_last_T + 60: brain_loop._zh_window_id = f"{_strike}_{now_ms_val}"
                brain_loop._zh_last_T = _cur_T
                with open("logs/z_history.jsonl", "a") as _f:
                    _f.write(_json.dumps({"ts": now_ms_val, "window_id": brain_loop._zh_window_id, "strike": _strike, "T": round(_cur_T, 1), "z": debug.get("z"), "z_ema": debug.get("z_ema"), "edge": debug.get("edge"), "edge_target": debug.get("edge_target"), "p_cone": debug.get("p_cone"), "spread_max": debug.get("spread_max"), "side": debug.get("side"), "reason": _reason, "oracle_price": round(oracle_px, 2) if np.isfinite(oracle_px) else None, "sigma": round(STATE.sigma_1m, 6), "regime": debug.get("regime"), "up_ask": round(eff_up_ask, 4), "dn_ask": round(eff_dn_ask, 4), "up_bid": round(eff_up_bid, 4), "dn_bid": round(eff_dn_bid, 4), "fired": str(_reason).startswith("FIRE")}) + "\n")
            except Exception: pass

        try:
            TELEMETRY.maybe_emit(portfolio=PORTFOLIO, up_bid=up_snap.best_bid, up_ask=up_snap.best_ask, dn_bid=dn_snap.best_bid, dn_ask=dn_snap.best_ask, extra={"z": debug.get("z"), "p_cone": debug.get("p_cone"), "T": round(STATE.sec_remaining, 1), "bias": IADL.window_bias, "regime": REGIME.label, "pos_state": {"up": POS_UP.inventory, "dn": POS_DOWN.inventory}, "fifo_state": {"up": POS_UP.inventory, "dn": POS_DOWN.inventory}})
        except Exception: pass

        if not str(debug.get("reason", "")).startswith("FIRE"):
            if (not EXEC_LOCKED and POS_UP.inventory > 0 and POS_DOWN.inventory > 0):
                total = POS_UP.inventory + POS_DOWN.inventory
                if total > 0 and max(POS_UP.inventory, POS_DOWN.inventory) / total >= 0.65:
                    ex_side, ex_snap, ex_tok, ex_qty = ("UP", up_snap, UP_TOKEN_ID, POS_UP.inventory - POS_DOWN.inventory) if POS_UP.inventory > POS_DOWN.inventory else ("DOWN", dn_snap, DOWN_TOKEN_ID, POS_DOWN.inventory - POS_UP.inventory)
                    if ex_snap.best_bid > 0.01 and ex_snap.best_ask - ex_snap.best_bid < 0.50:
                        await eq.put({"action": "PAIR_REBALANCE", "side": ex_side, "token_id": ex_tok, "price": round(max(0.01, ex_snap.best_bid - 0.01), 2), "size": max(1.0, math.floor(ex_qty * 0.5)), "order_side": "SELL", "edge": 0.0})
            continue

        _fire_now = ms_now()
        if not hasattr(brain_loop, '_last_fire_ms'): brain_loop._last_fire_ms = 0
        if _fire_now - brain_loop._last_fire_ms < 500: continue
        brain_loop._last_fire_ms = _fire_now

        LATEST_DEBUG["pcone_slope"], LATEST_DEBUG["burst_score"] = round(_pcone_slope, 4), round(max(up_sp, dn_sp) * 3.0, 2)
        if REGIME.sigma_ratio < 0.6: LATEST_DEBUG["sigma_regime"] = "CALM_SIGMA"
        elif REGIME.sigma_ratio > 1.8: LATEST_DEBUG["sigma_regime"] = "EXPLOSIVE_SIGMA"

        if not RTDS.is_fresh(RTDS_FRESH_MS): LATEST_DEBUG["reason"] = "NO_RTDS_NO_ENTRY"; continue

        if EXEC_LOCKED:
            if (ms_now() - EXEC_LOCK_MS) / 1000 > 5.0 or exec_lock_expired(): exec_unlock()
            else: LATEST_DEBUG["reason"] = "ORDER_INFLIGHT"; continue

        if check_emergency_exit(): LATEST_DEBUG["reason"] = "EMERGENCY_EXIT_MODE"; continue
        if CIRCUIT_BREAKER_ACTIVE: LATEST_DEBUG["reason"] = "CIRCUIT_BREAKER"; continue
        if not side: continue

        # FIX: Relaxed time window gates (15.0 and 275.0)
        _T = STATE.sec_remaining
        if not np.isnan(_T):
            if _T > WINDOW_HI: LATEST_DEBUG["reason"] = "WINDOW_EARLY"; continue
            if _T <= WINDOW_LO: LATEST_DEBUG["reason"] = "WINDOW_LATE"; continue

        tr = TAIL_RISK.check(ts_ms=now_ms_val, flip_rate=REGIME.flip_rate, sigma_eff=STATE.sigma_1m, spread=max(up_sp, dn_sp))
        LATEST_DEBUG["tail_risk"] = tr.reason if tr.reason else "OK"

        if str(debug.get("reason", "")).startswith("FIRE_MISPRICING"):
            exec_price, exec_token, exec_order_side, exec_implied, exec_limit = round(float(limit_price), 2), UP_TOKEN_ID if side == "UP" else DOWN_TOKEN_ID, "BUY", False, round(float(limit_price), 2)
        else: exec_price, exec_token, exec_order_side, exec_implied, exec_limit = best_executable_price(side, up_snap, dn_snap, allow_implied=False)

        if exec_price == float("inf"): LATEST_DEBUG["reason"] = "BOTH_BOOKS_ILLIQUID"; continue
        if exec_price < 0.05 or exec_price > 0.95:
            LATEST_DEBUG["reason"] = "EXTREME_PRICE"
            exec_unlock()
            continue

        if SHARPE_PAUSED: LATEST_DEBUG["reason"] = "SHARPE_PAUSE"; continue
        if DAY_KILL_ACTIVE and ms_now() < DAY_KILL_UNTIL_MS: LATEST_DEBUG["reason"] = "DAY_KILL"; continue
        elif DAY_KILL_ACTIVE and ms_now() >= DAY_KILL_UNTIL_MS: DAY_KILL_ACTIVE = False

        limit_price, edge_val, p_fire, z_fire = exec_limit, debug.get("edge", np.nan), debug.get("p_cone", np.nan), debug.get("z", np.nan)
        p_win = p_fire if side == "UP" else (1.0 - p_fire)
        f_kelly = edge_val / (p_win * (1.0 - p_win) + 1e-6) if (p_win * (1.0 - p_win) + 1e-6) > 0 else 0.0

        _kelly_base = 1.0 * SHARPE_MULT
        if debug.get("flow_confirms"): _kelly_base *= 1.2
        if REGIME.sigma_ratio < 0.6: _kelly_base *= 0.8
        _kelly_raw = _kelly_base * f_kelly

        _event_scale = 1.0 + min(1.5, REGIME.vol_detector.intensity) if REGIME.vol_detector.event_active else 1.0
        _jump_scale = 1.0 + min(2.0, JUMP_DETECTOR.jump_intensity / 5.0) if JUMP_DETECTOR.lambda_hat > 0.02 else 1.0
        _regime_mult = max(0.20, min(REGIME.current_params.kelly_multiplier, BIPOWER.kelly_multiplier, MOMENTUM.kelly_multiplier))
        _kelly_L1 = _kelly_raw * (_regime_mult * min(2.0, max(_event_scale, _jump_scale) if _event_scale > 1.0 or _jump_scale > 1.0 else 1.0))

        _fp = FILL_PROB.estimate(max(up_sp, dn_sp), min(up_snap.bid_size + up_snap.ask_size, dn_snap.bid_size + dn_snap.ask_size), bool(debug.get("flow_confirms", False)), float(FLOW.trade_velocity(window_s=5.0)))
        _lag_q = float(LAG_ADAPTIVE.p50_ms(side if side in ("UP", "DOWN") else "UP"))
        _q_score = quality_score(float(edge_val), float(max(up_sp, dn_sp)), float(_fp["p_fill"]), _lag_q)
        _kelly_L2 = _kelly_L1 * max(0.25, max(0.30, _fp["fill_kelly"]) * min(debug.get("spread_size_mult", 1.0), debug.get("late_kelly_mult", 1.0)))

        _kelly_L3 = _kelly_L2 * (min(SPRT.kelly_multiplier, LOSS_STREAK_KELLY_PENALTY) * tr.kelly_cap)
        _kelly_L4 = _kelly_L3 * EXPIRY.kelly_mult(STATE.sec_remaining)

        f_used = max(0.03 * _kelly_raw if _kelly_raw > 0 else 0.0, min(1.0, _kelly_L4))
        if _q_score >= 0.85: _q_floor, _q_cap = min(1.0, max(0.0, 0.15 * _kelly_raw)), 1.00
        elif _q_score >= 0.70: _q_floor, _q_cap = min(1.0, max(0.0, 0.10 * _kelly_raw)), 1.00
        elif _q_score >= 0.55: _q_floor, _q_cap = min(1.0, max(0.0, 0.04 * _kelly_raw)), 0.85
        elif _q_score >= 0.40: _q_floor, _q_cap = 0.0, 0.65
        else: _q_floor, _q_cap = 0.0, 0.40
        f_used = min(1.0, max(_q_floor, min(_q_cap, f_used)))

        spread_for_haircut = max(up_sp, dn_sp)
        if spread_for_haircut > 0.80: spread_haircut = 0.25
        elif spread_for_haircut > 0.40: spread_haircut = 0.50
        elif spread_for_haircut > 0.20: spread_haircut = 0.70
        else: spread_haircut = 1.0

        _flow_sz_scale, _z_scale = debug.get("flow_size_scale", 1.0), max(0.5, min(1.5, abs(debug.get("z", 0.0)) / 2.0))
        _sigma_scale = 1.0 + 0.5 * min(1.0, max(0.0, (STATE.sigma_fast / STATE.sigma_slow if STATE.sigma_slow > 1e-10 else 1.0) - 1.0))
        capped_size = max(max(1, math.ceil(1.0 / limit_price)), math.floor(math.floor(MAX_SPEND_PER_ORDER_USD / limit_price) * f_used * spread_haircut * _flow_sz_scale * _z_scale * _sigma_scale))

        if (int(debug.get("side_fires", 1)) - 1) > 0 and float(debug.get("edge", 0.0)) <= float(debug.get("prev_fire_edge", float(debug.get("edge", 0.0)))):
            capped_size = max(max(1, math.ceil(1.0 / limit_price)), int(capped_size * (0.70 ** (int(debug.get("side_fires", 1)) - 1))))

        bankroll = max(0.0, float(WINDOW_BANKROLL))
        if bankroll > 0: capped_size = max(max(1, math.ceil(1.0 / limit_price)), min(capped_size, math.floor(TAIL_RISK.clamp_size(capped_size * limit_price, bankroll) / limit_price)))

        collateral, liq = bankroll, portfolio_liquidation_value(up_snap.best_bid, dn_snap.best_bid)
        a_mult, b_mult, why = regime_risk_multipliers()
        equity = float(collateral + liq + SESSION_PNL)
        dd_mult = drawdown_risk_multiplier(equity)
        day_pnl, day_dd = update_day_kill_switch(equity)
        
        a = max(RISK_A_MIN, min(RISK_A_MAX, RISK_A_BASE * a_mult * dd_mult))
        b = max(RISK_B_MIN, min(RISK_B_MAX, RISK_B_BASE * b_mult * dd_mult))
        RISK_BUDGET = a * collateral + b * liq

        gross_before, net_before = liq, portfolio_net_shares()
        if gross_before + (capped_size * limit_price) > RISK_BUDGET:
            capped_size = int(max(max(1, math.ceil(1.0 / limit_price)), math.floor(max(0.0, RISK_BUDGET - gross_before) / max(0.01, limit_price))))

        MAX_GROSS_SHARES, MAX_NET_SHARES = int(os.getenv("MAX_GROSS_SHARES", "90")), int(os.getenv("MAX_NET_SHARES", "60"))
        if portfolio_gross_shares() + capped_size > MAX_GROSS_SHARES:
            capped_size = max(max(1, math.ceil(1.0 / limit_price)), int(MAX_GROSS_SHARES - portfolio_gross_shares()))
        if abs(portfolio_net_shares() + (capped_size if side == "UP" else -capped_size)) > MAX_NET_SHARES:
            capped_size = max(max(1, math.ceil(1.0 / limit_price)), int(min(capped_size, MAX_NET_SHARES - abs(portfolio_net_shares()))))

        log_risk_state(collateral, liq, RISK_BUDGET, gross_before, net_before, capped_size * limit_price, gross_before + (capped_size * limit_price), capped_size, limit_price, side)
        LAST_FIRE_SNAPSHOT = dict(debug)

        if SIMULATION_MODE: continue

        _mode = debug.get("mode", "regular")
        if not fire_allowed(exec_token, exec_order_side, _mode): LATEST_DEBUG["reason"] = "FIRE_DEBOUNCE"; continue
        if _pv["trap"] and _pv["spread_norm"] > 1.4: LATEST_DEBUG["reason"] = "PV_TRAP_BLOCK"; continue
        if BIPOWER.jump_regime and debug.get("mode", "impulse") not in ("drift", "mispricing"): LATEST_DEBUG["reason"] = "BIPOWER_MR_SUPPRESS"; continue
        if debug.get("mode", "impulse") not in ("drift", "mispricing") and not VWAP.mr_allowed(): LATEST_DEBUG["reason"] = "VWAP_INSIDE_FAIR"; continue

        _signal_snap = up_snap if exec_token == UP_TOKEN_ID else dn_snap
        _md_lag50 = float(LAG_ADAPTIVE.p50_ms(side if side in ("UP", "DOWN") else "UP"))
        _md_flow = float(debug.get("flow_centered", 0.0) or FLOW.imbalance(up_price=eff_up_bid, down_price=eff_dn_bid))
        _md_sp = float(_signal_snap.best_ask - _signal_snap.best_bid)

        _md = micro_edge_buffer(
            spread=_md_sp, bid_sz=float(getattr(_signal_snap, "bid_size", 0.0)) * 0.5, ask_sz=float(getattr(_signal_snap, "ask_size", 0.0)) * 0.5,
            sigma=float(STATE.sigma_1m), lag_p50_ms=_md_lag50, lag_p10_ms=float(LAG_ADAPTIVE.p10_ms(side if side in ("UP", "DOWN") else "UP")), trade_vel=float(FLOW.trade_velocity(window_s=5.0)),
            flow_bias=_md_flow, retries_expected=0.25 if debug.get("spread_tier", "taker_allowed") in ("maker_only", "maker_preferred") else 0.0,
        )

        _leak_hat = LEAK_MODEL.predict(lag50_ms=_md_lag50, sigma=float(STATE.sigma_1m), spread=_md_sp, flow=_md_flow, regime=REGIME.label)
        
        await eq.put({
            "action": "ORDER", "side": side, "token_id": exec_token, "price": limit_price, "size": capped_size, "order_side": exec_order_side, "edge": edge_val,
            "edge_target": float(debug.get("edge_target", 0.0) or 0.0), "edge_buffer_dyn": round(_md["buffer"], 5), "edge_target_exec": round(float(debug.get("edge_target", 0.0) or 0.0) + _md["buffer"], 5),
            "micro_decay_why": _md["why"], "lag50_ms": round(_md_lag50, 1), "p_cone": p_fire, "z": z_fire, "sigma_1m": STATE.sigma_1m, "z_ema": float(debug.get("z_ema", 0.0) or 0.0),
            "spread_tier": debug.get("spread_tier", "taker_allowed"), "mode": _mode, "regime": REGIME.label, "flow": round(FLOW.imbalance(up_price=eff_up_bid, down_price=eff_dn_bid), 3),
            "signal_type": debug.get("reason", ""), "signal_ask": round(float(_signal_snap.best_ask), 4), "signal_bid": round(float(_signal_snap.best_bid), 4), "fire_id": f"{int(ms_now())}_{side}_{exec_token}"
        })
        confirm_sniper_fire()

def _on_rtds_price(price: float, ts_ms: int) -> None:
    STRIKE_CAPTURE.feed(price, ts_ms)
    TAIL_RISK.feed_oracle(price, ts_ms)
    _update_sigma(price, ts_ms)
    if STATE.btc_ts_ms > 0 and not np.isnan(STATE.btc_price): BASIS.update(price, STATE.btc_price)

async def main() -> None:
    global MARKET_ID, UP_TOKEN_ID, DOWN_TOKEN_ID, WINDOW_OPEN_MS, FIRST_REAL_BOOK_MS
    logger.info("=" * 60)
    logger.info("PolyBot v2 starting — Chainlink RTDS oracle mode")
    logger.info("=" * 60)

    loop = asyncio.get_running_loop()
    info = None
    for _disc_attempt in range(60):
        info = await loop.run_in_executor(None, _auto_discover_market_sync)
        if info: break
        logger.warning(f"No active market found. Retrying in 10s… ({_disc_attempt+1}/60)")
        await asyncio.sleep(10)
    if not info: return

    MARKET_ID, UP_TOKEN_ID, DOWN_TOKEN_ID, WINDOW_OPEN_MS, FIRST_REAL_BOOK_MS = info["market_id"], info["up_token"], info["down_token"], ms_now(), 0
    for _tid in (UP_TOKEN_ID, DOWN_TOKEN_ID):
        if _tid not in POLY_STATE: POLY_STATE[_tid] = {"bid": 0.0, "ask": 1.0, "bid_size": 0.0, "ask_size": 0.0, "last_update": 0, "source": "init"}
    FLOW.set_tokens(UP_TOKEN_ID, DOWN_TOKEN_ID)
    GOLDSKY.set_market(MARKET_ID, UP_TOKEN_ID, DOWN_TOKEN_ID)

    await loop.run_in_executor(None, _seed_book_from_rest)
    global WINDOW_BANKROLL
    WINDOW_BANKROLL = min(float(await loop.run_in_executor(None, get_available_usdc_balance)), WINDOW_SPEND_CAP_USD)
    await loop.run_in_executor(None, _prime_tick_size_cache, UP_TOKEN_ID)
    await loop.run_in_executor(None, _prime_tick_size_cache, DOWN_TOKEN_ID)

    if client is not None and UP_TOKEN_ID:
        try: set_market_fee_rate(client.get_fee_rate_bps(UP_TOKEN_ID))
        except Exception: pass

    await asyncio.gather(
        chainlink_rtds_task(on_price=_on_rtds_price),
        coinbase_trade_task(),
        polymarket_book_task("wss://ws-subscriptions-clob.polymarket.com/ws/market"),
        execution_loop(),
        market_clock_task(info["end_time"]),
        brain_loop(execution_queue),
        GOLDSKY.poll_task(),
        watchdog_task(),
        dashboard_task(),
    )

async def watchdog_task():
    while True:
        await asyncio.sleep(1)
        rtds_age, cb_age_s = RTDS.age_ms() / 1000.0, (ms_now() - STATE.btc_ts_ms) / 1000.0 if STATE.btc_ts_ms > 0 else 999.0
        if rtds_age > 10.0 and cb_age_s > 10.0:
            if not STATE.trading_suspended:
                STATE.trading_suspended = True
        elif STATE.trading_suspended:
            STATE.trading_suspended = False

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: logger.info("Bot stopped by user.")