# calibration.py — Offline Threshold Recalibration from Production Logs
"""
Nightly recalibration job: reads ev_calibration.csv + trades.csv,
estimates realized EV per bucket, and outputs bucket-specific threshold
adjustments to logs/calibration_output.json.

Usage (standalone):
    python calibration.py [--cal-file logs/ev_calibration.csv] [--trades-file logs/trades.csv]

Usage (from main.py at startup):
    from calibration import load_calibration_overrides
    overrides = load_calibration_overrides("logs/calibration_output.json")
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ── Bucket Definitions ─────────────────────────────────────────────────

def t_bucket(sec_remaining: float) -> str:
    if sec_remaining <= 20:
        return "0_20"
    elif sec_remaining <= 60:
        return "20_60"
    elif sec_remaining <= 120:
        return "60_120"
    return "120_240"


def spread_bucket(spread: float) -> str:
    if spread < 0.03:
        return "tight"
    elif spread < 0.06:
        return "normal"
    return "wide"


def lag_bucket(lag_ms: float) -> str:
    if lag_ms < 450:
        return "fast"
    elif lag_ms < 900:
        return "normal"
    return "slow"


def sigma_bucket(sigma_ratio: float) -> str:
    if sigma_ratio < 0.8:
        return "low"
    elif sigma_ratio < 1.5:
        return "normal"
    return "high"


# ── Bucket Stats ───────────────────────────────────────────────────────

@dataclass
class BucketStats:
    """Accumulator for a single calibration bucket."""
    signals: int = 0
    fills: int = 0
    wins: int = 0
    total_edge_at_signal: float = 0.0
    total_realized_edge: float = 0.0
    total_pnl: float = 0.0

    def add_signal(self, edge: float):
        self.signals += 1
        self.total_edge_at_signal += edge

    def add_fill(self, edge_at_signal: float, realized_edge: float, pnl: float, won: bool):
        self.fills += 1
        self.total_realized_edge += realized_edge
        self.total_pnl += pnl
        if won:
            self.wins += 1

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.fills)

    @property
    def avg_signal_edge(self) -> float:
        return self.total_edge_at_signal / max(1, self.signals)

    @property
    def avg_realized_edge(self) -> float:
        return self.total_realized_edge / max(1, self.fills)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / max(1, self.fills)

    @property
    def leakage(self) -> float:
        if self.avg_signal_edge > 1e-9:
            return (self.avg_signal_edge - self.avg_realized_edge) / self.avg_signal_edge
        return 0.0


# ── Core Calibration ───────────────────────────────────────────────────

def bucket_key(side: str, t_b: str, sp_b: str, lag_b: str, sig_b: str) -> str:
    return f"{side}|{t_b}|{sp_b}|{lag_b}|{sig_b}"


def compute_adjustments(stats: BucketStats) -> dict:
    """Compute threshold adjustments from bucket stats.
    Positive edge_boost = raise the bar (harder to fire).
    Negative edge_boost = lower the bar (easier to fire)."""
    adj = {"edge_boost": 0.0, "min_p_shift": 0.0, "max_pay_shift": 0.0}

    if stats.fills < 5:
        return adj  # not enough data

    # If realized EV is strongly positive, lower edge threshold (fire more)
    if stats.avg_pnl > 0.02 and stats.win_rate > 0.55:
        adj["edge_boost"] = -0.002
        adj["max_pay_shift"] = 0.005
    # If realized EV is negative, raise edge threshold (fire less)
    elif stats.avg_pnl < -0.01 or stats.win_rate < 0.40:
        adj["edge_boost"] = 0.004
        adj["min_p_shift"] = 0.005
        adj["max_pay_shift"] = -0.008
    # If leakage is high, tighten max_pay
    if stats.leakage > 0.40:
        adj["max_pay_shift"] -= 0.005
    # If leakage is low (good execution), permit slightly more
    elif stats.leakage < 0.15 and stats.avg_pnl > 0:
        adj["max_pay_shift"] += 0.003

    return adj


def run_calibration(cal_file: str = "logs/ev_calibration.csv",
                    trades_file: str = "logs/trades.csv",
                    output_file: str = "logs/calibration_output.json") -> dict:
    """Main calibration routine. Reads logs, computes per-bucket stats, outputs adjustments."""
    buckets: dict[str, BucketStats] = defaultdict(BucketStats)

    # ── Read calibration samples (all signals, including rejected) ──
    if os.path.exists(cal_file):
        with open(cal_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    side = row["side"]
                    tb = row["t_bucket"]
                    spb = spread_bucket(float(row["spread"]))
                    lb = lag_bucket(float(row["lag_ms"]))
                    sb = sigma_bucket(float(row["sigma_ratio"]))
                    edge = float(row["edge"])
                    key = bucket_key(side, tb, spb, lb, sb)
                    buckets[key].add_signal(edge)
                except (KeyError, ValueError):
                    continue

    # ── Read trades (fills with outcomes) ──
    if os.path.exists(trades_file):
        with open(trades_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    if len(row) < 22:
                        continue
                    # trades.csv columns: ts, side, token, price, size, status, oid, exec_ms, _,
                    #   edge, p_cone, z, sigma, oracle_src, oracle_px, method, error, regime,
                    #   edge_target, mode, flow, realized_edge, edge_leak_bps, edge_leak_ratio
                    status = row[5]
                    if status != "FILLED":
                        continue
                    side = row[1]
                    edge_at_signal = float(row[9]) if row[9] else 0.0
                    price = float(row[3]) if row[3] else 0.0
                    sigma = float(row[11]) if row[11] else 0.001
                    # Reconstruct buckets from available data
                    # T bucket: not directly in trades.csv, use sigma as proxy for time
                    # For now, use a default bucket
                    tb = "60_120"  # default; ideally we'd join on timestamp
                    spb = "normal"
                    lb = "normal"
                    sb = sigma_bucket(sigma)

                    # EV attribution columns (P4)
                    realized_edge = float(row[21]) if len(row) > 21 and row[21] else 0.0
                    pnl = realized_edge * float(row[4]) if row[4] else 0.0
                    won = realized_edge > 0

                    key = bucket_key(side, tb, spb, lb, sb)
                    buckets[key].add_fill(edge_at_signal, realized_edge, pnl, won)
                except (IndexError, ValueError):
                    continue

    # ── Compute per-bucket adjustments ──
    output = {}
    for key, stats in buckets.items():
        adj = compute_adjustments(stats)
        output[key] = {
            "edge_boost": round(adj["edge_boost"], 5),
            "min_p_shift": round(adj["min_p_shift"], 5),
            "max_pay_shift": round(adj["max_pay_shift"], 5),
            "stats": {
                "signals": stats.signals,
                "fills": stats.fills,
                "win_rate": round(stats.win_rate, 3),
                "avg_signal_edge": round(stats.avg_signal_edge, 5),
                "avg_realized_edge": round(stats.avg_realized_edge, 5),
                "avg_pnl": round(stats.avg_pnl, 5),
                "leakage": round(stats.leakage, 3),
            },
        }

    # ── Write output ──
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    return output


def load_calibration_overrides(path: str = "logs/calibration_output.json") -> dict:
    """Load calibration output and return lookup dict of bucket adjustments.
    Returns empty dict if file doesn't exist (graceful fallback)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_bucket_adjustment(overrides: dict, side: str, sec_remaining: float,
                          spread: float, lag_ms: float, sigma_ratio: float) -> dict:
    """Look up calibration adjustment for the given market state.
    Falls back to default (zero shifts) if no match."""
    key = bucket_key(
        side,
        t_bucket(sec_remaining),
        spread_bucket(spread),
        lag_bucket(lag_ms),
        sigma_bucket(sigma_ratio),
    )
    entry = overrides.get(key, {})
    return {
        "edge_boost": entry.get("edge_boost", 0.0),
        "min_p_shift": entry.get("min_p_shift", 0.0),
        "max_pay_shift": entry.get("max_pay_shift", 0.0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalibrate edge thresholds from production logs")
    parser.add_argument("--cal-file", default="logs/ev_calibration.csv")
    parser.add_argument("--trades-file", default="logs/trades.csv")
    parser.add_argument("--output", default="logs/calibration_output.json")
    args = parser.parse_args()

    result = run_calibration(args.cal_file, args.trades_file, args.output)
    print(f"Calibration complete: {len(result)} buckets written to {args.output}")
    for key, data in sorted(result.items()):
        s = data["stats"]
        if s["fills"] >= 5:
            print(f"  {key}: fills={s['fills']} wr={s['win_rate']:.0%} "
                  f"leak={s['leakage']:.0%} adj_edge={data['edge_boost']:+.4f}")
