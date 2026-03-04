import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

def fee_per_share(price: float, rate: float = 0.0625) -> float:
    # Polymarket taker fee approximation used in your system
    return price * (1.0 - price) * rate

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

@dataclass
class Tick:
    ts_ms: int
    up_bid: float
    up_ask: float
    dn_bid: float
    dn_ask: float
    p_cone: float
    z_rtds: float
    cb_flow: float
    cb_z: float
    basis_stable: bool

    @property
    def up_mid(self) -> float:
        return 0.5 * (self.up_bid + self.up_ask)

    @property
    def dn_mid(self) -> float:
        return 0.5 * (self.dn_bid + self.dn_ask)

    @property
    def up_spread(self) -> float:
        return max(0.0, self.up_ask - self.up_bid)

    @property
    def dn_spread(self) -> float:
        return max(0.0, self.dn_ask - self.dn_bid)

    @property
    def spread_max(self) -> float:
        return max(self.up_spread, self.dn_spread)

def load_ticks_from_jsonl(path: str) -> List[Tick]:
    ticks: List[Tick] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue

            ts = int(d.get("ts") or d.get("ts_ms") or 0)
            if ts <= 0:
                continue

            # These field names match your z_history.jsonl writer style.
            up_bid = safe_float(d.get("up_bid"))
            up_ask = safe_float(d.get("up_ask"))
            dn_bid = safe_float(d.get("dn_bid"))
            dn_ask = safe_float(d.get("dn_ask"))
            p_cone = safe_float(d.get("p_cone"), 0.5)
            z = safe_float(d.get("z"), 0.0)

            cb_flow = safe_float(d.get("cb_flow"), 0.0)
            cb_z = safe_float(d.get("cb_z"), 0.0)
            basis_stable = bool(d.get("basis_stable", False))

            if None in (up_bid, up_ask, dn_bid, dn_ask):
                continue

            # sanity clamp
            up_bid = max(0.0, min(1.0, up_bid))
            up_ask = max(0.0, min(1.0, up_ask))
            dn_bid = max(0.0, min(1.0, dn_bid))
            dn_ask = max(0.0, min(1.0, dn_ask))

            # skip crossed
            if up_ask < up_bid or dn_ask < dn_bid:
                continue

            ticks.append(Tick(
                ts_ms=ts,
                up_bid=up_bid, up_ask=up_ask,
                dn_bid=dn_bid, dn_ask=dn_ask,
                p_cone=p_cone, z_rtds=z,
                cb_flow=cb_flow, cb_z=cb_z,
                basis_stable=basis_stable
            ))

    ticks.sort(key=lambda t: t.ts_ms)
    return ticks

def index_by_time(ticks: List[Tick]) -> List[int]:
    return [t.ts_ms for t in ticks]

def find_tick_at_or_after(ticks: List[Tick], times: List[int], ts_ms: int) -> Optional[int]:
    # binary search
    lo, hi = 0, len(times) - 1
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if times[mid] >= ts_ms:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

@dataclass
class EventResult:
    ts0: int
    side: str               # "UP" or "DOWN" we would buy
    cb_flow: float
    spread0: float
    entry_price: float      # maker-intent entry (ask)
    h_ms: int
    mid_move: float         # mid(t+h)-mid(t0)
    bid_exit: float         # bid at horizon
    taker_ev: float         # (bid - fee(bid)) - entry_price
    mid_ev_proxy: float     # mid_move - fee(mid) (rough proxy)

def run_backtest(
    ticks: List[Tick],
    flow_th: float,
    max_spread: float,
    horizons_ms: List[int],
) -> List[EventResult]:
    times = index_by_time(ticks)
    results: List[EventResult] = []

    for i, t0 in enumerate(ticks):
        # Event definition
        if not t0.basis_stable:
            continue
        if abs(t0.cb_flow) < flow_th:
            continue
        if t0.spread_max > max_spread:
            continue

        side = "UP" if t0.cb_flow > 0 else "DOWN"

        # Entry price = ask of chosen side (maker-intent optimistic)
        entry = t0.up_ask if side == "UP" else t0.dn_ask
        spread0 = t0.spread_max

        for h in horizons_ms:
            j = find_tick_at_or_after(ticks, times, t0.ts_ms + h)
            if j is None:
                continue
            th = ticks[j]

            mid0 = t0.up_mid if side == "UP" else t0.dn_mid
            mid1 = th.up_mid if side == "UP" else th.dn_mid
            bid1 = th.up_bid if side == "UP" else th.dn_bid

            mid_move = mid1 - mid0
            # realized exit value after taker fee
            exit_net = bid1 - fee_per_share(bid1)
            taker_ev = exit_net - entry

            # mid proxy (not tradeable, just direction strength)
            mid_ev_proxy = (mid1 - fee_per_share(mid1)) - entry

            results.append(EventResult(
                ts0=t0.ts_ms,
                side=side,
                cb_flow=t0.cb_flow,
                spread0=spread0,
                entry_price=entry,
                h_ms=h,
                mid_move=mid_move,
                bid_exit=bid1,
                taker_ev=taker_ev,
                mid_ev_proxy=mid_ev_proxy
            ))

    return results

def summarize(results: List[EventResult]) -> Dict:
    if not results:
        return {"n": 0}

    # group by horizon
    by_h: Dict[int, List[EventResult]] = {}
    for r in results:
        by_h.setdefault(r.h_ms, []).append(r)

    out = {"n_events_x_h": len(results), "by_horizon": {}}
    for h, rs in sorted(by_h.items()):
        n = len(rs)
        hit = sum(1 for r in rs if r.taker_ev > 0)
        avg_ev = sum(r.taker_ev for r in rs) / n
        avg_mid = sum(r.mid_move for r in rs) / n
        p25 = sorted(r.taker_ev for r in rs)[max(0, int(0.25*n)-1)]
        p50 = sorted(r.taker_ev for r in rs)[max(0, int(0.50*n)-1)]
        p75 = sorted(r.taker_ev for r in rs)[max(0, int(0.75*n)-1)]

        out["by_horizon"][str(h)] = {
            "n": n,
            "hit_rate": round(hit / n, 4),
            "avg_taker_ev": round(avg_ev, 6),
            "avg_mid_move": round(avg_mid, 6),
            "p25_taker_ev": round(p25, 6),
            "p50_taker_ev": round(p50, 6),
            "p75_taker_ev": round(p75, 6),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-history", default="logs/z_history.jsonl", help="Path to z_history.jsonl (must include cb_flow/cb_z/basis_stable)")
    ap.add_argument("--flow-th", type=float, default=1.5, help="OFI threshold in BTC over window (e.g. 1.5)")
    ap.add_argument("--max-spread", type=float, default=0.20, help="Skip events if spread_max exceeds this")
    ap.add_argument("--horizons", default="250,500,1000,2000,3000", help="Comma-separated horizons in ms")
    args = ap.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    ticks = load_ticks_from_jsonl(args.z_history)
    print(f"Loaded ticks: {len(ticks)}")

    results = run_backtest(
        ticks=ticks,
        flow_th=args.flow_th,
        max_spread=args.max_spread,
        horizons_ms=horizons,
    )

    summ = summarize(results)
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()