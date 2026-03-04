import os
import csv
import json
import time

# --- Baseline Calibration ---
BASELINE_AVG_WIN_USD = 2.79  # Your pristine backtest average win size
DECAY_ALARM_THRESHOLD = 0.40 # 40% threshold (The 0.35 cliff + 5% safety buffer)
TRADE_WINDOW = 50            # Look at the last N trades

def analyze_health():
    if not os.path.exists("logs/trades.csv"):
        return

    wins_usd = []
    losses_usd = []
    slippage_bps = []

    # Parse the CSV backwards to get the most recent N fills
    with open("logs/trades.csv", "r") as f:
        reader = list(csv.DictReader(f))
        fills = [row for row in reader if row.get("result") in ("FILLED", "FAK_AGGRESSIVE")]
        
        for row in fills[-TRADE_WINDOW:]:
            try:
                size = float(row.get("size", 0))
                realized_edge = float(row.get("realized_edge", 0))
                leak_bps = float(row.get("edge_leak_bps", 0))
                
                # Approximate USD value of the locked edge
                ev_usd = realized_edge * size
                
                if ev_usd > 0:
                    wins_usd.append(ev_usd)
                else:
                    losses_usd.append(ev_usd)
                    
                slippage_bps.append(max(0, leak_bps))
            except ValueError:
                continue

    if not wins_usd:
        return

    # Calculate Metrics
    win_rate = len(wins_usd) / max(1, (len(wins_usd) + len(losses_usd)))
    current_avg_win = sum(wins_usd) / len(wins_usd)
    avg_slip_bps = sum(slippage_bps) / len(slippage_bps) if slippage_bps else 0

    # THE DECAY FACTOR
    decay_factor = current_avg_win / BASELINE_AVG_WIN_USD

    # Console Output (clear screen for dashboard feel)
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*50)
    print(f"🔥 LIVE EDGE HEALTH MONITOR (Last {len(wins_usd) + len(losses_usd)} trades)")
    print("="*50)
    print(f"Win Rate:           {win_rate*100:.1f}%")
    print(f"Avg Win (USD):      ${current_avg_win:.2f} (Baseline: ${BASELINE_AVG_WIN_USD:.2f})")
    print(f"Avg Slippage:       {avg_slip_bps:.1f} bps")
    print("-" * 50)
    
    kill_switch = False
    if decay_factor <= DECAY_ALARM_THRESHOLD:
        print(f"🚨 DECAY ALERT: {decay_factor:.2f} <= {DECAY_ALARM_THRESHOLD:.2f}")
        print("🚨 EDGE HAS COLLAPSED TO FATAL LEVELS.")
        kill_switch = True
    elif decay_factor <= 0.55:
        print(f"⚠️ WARNING: Edge decaying ({decay_factor:.2f}). Approaching cliff.")
    else:
        print(f"✅ SYSTEM HEALTHY: Decay Factor = {decay_factor:.2f}")
    print("="*50)

    # Write to state file for main.py to read
    os.makedirs("config", exist_ok=True)
    with open("config/edge_health.json", "w") as f:
        json.dump({
            "decay_factor": round(decay_factor, 3),
            "kill_switch_active": kill_switch,
            "avg_slip_bps": round(avg_slip_bps, 1),
            "ts_ms": int(time.time() * 1000)
        }, f)

if __name__ == "__main__":
    print("Starting Edge Health Monitor...")
    while True:
        try:
            analyze_health()
        except Exception as e:
            print(f"Monitor error: {e}")
        time.sleep(10) # Update every 10 seconds