import json
import os
import math

# File paths
SESSION_FILE = "logs/calibration_output.json"
MASTER_FILE = "config/master_calibration.json"

# Max weight a single new session can have (e.g., 0.20 = 20%)
# Prevents one crazy volatility day from destroying your baseline
MAX_SESSION_WEIGHT = 0.20

def merge_calibrations():
    if not os.path.exists(SESSION_FILE):
        print(f"No session file found at {SESSION_FILE}. Exiting.")
        return

    # Load recent session
    with open(SESSION_FILE, "r") as f:
        session_data = json.load(f)

    # Load or initialize master file
    if os.path.exists(MASTER_FILE):
        with open(MASTER_FILE, "r") as f:
            master_data = json.load(f)
    else:
        master_data = {}

    merged_count = 0
    new_count = 0

    for bucket_key, s_stats in session_data.items():
        s_fills = s_stats.get("fills", 0)
        if s_fills == 0:
            continue

        if bucket_key not in master_data:
            # Brand new micro-state discovered
            master_data[bucket_key] = s_stats
            new_count += 1
        else:
            # Blend existing state
            m_stats = master_data[bucket_key]
            m_fills = m_stats.get("fills", 0)

            # Calculate blend weight based on relative sample size, capped at MAX_SESSION_WEIGHT
            # This ensures stable, institutional-grade smoothing
            total_fills = m_fills + s_fills
            alpha = min(s_fills / total_fills, MAX_SESSION_WEIGHT) if total_fills > 0 else 0

            # EMA Update for key metrics
            new_wr = (alpha * s_stats.get("wr", 0.5)) + ((1 - alpha) * m_stats.get("wr", 0.5))
            new_leak = (alpha * s_stats.get("leak", 0.0)) + ((1 - alpha) * m_stats.get("leak", 0.0))
            new_adj_edge = (alpha * s_stats.get("adj_edge", 0.0)) + ((1 - alpha) * m_stats.get("adj_edge", 0.0))

            master_data[bucket_key] = {
                "fills": total_fills,
                "wr": round(new_wr, 4),
                "leak": round(new_leak, 5),
                "adj_edge": round(new_adj_edge, 5)
            }
            merged_count += 1

    # Save the updated master map
    os.makedirs(os.path.dirname(MASTER_FILE), exist_ok=True)
    with open(MASTER_FILE, "w") as f:
        json.dump(master_data, f, indent=4)

    print("=" * 50)
    print("🧠 CALIBRATION MERGE COMPLETE")
    print("=" * 50)
    print(f"Master Database Size: {len(master_data)} buckets")
    print(f"Updated Existing: {merged_count}")
    print(f"Discovered New: {new_count}")
    print("-" * 50)

if __name__ == "__main__":
    merge_calibrations()