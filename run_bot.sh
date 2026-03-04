#!/bin/bash

mkdir -p logs
mkdir -p config

echo "Starting PolyBot v2 Institutional Stack..."

while true; do
    echo "[$(date)] Launching main.py..."

    # 1. Run the main trading engine
    python3 main.py

    EXIT_CODE=$?

    echo "[$(date)] Bot stopped with code $EXIT_CODE. Running end-of-session calibration..."

    # 2. Generate the daily session buckets
    python3 calibration.py

    # 3. Merge the daily buckets into the master database
    python3 merge_calibration.py

    # 4. Backup the logs so the next session starts fresh
    mv logs/trades.csv logs/trades_$(date +%Y%m%d_%H%M%S).csv 2>/dev/null

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Clean exit. Stopping."
        break
    else
        echo "[$(date)] Restarting engine in 5 seconds..."
        sleep 5
    fi
done
