# run_bot.ps1 — PolyBot v2 autonomous daily loop (Windows/PowerShell)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "config" | Out-Null

Write-Host "Starting PolyBot v2 Institutional Stack..."

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] Launching main.py..."

    # 1. Run the main trading engine
    python main.py
    $exitCode = $LASTEXITCODE

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] Bot stopped with code $exitCode. Running end-of-session calibration..."

    # 2. Generate the daily session buckets
    python calibration.py

    # 3. Merge the daily buckets into the master database
    python merge_calibration.py

    # 4. Backup the logs so the next session starts fresh
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    if (Test-Path "logs\trades.csv") {
        Move-Item "logs\trades.csv" "logs\trades_$stamp.csv" -Force
    }

    if ($exitCode -eq 0) {
        Write-Host "[$ts] Clean exit. Stopping."
        break
    } else {
        Write-Host "[$ts] Restarting engine in 5 seconds..."
        Start-Sleep -Seconds 5
    }
}
