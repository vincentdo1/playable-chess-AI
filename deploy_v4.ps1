# Deploy the Phase 2 v4-full model to the backend.
# Run AFTER training + gauntlet finish. Serves model/grandmaster_resnet_v4_full.pt
# with the strong config (MCTS 200, blunder guard OFF -- the guard hurts v4).
#
# Usage:  .\deploy_v4.ps1
# Stop the server with Ctrl+C. Runs in the foreground so you can see the
# "Model loaded ... epoch 2 | val_loss=..." line that confirms v4 is live.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Model  = "model/grandmaster_resnet_v4_full.pt"

if (-not (Test-Path $Python)) {
    Write-Host "FATAL: $Python not found -- expected the project venv." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $Model)) {
    Write-Host "FATAL: $Model not found -- has training finished?" -ForegroundColor Red
    exit 1
}

# --- Serving config for v4 ---
$env:MODEL_PATH              = $Model
$env:MAGNUS_USE_MCTS         = "1"     # strong search
$env:MAGNUS_MCTS_SIMULATIONS = "200"   # matches the gauntlet / Phase-1 config
$env:MAGNUS_BLUNDER_GUARD    = "0"     # guard hurts v4; MCTS bypasses it anyway
$env:MAGNUS_TEMPERATURE      = "0.0"   # greedy at the root
$env:PYTHONUNBUFFERED        = "1"

Write-Host ""
Write-Host "=== Deploying v4-full ===" -ForegroundColor Cyan
Write-Host "  MODEL_PATH              = $env:MODEL_PATH"
Write-Host "  MAGNUS_USE_MCTS         = $env:MAGNUS_USE_MCTS"
Write-Host "  MAGNUS_MCTS_SIMULATIONS = $env:MAGNUS_MCTS_SIMULATIONS"
Write-Host "  MAGNUS_BLUNDER_GUARD    = $env:MAGNUS_BLUNDER_GUARD"
Write-Host ""
Write-Host "Starting backend (Ctrl+C to stop)..." -ForegroundColor Green
Write-Host ""

& $Python -m backend.app
