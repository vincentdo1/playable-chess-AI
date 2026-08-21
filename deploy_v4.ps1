# Serve the v4 model with the gauntlet configuration.

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

# Serving configuration
$env:MODEL_PATH              = $Model
$env:MAGNUS_USE_MCTS         = "1"
$env:MAGNUS_MCTS_SIMULATIONS = "200"
$env:MAGNUS_BLUNDER_GUARD    = "0"
$env:MAGNUS_TEMPERATURE      = "0.0"
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
