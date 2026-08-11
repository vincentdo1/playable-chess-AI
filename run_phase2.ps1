# Phase 2: Full 20-shard ingest + v4 distillation training + gauntlet
# Run from anywhere: .\run_phase2.ps1  (it cd's to the repo root itself).
#
# Interpreter note: bare `python` on this machine is C:\Python314 with NO
# packages. The project environment is .venv (Python 3.12, torch 2.6+cu124,
# pyarrow 24) -- this script pins it explicitly.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = '1'

$Python    = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$LogDir    = "logs"
$OutDir    = "data/distill_chunks_v4_full"
$ModelOut  = "model/grandmaster_resnet_v4_full.pt"
$IngestLog = "$LogDir/ingest_phase2.log"
$TrainLog  = "$LogDir/train_phase2.log"

if (-not (Test-Path $Python)) {
    Write-Host "FATAL: $Python not found -- expected the project venv." -ForegroundColor Red
    exit 1
}
& $Python -c "import pyarrow, torch"
if ($LASTEXITCODE -ne 0) {
    Write-Host "FATAL: $Python cannot import pyarrow/torch -- wrong environment." -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path "model" | Out-Null

function Run-Step {
    param([string]$Label, [string[]]$Cmd, [string]$LogFile)
    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    & $Cmd[0] $Cmd[1..($Cmd.Length-1)] | Tee-Object -FilePath $LogFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED (exit $LASTEXITCODE) -- check $LogFile" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "$Label done." -ForegroundColor Green
}

# --- Step 1: Ingest ---
$existingShards = @(Get-ChildItem "$OutDir/*.parquet" -ErrorAction SilentlyContinue)
if ($existingShards.Count -gt 0) {
    Write-Host "Skipping ingest -- $OutDir already has $($existingShards.Count) shard(s)." -ForegroundColor Yellow
} else {
    Run-Step "Ingest (20 shards)" @(
        $Python, "-m", "training.ingest_lichess_evals",
        "--num_shards", "20",
        "--out_dir", $OutDir
    ) $IngestLog

    Write-Host ""
    Write-Host "Ingest stats:" -ForegroundColor Cyan
    Get-Content "$OutDir/ingest_stats.json"
}

# --- Step 2: Train (2 epochs) ---
# Resume semantics: a partially trained epoch counts as COMPLETE. If the run
# died during epoch 2, resuming with --epochs 2 trains nothing (the trainer
# prints a NOTE saying so) -- rerun with --epochs 3 by hand in that case.
$resumeFlag = @()
if (Test-Path $ModelOut) {
    Write-Host "Found existing checkpoint $ModelOut -- resuming." -ForegroundColor Yellow
    $resumeFlag = @("--resume", $ModelOut)
}

Run-Step "Train v4 (2 epochs)" (@(
    $Python, "-m", "training.train_distill",
    "--data_dir",  $OutDir,
    "--output",    $ModelOut,
    "--epochs",    "2"
) + $resumeFlag) $TrainLog

# --- Step 3: Gauntlet ---
Write-Host ""
Write-Host "=== Gauntlet ===" -ForegroundColor Cyan

foreach ($elo in @(2000, 2300, 2500)) {
    Write-Host ""
    Write-Host "--- vs UCI_Elo $elo ---" -ForegroundColor Yellow
    & $Python -m evaluation.vs_stockfish `
        --model $ModelOut `
        --mode mcts --sims 200 `
        --uci_elo $elo `
        --games 24 `
        --no_blunder_guard `
        | Tee-Object -FilePath "$LogDir/gauntlet_${elo}.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Gauntlet vs $elo failed -- check $LogDir/gauntlet_${elo}.log" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Phase 2 complete ===" -ForegroundColor Green
Write-Host "Model: $ModelOut"
Write-Host "Logs:  $LogDir/"
