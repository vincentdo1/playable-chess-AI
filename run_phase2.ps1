# Phase 2: Full 20-shard ingest + v4 distillation training + gauntlet
# Run from anywhere: .\run_phase2.ps1  (it cd's to the repo root itself).
#
# Interpreter note: bare `python` on this machine is C:\Python314 with NO
# packages. The project environment is .venv (Python 3.12, torch 2.6+cu124,
# pyarrow 24) -- this script pins it explicitly.

param(
    [switch]$Execute,
    [string]$SourceRevision = $env:LICHESS_EVAL_REVISION,
    [string]$ResumeCheckpoint = ""
)

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
$Manifest  = "$OutDir/ingest_manifest.json"

if (-not $Execute) {
    Write-Host "FATAL: Phase 2 downloads ~41 GB and starts a multi-day GPU run." -ForegroundColor Red
    Write-Host "Re-run with -Execute and an immutable -SourceRevision <HF commit SHA>." -ForegroundColor Yellow
    exit 2
}
if ([string]::IsNullOrWhiteSpace($SourceRevision) -or
    $SourceRevision -notmatch '^[0-9a-fA-F]{40,64}$') {
    Write-Host "FATAL: -SourceRevision (or LICHESS_EVAL_REVISION) must be a 40-64 character hexadecimal Hugging Face commit SHA; 'main' is not allowed." -ForegroundColor Red
    exit 2
}
$SourceRevision = $SourceRevision.ToLowerInvariant()

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
    if (-not (Test-Path $Manifest)) {
        Write-Host "FATAL: $OutDir has shards but no completed ingest manifest. Refusing to train on a partial or unproven corpus." -ForegroundColor Red
        exit 1
    }
    $manifestData = Get-Content $Manifest -Raw | ConvertFrom-Json
    if ($manifestData.status -ne 'complete') {
        Write-Host "FATAL: $Manifest does not report status=complete." -ForegroundColor Red
        exit 1
    }
    if ($manifestData.source.revision -ne $SourceRevision) {
        Write-Host "FATAL: existing corpus revision $($manifestData.source.revision) does not match requested $SourceRevision." -ForegroundColor Red
        exit 1
    }
    $trainShards = @(Get-ChildItem "$OutDir/train_*.parquet" -ErrorAction SilentlyContinue)
    $valShards = @(Get-ChildItem "$OutDir/val_*.parquet" -ErrorAction SilentlyContinue)
    if ($trainShards.Count -eq 0 -or $valShards.Count -eq 0) {
        Write-Host "FATAL: completed manifest exists but train/validation shards are missing." -ForegroundColor Red
        exit 1
    }
    Write-Host "Skipping ingest -- verified completed corpus at revision $SourceRevision ($($existingShards.Count) shards)." -ForegroundColor Yellow
} else {
    if (Test-Path $Manifest) {
        Write-Host "FATAL: $Manifest exists but no Parquet shards were found. Use a fresh output directory." -ForegroundColor Red
        exit 1
    }
    Run-Step "Ingest (20 shards)" @(
        $Python, "-m", "training.ingest_lichess_evals",
        "--num_shards", "20",
        "--out_dir", $OutDir,
        "--source_revision", $SourceRevision
    ) $IngestLog

    Write-Host ""
    Write-Host "Ingest stats:" -ForegroundColor Cyan
    Get-Content "$OutDir/ingest_stats.json"
}

# --- Step 2: Train (2 epochs) ---
$resumeFlag = @()
if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    if (-not (Test-Path $ResumeCheckpoint)) {
        Write-Host "FATAL: explicit -ResumeCheckpoint not found: $ResumeCheckpoint" -ForegroundColor Red
        exit 1
    }
    Write-Host "Explicitly resuming from $ResumeCheckpoint." -ForegroundColor Yellow
    $resumeFlag = @("--resume", $ResumeCheckpoint)
} elseif (Test-Path $ModelOut) {
    Write-Host "FATAL: $ModelOut already exists. Refusing to overwrite or implicitly resume it; pass -ResumeCheckpoint with the exact best/latest checkpoint you intend." -ForegroundColor Red
    exit 1
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
    $evalArgs = @(
        "-m", "evaluation.vs_stockfish",
        "--model", $ModelOut,
        "--mode", "mcts", "--sims", "200",
        "--uci_elo", "$elo",
        "--games", "24",
        "--no_blunder_guard",
        "--output_dir", "evaluation/results/phase2_${elo}",
        "--require_p95_seconds", "2.0"
    )
    if ($elo -eq 2500) {
        # Predeclared definition of done: the paired-game 95% lower score bound
        # must clear 50%, not merely the noisy point estimate.
        $evalArgs += @("--require_score_lower_bound", "0.50")
    }
    & $Python @evalArgs | Tee-Object -FilePath "$LogDir/gauntlet_${elo}.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Gauntlet vs $elo failed -- check $LogDir/gauntlet_${elo}.log" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "=== Phase 2 complete ===" -ForegroundColor Green
Write-Host "Model: $ModelOut"
Write-Host "Logs:  $LogDir/"
