<#
.SYNOPSIS
  Compare a resumed model with base and self-play checkpoints.

.DESCRIPTION
  Runs paired-opening matches in MCTS and raw-policy modes, writes each match
  log, and prints a consolidated Elo summary.
#>

param(
    [Parameter(Mandatory = $true)] [string]$Resumed,
    [Parameter(Mandatory = $true)] [string]$Base,
    [string]$SelfPlay = "model\selfplay_checkpoints\selfplay_iter0020.pt",
    [int]$Sims = 200,
    [int]$Games = 32,            # 0 = full paired suite (32 games)
    [int]$MctsBatchSize = 16,
    [double]$PolicyTemperature = 0.3,
    [string]$LogDir = "eval_logs"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# A is the resumed model in every match.
$matrix = @(
    @{ Name = "resumed_vs_base_mcts";       Opp = $Base;     Method = "mcts"   },
    @{ Name = "resumed_vs_base_policy";     Opp = $Base;     Method = "policy" },
    @{ Name = "resumed_vs_selfplay_mcts";   Opp = $SelfPlay; Method = "mcts"   },
    @{ Name = "resumed_vs_selfplay_policy"; Opp = $SelfPlay; Method = "policy" }
)

$summary = @()

foreach ($m in $matrix) {
    $log = Join-Path $LogDir "$($m.Name).log"
    Write-Host "`n=== Running $($m.Name) ===" -ForegroundColor Cyan

    $common = @(
        "-m", "evaluation.eval_arena",
        "--model_a", $Resumed,
        "--model_b", $m.Opp,
        "--method_a", $m.Method,
        "--method_b", $m.Method,
        "--paired",
        "--games", $Games,
        "--sims", $Sims,
        "--mcts_batch_size", $MctsBatchSize,
        "--policy_temperature", "1.5",
        "--temperature", $PolicyTemperature
    )

    python @common 2>&1 | Tee-Object -FilePath $log

    $eloLine = Select-String -Path $log -Pattern "Elo\(A - B\)" | Select-Object -Last 1
    $verdict = Select-String -Path $log -Pattern "=> " | Select-Object -Last 1
    $summary += [pscustomobject]@{
        Match   = $m.Name
        Elo     = if ($eloLine) { $eloLine.Line.Trim() } else { "n/a" }
        Verdict = if ($verdict) { $verdict.Line.Trim() } else { "n/a" }
    }
}

Write-Host "`n`n==================== SUMMARY ====================" -ForegroundColor Green
Write-Host "A = resumed model in every row; positive Elo = resumed is stronger.`n"
$summary | Format-Table -AutoSize -Wrap
Write-Host "Full logs in: $LogDir" -ForegroundColor Green
