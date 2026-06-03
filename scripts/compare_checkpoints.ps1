<#
.SYNOPSIS
  Run the full strength-comparison matrix for the resumed supervised model.

.DESCRIPTION
  Plays paired-opening matches (via eval_arena.py) of the resumed checkpoint
  against BOTH the supervised base and the best self-play checkpoint, in BOTH
  MCTS and raw-policy modes. Raw policy matters because the deployed app uses
  the policy/value head directly; MCTS matters because that is the strongest
  setting. A model can win one mode and lose the other, so we measure both.

  Each sub-match writes its full output to a log file and the script prints a
  consolidated summary (the "Elo(A - B)" line) at the end.

.EXAMPLE
  .\compare_checkpoints.ps1 `
    -Resumed model\grandmaster_resnet_v2_resumed.pt `
    -Base    model\grandmaster_model_perspective_resnet_negatives_v2.pt `
    -SelfPlay model\selfplay_checkpoints\selfplay_iter0020.pt `
    -Sims 200 -Games 32
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

# A=Resumed in every match, so a POSITIVE Elo always means "resumed is stronger".
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
        "eval_arena.py",
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

    # Run, tee to console and log.
    python @common 2>&1 | Tee-Object -FilePath $log

    # Extract the headline Elo line for the summary table.
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
