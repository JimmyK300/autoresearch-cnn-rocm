param(
    [string]$Config = "configs/default.yaml",
    [int]$Runs = 1,
    [int]$PauseSeconds = 1,
    [string]$Hypothesis = ""
)

$ErrorActionPreference = "Stop"

for ($runIndex = 1; $runIndex -le $Runs; $runIndex++) {
    Write-Host ""
    Write-Host "=== Experiment $runIndex / $Runs ==="
    Write-Host "Config: $Config"

    $command = @("-m", "mnist_cnn", "train", "--config", $Config)
    if ($Hypothesis) {
        $command += @("--hypothesis", $Hypothesis)
    }

    & py @command
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }

    & py -m mnist_cnn evaluate --run best
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed with exit code $LASTEXITCODE"
    }

    if ($runIndex -lt $Runs -and $PauseSeconds -gt 0) {
        Start-Sleep -Seconds $PauseSeconds
    }
}
