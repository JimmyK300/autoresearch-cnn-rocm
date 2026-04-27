param(
    [string]$Config = "configs/default.yaml",
    [string]$SearchPlan = "configs/search_plan.json",
    [int]$PauseSeconds = 1
)

$ErrorActionPreference = "Stop"

& py -m mnist_cnn sweep --config $Config --search-plan $SearchPlan --pause-seconds $PauseSeconds
if ($LASTEXITCODE -ne 0) {
    throw "Sweep failed with exit code $LASTEXITCODE"
}
