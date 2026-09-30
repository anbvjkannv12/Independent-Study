# Scheduler entry point: powershell -File scripts\live_run.ps1 update|retrain
param([Parameter(Mandatory = $true)][ValidateSet("update", "retrain")][string]$Command)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& "$root\.venv-transformer\Scripts\python.exe" "$root\scripts\live_pipeline.py" $Command
exit $LASTEXITCODE
