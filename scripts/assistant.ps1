param(
    [string]$Query = "",
    [switch]$Interactive,
    [int]$MaxToolCalls = 5
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonCommand = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCommand = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCommand = "py"
}

if (-not $pythonCommand) {
    throw "Python is not installed or not on PATH. Install Python 3.11+ and retry."
}

if ($Interactive) {
    & $pythonCommand -m src.assistant.cli --interactive --max-tool-calls $MaxToolCalls
    exit $LASTEXITCODE
}

if (-not $Query) {
    throw "Provide -Query 'your question' or use -Interactive."
}

& $pythonCommand -m src.assistant.cli --query $Query --max-tool-calls $MaxToolCalls
exit $LASTEXITCODE
