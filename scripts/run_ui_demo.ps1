[CmdletBinding()]
param(
    [int]$Port = 8100,
    [string]$StateDir = "",
    [switch]$RebuildFrontend
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VirtualEnvironment = Join-Path $ProjectRoot ".venv-ui312"
$Python = Join-Path $VirtualEnvironment "Scripts\python.exe"
$Frontend = Join-Path $ProjectRoot "ui\frontend"

Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $Python)) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -eq $PyLauncher) {
        throw "Python 3.12 is required. Install it, then run this script again."
    }
    Write-Host "[AlphaBrain] Creating isolated Python 3.12 environment..."
    & py -3.12 -m venv $VirtualEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv-ui312 with Python 3.12."
    }
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements-ui.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "UI dependency installation failed."
    }
}

& $Python -c "import fastapi, sqlalchemy, uvicorn, websockets, msgpack"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[AlphaBrain] Installing missing UI dependencies..."
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements-ui.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "UI dependency installation failed."
    }
}

$IndexHtml = Join-Path $Frontend "dist\index.html"
if ($RebuildFrontend -or -not (Test-Path -LiteralPath $IndexHtml)) {
    Push-Location $Frontend
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $Frontend "node_modules"))) {
            npm.cmd install
            if ($LASTEXITCODE -ne 0) {
                throw "Frontend dependency installation failed."
            }
        }
        npm.cmd run build
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build failed."
        }
    }
    finally {
        Pop-Location
    }
}

$LaunchArguments = @("-m", "alphabrain_ui", "--demo", "--host", "127.0.0.1", "--port", "$Port")
if ($StateDir) {
    $LaunchArguments += @("--state-dir", $StateDir)
}

Write-Host ""
Write-Host "[AlphaBrain] Demo is local-only and uses simulated GPUs/model compute."
Write-Host "[AlphaBrain] Open http://127.0.0.1:$Port"
Write-Host "[AlphaBrain] Press Ctrl+C to stop the UI."
Write-Host ""

& $Python @LaunchArguments
exit $LASTEXITCODE
