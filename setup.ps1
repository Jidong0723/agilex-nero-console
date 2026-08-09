param(
  [string]$PythonExe = $env:NERO_PYTHON_EXE
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
  if (-not $PythonExe) {
    $candidate = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $candidate) { $PythonExe = $candidate.Trim() }
  }
  if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    throw 'Provide a Python 3.12 executable with -PythonExe, or install Python 3.12 for the py launcher.'
  }
  & $PythonExe -m venv (Join-Path $root '.venv')
}

& $venvPython -m pip install --upgrade pip
Push-Location $root
try { & $venvPython -m pip install -r 'requirements.txt' }
finally { Pop-Location }
Write-Host 'Control Runtime Python environment is ready.'
