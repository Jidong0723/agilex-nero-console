$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$prefix = Join-Path $root '.conda\nero-kinematics'
if (Test-Path $prefix) {
  conda env update --prefix $prefix --file (Join-Path $root 'environment-kinematics.yml') --prune
} else {
  conda env create --prefix $prefix --file (Join-Path $root 'environment-kinematics.yml')
}
Write-Host "Kinematics environment created at $prefix"
