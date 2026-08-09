$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$prefix = Join-Path $root '.conda\nero-kinematics'
conda env create --prefix $prefix --file (Join-Path $root 'environment-kinematics.yml')
Write-Host "Kinematics environment created at $prefix"
