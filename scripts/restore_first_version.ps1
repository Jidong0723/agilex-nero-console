param(
    [string]$ArchivePath = "",
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$ExpectedSha256 = "641F72BFCE1E523771E68C13A3E786CE5A52424125DE755C15E11E9E4A1CECB0"
$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
if ([string]::IsNullOrWhiteSpace($ArchivePath)) {
    $ArchivePath = Join-Path $workspaceRoot "agilex-nero-console.zip"
}
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $workspaceRoot "restored\agilex-nero-console-first-version"
}
$archive = [System.IO.Path]::GetFullPath($ArchivePath)
$destinationPath = [System.IO.Path]::GetFullPath($Destination)

if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    throw "First-version archive not found: $archive"
}
if (Test-Path -LiteralPath $destinationPath) {
    throw "Restore destination already exists; refusing to overwrite it: $destinationPath"
}
$actualSha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToUpperInvariant()
if ($actualSha256 -ne $ExpectedSha256) {
    throw "Archive checksum mismatch. Expected $ExpectedSha256, got $actualSha256"
}

$destinationParent = Split-Path -Parent $destinationPath
New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
$temporary = Join-Path $destinationParent ("first-version-restore-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    Expand-Archive -LiteralPath $archive -DestinationPath $temporary
    $source = Join-Path $temporary "agilex-nero-console-master"
    if (-not (Test-Path -LiteralPath (Join-Path $source "run_console.cmd") -PathType Leaf)) {
        throw "Archive structure verification failed: run_console.cmd is missing"
    }
    Move-Item -LiteralPath $source -Destination $destinationPath
}
finally {
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Recurse -Force
    }
}

Write-Output "First version restored to: $destinationPath"
Write-Output "The active console directory was not modified."
