param(
  [string]$Adb = "adb",
  [int]$Port = 8768
)

$ErrorActionPreference = "Stop"

if ($Adb -in @("adb", "adb.exe") -and -not (Get-Command $Adb -ErrorAction SilentlyContinue)) {
  $candidates = @(
    (Join-Path $env:APPDATA "PICO_Developer_Center_Downloads\cn\adb\platform-tools\adb.exe"),
    (Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe")
  )
  $tuanjie = "C:\Program Files\Tuanjie\Hub\Editor"
  if (Test-Path -LiteralPath $tuanjie) {
    $candidates += Get-ChildItem -LiteralPath $tuanjie -Filter adb.exe -Recurse -ErrorAction SilentlyContinue | ForEach-Object FullName
  }
  $found = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
  if ($found) { $Adb = $found }
}

try {
  & $Adb version *> $null
} catch {
  throw "adb was not found. Install Android Platform Tools and add adb to PATH."
}
if ($LASTEXITCODE -ne 0) {
  throw "Could not execute adb. Check the Android Platform Tools installation."
}

$devices = @(& $Adb devices | Select-String "`tdevice$")
if ($devices.Count -eq 0) {
  throw "No authorized PICO USB device found. Connect the headset and allow USB debugging."
}
if ($devices.Count -ne 1) {
  throw "Detected $($devices.Count) ADB devices. Keep exactly one PICO device connected."
}

& $Adb reverse "tcp:$Port" "tcp:$Port"
if ($LASTEXITCODE -ne 0) {
  throw "ADB USB port forwarding failed: tcp:$Port -> tcp:$Port"
}

$forward = @(& $Adb reverse --list | Select-String "tcp:$Port")
if ($forward.Count -eq 0) {
  throw "ADB did not report tcp:$Port forwarding; the USB channel could not be verified."
}

Write-Output "PICO USB/ADB connection established."
Write-Output "PICO WebSocket address: ws://127.0.0.1:$Port"
Write-Output "The APK must send type=input_frame as its first message; no pairing code or QR code is required."
