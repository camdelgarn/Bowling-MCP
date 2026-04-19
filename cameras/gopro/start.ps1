# GoPro session launcher — always runs with system Python 3.12
# Usage from project root:
#   .\cameras\gopro\start.ps1 stream        # start all 3 GoPros streaming
#   .\cameras\gopro\start.ps1 capture       # open 3-window recorder
#   .\cameras\gopro\start.ps1 stop          # stop all streams
#   .\cameras\gopro\start.ps1 scan          # BLE scan only
#   .\cameras\gopro\start.ps1 nginx         # start nginx only

$PY = "C:\Users\grass\AppData\Local\Programs\Python\Python312\python.exe"
$ROOT = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

if (-not (Test-Path $PY)) {
    Write-Error "Python 3.12 not found at $PY"
    exit 1
}

Set-Location $ROOT

switch ($args[0]) {
    "nginx"   { & $PY -m cameras.gopro.gopro start-nginx }
    "stream"  { & $PY -m cameras.gopro.livestream_ble }
    "stop"    { & $PY -m cameras.gopro.livestream_ble --stop }
    "capture" { & $PY -m cameras.gopro.multi_stream_capture }
    "scan"    { & $PY -m cameras.gopro.gopro scan --timeout 15 }
    default   {
        Write-Host "Usage: .\cameras\gopro\start.ps1 <command>"
        Write-Host "Commands: nginx | stream | stop | capture | scan"
    }
}
