@echo off
REM Quick RTMP Stream Test
echo RTMP Stream Test
echo ================
echo.
echo This will check if your RTMP server is receiving a stream.
echo.
echo Make sure your GoPro/camera is streaming to:
echo   rtmp://localhost/live/stream
echo.
echo Then run this test...
echo.
pause

echo Checking RTMP server status...
python rtmp_server_manager.py status
echo.

echo Checking for active streams...
powershell -Command "& { try { $response = Invoke-WebRequest -Uri 'http://localhost:8080/stat' -UseBasicParsing; $xml = [xml]$response.Content; $live = $xml.rtmp.server.application | Where-Object { $_.name -eq 'live' }; $clients = $live.live.nclients; Write-Host \"Live stream clients: $clients\" -ForegroundColor $(if ($clients -gt 0) { 'Green' } else { 'Red' }) } catch { Write-Host 'Cannot check stats - server may not be running' -ForegroundColor Red } }"
echo.

echo Testing stream connection...
python test_rtmp_connection.py quick
echo.

echo Done! Check the results above.
echo If no clients are connected, make sure your device is streaming.
pause