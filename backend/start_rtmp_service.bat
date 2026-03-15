@echo off
REM RTMP Server Service Launcher
REM This starts nginx RTMP server as a Windows service

echo Starting GoPro RTMP Server Service...
echo.

REM Check if nginx config exists
if not exist "nginx_rtmp.conf" (
    echo Creating nginx RTMP configuration...
    call python setup_rtmp_server.py
    echo.
)

REM Start nginx RTMP server
echo Starting RTMP server on port 1935...
nginx -c nginx_rtmp.conf

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ RTMP Server started successfully!
    echo   RTMP URL: rtmp://localhost/live/stream
    echo   Web Interface: http://localhost:8080/stat
    echo.
    echo Press Ctrl+C to stop the server
    echo.
    pause
) else (
    echo.
    echo ❌ Failed to start RTMP server
    echo Make sure nginx is installed and in PATH
    echo.
    pause
)