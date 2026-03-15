@echo off
REM RTMP Server Setup Script for Windows
REM Downloads and configures nginx with RTMP module

echo RTMP Server Setup for Bowling MCP
echo ==================================

set NGINX_DIR=%~dp0..\rtmp
set NGINX_EXE=%NGINX_DIR%\nginx.exe
set DOWNLOAD_URL=https://github.com/illuspas/nginx-rtmp-win32/archive/refs/tags/v1.2.1.zip

echo Checking if nginx is already installed...

if exist "%NGINX_EXE%" (
    echo ✓ Nginx RTMP already installed at %NGINX_DIR%
    goto :check_config
)

echo Downloading nginx with RTMP module...
echo URL: %DOWNLOAD_URL%

powershell -Command "& {Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile 'nginx-rtmp.zip'}"

if not exist "nginx-rtmp.zip" (
    echo ❌ Download failed
    echo Please download manually from:
    echo %DOWNLOAD_URL%
    pause
    exit /b 1
)

echo Extracting nginx...
powershell -Command "& {Expand-Archive -Path 'nginx-rtmp.zip' -DestinationPath '.' -Force}"

if not exist "nginx-rtmp-win32-1.2.1" (
    echo ❌ Extraction failed
    pause
    exit /b 1
)

rename nginx-rtmp-win32-1.2.1 nginx-rtmp
del nginx-rtmp.zip

echo ✓ Nginx RTMP installed successfully

:check_config
echo Checking configuration...

if not exist "nginx_rtmp.conf" (
    echo ❌ nginx_rtmp.conf not found
    echo Please ensure nginx_rtmp.conf is in the same directory
    pause
    exit /b 1
)

echo ✓ Configuration file found

:test_server
echo Testing nginx configuration...

"%NGINX_EXE%" -t -c "%~dp0nginx_rtmp.conf"

if %ERRORLEVEL% EQU 0 (
    echo ✓ Configuration test passed
) else (
    echo ❌ Configuration test failed
    echo Please check nginx_rtmp.conf for errors
    pause
    exit /b 1
)

echo.
echo RTMP Server setup complete!
echo.
echo To start the server:
echo   python rtmp_server_manager.py start
echo.
echo Or run interactively:
echo   python rtmp_server_manager.py
echo.
echo Server URLs when running:
echo   RTMP: rtmp://localhost/live/stream
echo   HLS:  http://localhost:8080/hls/stream.m3u8
echo   Stats: http://localhost:8080/stat
echo.
pause