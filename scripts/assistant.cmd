@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

where py >nul 2>nul
if not errorlevel 1 (
    py -m src.assistant.cli %*
    exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if not errorlevel 1 (
    python -m src.assistant.cli %*
    exit /b %ERRORLEVEL%
)

echo Python is not installed or not on PATH. Install Python 3.11+ and retry.
exit /b 1
