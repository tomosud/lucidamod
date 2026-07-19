@echo off
chcp 65001 >nul
setlocal EnableExtensions
pushd "%~dp0"

echo ======================================================================
echo Lucida Background Remover
echo ======================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The virtual environment was not found.
    echo Run setup.bat first.
    echo.
    popd
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "LUCIDA_HOST=127.0.0.1"
set "LUCIDA_PORT=8756"

echo Opening http://%LUCIDA_HOST%:%LUCIDA_PORT%/
echo Keep this window open while using Lucida. Press Ctrl+C to stop.
echo.

start "" "http://%LUCIDA_HOST%:%LUCIDA_PORT%/"
".venv\Scripts\python.exe" -m uvicorn serving.app:app --host %LUCIDA_HOST% --port %LUCIDA_PORT%
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if %EXIT_CODE%==0 (
    echo Lucida stopped.
) else (
    echo ERROR: Lucida exited with code %EXIT_CODE%.
)

popd
pause
exit /b %EXIT_CODE%
