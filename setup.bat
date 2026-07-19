@echo off
chcp 65001 >nul
setlocal EnableExtensions
pushd "%~dp0"

echo ======================================================================
echo Lucida Background Remover - Setup
echo ======================================================================
echo.

set "PYTHON_CMD="
py -3.12 -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py -3.12"
if not defined PYTHON_CMD py -3.11 -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD python -c "import sys; raise SystemExit(sys.version_info ^< (3, 11))" >nul 2>&1 && set "PYTHON_CMD=python"

if not defined PYTHON_CMD (
    echo ERROR: Python 3.11 or newer was not found.
    echo Install Python and make either "py" or "python" available, then retry.
    popd
    pause
    exit /b 1
)

if exist ".venv" (
    echo Removing the existing virtual environment...
    rmdir /s /q ".venv"
    if exist ".venv" (
        echo ERROR: Could not remove .venv. Close programs using it and retry.
        popd
        pause
        exit /b 1
    )
)

echo Creating .venv with %PYTHON_CMD%...
%PYTHON_CMD% -m venv ".venv"
if errorlevel 1 goto :error

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo.
echo ----------------------------------------------------------------------
echo Installing dependencies...
echo ----------------------------------------------------------------------
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo ======================================================================
echo Setup completed. Run run.bat to start Lucida.
echo ======================================================================
popd
pause
exit /b 0

:error
echo.
echo ======================================================================
echo ERROR: Setup failed. Review the messages above.
echo ======================================================================
popd
pause
exit /b 1
