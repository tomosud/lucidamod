@echo off
chcp 65001 >nul
setlocal EnableExtensions
pushd "%~dp0"

echo ======================================================================
echo Lucida ONNX Browser Test
echo ======================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv was not found. Run setup.bat first.
    popd
    pause
    exit /b 1
)
if not exist "models\lucida-web-512-fp32.onnx" (
    echo ERROR: models\lucida-web-512-fp32.onnx was not found.
    echo Run the ONNX export first. See docs\onnx.md.
    popd
    pause
    exit /b 1
)

set "WEB_URL=http://127.0.0.1:8760/web_onnx/"
echo Opening %WEB_URL%
echo Keep this window open. Press Ctrl+C to stop.
echo.
start "" "%WEB_URL%"
".venv\Scripts\python.exe" scripts\serve_onnx_web.py
set "EXIT_CODE=%ERRORLEVEL%"
popd
pause
exit /b %EXIT_CODE%
