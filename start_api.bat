@echo off
title VigiLens - REST API & Swagger UI
color 0E
cd /d "%~dp0"

echo ============================================================
echo           [ V I G I L E N S   A I   E N G I N E ]
echo               RESTful API & Swagger Documentation
echo ============================================================
echo.

set "PY_EXE=.venv\Scripts\python.exe"
if not exist "%PY_EXE%" (
    set "PY_EXE=python"
)

echo [INFO] Launching FastAPI engine on http://localhost:8000 ...
echo [INFO] Opening Swagger Documentation UI in default browser...
start http://localhost:8000/docs
echo.
"%PY_EXE%" -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

pause
