@echo off
title VigiLens AI Engine - Autonomous Vision Platform
color 0B
cd /d "%~dp0"

:: Check virtual environment
set "PY_EXE=.venv\Scripts\python.exe"
set "UVI_EXE=.venv\Scripts\uvicorn.exe"

if not exist "%PY_EXE%" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Attempting to fall back to system python...
    set "PY_EXE=python"
    set "UVI_EXE=uvicorn"
)

:: Default model set to highest-accuracy flagship
set "ACTIVE_MODEL=weights\yolo11x.pt"
set "ACTIVE_TIER=Extra Large (56.9M, Flagship Max ~54.7%% mAP - Highest Accuracy)"

:MENU
cls
echo ============================================================
echo           [ V I G I L E N S   A I   E N G I N E ]
echo     Real-Time Object Detection, Tracking ^& Geofencing
echo ============================================================
echo  Active Model: %ACTIVE_MODEL%
echo  Tier Info   : %ACTIVE_TIER%
echo ============================================================
echo.
echo  Select a demonstration mode to run (Press Enter for default [1]):
echo.
echo   [1] Live Webcam + Restricted Hazard Zone Geofencing (DEFAULT)
echo   [2] Live Webcam + Virtual Tripwire Directional Counter
echo   [3] Live Webcam + Object Tracking ^& Accuracy Percentages
echo   [4] Test Inference on Sample Images (Traffic / Pedestrians)
echo   [5] Start FastAPI REST Microservice (Opens Swagger UI)
echo   [6] Run Complete Verification Test Suite (36 Tests)
echo   [7] Export Model to ONNX Runtime Engine
echo   [8] Switch Model Tier (Nano vs Medium vs Extra-Large)
echo   [0] Exit
echo.
echo ============================================================
set "choice=1"
set /p choice="Enter your choice [1-8, 0, Default=1]: "

if "%choice%"=="1" goto ZONE
if "%choice%"=="2" goto TRIPWIRE
if "%choice%"=="3" goto TRACK
if "%choice%"=="4" goto SAMPLES
if "%choice%"=="5" goto API
if "%choice%"=="6" goto TEST
if "%choice%"=="7" goto EXPORT
if "%choice%"=="8" goto SWITCH_MODEL
if "%choice%"=="0" goto EXIT

echo Invalid option. Please select 1-8 or 0.
timeout /t 2 >nul
goto MENU

:ZONE
cls
echo ============================================================
echo  Starting Live Webcam with Restricted Hazard Zone...
echo  Model: %ACTIVE_MODEL%
echo  (Press 'q' inside the video window anytime to exit)
echo ============================================================
echo.
"%PY_EXE%" scripts\predict.py --source 0 --display --zone preset --track --model "%ACTIVE_MODEL%"
echo.
pause
goto MENU

:TRIPWIRE
cls
echo ============================================================
echo  Starting Live Webcam with Virtual Tripwire Counter...
echo  Model: %ACTIVE_MODEL%
echo  (Press 'q' inside the video window anytime to exit)
echo ============================================================
echo.
"%PY_EXE%" scripts\predict.py --source 0 --display --tripwire preset --model "%ACTIVE_MODEL%"
echo.
pause
goto MENU

:TRACK
cls
echo ============================================================
echo  Starting Live Webcam with Object Tracking ^& Accuracy HUD...
echo  Model: %ACTIVE_MODEL%
echo  (Press 'q' inside the video window anytime to exit)
echo ============================================================
echo.
"%PY_EXE%" scripts\predict.py --source 0 --display --track --model "%ACTIVE_MODEL%"
echo.
pause
goto MENU

:SAMPLES
cls
echo ============================================================
echo  Running Detection on Sample Urban Traffic Scene...
echo  Model: %ACTIVE_MODEL%
echo ============================================================
echo.
"%PY_EXE%" scripts\predict.py --source data\samples\traffic.jpg --conf 0.25 --model "%ACTIVE_MODEL%"
echo.
echo [INFO] Annotated result saved to outputs\predictions\pred_traffic.jpg
pause
goto MENU

:API
cls
echo ============================================================
echo  Launching FastAPI REST Microservice on http://localhost:8000
echo ============================================================
echo.
echo Opening interactive Swagger documentation in your browser...
start http://localhost:8000/docs
"%PY_EXE%" -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
pause
goto MENU

:TEST
cls
echo ============================================================
echo  Running All 36 Automated Unit ^& Integration Tests...
echo ============================================================
echo.
"%PY_EXE%" -m pytest -v
echo.
pause
goto MENU

:EXPORT
cls
echo ============================================================
echo  Exporting Best Weights to ONNX Runtime...
echo ============================================================
echo.
"%PY_EXE%" scripts\export_model.py --model runs\checkpoints\best.pt --format onnx --output-dir outputs\exports
echo.
pause
goto MENU

:SWITCH_MODEL
cls
echo ============================================================
echo           [ S E L E C T   M O D E L   T I E R ]
echo ============================================================
echo.
echo   [1] YOLO11-Nano (2.6M params, ~39.5%% mAP - Fastest / Low CPU usage)
echo   [2] YOLO11-Medium (20.1M params, ~51.5%% mAP - Recommended High Accuracy)
echo   [3] YOLO11-Extra Large (56.9M params, ~54.7%% mAP - Maximum Flagship Accuracy)
echo.
echo ============================================================
set /p mchoice="Select tier [1-3]: "
if "%mchoice%"=="1" (
    set "ACTIVE_MODEL=weights\yolo11n.pt"
    set "ACTIVE_TIER=Nano (2.6M, Ultra-Fast ~39.5%% mAP)"
)
if "%mchoice%"=="2" (
    set "ACTIVE_MODEL=weights\yolo11m.pt"
    set "ACTIVE_TIER=Medium (20.1M, High-Accuracy ~51.5%% mAP)"
)
if "%mchoice%"=="3" (
    set "ACTIVE_MODEL=weights\yolo11x.pt"
    set "ACTIVE_TIER=Extra Large (56.9M, Flagship Max ~54.7%% mAP)"
)
goto MENU

:EXIT
echo Exiting VigiLens. Goodbye!
exit /b 0
