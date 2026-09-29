@echo off
title VigiLens - Live AI Vision & Geofencing System
color 0A
cd /d "%~dp0"

:: Locate Python in virtual environment
set "PY_EXE=.venv\Scripts\python.exe"
if not exist "%PY_EXE%" (
    set "PY_EXE=python"
)

cls
echo ============================================================
echo           [ V I G I L E N S   A I   E N G I N E ]
echo     Real-Time Object Detection, Tracking ^& Geofencing
echo ============================================================
echo.
echo  SELECT DETECTION MODEL (Press ENTER for Default [1]):
echo.
echo   [1] YOLO11-Extra Large (56.9M params, 54.7%% mAP - HIGHEST ACCURACY ^& DETECTS MOST OBJECTS) [RECOMMENDED]
echo   [2] YOLO11-Medium      (20.1M params, 51.5%% mAP - BALANCED HIGH-SPEED 70+ FPS)
echo   [3] YOLO11-Nano        ( 2.6M params, 39.5%% mAP - ULTRA-FAST LIGHTWEIGHT)
echo.
echo ============================================================
set "m_choice=1"
set /p m_choice="Enter choice [1-3, Default=1]: "

set "MODEL=weights\yolo11x.pt"
set "MODEL_NAME=YOLO11-Extra Large (56.9M, 54.7%% mAP)"

if "%m_choice%"=="2" (
    set "MODEL=weights\yolo11m.pt"
    set "MODEL_NAME=YOLO11-Medium (20.1M, 51.5%% mAP)"
)
if "%m_choice%"=="3" (
    set "MODEL=weights\yolo11n.pt"
    set "MODEL_NAME=YOLO11-Nano (2.6M, 39.5%% mAP)"
)

cls
echo ============================================================
echo  Starting Live Webcam with Hazard Geofencing ^& Tracking...
echo  Selected Model: %MODEL_NAME%
echo  File Path     : %MODEL%
echo.
echo  Controls: Press 'q' inside video window anytime to quit.
echo ============================================================
echo.

"%PY_EXE%" scripts\predict.py --source 0 --display --zone preset --track --model "%MODEL%"

:: Check if webcam failed (e.g. no hardware camera attached)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo [NOTICE] Webcam (index 0) was unavailable or closed.
    echo Running inference demonstration on sample traffic image...
    echo ============================================================
    echo.
    "%PY_EXE%" scripts\predict.py --source data\samples\traffic.jpg --conf 0.25 --model "%MODEL%"
)

echo.
echo ============================================================
echo  VigiLens execution finished.
echo ============================================================
pause
