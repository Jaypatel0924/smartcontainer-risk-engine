@echo off
REM SmartContainer Risk Engine - Quick Start Script (Windows)

setlocal enabledelayedexpansion

echo ======================================
echo SmartContainer Risk Engine
echo HackaMINEd-2026 Hackathon
echo ======================================

REM Check Python
echo.
echo [CHECK] Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found
    exit /b 1
)

for /f "delims= " %%A in ('python --version') do (
    echo [OK] %%A
)

REM Create venv
echo.
echo [SETUP] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate venv
echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo [SETUP] Installing dependencies...
pip install -q -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Installation failed
    exit /b 1
)
echo [OK] Dependencies installed

REM Train model
echo.
echo [STEP 1] Training ML models...
python model_training.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Model training failed
    exit /b 1
)

REM Generate predictions
echo.
echo [STEP 2] Generating predictions...
python predict.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Prediction generation failed
    exit /b 1
)

echo.
echo [SUCCESS] Pipeline completed successfully!
echo.
echo [INFO] Generated files:
echo   - models/random_forest_model.pkl
echo   - models/isolation_forest_model.pkl
echo   - output/risk_predictions.csv
echo.
echo [NEXT] Launch dashboard:
echo   streamlit run dashboard.py
echo.

REM Keep window open
pause
