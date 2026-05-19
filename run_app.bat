@echo off
title AuraPalm AI Platform - Startup Script
echo ==================================================
echo         AuraPalm AI Platform Local Startup
echo ==================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

:: Setup virtual environment if missing
if not exist ".venv" (
    echo [+] Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate and install dependencies
echo [+] Updating dependencies...
call .venv\Scripts\python -m pip install --upgrade pip
call .venv\Scripts\python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: Launch browser after a short delay
echo [+] Launching UI browser...
start http://127.0.0.1:8000

:: Start server
echo [+] Starting AuraPalm AI FastAPI Server...
call .venv\Scripts\uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

pause
