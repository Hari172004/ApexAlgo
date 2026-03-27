@echo off
title Agni-V Trading Bot
cd /d "%~dp0"

echo [1/2] Activating Virtual Environment...
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] .venv not found. Running with system python...
)

echo [2/2] Starting Agni-V Bot...
python run_bot.py

echo.
echo Bot session ended.
pause
