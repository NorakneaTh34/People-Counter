@echo off
:: ============================================================
::  People Counter — Launch (Windows)
::  Activates the venv and starts the app.
:: ============================================================

if not exist venv (
    echo  [ERROR] venv not found. Run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo  [INFO] Starting People Counter ...
python people_counter.py
