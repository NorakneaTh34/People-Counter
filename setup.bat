@echo off
:: ============================================================
::  People Counter — Windows Setup Script
::  Run this ONCE to create the venv and install everything.
::  After setup, use run.bat to launch the app.
:: ============================================================

setlocal enabledelayedexpansion

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║   People Counter  —  Windows Setup           ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ── 1. Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo  Download Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo  Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER% found

:: ── 2. Create virtual environment ────────────────────────────────────────────
if exist venv (
    echo  [INFO] venv already exists — skipping creation
) else (
    echo  [INFO] Creating virtual environment ...
    python -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create venv
        pause
        exit /b 1
    )
    echo  [OK] venv created
)

:: ── 3. Activate venv ─────────────────────────────────────────────────────────
call venv\Scripts\activate.bat
echo  [OK] venv activated

:: ── 4. Upgrade pip ───────────────────────────────────────────────────────────
echo  [INFO] Upgrading pip ...
python -m pip install --upgrade pip --quiet

:: ── 5. Detect GPU ────────────────────────────────────────────────────────────
echo  [INFO] Checking for NVIDIA GPU ...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo  [INFO] No NVIDIA GPU detected — installing CPU-only PyTorch
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
) else (
    echo  [OK] NVIDIA GPU found — installing CUDA PyTorch (CUDA 12.1)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
)

:: ── 6. Install remaining deps ────────────────────────────────────────────────
echo  [INFO] Installing remaining dependencies ...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] pip install failed — check your internet connection
    pause
    exit /b 1
)

:: ── 7. Pre-download YOLO model ───────────────────────────────────────────────
echo  [INFO] Pre-downloading YOLOv8n model (6 MB) ...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>nul
echo  [OK] Model ready

echo.
echo  ══════════════════════════════════════════════
echo   Setup complete!  Run:  run.bat
echo  ══════════════════════════════════════════════
echo.
pause
