#!/usr/bin/env bash
# ============================================================
#  People Counter — macOS / Linux Setup Script
#  Run ONCE.  After setup use:  ./run.sh
# ============================================================

set -e   # exit on any error

PYTHON=""

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   People Counter  —  macOS/Linux Setup       ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Find Python 3.10 or 3.11 ─────────────────────────────────────────────
for cmd in python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(sys.version_info[:2])")
        if [[ "$VER" == "(3, 10)" || "$VER" == "(3, 11)" || "$VER" == "(3, 12)" ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "  [ERROR] Python 3.10/3.11/3.12 not found."
    echo ""
    echo "  macOS  : brew install python@3.11"
    echo "  Ubuntu : sudo apt install python3.11 python3.11-venv"
    echo ""
    exit 1
fi

echo "  [OK] Using $PYTHON ($($PYTHON --version))"

# ── 2. Check tkinter (Linux often needs it separately) ───────────────────────
if ! $PYTHON -c "import tkinter" &>/dev/null; then
    echo ""
    echo "  [ERROR] tkinter not found."
    echo "  Ubuntu/Debian : sudo apt install python3-tk"
    echo "  Fedora        : sudo dnf install python3-tkinter"
    echo "  macOS         : tkinter ships with the python.org installer"
    echo ""
    exit 1
fi
echo "  [OK] tkinter available"

# ── 3. Create venv ───────────────────────────────────────────────────────────
if [ -d "venv" ]; then
    echo "  [INFO] venv already exists — skipping creation"
else
    echo "  [INFO] Creating virtual environment ..."
    $PYTHON -m venv venv
    echo "  [OK] venv created"
fi

source venv/bin/activate
echo "  [OK] venv activated"

# ── 4. Upgrade pip ───────────────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── 5. Detect GPU and install correct PyTorch ────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo "  [OK] NVIDIA GPU found — installing CUDA PyTorch (CUDA 12.1)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "  [INFO] macOS — installing PyTorch with MPS (Apple Silicon) support"
    pip install torch torchvision --quiet
else
    echo "  [INFO] No NVIDIA GPU — installing CPU-only PyTorch"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# ── 6. Install remaining deps ────────────────────────────────────────────────
echo "  [INFO] Installing remaining dependencies ..."
pip install -r requirements.txt --quiet

# ── 7. Pre-download YOLO model ───────────────────────────────────────────────
echo "  [INFO] Pre-downloading YOLOv8n model (6 MB) ..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || true
echo "  [OK] Model ready"

echo ""
echo "  ══════════════════════════════════════════════"
echo "   Setup complete!  Run:  ./run.sh"
echo "  ══════════════════════════════════════════════"
echo ""
