#!/usr/bin/env bash
# ============================================================
#  People Counter — Launch (macOS / Linux)
# ============================================================

if [ ! -d "venv" ]; then
    echo "  [ERROR] venv not found. Run:  bash setup.sh"
    exit 1
fi

source venv/bin/activate
echo "  [INFO] Starting People Counter ..."
python people_counter.py
