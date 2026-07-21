#!/usr/bin/env bash
# Amber — one-click launcher for macOS/Linux. Double-click (macOS) or run ./launch.command
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "First run: creating virtual environment and installing dependencies..."
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt -r requirements-dashboard.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "Starting Amber dashboard... open http://localhost:8501"
python -m streamlit run amber/dashboard/app.py
