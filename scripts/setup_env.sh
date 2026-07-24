#!/usr/bin/env bash
set -euo pipefail

# Ubuntu/Debian ship python3, not python — use python3 to create the venv;
# inside the venv `python` exists.
PY="$(command -v python3 || command -v python)"
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready. Activate with: source .venv/bin/activate"
