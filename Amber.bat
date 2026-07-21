@echo off
REM Amber — one-click launcher for Windows. Double-click this file.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating virtual environment and installing dependencies...
  python -m venv .venv
  call ".venv\Scripts\activate.bat"
  python -m pip install --upgrade pip
  pip install -r requirements.txt -r requirements-dashboard.txt
) else (
  call ".venv\Scripts\activate.bat"
)

echo Starting Amber dashboard... a browser tab will open at http://localhost:8501
python -m streamlit run amber\dashboard\app.py
pause
