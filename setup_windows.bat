@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
if not exist .env copy .env.example .env >nul
echo Setup completed. Run run_app.bat to open FruitCareAI.
pause
exit /b 0
:fail
echo Setup failed. Read the error above. Install Python 3.11 64-bit if missing.
pause
exit /b 1
