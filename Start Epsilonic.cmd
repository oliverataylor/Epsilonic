@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto ready
where py >nul 2>nul
if errorlevel 1 goto python
py -3 -m venv .venv
if errorlevel 1 goto failed
goto install
:python
python -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
goto launch
:ready
".venv\Scripts\python.exe" -c "import numpy, scipy, sounddevice, PIL"
if errorlevel 1 goto install
:launch
".venv\Scripts\python.exe" epsilonic.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Epsilonic could not start. Check the error above and ensure Python 3.11 or newer is installed.
pause
exit /b 1
