@echo off
rem Still Life AI launcher.
rem First start: creates .venv with Python 3.10-3.12 and installs requirements.txt.
rem Later starts: opens the app right away (reinstalls only if requirements.txt changed).
setlocal EnableExtensions
cd /d "%~dp0"
title Still Life AI

set "VENV_PY=.venv\Scripts\python.exe"
set "STAMP=.venv\requirements.installed"

if not exist "%VENV_PY%" (
    call :create_venv || goto :failed
)
fc /b requirements.txt "%STAMP%" >nul 2>&1
if errorlevel 1 (
    call :install || goto :failed
)

start "" ".venv\Scripts\pythonw.exe" main.py %*
exit /b 0


:create_venv
echo First start: setting up Still Life AI in "%CD%\.venv"
call :find_python || exit /b 1
echo Using: %PY_CMD%
%PY_CMD% -m venv .venv || exit /b 1
exit /b 0


:find_python
set "PY_CMD="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY_CMD (
        py -%%V -c "import sys" >nul 2>&1 && set "PY_CMD=py -%%V"
    )
)
if not defined PY_CMD (
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1 && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo.
    echo Python 3.10, 3.11 or 3.12 was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo ^(tick "Add python.exe to PATH" in the installer^) and run run.bat again.
    exit /b 1
)
exit /b 0


:install
echo.
echo Installing dependencies: about 400 MB, this takes a few minutes.
echo.
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt || exit /b 1
copy /y requirements.txt "%STAMP%" >nul
echo.
echo Done. Starting Still Life AI...
exit /b 0


:failed
echo.
echo Setup failed, see the messages above.
pause
exit /b 1
