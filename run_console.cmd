@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Control Runtime is not installed. Run setup.ps1 first.
  exit /b 1
)
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
".venv\Scripts\python.exe" -m nero_console serve
set "NERO_EXIT_CODE=%ERRORLEVEL%"
if not "%NERO_EXIT_CODE%"=="0" (
  echo.
  echo NERO control console failed to start. Review the message above.
  pause
)
exit /b %NERO_EXIT_CODE%
