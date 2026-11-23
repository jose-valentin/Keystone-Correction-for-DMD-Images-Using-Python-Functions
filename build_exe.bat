@echo off
rem Builds KeystoneDashboard.exe on Windows. Run this from the folder with dashboard.py.
rem Needs the regular 64-bit Python from python.org (it installs the "py" launcher).
rem The Python that comes with MSYS2 will not work for this.

where py >nul 2>nul
if errorlevel 1 (
    echo Could not find the "py" launcher.
    echo Install Python from https://www.python.org/downloads/ and run this file again.
    pause
    exit /b 1
)

py -3 -m pip install -r requirements-dashboard.txt
if errorlevel 1 (
    echo.
    echo Something went wrong while installing. Read the messages above. The exe was NOT built.
    pause
    exit /b 1
)

py -3 -m PyInstaller --onefile --windowed --name KeystoneDashboard dashboard.py
if errorlevel 1 (
    echo.
    echo Something went wrong while building. Read the messages above. The exe was NOT built.
    pause
    exit /b 1
)

echo.
echo Done. The program is in the dist folder: dist\KeystoneDashboard.exe
pause
