@echo off
rem Double-click this to start AI Face Rater without a terminal window staying open
cd /d "%~dp0"
rem The py launcher (pyw) finds Python even when it isn't on PATH, so use it when it's installed
where pyw >nul 2>nul && (start "" pyw -3 face_rater_app.py) || start "" pythonw face_rater_app.py
