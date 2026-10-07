@echo off
rem Double-click this to start AI Face Rater without a terminal window staying open
cd /d "%~dp0"
rem With several Pythons installed, the default one may not have the packages, so use the first Python that has them
rem (the newest first). The py launcher finds Python even when it is not on PATH.
for %%v in (3.14 3.13 3.12 3.11 3.10) do (
    py -%%v -c "import cv2, mediapipe, numpy, PIL" >nul 2>nul && (
        start "" pyw -%%v face_rater_app.py
        exit /b
    )
)
rem No Python with the packages: start anyway, so the app can say which package is missing
start "" pythonw face_rater_app.py
