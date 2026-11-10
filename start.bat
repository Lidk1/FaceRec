@echo off
chcp 65001 >nul
cd /d "%~dp0"
FaceRec.exe --no-browser %*
pause
