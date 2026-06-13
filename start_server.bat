@echo off
title LLM Proxy Server
echo Starting LLM Proxy Server...
cd /d "%~dp0"

if not exist ".venv" (
    echo Virtual environment (.venv) not found. Creating it...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo.
echo Server is launching. To access the portal, go to:
echo http://127.0.0.1:8000
echo.
echo Press Ctrl+C in this window to stop the server.
echo.

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

echo.
echo Server stopped.
pause
