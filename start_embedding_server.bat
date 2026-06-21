@echo off
title Embedding Proxy Server
echo Starting Embedding Proxy Server on port 8081...
cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

.venv\Scripts\python -m uvicorn embedding_main:app --host 127.0.0.1 --port 8081

echo Embedding Proxy Server stopped.
pause
