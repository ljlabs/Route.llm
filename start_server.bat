@echo off
title LLM Proxy Server
echo Starting LLM Proxy Server...
cd /d "%~dp0"

if exist ".venv" goto :start

echo Virtual environment (.venv) not found. Creating it...
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt

:start
echo(
echo Server is launching. To access the portal, go to:
echo http://0.0.0.0:8000
echo(
echo Press Ctrl+C in this window to stop the server.
echo(

.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000

echo(
echo Server stopped.
pause