@echo off
REM Run the proxy router with verbose streaming logging enabled
setlocal enabledelayedexpansion

echo.
echo 🔍 Starting LLM Proxy Router with VERBOSE STREAMING ENABLED
echo.
echo To make it less verbose, run instead:
echo   python -m uvicorn main:app --host 127.0.0.1 --port 8000
echo.

set VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000

pause
