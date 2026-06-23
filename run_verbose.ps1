# Run the proxy router with verbose streaming logging enabled

Write-Host ""
Write-Host "🔍 Starting LLM Proxy Router with VERBOSE STREAMING ENABLED" -ForegroundColor Cyan
Write-Host ""
Write-Host "To make it less verbose, run instead:" -ForegroundColor Yellow
Write-Host "  python -m uvicorn main:app --host 127.0.0.1 --port 8000"
Write-Host ""

$env:VERBOSE_STREAMING = "true"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
