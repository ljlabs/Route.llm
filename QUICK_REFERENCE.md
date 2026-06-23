# Quick Reference: Verbose Streaming Logging

## Enable Verbose Logging

### Windows Command Prompt
```batch
set VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Windows PowerShell
```powershell
$env:VERBOSE_STREAMING="true"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Linux / macOS
```bash
export VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Quick Scripts (Windows)
```bash
# Command Prompt
run_verbose.bat

# PowerShell
.\run_verbose.ps1
```

---

## Disable Verbose Logging (Default)

Just run normally without setting the environment variable:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

---

## Log Output Format

```
[LLM → ROUTER] Chunk N from Provider: {...}        # Raw from provider
[ROUTER → CLIENT] Output chunk: {...}               # Processed to client
[STREAM COMPLETE] Total chunks: X, Latency: Xms    # Summary
```

---

## What You'll See

### Example 1: Single Chunk
```
2026-06-23 21:00:05,123 - streaming - DEBUG - [LLM → ROUTER] Chunk 1 from Nvidia NIM: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}

2026-06-23 21:00:05,124 - streaming - DEBUG - [ROUTER → CLIENT] Output chunk: event: message_start
data: {"type": "message_start", "message": {"id": "msg_abc", ...}}
```

### Example 2: Stream Complete
```
2026-06-23 21:00:05,300 - streaming - DEBUG - [STREAM COMPLETE] Total chunks: 42, Latency: 177ms
2026-06-23 21:00:05,301 - streaming - DEBUG - [LLM → ROUTER] Stream completed after 42 chunks from Nvidia NIM
```

---

## Debugging Checklist

- [ ] Enable verbose mode with `VERBOSE_STREAMING=true`
- [ ] Make a streaming request
- [ ] Watch terminal for chunks
- [ ] Count `[LLM → ROUTER]` chunks (what provider sent)
- [ ] Count `[ROUTER → CLIENT]` chunks (what client receives)
- [ ] Look for format (OpenAI vs Anthropic)
- [ ] Check latency at end
- [ ] Search for errors or warnings

---

## Common Issues

| Issue | Fix |
|-------|-----|
| No logs showing | Verify `VERBOSE_STREAMING=true` is set |
| Terminal too cluttered | Redirect to file: `> debug.log 2>&1` |
| Logs cut off | Scroll up or check full file |
| Environment variable not working | Restart terminal/IDE |

---

## Advanced: Save to File

```bash
# Windows
set VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000 > debug.log 2>&1

# Linux/Mac
export VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000 > debug.log 2>&1

# Then view with:
tail -f debug.log          # Linux/Mac (live)
Get-Content debug.log -Tail 50 -Wait  # PowerShell (live)
type debug.log             # Windows CMD
```

---

## What Gets Logged

✓ Every SSE event chunk from provider  
✓ Every processed chunk to client  
✓ Stream completion stats  
✓ Chunk count and timing  

❌ Response bodies (too much noise)  
❌ Headers  
❌ Raw HTTP details  

---

## Reference Files

- `VERBOSE_LOGGING_GUIDE.md` - Comprehensive guide
- `README_VERBOSE_STREAMING.md` - Real-world examples  
- `run_verbose.bat` - Windows CMD startup script
- `run_verbose.ps1` - PowerShell startup script
