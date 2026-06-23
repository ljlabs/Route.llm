# Verbose Streaming Logging Feature

## Quick Start

Enable verbose logging to see every SSE chunk flowing through the router:

### Windows Command Prompt
```bash
set VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Windows PowerShell
```powershell
$env:VERBOSE_STREAMING="true"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Linux/Mac
```bash
export VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Quick Scripts (Windows)
```bash
# Use provided startup script
run_verbose.bat      # For Command Prompt
.\run_verbose.ps1    # For PowerShell
```

## What You'll See

### Raw Provider Chunks (LLM → Router)
```
2026-06-23 21:00:05,123 - streaming - DEBUG - [LLM → ROUTER] Chunk 1 from Nvidia NIM (MiniMax M3): {"id": "chatcmpl-8b4a9c", "object": "chat.completion.chunk", ...
```

### Processed Client Chunks (Router → Client)
```
2026-06-23 21:00:05,124 - streaming - DEBUG - [ROUTER → CLIENT] Output chunk: event: message_start
data: {"type": "message_start", "message": {"id": "msg_abc", ...
```

### Stream Summary
```
2026-06-23 21:00:05,300 - streaming - DEBUG - [STREAM COMPLETE] Total chunks: 42, Latency: 197ms
```

## Why This Is Useful

### Problem: Web UI Says "Streaming — body accumulating"
Before: You can't see what chunks are actually flowing through  
After: Every chunk is logged to terminal for easy debugging

### Debugging Translation Issues
- See exactly what OpenAI provider sends
- See exactly what Anthropic client receives
- Spot format mismatches instantly

### Performance Analysis
- See how many chunks were sent
- Total latency from first to last byte
- Identify slow providers or processing

### Development
- Verify streaming works correctly
- Test format conversions
- Ensure schema validation passes

## Modes

### Verbose Mode (Enabled)
```bash
python -m uvicorn main:app --verbose
```
- All SSE chunks logged to terminal
- Shows [LLM → ROUTER] and [ROUTER → CLIENT] labels
- Full chunk content in debug logs

### Silent Mode (Default)
```bash
python -m uvicorn main:app
```
- No streaming logs (clean terminal)
- Application still works normally
- Better for production

## Log Entry Breakdown

```
2026-06-23 21:00:05,123 - streaming - DEBUG - [LLM → ROUTER] Chunk 1: {...}
│                 │        │          │       │             │ │
│                 │        │          │       │             │ └─ Chunk number
│                 │        │          │       │             └─── Direction
│                 │        │          │       └─────────────────── Label
│                 │        │          └──────────────────────────── Log level
│                 │        └──────────────────────────────────────── Logger name
│                 └─────────────────────────────────────────────── Timestamp
└────────────────────────────────────────────────────────────────── Prefix
```

## Understanding Chunk Flow

### Single Provider (Passthrough)
```
LLM sends chunk → [LLM → ROUTER] → [ROUTER → CLIENT] → Client receives chunk
```
No translation, just logging and validation.

### Multiple Providers (Translation)
```
LLM sends chunk → [LLM → ROUTER] → Translation → [ROUTER → CLIENT] → Client receives chunk
Example: OpenAI provider → Anthropic format → Anthropic client
```

## Common Debugging Scenarios

### Scenario 1: Missing Chunks
```
[LLM → ROUTER] Chunk 1: {...}
[ROUTER → CLIENT] Output chunk: event: message_start
data: {...}
[LLM → ROUTER] Chunk 2: {...}   ← Notice jump from 1 to 2, no chunk between
[ROUTER → CLIENT] Output chunk: event: message_stop
```
Check if chunks are being filtered somewhere.

### Scenario 2: Slow Response
```
[LLM → ROUTER] Chunk 1: {...}
[LLM → ROUTER] Chunk 2: {...}   ← 2 seconds gap between chunks
[LLM → ROUTER] Chunk 3: {...}
[STREAM COMPLETE] Total chunks: 42, Latency: 125000ms
```
Provider is slow, not your router.

### Scenario 3: Format Mismatch
```
[LLM → ROUTER] Chunk: {"id": "chatcmpl-...", "object": "chat.completion.chunk", ...}  ← OpenAI format
[ROUTER → CLIENT] Output chunk: event: message_start
data: {"type": "message_start", ...}  ← Anthropic format
```
Confirms translation is working correctly.

## Output Redirection

### Save to File
```bash
python -m uvicorn main:app --verbose > stream_debug.log 2>&1
```

### Filter Specific Messages
```bash
# Only show chunk summary
python -m uvicorn main:app --verbose 2>&1 | grep "STREAM COMPLETE"

# Show all streaming chunks
python -m uvicorn main:app --verbose 2>&1 | grep "\[LLM\|ROUTER\|COMPLETE\]"
```

## Performance Note

- **Verbose OFF** (default): Minimal overhead, suitable for production
- **Verbose ON**: Debug-level I/O, only use for development/troubleshooting
- **Disable with**: Simply don't use `--verbose` flag (default)

## Enabling Methods (Pick One)

### Method 1: Command Line (Windows CMD)
```bash
set VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
✓ Simplest for one-off debugging  
✓ Easy to remember  

### Method 2: PowerShell
```powershell
$env:VERBOSE_STREAMING="true"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
✓ Works in modern Windows terminals  
✓ Can persist in session  

### Method 3: Linux/Mac
```bash
export VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
✓ Standard Unix approach  
✓ Persists for session  

### Method 4: Use Startup Scripts
```bash
# Windows CMD
run_verbose.bat

# Windows PowerShell
.\run_verbose.ps1
```
✓ Easy to toggle between modes  
✓ Scripts provided  
✓ Can be added to development workflow

## Real World Example

### Terminal Session
```
$ python -m uvicorn main:app --verbose --host 127.0.0.1 --port 8000
...
2026-06-23 21:00:00,100 - main - INFO - 🔍 VERBOSE STREAMING LOGGING ENABLED

# Now make a streaming request in another terminal or browser
# curl http://127.0.0.1:8000/v1/messages \
#   -H "Content-Type: application/json" \
#   -d '{"messages": [{"role": "user", "content": "hello"}], "stream": true}'

# Watch the terminal...

2026-06-23 21:00:05,123 - streaming - DEBUG - [LLM → ROUTER] Chunk 1 from Nvidia NIM (MiniMax M3): {"id": "chatcmpl-8b4a9c", "object": "chat.completion.chunk", "created": 1719177605, "model": "minimaxai/minimax-m3", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}
2026-06-23 21:00:05,124 - streaming - DEBUG - [ROUTER → CLIENT] Output chunk: event: message_start
data: {"type": "message_start", "message": {"id": "msg_abc123xyz", "type": "message", "role": "assistant", "model": "minimaxai/minimax-m3", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 0, "output_tokens": 0}}}

2026-06-23 21:00:05,126 - streaming - DEBUG - [LLM → ROUTER] Chunk 2 from Nvidia NIM (MiniMax M3): {"id": "chatcmpl-8b4a9c", "object": "chat.completion.chunk", "created": 1719177605, "model": "minimaxai/minimax-m3", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}
2026-06-23 21:00:05,127 - streaming - DEBUG - [ROUTER → CLIENT] Output chunk: event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

2026-06-23 21:00:05,300 - streaming - DEBUG - [STREAM COMPLETE] Total chunks: 3, Latency: 177ms
2026-06-23 21:00:05,301 - streaming - DEBUG - [LLM → ROUTER] Stream completed after 3 chunks from Nvidia NIM (MiniMax M3)
```

Perfect! You can now see:
1. ✓ What OpenAI format the LLM sent
2. ✓ How it was translated to Anthropic format
3. ✓ Exact content of each chunk
4. ✓ Total streaming latency

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Logs not showing | Verify `--verbose` flag in command |
| Terminal too cluttered | Pipe to file: `> debug.log 2>&1` |
| Want less detail | Not available, this is the detail level for debug mode |
| Performance degradation | Don't use in production; use verbose only for testing |
| Can't find chunk | Search log file: `grep "specific_text" debug.log` |

## Next Steps

1. Run server with `--verbose`
2. Make streaming request from web UI or curl
3. Watch terminal output to debug
4. Compare LLM chunks vs client chunks
5. Identify issues or verify correctness
