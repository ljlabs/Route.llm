# Verbose Streaming Logging Guide

## Overview

The verbose streaming logging feature allows you to see every single SSE chunk flowing through the router:
- **LLM → ROUTER**: Raw chunks received from the LLM provider
- **ROUTER → CLIENT**: Processed/translated chunks sent to the client
- **CHUNK DETAILS**: Full content of each chunk for debugging

This is extremely helpful for debugging streaming response issues and seeing the difference between what the LLM sends and what the client receives.

## How to Enable

### Option 1: Command Line (Windows CMD)
```bash
set VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Option 2: PowerShell
```powershell
$env:VERBOSE_STREAMING="true"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Option 3: Linux/Mac
```bash
export VERBOSE_STREAMING=true
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Option 4: Use Startup Script (Windows)
```bash
# For Command Prompt
run_verbose.bat

# For PowerShell
.\run_verbose.ps1
```

## Log Output Format

When verbose logging is enabled, you'll see entries like:

### Incoming Chunks from LLM
```
2026-06-23 21:00:00,123 - streaming - DEBUG - [LLM → ROUTER] Chunk 1 from Nvidia NIM (MiniMax M3): {"id": "chatcmpl-123", "object": "chat.completion.chunk"...
2026-06-23 21:00:00,125 - streaming - DEBUG - [LLM → ROUTER] Chunk 2 from Nvidia NIM (MiniMax M3): {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices":...
2026-06-23 21:00:00,127 - streaming - DEBUG - [LLM → ROUTER] Chunk 3 from Nvidia NIM (MiniMax M3): {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices":...
```

### Translated/Processed Chunks to Client
```
2026-06-23 21:00:00,126 - streaming - DEBUG - [ROUTER → CLIENT] Output chunk: event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta"...
2026-06-23 21:00:00,128 - streaming - DEBUG - [ROUTER → CLIENT] Output chunk: event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta"...
```

### Stream Completion
```
2026-06-23 21:00:00,200 - streaming - DEBUG - [STREAM COMPLETE] Total chunks: 45, Latency: 3456ms
2026-06-23 21:00:00,215 - streaming - DEBUG - [LLM → ROUTER] Stream completed after 45 chunks from Nvidia NIM (MiniMax M3)
```

## Log Fields Explained

- **[LLM → ROUTER]**: Chunks received directly from the provider (before any translation)
- **Chunk N**: Sequential chunk number
- **from {Provider Name}**: Which provider sent this chunk
- **{First 200 chars}...**: Preview of the chunk content
- **[ROUTER → CLIENT]**: Chunk after translation/processing (what client receives)
- **Output chunk**: Full formatted SSE event being sent to client
- **[STREAM COMPLETE]**: Summary of stream completion
  - `Total chunks`: Number of SSE events received
  - `Latency`: Total time from first to last byte

## Example: Debugging Translation Issues

If you're converting between OpenAI and Anthropic formats:

1. **LLM sends OpenAI format** (from compatible provider):
   ```
   [LLM → ROUTER] Chunk 1: {"id": "chatcmpl-...", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Hello"}}]}
   ```

2. **Router translates to Anthropic** (for client):
   ```
   [ROUTER → CLIENT] Output chunk: event: content_block_delta
   data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
   ```

3. **You can see the difference** between input/output formats

## Example: Debugging Streaming Issues

With verbose logging enabled, you can:

1. **Count total chunks**: Look for `[STREAM COMPLETE]` to see total chunks
2. **Spot delays**: Times between consecutive chunks show where delays occur
3. **Find malformed chunks**: See exact chunk content that fails validation
4. **Track transformations**: See exactly how chunks change through translation

## Performance Note

Verbose logging adds I/O overhead and should **not** be used in production. It's meant for:
- Development and debugging
- Testing streaming responses
- Verifying format conversions
- Troubleshooting streaming issues

For production, either:
- Don't use `--verbose` flag
- Use selective logging at WARNING level instead

## Terminal Output Example

```bash
$ python -m uvicorn main:app --verbose
...
2026-06-23 21:00:00,100 - main - INFO - 🔍 VERBOSE STREAMING LOGGING ENABLED - All SSE chunks will be logged to terminal

[Making a streaming request...]

2026-06-23 21:00:05,123 - streaming - DEBUG - [LLM → ROUTER] Chunk 1 from Nvidia NIM (MiniMax M3): {"id": "chatcmpl-8b4a9c", "object": "chat.completion.chunk", "created": 1719177605, "model": "minimaxai/minimax-m3", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}
2026-06-23 21:00:05,124 - streaming - DEBUG - [ROUTER → CLIENT] Output chunk: event: message_start
data: {"type": "message_start", "message": {"id": "msg_abc123...
2026-06-23 21:00:05,126 - streaming - DEBUG - [LLM → ROUTER] Chunk 2 from Nvidia NIM (MiniMax M3): {"id": "chatcmpl-8b4a9c", "object": "chat.completion.chunk", "created": 1719177605, "model": "minimaxai/minimax-m3", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}
2026-06-23 21:00:05,127 - streaming - DEBUG - [ROUTER → CLIENT] Output chunk: event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
2026-06-23 21:00:05,300 - streaming - DEBUG - [STREAM COMPLETE] Total chunks: 3, Latency: 197ms
2026-06-23 21:00:05,301 - streaming - DEBUG - [LLM → ROUTER] Stream completed after 3 chunks from Nvidia NIM (MiniMax M3)
```

## Interpreting the Logs

1. **First chunk from LLM** - Role/header information
2. **Router translates it** - Outputs `message_start` event
3. **Content chunks flow** - Each delta gets transformed
4. **Stream completes** - Total count and timing shown

This makes it easy to spot:
- Missing chunks
- Format mismatches
- Ordering issues
- Performance bottlenecks
- Validation errors
