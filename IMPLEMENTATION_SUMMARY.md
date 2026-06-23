# Verbose Streaming Logging - Implementation Summary

## What Was Implemented

Added comprehensive streaming logging to visualize the flow of SSE chunks through the router with a `--verbose` flag.

## Files Modified

### 1. **main.py**
- Added `--verbose` flag detection from command line arguments
- Added `VERBOSE_STREAMING` global variable
- Added `stream_logger` with dynamic level setting
- Log startup message indicating verbose mode status

### 2. **core/router.py**
- Added `stream_logger` import
- Enhanced `stream_generator()` to log each chunk:
  - `[ROUTER → CLIENT]` logs for processed chunks sent to client
  - Includes chunk count and latency summary
  - Truncated first 200 chars for readability

### 3. **core/translation/stream_base.py**
- **PassthroughStreamTranslator**: Added logging for LLM chunks
  - `[LLM → ROUTER]` logs for incoming raw chunks
  - Chunk count tracking
  - Stream completion summary

- **AnthropicToOpenAIStreamTranslator**: Added logging for both directions
  - Logs both incoming and outgoing chunks
  - Tracks translation in both directions

## How It Works

### Logging Levels
- **DEBUG** (`--verbose`): See all SSE chunks and translations
- **WARNING** (default): Silent operation

### Log Format
```
[TIMESTAMP] - streaming - DEBUG - [DIRECTION] [INFO]
```

### Directions
- `[LLM → ROUTER]`: Raw chunks from provider
- `[ROUTER → CLIENT]`: Processed chunks to client
- `[STREAM COMPLETE]`: Stream summary

## Usage

### Enable Verbose Logging
```bash
# Option 1: Command line flag
python -m uvicorn main:app --verbose

# Option 2: Environment variable
export VERBOSE_STREAMING=true
python -m uvicorn main:app

# Option 3: Windows PowerShell
$env:VERBOSE_STREAMING="true"
python -m uvicorn main:app
```

### Disable (Default)
```bash
python -m uvicorn main:app
```

## Example Output

When verbose logging is enabled, you'll see:

```
[LLM → ROUTER] Chunk 1 from Nvidia NIM: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
[ROUTER → CLIENT] Output chunk: event: message_start
data: {"type": "message_start", "message": {"id": "msg_abc", "type": "message", "role": "assistant"...

[LLM → ROUTER] Chunk 2 from Nvidia NIM: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Hello"}}]}
[ROUTER → CLIENT] Output chunk: event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

[STREAM COMPLETE] Total chunks: 42, Latency: 3456ms
[LLM → ROUTER] Stream completed after 42 chunks from Nvidia NIM
```

## Benefits

### Debugging
- **See what LLM sends**: Exact raw chunks from provider
- **See what client gets**: Exact processed/translated chunks
- **Spot differences**: Compare input vs output to find issues

### Performance
- **Latency tracking**: See how long streaming takes
- **Chunk count**: Understand request/response sizes

### Testing
- **Verify translations**: Confirm OpenAI↔Anthropic conversion
- **Validate format**: Check if chunks match expected format
- **Error detection**: Spot where things break

## Technical Details

### Logger Names
- `main`: Main application logger (startup messages)
- `streaming`: Streaming-specific logger (chunk logs)

### Log Levels
- When `--verbose` is used: `streaming` logger set to DEBUG
- When not used: `streaming` logger set to WARNING (silent)

### Performance Impact
- Minimal when disabled (just logger level check)
- Moderate when enabled (writes debug logs to stdout)
- **Not recommended for production**

## Integration with Existing Logging

The verbose logging integrates with the existing logging system:
- Same format as other logs
- Uses standard Python logging
- Can be redirected to files if needed
- Works with existing log level configuration

## Troubleshooting

### Logs not appearing?
- Check if `--verbose` flag was used
- Verify application restarted after flag change
- Check terminal/console for output

### Too much output?
- That's expected! Verbose mode logs every chunk
- Redirect to file: `python -m uvicorn main:app --verbose > stream_debug.log 2>&1`

### Want to log to file?
- Use shell redirection: `> streaming_debug.log 2>&1`
- Or update logging config in main.py to add file handler
