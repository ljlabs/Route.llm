# SSE Response Schema Validation - Complete Guide

## Summary

Mimo successfully implemented schema validation for Anthropic and OpenAI SSE responses. The validation system is **fully working and operational**.

### ✅ What Was Implemented

1. **Schema Validation Functions** (`core/translation/response_schemas.py`)
   - `validate_anthropic_sse_line()` - Validates individual SSE lines against Anthropic spec
   - `validate_openai_sse_line()` - Validates individual SSE lines against OpenAI spec
   - Uses Pydantic models for strict schema validation
   - Logs warnings when validation fails

2. **Stream Translators** (`core/translation/stream_base.py`)
   - `PassthroughStreamTranslator` - Optionally validates while passing through
   - `AnthropicToOpenAIStreamTranslator` - Validates during translation
   - Both track:
     - `validation_checked` - Number of SSE events validated
     - `validation_warnings` - List of invalid events

3. **Router Integration** (`core/router.py`, lines 455-517)
   - Reads `response_format` setting from database
   - Passes it to stream translator as `validate_format` parameter
   - After streaming completes, logs validation results in `sse_validation` stage

4. **Web UI Support** (`static/app.js`, lines 539-548)
   - Timeline shows `sse_validation` stage
   - Displays validation results with status badge
   - Shows event count and any warnings

## Why You Don't See Validation Logs

**The validation only triggers on STREAMING requests**, not non-streaming requests.

### Non-Streaming Request (❌ No Validation)
```bash
curl -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false
  }'
```
Log stages: `router_received` → `provider_request` → `provider_response` → `client_response`

### Streaming Request (✅ With Validation)
```bash
curl -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'
```
Log stages: `router_received` → `provider_request` → `provider_response` → **`sse_validation`** → `client_response`

## How to See Validation Logs

### Option 1: Via Web UI
1. Start the server: `python -m uvicorn main:app --host 127.0.0.1 --port 8000`
2. Open browser: `http://127.0.0.1:8000`
3. Go to **Logs** tab
4. Make a **streaming** request (with `"stream": true`)
5. Find the latest log in the UI
6. Look for the **SSE Validation** stage in the timeline

### Option 2: Via Terminal (curl)
```bash
# Make a streaming request
curl -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 100,
    "stream": true
  }'
```

Then query the database:
```bash
python -c "
import database as db
logs = db.get_logs()
for log in logs[:1]:
    for evt in log['events']:
        if evt['stage'] == 'sse_validation':
            print('✓ Validation found!')
            print(evt['body'])
"
```

### Option 3: Via Python (Direct Test)
```python
import asyncio
import json
from core.router import get_router_service, init_router_service
from core.providers.service import get_provider_service
from core.rate_limiter import init_rate_limiter, get_per_provider_limiter
from infrastructure.http_client import init_http_client
import database as db

async def test():
    # Initialize
    db.init_db()
    http_client = init_http_client()
    rate_limiter = init_rate_limiter()
    per_provider_limiter = get_per_provider_limiter()
    init_router_service(http_client, rate_limiter, per_provider_limiter)
    
    # Set format
    db.set_response_format('anthropic')
    
    # Get services
    router = get_router_service()
    provider_service = get_provider_service()
    provider = provider_service.get_active_provider()
    
    # Make streaming request
    request = {
        "model": provider.model_name,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True
    }
    
    response = await router.route_anthropic_request(request, stream=True)
    
    # Consume stream
    async for chunk in response.body_iterator:
        pass
    
    # Check logs
    logs = db.get_logs()
    for evt in logs[0]['events']:
        if evt['stage'] == 'sse_validation':
            print(f"✓ Validation: {json.loads(evt['body'])}")

asyncio.run(test())
```

## Validation Output Example

When validation completes successfully:
```json
{
  "format": "anthropic",
  "events_checked": 4,
  "warnings": [],
  "status": "pass"
}
```

When validation finds issues:
```json
{
  "format": "anthropic",
  "events_checked": 5,
  "warnings": [
    "data: {\"type\": \"unknown_event\", ...}"
  ],
  "status": "fail"
}
```

## Configuration

### Change Response Format
```python
import database as db
db.set_response_format('anthropic')  # or 'openai'
```

Or via Web UI:
1. Click **Settings** in sidebar
2. Set "Response Format" dropdown
3. Save

### Enable/Disable Validation
Validation is **always enabled** when a response format is set. It validates SSE events based on the target format.

## Implementation Details

### Where Validation Happens
1. **Input**: Each SSE line from provider's streaming response
2. **Validation**: Checked against Pydantic schema (Anthropic or OpenAI)
3. **Logging**: 
   - If `validation_format` is set, events are validated
   - Warnings accumulated for invalid events
   - Results logged in `sse_validation` stage after stream completes

### Performance Impact
- Minimal: JSON schema validation only on SSE data lines
- Runs during streaming, after line is yielded
- Does not block response streaming

### Supported Validation Types

**Anthropic SSE Events:**
- `message_start`
- `content_block_start`
- `content_block_delta`
- `content_block_stop`
- `message_delta`
- `message_stop`
- `ping`
- `error`

**OpenAI SSE Events:**
- `chat.completion.chunk` (validated via `object` field)

## Troubleshooting

### "No sse_validation stage found"
- **Cause**: Making non-streaming request
- **Fix**: Add `"stream": true` to request

### "No validation logs in database"
- **Cause**: Response format not set in database
- **Check**: `db.get_response_format()` should return 'anthropic' or 'openai'
- **Fix**: `db.set_response_format('anthropic')`

### "Validation warnings found"
- This is expected if the SSE format doesn't match the schema
- Check the `warnings` array in the validation log
- Each warning contains the invalid SSE line

## Next Steps

To fully test validation across all providers:
1. Make streaming requests to each configured provider
2. Check logs for `sse_validation` stage
3. Verify all events pass validation
4. Integration tests in `tests/test_response_schemas.py` cover schema validation

