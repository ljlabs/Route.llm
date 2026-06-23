#!/bin/bash
# Test streaming request to see validation logs

echo "Making streaming request..."
curl -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Say hello briefly"
      }
    ],
    "max_tokens": 100,
    "stream": true
  }' | head -20

echo ""
echo "✓ Request sent"
echo "✓ Check the web UI logs tab - you should see the sse_validation stage"
