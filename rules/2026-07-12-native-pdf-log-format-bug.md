# Native PDF log stored raw Gemini response instead of final_response

**Symptom**: Dashboard "Response to Client" showed Gemini `candidates` format instead of OpenAI `choices`, even though the actual HTTP response was correct.

**Failed approaches**: N/A — this was a straightforward variable naming bug, not a multi-attempt debugging loop.

**Fix**: Changed `json.dumps(gemini_response, indent=2)` to `json.dumps(final_response, indent=2)` at `core/router.py:444`. Same pattern existed at line 359 in `_handle_non_streaming` where `json.dumps(response_json)` was logged instead of `json.dumps(final_response)`.

**Signal**: When a log entry and the HTTP response show different formats, check whether the log stores the *intermediate* variable (raw provider response) instead of the *final* variable (post-translation response). Look for `json.dumps(some_variable)` near `complete_request_log` calls — it should always use the variable that matches what `JSONResponse(content=...)` returns.
