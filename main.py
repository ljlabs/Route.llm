from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx
import json
import asyncio
import os
import sys
import time
import traceback
import database as db
import translator as ts

app = FastAPI(title="LLM Proxy & Router")

# Global client to handle connection pooling and keep-alive for streams
http_client = None

# Enable CORS for convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database and HTTP client on startup
@app.on_event("startup")
async def startup_event():
    db.init_db()
    global http_client
    # Disable timeout limits or set high timeout for streaming
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0, read=120.0))

# Close client on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    global http_client
    if http_client:
        await http_client.aclose()

# API: Providers Management
@app.get("/api/providers")
def get_providers():
    return db.get_providers()

@app.post("/api/providers")
async def add_provider(request: Request):
    data = await request.json()
    db.add_provider(
        name=data["name"],
        api_type=data["api_type"],
        endpoint_url=data["endpoint_url"],
        api_key=data["api_key"],
        model_name=data["model_name"],
        is_active=data.get("is_active", 0)
    )
    return {"status": "success"}

@app.put("/api/providers/{provider_id}")
async def update_provider(provider_id: int, request: Request):
    data = await request.json()
    db.update_provider(
        provider_id=provider_id,
        name=data["name"],
        api_type=data["api_type"],
        endpoint_url=data["endpoint_url"],
        api_key=data["api_key"],
        model_name=data["model_name"],
        is_active=data.get("is_active", 0)
    )
    return {"status": "success"}

@app.delete("/api/providers/{provider_id}")
def delete_provider(provider_id: int):
    db.delete_provider(provider_id)
    return {"status": "success"}

@app.post("/api/providers/{provider_id}/active")
def activate_provider(provider_id: int):
    db.set_active_provider(provider_id)
    return {"status": "success"}

# API: Settings Management
@app.get("/api/settings")
def get_settings():
    return {"log_limit": db.get_log_limit()}

@app.post("/api/settings")
async def set_settings(request: Request):
    data = await request.json()
    if "log_limit" in data:
        db.set_log_limit(int(data["log_limit"]))
    return {"status": "success"}

# API: Logs
@app.get("/api/logs")
def get_logs():
    return db.get_logs()

@app.delete("/api/logs")
def clear_logs():
    db.clear_logs()
    return {"status": "success"}

# API: Test chat endpoint
@app.post("/api/chat")
async def test_chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    active_prov = db.get_active_provider()
    if not active_prov:
        raise HTTPException(status_code=400, detail="No active provider configured")
        
    # Build standard OpenAI request format for frontend testing
    test_req = {
        "model": active_prov["model_name"],
        "messages": [{"role": "user", "content": message}],
        "stream": False
    }
    
    url = active_prov["endpoint_url"]
    
    try:
        if active_prov["api_type"] in ["openai", "gemini", "mistral"]:
            headers = {"Authorization": f"Bearer {active_prov['api_key']}"}
            is_gemini = active_prov["api_type"] == "gemini"
            payload = ts.sanitize_openai_payload(test_req, is_gemini=is_gemini)
        else: # anthropic
            payload = ts.openai_to_anthropic_request(test_req, active_prov["model_name"])
            headers = {
                "x-api-key": active_prov["api_key"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        
        print(f"\n--- OUTGOING TEST REQUEST TO PROVIDER [{active_prov['name']}] ---")
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        resp = await http_client.post(url, json=payload, headers=headers)
        
        print(f"Response status: {resp.status_code}")
        
        # Handle non-200 responses gracefully
        if resp.status_code != 200:
            error_detail = f"Backend returned {resp.status_code}"
            try:
                error_json = resp.json()
                # Try to extract message from various common error formats
                if "error" in error_json:
                    if isinstance(error_json["error"], dict):
                        error_detail += f": {error_json['error'].get('message', '')}"
                    else:
                        error_detail += f": {error_json['error']}"
                elif "detail" in error_json:
                    error_detail += f": {error_json['detail']}"
                elif "message" in error_json:
                    error_detail += f": {error_json['message']}"
            except Exception:
                # If not JSON, use the raw text truncated
                if resp.text:
                    error_detail += f": {resp.text[:200]}"

            # Add human-friendly hints for common status codes
            if resp.status_code == 401:
                error_detail = f"Unauthorized (401): Check your API Key. {error_detail}"
            elif resp.status_code == 404:
                error_detail = f"Not Found (404): Check your Endpoint URL or Model Name. {error_detail}"
            elif resp.status_code == 403:
                error_detail = f"Forbidden (403): Access denied. {error_detail}"
            elif resp.status_code == 429:
                error_detail = f"Rate Limited (429): Too many requests. {error_detail}"
            elif 400 <= resp.status_code < 500:
                error_detail = f"Request Error ({resp.status_code}): {error_detail}"
            elif resp.status_code >= 500:
                error_detail = f"Server Error ({resp.status_code}): Backend is down. {error_detail}"

            db.add_log(
                provider_name=active_prov["name"],
                request_method="POST",
                request_path="/api/chat",
                request_body=json.dumps(payload, indent=2),
                response_status=resp.status_code,
                response_body=resp.text
            )
            raise HTTPException(status_code=resp.status_code, detail=error_detail)

        try:
            resp_json = resp.json()
        except Exception as json_err:
            err_msg = f"JSON Decode Error: {str(json_err)}\n\nResponse Text:\n{resp.text}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            db.add_log(
                provider_name=active_prov["name"],
                request_method="POST",
                request_path="/api/chat",
                request_body=json.dumps(payload, indent=2),
                response_status=resp.status_code,
                response_body=err_msg
            )
            raise HTTPException(status_code=500, detail=err_msg)
        
        # Process success response
        try:
            if active_prov["api_type"] in ["openai", "gemini", "mistral"]:
                content = resp_json["choices"][0]["message"]["content"]
            else:
                openai_res = ts.anthropic_to_openai_response(resp_json)
                content = openai_res["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as parse_err:
            err_msg = f"Response Parsing Error: {str(parse_err)}. The provider returned a format we didn't expect."
            db.add_log(
                provider_name=active_prov["name"],
                request_method="POST",
                request_path="/api/chat",
                request_body=json.dumps(payload, indent=2),
                response_status=resp.status_code,
                response_body=json.dumps(resp_json, indent=2)
            )
            raise HTTPException(status_code=500, detail=err_msg)
            
        db.add_log(
            provider_name=active_prov["name"],
            request_method="POST",
            request_path="/api/chat",
            request_body=json.dumps(payload, indent=2),
            response_status=resp.status_code,
            response_body=json.dumps(resp_json, indent=2)
        )
        return {"response": content, "provider": active_prov["name"]}
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        err_msg = f"Exception: {str(e)}\n\n{traceback.format_exc()}"
        print(f"ERROR during connection/request: {err_msg}", file=sys.stderr)
        db.add_log(
            provider_name=active_prov["name"],
            request_method="POST",
            request_path="/api/chat",
            request_body=json.dumps(test_req, indent=2),
            response_status=500,
            response_body=err_msg
        )
        raise HTTPException(status_code=500, detail=str(e))

# Proxy Router Functions
async def stream_generator(response, active_prov, is_incoming_anthropic, is_backend_openai, req_body_str):
    accumulated_response_blocks = []
    sent_start = False
    
    # Track streamed tool call states (OpenAI index -> Anthropic block index / details)
    tool_idx_map = {}
    next_anth_block_idx = 1 # Index 0 is text content block
    
    is_gemini = active_prov.get("api_type") == "gemini"
    is_mistral = active_prov.get("api_type") == "mistral"
    
    try:
        async for line in response.aiter_lines():
            if not line:
                continue
                
            # Translation: Incoming Anthropic, Backend OpenAI (e.g. OpenRouter, Gemini, Mistral)
            if is_incoming_anthropic and (is_backend_openai or is_gemini or is_mistral):
                if line.startswith("data:"):
                    data_content = line.replace("data:", "").strip()
                    if data_content == "[DONE]":
                        break
                    try:
                        data = json.loads(data_content)
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        
                        # Generate message start events once
                        if not sent_start:
                            msg_id = data.get("id", f"msg_local_{os.urandom(8).hex()}")
                            model_name = data.get("model", active_prov["model_name"])
                            yield f"event: message_start\ndata: " + json.dumps({
                                "type": "message_start",
                                "message": {
                                    "id": msg_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "model": model_name,
                                    "content": [],
                                    "stop_reason": None,
                                    "stop_sequence": None,
                                    "usage": {"input_tokens": 0, "output_tokens": 0}
                                }
                            }) + "\n\n"
                            
                            # Start default text content block at index 0
                            yield f"event: content_block_start\ndata: " + json.dumps({
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""}
                            }) + "\n\n"
                            sent_start = True
                        
                        # Yield standard text content chunk
                        text = delta.get("content", "")
                        if text:
                            # Accumulate response block text
                            if len(accumulated_response_blocks) <= 0:
                                accumulated_response_blocks.append({"type": "text", "text": ""})
                            accumulated_response_blocks[0]["text"] += text
                            
                            yield f"event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": text}
                            }) + "\n\n"
                            
                        # Yield tool calls chunks if present
                        tool_calls = delta.get("tool_calls", [])
                        for call in tool_calls:
                            call_idx = call.get("index", 0)
                            
                            # If new tool call index, send start event
                            if call_idx not in tool_idx_map:
                                anth_block_id = f"toolu_{os.urandom(8).hex()}"
                                tool_name = call.get("function", {}).get("name", "unknown_tool")
                                
                                signature = call.get("extra_content", {}).get("google", {}).get("thought_signature")
                                if signature:
                                    anth_block_id = f"{anth_block_id}{ts.SIGNATURE_SEPARATOR}{signature}"
                                    
                                tool_idx_map[call_idx] = {
                                    "id": anth_block_id,
                                    "name": tool_name,
                                    "arguments_accum": "",
                                    "anth_idx": next_anth_block_idx
                                }
                                
                                yield f"event: content_block_start\ndata: " + json.dumps({
                                    "type": "content_block_start",
                                    "index": next_anth_block_idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": anth_block_id,
                                        "name": tool_name,
                                        "input": {}
                                    }
                                }) + "\n\n"
                                next_anth_block_idx += 1
                            
                            # Yield argument json content deltas
                            arg_delta = call.get("function", {}).get("arguments", "")
                            if arg_delta:
                                tool_idx_map[call_idx]["arguments_accum"] += arg_delta
                                yield f"event: content_block_delta\ndata: " + json.dumps({
                                    "type": "content_block_delta",
                                    "index": tool_idx_map[call_idx]["anth_idx"],
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": arg_delta
                                    }
                                }) + "\n\n"
                            
                        # Handle stop reasons
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice.get("finish_reason")
                            stop_reason = "end_turn"
                            
                            # Stop text content block
                            yield f"event: content_block_stop\ndata: " + json.dumps({
                                "type": "content_block_stop",
                                "index": 0
                            }) + "\n\n"
                            
                            # Stop any active tool blocks
                            for c_idx, t_data in tool_idx_map.items():
                                yield f"event: content_block_stop\ndata: " + json.dumps({
                                    "type": "content_block_stop",
                                    "index": t_data["anth_idx"]
                                }) + "\n\n"
                                
                                # Try parsing accumulated json arguments
                                try:
                                    parsed_args = json.loads(t_data["arguments_accum"])
                                except Exception:
                                    parsed_args = t_data["arguments_accum"]
                                accumulated_response_blocks.append({
                                    "type": "tool_use",
                                    "id": t_data["id"],
                                    "name": t_data["name"],
                                    "input": parsed_args
                                })
                            
                            if finish_reason == "tool_calls" or tool_idx_map:
                                stop_reason = "tool_use"
                            elif finish_reason == "length":
                                stop_reason = "max_tokens"
                            
                            yield f"event: message_delta\ndata: " + json.dumps({
                                "type": "message_delta",
                                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                "usage": {"output_tokens": 0}
                            }) + "\n\n"
                            yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"
                    except Exception:
                        pass
                        
            # Translation: Incoming OpenAI, Backend Anthropic
            elif not is_incoming_anthropic and not (is_backend_openai or is_gemini):

                if line.startswith("data:"):
                    data_content = line.replace("data:", "").strip()
                    try:
                        data = json.loads(data_content)
                        event_type = data.get("type")
                        
                        if event_type == "message_start":
                            msg_id = data.get("message", {}).get("id", f"chatcmpl-{os.urandom(8).hex()}")
                            model_name = data.get("message", {}).get("model", active_prov["model_name"])
                            yield "data: " + json.dumps({
                                "id": msg_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None
                                }]
                            }) + "\n\n"
                            sent_start = True
                        elif event_type == "content_block_delta":
                            text = data.get("delta", {}).get("text", "")
                            if text:
                                if len(accumulated_response_blocks) <= 0:
                                    accumulated_response_blocks.append({"type": "text", "text": ""})
                                accumulated_response_blocks[0]["text"] += text
                                yield "data: " + json.dumps({
                                    "id": f"chatcmpl-{os.urandom(8).hex()}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None
                                    }]
                                }) + "\n\n"
                        elif event_type == "message_delta":
                            anth_stop = data.get("delta", {}).get("stop_reason")
                            stop_reason = "stop"
                            if anth_stop == "end_turn":
                                stop_reason = "stop"
                            elif anth_stop == "max_tokens":
                                stop_reason = "length"
                            
                            yield "data: " + json.dumps({
                                "id": f"chatcmpl-{os.urandom(8).hex()}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": stop_reason
                                }]
                            }) + "\n\n"
                            yield "data: [DONE]\n\n"
                    except Exception:
                        pass
                        
            # Format types match directly (No translation needed)
            else:
                yield line + "\n"
                if line.startswith("data:"):
                    data_content = line.replace("data:", "").strip()
                    if data_content != "[DONE]":
                        try:
                            data = json.loads(data_content)
                            if is_incoming_anthropic:
                                if data.get("type") == "content_block_delta":
                                    text = data.get("delta", {}).get("text", "")
                                    if len(accumulated_response_blocks) <= 0:
                                        accumulated_response_blocks.append({"type": "text", "text": ""})
                                    accumulated_response_blocks[0]["text"] += text
                            else:
                                text = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if text:
                                    if len(accumulated_response_blocks) <= 0:
                                        accumulated_response_blocks.append({"type": "text", "text": ""})
                                    accumulated_response_blocks[0]["text"] += text
                        except Exception:
                            pass
                            
    except Exception as stream_err:
        err_msg = f"Stream interrupted: {str(stream_err)}\n\n{traceback.format_exc()}"
        print(f"STREAM ERROR: {err_msg}", file=sys.stderr)
        if is_incoming_anthropic:
            yield "event: error\ndata: " + json.dumps({"type": "error", "error": {"type": "api_error", "message": str(stream_err)}}) + "\n\n"
        else:
            yield "data: " + json.dumps({"choices": [{"delta": {"content": f"\n[Stream Error: {str(stream_err)}]"}, "finish_reason": "error"}]}) + "\n\n"
    finally:
        await response.aclose()

    db.add_log(
        provider_name=active_prov["name"],
        request_method="POST",
        request_path="/v1/messages" if is_incoming_anthropic else "/v1/chat/completions",
        request_body=req_body_str,
        response_status=200,
        response_body=json.dumps(accumulated_response_blocks, indent=2) if accumulated_response_blocks else "[Streamed response logs]"
    )

# Anthropic Messages Proxy Endpoint
@app.post("/v1/messages")
async def proxy_anthropic_messages(request: Request):
    active_prov = db.get_active_provider()
    if not active_prov:
        raise HTTPException(status_code=400, detail="No active provider configured")
        
    req_body = await request.json()
    req_body_str = json.dumps(req_body, indent=2)
    stream_requested = req_body.get("stream", False)
    
    is_backend_openai = active_prov["api_type"] in ["openai", "gemini", "mistral"]
    url = active_prov["endpoint_url"]
    
    try:
        if is_backend_openai:
            openai_req = ts.anthropic_to_openai_request(req_body, active_prov["model_name"])
            is_gemini = active_prov["api_type"] == "gemini"
            openai_req = ts.sanitize_openai_payload(openai_req, is_gemini=is_gemini)
            headers = {"Authorization": f"Bearer {active_prov['api_key']}"}
            
            print(f"\n--- PROXYING ANTHROPIC TO OPENAI [{active_prov['name']}] ---")
            print(f"Target URL: {url}")
            
            if stream_requested:
                req = http_client.build_request("POST", url, json=openai_req, headers=headers)
                resp = await http_client.send(req, stream=True)
                
                if resp.status_code >= 400:
                    await resp.aread()
                    resp_text = resp.text
                    print(f"Backend returned error status code: {resp.status_code}\nBody: {resp_text}", file=sys.stderr)
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/messages",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=resp_text
                    )
                    await resp.aclose()
                    return JSONResponse(status_code=resp.status_code, content={"type": "error", "error": {"type": "api_error", "message": f"Backend returned {resp.status_code}: {resp_text}"}})
                
                return StreamingResponse(
                    stream_generator(resp, active_prov, True, True, req_body_str),
                    media_type="text/event-stream"
                )
            else:
                resp = await http_client.post(url, json=openai_req, headers=headers)
                try:
                    resp_json = resp.json()
                    anth_resp = ts.openai_to_anthropic_response(resp_json)
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/messages",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=json.dumps(anth_resp, indent=2)
                    )
                    return JSONResponse(content=anth_resp, status_code=200)
                except Exception as json_err:
                    err_msg = f"JSON Decode Error: {str(json_err)}\n\nResponse Text:\n{resp.text}"
                    print(f"ERROR: {err_msg}", file=sys.stderr)
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/messages",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=err_msg
                    )
                    raise HTTPException(status_code=500, detail=err_msg)
        else:
            req_body["model"] = active_prov["model_name"]
            headers = {
                "x-api-key": active_prov["api_key"],
                "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
                "content-type": "application/json"
            }
            
            print(f"\n--- PROXYING ANTHROPIC TO ANTHROPIC [{active_prov['name']}] ---")
            print(f"Target URL: {url}")
            
            if stream_requested:
                req = http_client.build_request("POST", url, json=req_body, headers=headers)
                resp = await http_client.send(req, stream=True)
                
                if resp.status_code >= 400:
                    await resp.aread()
                    resp_text = resp.text
                    print(f"Backend returned error status code: {resp.status_code}\nBody: {resp_text}", file=sys.stderr)
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/messages",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=resp_text
                    )
                    await resp.aclose()
                    return JSONResponse(status_code=resp.status_code, content={"type": "error", "error": {"type": "api_error", "message": f"Backend returned {resp.status_code}: {resp_text}"}})
                
                return StreamingResponse(
                    stream_generator(resp, active_prov, True, False, req_body_str),
                    media_type="text/event-stream"
                )
            else:
                resp = await http_client.post(url, json=req_body, headers=headers)
                try:
                    resp_json = resp.json()
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/messages",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=json.dumps(resp_json, indent=2)
                    )
                    return JSONResponse(content=resp_json, status_code=resp.status_code)
                except Exception as json_err:
                    err_msg = f"JSON Decode Error: {str(json_err)}\n\nResponse Text:\n{resp.text}"
                    print(f"ERROR: {err_msg}", file=sys.stderr)
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/messages",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=err_msg
                    )
                    raise HTTPException(status_code=500, detail=err_msg)
                
    except Exception as e:
        err_msg = f"Exception: {str(e)}\n\n{traceback.format_exc()}"
        print(f"ERROR during messages proxy: {err_msg}", file=sys.stderr)
        db.add_log(
            provider_name=active_prov["name"],
            request_method="POST",
            request_path="/v1/messages",
            request_body=req_body_str,
            response_status=500,
            response_body=err_msg
        )
        raise HTTPException(status_code=500, detail=str(e))

# OpenAI Chat Completions Proxy Endpoint
@app.post("/v1/chat/completions")
async def proxy_openai_chat_completions(request: Request):
    active_prov = db.get_active_provider()
    if not active_prov:
        raise HTTPException(status_code=400, detail="No active provider configured")
        
    req_body = await request.json()
    req_body_str = json.dumps(req_body, indent=2)
    stream_requested = req_body.get("stream", False)
    
    is_backend_openai = active_prov["api_type"] in ["openai", "gemini", "mistral"]
    url = active_prov["endpoint_url"]
    
    try:
        if not is_backend_openai:
            anth_req = ts.openai_to_anthropic_request(req_body, active_prov["model_name"])
            headers = {
                "x-api-key": active_prov["api_key"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            print(f"\n--- PROXYING OPENAI TO ANTHROPIC [{active_prov['name']}] ---")
            print(f"Target URL: {url}")
            
            if stream_requested:
                req = http_client.build_request("POST", url, json=anth_req, headers=headers)
                resp = await http_client.send(req, stream=True)
                
                if resp.status_code >= 400:
                    await resp.aread()
                    resp_text = resp.text
                    print(f"Backend returned error status code: {resp.status_code}\nBody: {resp_text}", file=sys.stderr)
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/chat/completions",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=resp_text
                    )
                    await resp.aclose()
                    return JSONResponse(status_code=resp.status_code, content={"error": {"message": f"Backend returned {resp.status_code}: {resp_text}", "type": "api_error"}})
                
                return StreamingResponse(
                    stream_generator(resp, active_prov, False, False, req_body_str),
                    media_type="text/event-stream"
                )
            else:
                resp = await http_client.post(url, json=anth_req, headers=headers)
                try:
                    resp_json = resp.json()
                    openai_resp = ts.anthropic_to_openai_response(resp_json)
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/chat/completions",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=json.dumps(openai_resp, indent=2)
                    )
                    return JSONResponse(content=openai_resp, status_code=200)
                except Exception as json_err:
                    err_msg = f"JSON Decode Error: {str(json_err)}\n\nResponse Text:\n{resp.text}"
                    print(f"ERROR: {err_msg}", file=sys.stderr)
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/chat/completions",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=err_msg
                    )
                    raise HTTPException(status_code=500, detail=err_msg)
        else:
            req_body["model"] = active_prov["model_name"]
            is_gemini = active_prov["api_type"] == "gemini"
            req_body = ts.sanitize_openai_payload(req_body, is_gemini=is_gemini)
            headers = {"Authorization": f"Bearer {active_prov['api_key']}"}
            
            print(f"\n--- PROXYING OPENAI TO OPENAI [{active_prov['name']}] ---")
            print(f"Target URL: {url}")
            
            if stream_requested:
                req = http_client.build_request("POST", url, json=req_body, headers=headers)
                resp = await http_client.send(req, stream=True)
                
                if resp.status_code >= 400:
                    await resp.aread()
                    resp_text = resp.text
                    print(f"Backend returned error status code: {resp.status_code}\nBody: {resp_text}", file=sys.stderr)
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/chat/completions",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=resp_text
                    )
                    await resp.aclose()
                    return JSONResponse(status_code=resp.status_code, content={"error": {"message": f"Backend returned {resp.status_code}: {resp_text}", "type": "api_error"}})
                
                return StreamingResponse(
                    stream_generator(resp, active_prov, False, True, req_body_str),
                    media_type="text/event-stream"
                )
            else:
                resp = await http_client.post(url, json=req_body, headers=headers)
                try:
                    resp_json = resp.json()
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/chat/completions",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=json.dumps(resp_json, indent=2)
                    )
                    return JSONResponse(content=resp_json, status_code=resp.status_code)
                except Exception as json_err:
                    err_msg = f"JSON Decode Error: {str(json_err)}\n\nResponse Text:\n{resp.text}"
                    print(f"ERROR: {err_msg}", file=sys.stderr)
                    db.add_log(
                        provider_name=active_prov["name"],
                        request_method="POST",
                        request_path="/v1/chat/completions",
                        request_body=req_body_str,
                        response_status=resp.status_code,
                        response_body=err_msg
                    )
                    raise HTTPException(status_code=500, detail=err_msg)
                
    except Exception as e:
        err_msg = f"Exception: {str(e)}\n\n{traceback.format_exc()}"
        print(f"ERROR during completions proxy: {err_msg}", file=sys.stderr)
        db.add_log(
            provider_name=active_prov["name"],
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body=req_body_str,
            response_status=500,
            response_body=err_msg
        )
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for the HTML Dashboard portal
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"), html=True), name="static")
