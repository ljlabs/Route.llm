"""
Chat API Endpoints

Test endpoint for trying out the proxy with the active provider.
"""

import json
import sys
import traceback
from fastapi import APIRouter, HTTPException, Request
from core.providers.service import get_provider_service
from core.rate_limiter import get_rate_limiter
from infrastructure.http_client import HTTPClient
import database as db
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

provider_service = get_provider_service()


@router.post("/api/chat")
async def test_chat(request: Request):
    """Test chat endpoint to verify provider setup."""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Get active provider
        active_prov = provider_service.get_active_provider()
        if not active_prov:
            raise HTTPException(status_code=400, detail="No active provider configured")
        
        # Build test request (use Anthropic format)
        effective_max_tokens = active_prov.max_tokens or db.get_max_tokens()
        test_req = {
            "model": active_prov.model_name,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": effective_max_tokens,
            "stream": False
        }
        
        # Wrap request to provider format
        wrapped_request = active_prov.wrap_request(test_req)
        
        req_body_str = json.dumps(test_req, indent=2)
        provider_req_str = json.dumps(wrapped_request, indent=2)

        logger.info(f"Test chat request to {active_prov.name}")
        logger.debug(f"Wrapped request: {provider_req_str}")
        
        # Stage 1 — Router received
        request_id = db.start_request_log(
            provider_name=active_prov.name,
            request_method="POST",
            request_path="/api/chat",
            request_body=req_body_str,
        )

        # Get HTTP client
        http_client = HTTPClient()
        await http_client.__aenter__()
        
        # Get rate limiter and apply it
        rate_limiter = get_rate_limiter()
        await rate_limiter.wait()
        
        try:
            # Stage 2 — About to send to provider
            db.add_log_event(request_id, stage="provider_request", body=provider_req_str)

            # Send request
            response = await http_client.post(
                active_prov.endpoint_url,
                json=wrapped_request,
                headers=active_prov.get_headers()
            )
            
            logger.debug(f"Response status: {response.status_code}")

            # Stage 3 — Response received from provider
            db.add_log_event(
                request_id,
                stage="provider_response",
                body=response.text,
                status_code=response.status_code,
            )
            
            # Handle non-200 responses
            if response.status_code != 200:
                error_detail = f"Backend returned {response.status_code}"
                try:
                    error_json = response.json()
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
                    if response.text:
                        error_detail += f": {response.text[:200]}"
                
                # Add human-friendly hints
                if response.status_code == 401:
                    error_detail = f"Unauthorized (401): Check your API Key. {error_detail}"
                elif response.status_code == 404:
                    error_detail = f"Not Found (404): Check your Endpoint URL or Model Name. {error_detail}"
                elif response.status_code == 403:
                    error_detail = f"Forbidden (403): Access denied. {error_detail}"
                elif response.status_code == 429:
                    error_detail = f"Rate Limited (429): Too many requests. {error_detail}"
                
                # Stage 4 — Error response to client
                db.complete_request_log(
                    request_id=request_id,
                    response_status=response.status_code,
                    response_body=response.text,
                )
                
                raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            # Parse response
            try:
                response_json = response.json()
            except Exception as json_err:
                err_msg = f"JSON Decode Error: {str(json_err)}\n\nResponse Text:\n{response.text}"
                logger.error(err_msg)
                db.complete_request_log(
                    request_id=request_id,
                    response_status=response.status_code,
                    response_body=err_msg,
                )
                raise HTTPException(status_code=500, detail=err_msg)
            
            # Unwrap response to Anthropic format
            try:
                anthropic_response = active_prov.unwrap_response(response_json)
                content = anthropic_response.get("content", [])
                if content and isinstance(content, list):
                    text_content = ""
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content += block.get("text", "")
                else:
                    text_content = str(content)
            except (KeyError, IndexError, TypeError) as parse_err:
                err_msg = f"Response Parsing Error: {str(parse_err)}. The provider returned a format we didn't expect."
                logger.error(err_msg)
                db.complete_request_log(
                    request_id=request_id,
                    response_status=response.status_code,
                    response_body=json.dumps(response_json, indent=2),
                )
                raise HTTPException(status_code=500, detail=err_msg)
            
            # Stage 4 — Successful response to client
            db.complete_request_log(
                request_id=request_id,
                response_status=response.status_code,
                response_body=json.dumps(response_json, indent=2),
            )
            
            return {
                "response": text_content,
                "provider": active_prov.name
            }
        
        finally:
            await http_client.close()
    
    except HTTPException:
        raise
    except Exception as e:
        err_msg = f"Exception: {str(e)}\n\n{traceback.format_exc()}"
        logger.error(f"Error during test chat: {err_msg}")
        raise HTTPException(status_code=500, detail=str(e))