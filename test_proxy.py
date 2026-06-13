import pytest
import json
import time
import database as db
import translator as ts

# --- Unit Tests for Translator ---

def test_anthropic_to_openai_request_translation():
    # Simple message
    anth_req = {
        "model": "claude-3-5-sonnet",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": True
    }
    
    openai_req = ts.anthropic_to_openai_request(anth_req, "gpt-4o")
    
    assert openai_req["model"] == "gpt-4o"
    assert len(openai_req["messages"]) == 2
    assert openai_req["messages"][0]["role"] == "system"
    assert openai_req["messages"][0]["content"] == "You are a helpful assistant."
    assert openai_req["messages"][1]["role"] == "user"
    assert openai_req["messages"][1]["content"] == "Hello!"
    assert openai_req["max_tokens"] == 1024
    assert openai_req["temperature"] == 0.7
    assert openai_req["stream"] is True

def test_anthropic_to_openai_request_with_tools():
    # Message with tool use definitions and tool results
    anth_req = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "Command output text here"
                    }
                ]
            }
        ],
        "tools": [
            {
                "name": "run_command",
                "description": "Runs a command in shell",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        ]
    }
    
    openai_req = ts.anthropic_to_openai_request(anth_req, "gpt-4o")
    
    assert len(openai_req["tools"]) == 1
    assert openai_req["tools"][0]["type"] == "function"
    assert openai_req["tools"][0]["function"]["name"] == "run_command"
    assert openai_req["tools"][0]["function"]["parameters"]["properties"]["command"]["type"] == "string"
    
    # Check messages array has the role tool
    assert len(openai_req["messages"]) == 1
    assert openai_req["messages"][0]["role"] == "tool"
    assert openai_req["messages"][0]["tool_call_id"] == "toolu_123"
    assert openai_req["messages"][0]["content"] == "Command output text here"

def test_openai_to_anthropic_response_translation():
    # OpenAI response with tool calls
    openai_res = {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Calling command now...",
                    "tool_calls": [
                        {
                            "id": "call_999",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": "{\"command\": \"ls\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25
        }
    }
    
    anth_res = ts.openai_to_anthropic_response(openai_res)
    
    assert anth_res["role"] == "assistant"
    assert anth_res["stop_reason"] == "tool_use"
    assert len(anth_res["content"]) == 2
    
    assert anth_res["content"][0]["type"] == "text"
    assert anth_res["content"][0]["text"] == "Calling command now..."
    
    assert anth_res["content"][1]["type"] == "tool_use"
    assert anth_res["content"][1]["id"] == "call_999"
    assert anth_res["content"][1]["name"] == "run_command"
    assert anth_res["content"][1]["input"] == {"command": "ls"}
    
    assert anth_res["usage"]["input_tokens"] == 15
    assert anth_res["usage"]["output_tokens"] == 25

def test_gemini_thought_signature_translation():
    # 1. Test translation from Gemini response (OpenAI format) to Anthropic response
    # This simulates receiving a tool call from Gemini with a signature
    gemini_res = {
        "id": "chatcmpl-gemini",
        "model": "gemini-2.0-flash-thinking",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Thinking...",
                    "tool_calls": [
                        {
                            "id": "call_gemini_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"London\"}"
                            },
                            "extra_content": {
                                "google": {
                                    "thought_signature": "SIG_DATA_ABC_123"
                                }
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20}
    }
    
    anth_res = ts.openai_to_anthropic_response(gemini_res)
    
    # The ID should now be "mangled" with the signature
    expected_id = f"call_gemini_123{ts.SIGNATURE_SEPARATOR}SIG_DATA_ABC_123"
    assert anth_res["content"][1]["id"] == expected_id
    
    # 2. Test translation from Anthropic request (with mangled ID) back to Gemini (OpenAI format)
    # This simulates the next turn where the client sends back history or a tool result
    anth_req = {
        "model": "gemini-2.0-flash-thinking",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Thinking..."},
                    {
                        "type": "tool_use",
                        "id": expected_id,
                        "name": "get_weather",
                        "input": {"location": "London"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": expected_id,
                        "content": "Sunny, 20C"
                    }
                ]
            }
        ]
    }
    
    openai_req = ts.anthropic_to_openai_request(anth_req, "gemini-2.0-flash-thinking")
    
    # Check assistant message has the restored signature in extra_content
    assistant_msg = openai_req["messages"][0]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "call_gemini_123"
    assert assistant_msg["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG_DATA_ABC_123"
    
    # Check tool result message has the cleaned ID (no signature)
    tool_msg = openai_req["messages"][1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_gemini_123"
    assert tool_msg["content"] == "Sunny, 20C"

def test_mistral_sanitization():
    # Simulate an Anthropic request with cache_control and list-based system prompt
    anth_req = {
        "model": "mistral-large-latest",
        "system": [{"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"call_123{ts.SIGNATURE_SEPARATOR}SIG_STUFF",
                        "name": "tool",
                        "input": {}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}
                ]
            }
        ]
    }
    
    # 1. Translate to OpenAI
    openai_req = ts.anthropic_to_openai_request(anth_req, "mistral-large-latest")
    
    # 2. Sanitize for Mistral (is_gemini=False)
    sanitized_mistral = ts.sanitize_openai_payload(openai_req, is_gemini=False)
    
    # System prompt should be a string and clean
    assert sanitized_mistral["messages"][0]["role"] == "system"
    assert sanitized_mistral["messages"][0]["content"] == "System prompt"
    
    # Assistant message should have extra_content STRIPPED
    assert sanitized_mistral["messages"][1]["role"] == "assistant"
    assert "extra_content" not in sanitized_mistral["messages"][1]["tool_calls"][0]
    
    # User message should be clean
    assert sanitized_mistral["messages"][2]["role"] == "user"
    assert "cache_control" not in sanitized_mistral["messages"][2]

    # 3. Sanitize for Gemini (is_gemini=True)
    openai_req_gemini = ts.anthropic_to_openai_request(anth_req, "gemini-model")
    sanitized_gemini = ts.sanitize_openai_payload(openai_req_gemini, is_gemini=True)
    
    # Assistant message should have extra_content PRESERVED
    assert sanitized_gemini["messages"][1]["role"] == "assistant"
    assert "extra_content" in sanitized_gemini["messages"][1]["tool_calls"][0]
    assert sanitized_gemini["messages"][1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG_STUFF"

# --- Unit Tests for DB Limits and Logging ---

def test_database_logging_limits(tmp_path):
    # Override database path for isolated test
    import os
    db_file = os.path.join(tmp_path, "test_proxy.db")
    db.DB_PATH = db_file
    
    db.init_db()
    
    # Ensure starting empty
    db.clear_logs()
    assert len(db.get_logs()) == 0
    
    # Set limit to 3 logs
    db.set_log_limit(3)
    assert db.get_log_limit() == 3
    
    # Insert 5 logs
    for i in range(5):
        db.add_log(
            provider_name=f"Provider {i}",
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body="{}",
            response_status=200,
            response_body=f"Response {i}"
        )
        # Add a tiny sleep to guarantee monotonic timestamps order
        time.sleep(0.01)
        
    logs = db.get_logs()
    # Should be capped at 3
    assert len(logs) == 3
    
    # Should keep the newest ones (2, 3, 4)
    provider_names = [l["provider_name"] for l in logs]
    assert "Provider 4" in provider_names
    assert "Provider 3" in provider_names
    assert "Provider 2" in provider_names
    assert "Provider 0" not in provider_names
    
    # Set limit to -1 (disabled)
    db.set_log_limit(-1)
    db.add_log(
        provider_name="Should Not Log",
        request_method="POST",
        request_path="/v1/chat/completions",
        request_body="{}",
        response_status=200,
        response_body="No-op"
    )
    # Check that database has 0 logs now
    assert len(db.get_logs()) == 0
