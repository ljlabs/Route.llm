"""
LLM Agent Harness Client Demo

Demonstrates how to interact with the model_router using the Anthropic
Messages API format with tool calling. Validates that tool call responses
are structurally correct before attempting execution.

Usage:
    1. Start the router: python -m uvicorn main:app --host 127.0.0.1 --port 8000
    2. Run this script: python client_demo.py
"""

import httpx
import json
import sys

ROUTER_URL = "http://127.0.0.1:8000"
ANTHROPIC_ENDPOINT = f"{ROUTER_URL}/v1/messages"

# Tool definition in Anthropic format
BASH_TOOL = {
    "name": "bash",
    "description": "Execute a bash command in the terminal and return the output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute"
            }
        },
        "required": ["command"]
    }
}

# The prompt that should trigger a tool_use response
USER_PROMPT = "use your bash tool to run the command pwd and tell me the response"


def build_request(prompt: str, model: str = "default") -> dict:
    """Build an Anthropic Messages API request body."""
    return {
        "model": model,
        "max_tokens": 1024,
        "tools": [BASH_TOOL],
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }


def validate_tool_call(response: dict) -> tuple[bool, str]:
    """
    Validate that a response contains a properly structured tool_use block.
    Returns (is_valid, message).
    """
    if "content" not in response:
        return False, "Response missing 'content' field"

    content = response["content"]
    if not isinstance(content, list):
        return False, f"'content' is not a list, got {type(content).__name__}"

    tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

    if not tool_use_blocks:
        # Check if it's a text-only response (tool_use wasn't triggered)
        text_blocks = [b for b in content if b.get("type") == "text"]
        if text_blocks:
            text = text_blocks[0].get("text", "")
            return False, f"LLM responded with text instead of tool_use:\n{text}"
        return False, "No tool_use blocks found in response"

    block = tool_use_blocks[0]

    # Validate required fields
    # Strip the ____ts____ signature separator that Gemini providers embed
    raw_id = block.get("id", "")
    base_id = raw_id.split("____ts____")[0]
    errors = []
    if not base_id:
        errors.append(f"Missing 'id'")
    if block.get("name") != "bash":
        errors.append(f"Wrong tool name (got: {block.get('name')}, expected: bash)")
    if "input" not in block or not isinstance(block["input"], dict):
        errors.append(f"Missing or invalid 'input' dict")
    elif "command" not in block["input"]:
        errors.append("Missing 'command' in input")
    else:
        cmd = block["input"]["command"]
        if not isinstance(cmd, str) or not cmd.strip():
            errors.append(f"Invalid command value: {cmd!r}")
        elif "pwd" not in cmd:
            errors.append(f"Command does not contain 'pwd': {cmd!r}")

    if errors:
        return False, "Tool call validation failed:\n" + "\n".join(f"  - {e}" for e in errors)

    id_note = " (has signature)" if "____ts____" in raw_id else ""
    return True, (
        f"Valid tool_use block:\n"
        f"  id:      {block["id"]}\n"
        f"  name:    {block['name']}\n"
        f"  command: {block['input']['command']!r}"
    )


def main():
    print("=" * 60)
    print("LLM Agent Harness - Tool Calling Validation")
    print("=" * 60)
    print(f"Router:   {ROUTER_URL}")
    print(f"Endpoint: {ANTHROPIC_ENDPOINT}")
    print(f"Prompt:   {USER_PROMPT}")
    print(f"Tool:     bash (terminal executor)")
    print("-" * 60)

    request_body = build_request(USER_PROMPT)
    print("\nRequest payload:")
    print(json.dumps(request_body, indent=2))

    print("\nSending request to router...")

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(ANTHROPIC_ENDPOINT, json=request_body)
            resp.raise_for_status()
    except httpx.ConnectError:
        print("\nERROR: Could not connect to router.")
        print(f"Make sure the server is running: python -m uvicorn main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"\nERROR: HTTP {e.response.status_code}")
        print(e.response.text)
        sys.exit(1)

    print(f"\nResponse status: {resp.status_code}")
    response = resp.json()
    print("Response body:")
    print(json.dumps(response, indent=2))

    print("\n" + "-" * 60)
    print("Validating tool call structure...")
    print("-" * 60)

    is_valid, message = validate_tool_call(response)
    print(message)

    if is_valid:
        print("\nSUCCESS: Router returned a valid tool call.")
        print("The LLM client harness can safely parse and execute this tool_use block.")
        sys.exit(0)
    else:
        print("\nFAILURE: Tool call validation failed.")
        print("Check your provider configuration and model support for tool calling.")
        sys.exit(1)


if __name__ == "__main__":
    main()
