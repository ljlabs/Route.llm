"""
LLM Agent Harness - Streaming Client Demo

Tests the streaming response version of tool calling through the model_router.
Parses SSE events, reassembles tool_use blocks from deltas, and validates
the final tool call structure.

Usage:
    1. Start the router: python -m uvicorn main:app --host 127.0.0.1 --port 8000
    2. Run this script: python client_demo_streaming.py
"""

import httpx
import json
import sys
from typing import Generator

ROUTER_URL = "http://127.0.0.1:8000"
ANTHROPIC_ENDPOINT = f"{ROUTER_URL}/v1/messages"

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

USER_PROMPT = "use your bash tool to run the command pwd and tell me the response"


def build_request(prompt: str, model: str = "default") -> dict:
    """Build an Anthropic Messages API request body with streaming enabled."""
    return {
        "model": model,
        "max_tokens": 1024,
        "stream": True,
        "tools": [BASH_TOOL],
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }


def parse_sse_events(lines: Generator[str, None, None]) -> Generator[tuple[str, dict], None, None]:
    """
    Parse SSE lines into (event_type, data) tuples.
    Handles the event:/data: line pairs and blank-line separators.
    """
    current_event = "message"  # default if no event: line
    current_data = []

    for line in lines:
        line = line.rstrip("\n\r")

        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data.append(line[len("data:"):].strip())
        elif line == "":
            # Blank line = end of this SSE message
            if current_data:
                data_str = "\n".join(current_data)
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    data = {"_raw": data_str}
                yield current_event, data
            current_event = "message"
            current_data = []


def reassemble_tool_calls(events: list[tuple[str, dict]]) -> dict:
    """
    Reassemble the final message from SSE events.
    Tracks content blocks by index and builds the tool_use blocks
    from content_block_start + input_json_delta events.
    """
    message = {
        "id": None,
        "role": "assistant",
        "model": "",
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }

    # Track blocks by index
    blocks = {}  # index -> block dict
    block_args = {}  # index -> accumulated argument string

    for event_type, data in events:
        if event_type == "message_start":
            msg = data.get("message", {})
            message["id"] = msg.get("id")
            message["model"] = msg.get("model", "")
            message["usage"] = msg.get("usage", message["usage"])

        elif event_type == "content_block_start":
            idx = data.get("index", 0)
            cb = data.get("content_block", {})
            blocks[idx] = cb
            if cb.get("type") == "tool_use":
                block_args[idx] = ""

        elif event_type == "content_block_delta":
            idx = data.get("index", 0)
            delta = data.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "input_json_delta" and idx in block_args:
                block_args[idx] += delta.get("partial_json", "")
            elif delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    if idx not in blocks:
                        blocks[idx] = {"type": "text", "text": ""}
                    blocks[idx]["text"] = blocks[idx].get("text", "") + text

        elif event_type == "content_block_stop":
            idx = data.get("index", 0)
            if idx in blocks and blocks[idx].get("type") == "tool_use":
                # Finalize the tool_use block with parsed arguments
                raw_args = block_args.get(idx, "{}")
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {"_raw": raw_args}
                blocks[idx]["input"] = parsed_args
                message["content"].append(blocks[idx])

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            message["stop_reason"] = delta.get("stop_reason")
            usage = data.get("usage", {})
            if usage.get("output_tokens"):
                message["usage"]["output_tokens"] = usage["output_tokens"]

    # Append any text blocks that weren't already added
    for idx in sorted(blocks.keys()):
        if blocks[idx].get("type") == "text":
            text = blocks[idx].get("text", "")
            if text:
                # Check if already added
                already = any(
                    b.get("type") == "text" and b.get("text") == text
                    for b in message["content"]
                )
                if not already:
                    message["content"].insert(0, blocks[idx])

    return message


def validate_tool_call(response: dict) -> tuple[bool, str]:
    """Validate that a reassembled response contains a properly structured tool_use block."""
    content = response.get("content", [])
    if not isinstance(content, list):
        return False, f"'content' is not a list, got {type(content).__name__}"

    tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

    if not tool_use_blocks:
        text_blocks = [b for b in content if b.get("type") == "text"]
        if text_blocks:
            return False, f"LLM responded with text instead of tool_use:\n{text_blocks[0].get('text', '')}"
        return False, "No tool_use blocks found in reassembled response"

    if response.get("stop_reason") != "tool_use":
        return False, f"stop_reason is {response.get('stop_reason')!r}, expected 'tool_use'"

    block = tool_use_blocks[0]

    raw_id = block.get("id", "")
    base_id = raw_id.split("____ts____")[0]
    errors = []
    if not base_id:
        errors.append("Missing 'id'")
    elif not base_id.startswith("toolu_"):
        errors.append(f"Invalid 'id' prefix (got: {base_id!r}, expected 'toolu_...')")
    if block.get("name") != "bash":
        errors.append(f"Wrong tool name (got: {block.get('name')}, expected: bash)")
    if "input" not in block or not isinstance(block["input"], dict):
        errors.append("Missing or invalid 'input' dict")
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
        f"Valid tool_use block (reassembled from stream):\n"
        f"  id:      {base_id}{id_note}\n"
        f"  name:    {block['name']}\n"
        f"  command: {block['input']['command']!r}\n"
        f"  stop_reason: {response['stop_reason']}"
    )


def main():
    print("=" * 60)
    print("LLM Agent Harness - Streaming Tool Call Validation")
    print("=" * 60)
    print(f"Router:   {ROUTER_URL}")
    print(f"Endpoint: {ANTHROPIC_ENDPOINT}")
    print(f"Prompt:   {USER_PROMPT}")
    print(f"Tool:     bash (terminal executor)")
    print(f"Stream:   true")
    print("-" * 60)

    request_body = build_request(USER_PROMPT, model="gemma-4-31b")
    print("\nRequest payload:")
    print(json.dumps(request_body, indent=2))

    print("\nSending streaming request to router...")

    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", ANTHROPIC_ENDPOINT, json=request_body) as resp:
                resp.raise_for_status()
                print(f"Response status: {resp.status_code}")
                print(f"Content-Type: {resp.headers.get('content-type', 'unknown')}")
                print("-" * 60)
                print("SSE events received:")
                print("-" * 60)

                events = []
                for event_type, data in parse_sse_events(resp.iter_lines()):
                    events.append((event_type, data))

                    # Print compact view of each event
                    etype = data.get("type", event_type)
                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        dt = delta.get("type", "")
                        if dt == "text_delta":
                            print(f"  [{event_type}] text: {delta.get('text', '')!r}")
                        elif dt == "input_json_delta":
                            print(f"  [{event_type}] input_json: {delta.get('partial_json', '')!r}")
                        else:
                            print(f"  [{event_type}] {json.dumps(delta)[:120]}")
                    elif event_type == "content_block_start":
                        cb = data.get("content_block", {})
                        print(f"  [{event_type}] type={cb.get('type')} id={cb.get('id', '')[:30]}...")
                    elif event_type == "message_start":
                        msg = data.get("message", {})
                        print(f"  [{event_type}] id={msg.get('id', '')} model={msg.get('model', '')}")
                    elif event_type == "message_delta":
                        d = data.get("delta", {})
                        print(f"  [{event_type}] stop_reason={d.get('stop_reason')}")
                    elif event_type == "message_stop":
                        print(f"  [{event_type}]")
                    else:
                        print(f"  [{event_type}] {json.dumps(data)[:120]}")

    except httpx.ConnectError:
        print("\nERROR: Could not connect to router.")
        print(f"Make sure the server is running: python -m uvicorn main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"\nERROR: HTTP {e.response.status_code}")
        print(e.response.read().decode())
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Reassembling message from stream events...")
    print("=" * 60)

    reassembled = reassemble_tool_calls(events)

    print("\nReassembled message:")
    print(json.dumps(reassembled, indent=2))

    print("\n" + "-" * 60)
    print("Validating tool call structure...")
    print("-" * 60)

    is_valid, message = validate_tool_call(reassembled)
    print(message)

    if is_valid:
        print("\nSUCCESS: Streaming tool call reassembly and validation passed.")
        print("The client harness can safely parse streaming tool_use blocks.")
        sys.exit(0)
    else:
        print("\nFAILURE: Tool call validation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
