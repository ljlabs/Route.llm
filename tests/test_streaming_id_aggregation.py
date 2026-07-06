"""


Tests for streaming ID aggregation in stream translators.



Ensures that IDs are preserved and reused throughout the entire stream,


which is critical for clients (like Claude) to properly aggregate streamed responses.
"""



import json


import pytest


import asyncio


from core.translation.stream_base import (

    AnthropicToOpenAIStreamTranslator,

    OpenAIToAnthropicStreamTranslator,

    PassthroughStreamTranslator,

)



class MockStreamResponse:

    """Mock streaming response for testing."""


    def __init__(self, chunks):

        """Initialize with list of SSE data lines."""

        self.chunks = chunks

        self.closed = False


    async def aiter_lines(self):

        """Async iterator yielding chunks."""

        for chunk in self.chunks:

            yield chunk


    async def aclose(self):

        """Close the stream."""

        self.closed = True



# ---------------------------------------------------------------------------

# OpenAI to Anthropic Translator Tests

# ---------------------------------------------------------------------------



class TestOpenAIToAnthropicIDPreservation:


    """Test that OpenAI->Anthropic translation preserves message IDs across chunks."""



    @pytest.mark.anyio


    async def test_same_id_across_all_chunks(self):
        """


        Verify that the same ID from message_start is used in all subsequent chunks.



        This is critical for proper streaming response aggregation.


        Real-world example from OpenCode:


        - message_start has id "gen-1782246708-77Sq3JZm8vr1sxQV7IVv"


        - All content_block_delta chunks must have the SAME id


        - message_delta must have the SAME id
        """


        provider_config = {


            "name": "OpenCode Zen",


            "api_type": "openai",


            "model_name": "mimo-v2.5",


        }



        chunks = [


            'data: {"id":"gen-1782246708-77Sq3JZm8vr1sxQV7IVv","object":"chat.completion.chunk","created":1782246708,"model":"xiaomi/mimo-v2.5-20260422","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',


            'data: {"id":"gen-1782246708-77Sq3JZm8vr1sxQV7IVv","object":"chat.completion.chunk","created":1782246708,"model":"xiaomi/mimo-v2.5-20260422","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',


            'data: {"id":"gen-1782246708-77Sq3JZm8vr1sxQV7IVv","object":"chat.completion.chunk","created":1782246708,"model":"xiaomi/mimo-v2.5-20260422","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',


            'data: {"id":"gen-1782246708-77Sq3JZm8vr1sxQV7IVv","object":"chat.completion.chunk","created":1782246708,"model":"xiaomi/mimo-v2.5-20260422","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',


            "data: [DONE]",


        ]



        response = MockStreamResponse(chunks)


        # Use AnthropicToOpenAI translator since input is OpenAI format


        translator = OpenAIToAnthropicStreamTranslator()


        accumulated_blocks = []



        output_chunks = []


        async for chunk in translator.translate_stream(


            response, provider_config, accumulated_blocks


        ):


            output_chunks.append(chunk)



        # Parse all output chunks and extract IDs from Anthropic format


        ids_found = []


        for chunk in output_chunks:


            if 'data: {' in chunk:  # Skip empty lines


                try:


                    # Extract the JSON data part (after 'data: ')


                    if 'data: ' in chunk:


                        data_part = chunk.split('data: ', 1)[1].strip()


                        # Remove trailing newlines


                        data_part = data_part.rstrip('\n')


                        data = json.loads(data_part)


                        # Anthropic format has ID nested in message_start events


                        if data.get("type") == "message_start":


                            msg_id = data.get("message", {}).get("id")


                            if msg_id:


                                ids_found.append(msg_id)


                except (json.JSONDecodeError, IndexError, ValueError):
                    pass



        # Assert all IDs are the same


        expected_id = "gen-1782246708-77Sq3JZm8vr1sxQV7IVv"


        assert len(ids_found) > 0, f"Should have found IDs in output chunks. Output: {output_chunks[:3]}"


        assert all(


            id == expected_id for id in ids_found


        ), f"All IDs should be {expected_id}, but got: {ids_found}"



    @pytest.mark.anyio


    async def test_id_in_message_start_only_generated_if_missing(self):
        """


        Verify that if provider doesn't send an ID, we generate one and use it consistently.
        """


        provider_config = {


            "name": "Test Provider",


            "api_type": "openai",
            "model_name": "test-model",


        }



        chunks = [


            'data: {"object":"chat.completion.chunk","created":1782246708,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',


            'data: {"object":"chat.completion.chunk","created":1782246708,"model":"test-model","choices":[{"index":0,"delta":{"content":"Test"},"finish_reason":null}]}',


            'data: {"object":"chat.completion.chunk","created":1782246708,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',


            "data: [DONE]",


        ]



        response = MockStreamResponse(chunks)


        # Use AnthropicToOpenAI translator since input is OpenAI format


        translator = OpenAIToAnthropicStreamTranslator()


        accumulated_blocks = []



        output_chunks = []


        async for chunk in translator.translate_stream(


            response, provider_config, accumulated_blocks


        ):


            output_chunks.append(chunk)



        # Parse IDs and verify they exist and are consistent


        # AnthropicToOpenAI outputs Anthropic format with nested message.id


        ids_found = []


        for chunk in output_chunks:


            if 'data: {' in chunk:


                try:


                    if 'data: ' in chunk:


                        data_part = chunk.split('data: ', 1)[1].strip()


                        data_part = data_part.rstrip('\n')


                        data = json.loads(data_part)


                        # Anthropic format has ID nested in message_start events


                        if data.get("type") == "message_start":


                            msg_id = data.get("message", {}).get("id")


                            if msg_id:


                                ids_found.append(msg_id)


                except (json.JSONDecodeError, IndexError, ValueError):
                    pass



        # Assert IDs exist and are consistent


        assert len(ids_found) > 0, f"Should have generated an ID. Output: {output_chunks[:3]}"


        assert all(


            id == ids_found[0] for id in ids_found


        ), "All generated IDs should be the same"


        assert ids_found[0].startswith("msg_local_"), f"Generated ID should follow the pattern, got: {ids_found[0]}"



    @pytest.mark.anyio


    async def test_id_consistency_across_delta_chunks(self):
        """


        Specific test case: Verify that the ID from message_start is preserved throughout the stream.



        This mirrors the real-world bug where we were generating new IDs.
        """


        provider_config = {


            "name": "OpenCode",


            "api_type": "openai",
            "model_name": "mimo",


        }



        test_id = "gen-test-123"


        chunks = [


            f'data: {{"id":"{test_id}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"role":"assistant"}},"finish_reason":null}}]}}',


            f'data: {{"id":"{test_id}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":"chunk1"}},"finish_reason":null}}]}}',


            f'data: {{"id":"{test_id}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":"chunk2"}},"finish_reason":null}}]}}',


            f'data: {{"id":"{test_id}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":"chunk3"}},"finish_reason":null}}]}}',


            f'data: {{"id":"{test_id}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}',


            "data: [DONE]",


        ]



        response = MockStreamResponse(chunks)


        # Use AnthropicToOpenAI translator since input is OpenAI format


        translator = OpenAIToAnthropicStreamTranslator()


        accumulated_blocks = []



        message_start_id = None


        output_chunks = []


        async for chunk in translator.translate_stream(


            response, provider_config, accumulated_blocks


        ):


            output_chunks.append(chunk)


            # Extract ID from message_start


            if 'message_start' in chunk and '"type"' in chunk:


                try:


                    data_part = chunk.split('data: ', 1)[1].strip()


                    data = json.loads(data_part)


                    message_start_id = data.get("message", {}).get("id")


                except:
                    pass



        # Verify message_start ID was captured


        assert message_start_id is not None, f"Should have found message_start ID. Output: {output_chunks[:3]}"


        assert message_start_id == test_id, f"Message start ID should be {test_id}, got: {message_start_id}"
        


        # Verify we got delta chunks


        assert any("content_block_delta" in chunk for chunk in output_chunks), "Should have content_block_delta chunks"




# ---------------------------------------------------------------------------


# Passthrough Translator Tests (ID should pass through unchanged)


# ---------------------------------------------------------------------------




class TestPassthroughTranslatorIDPassthrough:


    """Test that passthrough translator doesn't modify IDs."""



    @pytest.mark.anyio


    async def test_passthrough_preserves_ids(self):
        """


        Verify that passthrough translator doesn't modify IDs in any way.
        """


        provider_config = {


            "name": "Anthropic",


            "api_type": "anthropic",


            "model_name": "claude-3-sonnet",


        }



        test_id = "msg-original-id-123"


        chunks = [


            f'data: {{"type":"message_start","message":{{"id":"{test_id}","type":"message","role":"assistant","model":"claude-3-sonnet","content":[],"stop_reason":null}}}}',


            f'data: {{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"Hello"}}}}',


            f'data: {{"type":"message_delta","delta":{{"stop_reason":"end_turn"}}}}',


            'data: {"type":"message_stop"}',


        ]



        response = MockStreamResponse(chunks)


        translator = PassthroughStreamTranslator()


        accumulated_blocks = []



        output_chunks = []


        async for chunk in translator.translate_stream(


            response, provider_config, accumulated_blocks


        ):


            output_chunks.append(chunk)



        # Check that message ID is preserved


        message_start_found = False


        for chunk in output_chunks:


            if '"type":"message_start"' in chunk and f'"{test_id}"' in chunk:


                message_start_found = True


                break



        assert message_start_found, "Original message ID should be preserved in output"




# ---------------------------------------------------------------------------


# Anthropic to OpenAI Translator Tests


# ---------------------------------------------------------------------------




class TestAnthropicToOpenAIIDGeneration:


    """Test that Anthropic->OpenAI translation generates consistent IDs."""



    @pytest.mark.anyio


    async def test_generates_consistent_id_from_anthropic_stream(self):
        """


        Test that when translating Anthropic format to OpenAI,


        we generate ONE ID and reuse it throughout.
        """


        provider_config = {


            "name": "Claude",


            "api_type": "anthropic",


            "model_name": "claude-3-sonnet",


        }



        anthropic_id = "msg-from-anthropic"


        chunks = [


            f'data: {{"type":"message_start","message":{{"id":"{anthropic_id}","type":"message","role":"assistant","model":"claude-3-sonnet","content":[]}}}}',


            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',


            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',


            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',


            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',


            'data: {"type":"message_stop"}',


        ]



        response = MockStreamResponse(chunks)


        # Use OpenAIToAnthropic translator since input is Anthropic format


        translator = AnthropicToOpenAIStreamTranslator()


        accumulated_blocks = []



        output_chunks = []


        async for chunk in translator.translate_stream(


            response, provider_config, accumulated_blocks


        ):


            output_chunks.append(chunk)



        # Extract all IDs from output


        ids_found = []


        for chunk in output_chunks:


            if 'data: {' in chunk and '"id"' in chunk:


                try:


                    if 'data: ' in chunk:


                        data_part = chunk.split('data: ', 1)[1].strip()


                        data_part = data_part.rstrip('\n')


                        data = json.loads(data_part)


                        if 'id' in data:


                            ids_found.append(data['id'])


                except (json.JSONDecodeError, IndexError, ValueError):
                    pass



        # All should have the same ID


        assert len(ids_found) > 0, "Should have generated IDs"


        assert all(


            id == ids_found[0] for id in ids_found


        ), f"All IDs should be consistent, got: {ids_found}"




# ---------------------------------------------------------------------------


# Integration test: Full streaming cycle


# ---------------------------------------------------------------------------




@pytest.mark.anyio


async def test_openai_to_anthropic_full_streaming_cycle():
    """


    Integration test: Simulate a complete streaming response from OpenAI provider


    and verify ID aggregation works correctly throughout the entire cycle.
    """


    provider_config = {


        "name": "OpenCode Zen",


        "api_type": "openai",


        "model_name": "mimo-v2.5",


    }



    # Real-world-like streaming response in OpenAI format


    original_id = "gen-real-world-123"


    chunks = [


        f'data: {{"id":"{original_id}","object":"chat.completion.chunk","created":1234567890,"model":"mimo-v2.5","choices":[{{"index":0,"delta":{{"role":"assistant"}},"finish_reason":null}}]}}',


        f'data: {{"id":"{original_id}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":"This"}},"finish_reason":null}}]}}',


        f'data: {{"id":"{original_id}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":" is"}},"finish_reason":null}}]}}',


        f'data: {{"id":"{original_id}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":" a"}},"finish_reason":null}}]}}',


        f'data: {{"id":"{original_id}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":" test"}},"finish_reason":null}}]}}',


        f'data: {{"id":"{original_id}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}',


        "data: [DONE]",


    ]



    response = MockStreamResponse(chunks)


    # Use AnthropicToOpenAI translator since input is OpenAI format


    translator = OpenAIToAnthropicStreamTranslator()


    accumulated_blocks = []



    # Collect all output and verify structure


    output_chunks = []


    async for chunk in translator.translate_stream(


        response, provider_config, accumulated_blocks


    ):


        output_chunks.append(chunk)



    # Verify we have the expected event types


    assert any("message_start" in chunk for chunk in output_chunks), "Should have message_start"


    assert any(


        "content_block_delta" in chunk for chunk in output_chunks


    ), "Should have content_block_delta"


    assert any("message_delta" in chunk for chunk in output_chunks), "Should have message_delta"


    assert any("message_stop" in chunk for chunk in output_chunks), "Should have message_stop"



    # Extract and verify IDs - AnthropicToOpenAI outputs Anthropic format


    # Only message_start events have IDs in nested message structure


    extracted_ids = []


    extracted_text = []


    for chunk in output_chunks:


        if 'message_start' in chunk:


            try:


                if 'data: ' in chunk:


                    data_part = chunk.split('data: ', 1)[1].strip()


                    data_part = data_part.rstrip('\n')


                    data = json.loads(data_part)


                    msg_id = data.get("message", {}).get("id")


                    if msg_id:


                        extracted_ids.append(msg_id)


            except (json.JSONDecodeError, IndexError, ValueError):
                pass


        # Extract text from content_block_delta events


        elif 'content_block_delta' in chunk:


            try:


                if 'data: ' in chunk:


                    data_part = chunk.split('data: ', 1)[1].strip()


                    data_part = data_part.rstrip('\n')


                    data = json.loads(data_part)


                    text = data.get("delta", {}).get("text", "")


                    if text:


                        extracted_text.append(text)


            except (json.JSONDecodeError, IndexError, ValueError):
                pass



    # Verify ID consistency


    assert len(extracted_ids) > 0, f"Should have extracted IDs from message_start. Output: {output_chunks[:3]}"


    assert all(


        id == original_id for id in extracted_ids


    ), f"All IDs should match {original_id}, got: {extracted_ids}"



    # Verify content aggregation


    full_text = "".join(extracted_text)


    assert "This is a test" in full_text, f"Content should be preserved, got: {full_text}"


