#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from core.providers.translation import anthropic_to_openai_request

TINY_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

anth_req = {
    'model': 'claude-3-5-sonnet',
    'messages': [
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'Let me read that file'},
                {
                    'type': 'tool_use',
                    'id': 'toolu_read_123',
                    'name': 'Read',
                    'input': {'file_path': '/tmp/image.jpg'}
                }
            ]
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': 'toolu_read_123',
                    'content': [
                        {'type': 'text', 'text': 'File contents:'},
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/jpeg',
                                'data': TINY_BASE64
                            }
                        }
                    ]
                }
            ]
        }
    ],
    'max_tokens': 1024
}

openai_req = anthropic_to_openai_request(anth_req, 'gpt-4o')
print('Messages:', len(openai_req['messages']))
for i, msg in enumerate(openai_req['messages']):
    role = msg.get('role')
    content_type = type(msg.get('content')).__name__
    print(f'  {i}: role={role}, content_type={content_type}')
    if isinstance(msg.get('content'), list):
        print(f'     content items: {len(msg["content"])}')
        for j, item in enumerate(msg['content']):
            if isinstance(item, dict):
                print(f'       {j}: type={item.get("type")}')
            else:
                print(f'       {j}: {type(item).__name__}')

# Check tool message
tool_msg = None
for msg in openai_req['messages']:
    if msg.get('role') == 'tool':
        tool_msg = msg
        break

if tool_msg:
    print('\nTool message found:')
    print('  tool_call_id:', tool_msg.get('tool_call_id'))
    print('  content type:', type(tool_msg.get('content')).__name__)
    if isinstance(tool_msg.get('content'), list):
        print('  content items:', len(tool_msg['content']))
        for j, item in enumerate(tool_msg['content']):
            if isinstance(item, dict):
                print(f'    {j}: type={item.get("type")}')
else:
    print('\nNo tool message found!')
