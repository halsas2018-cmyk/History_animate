#!/usr/bin/env python3
"""Quick test script for llm_client.py"""

import sys
sys.path.insert(0, '/root/whiteboard_anime/dynamic_history_video_project')

# Test 1: model registry resolution
from llm_client import MODEL_REGISTRY, DEFAULT_MODEL_KEY, LLMClient
client = LLMClient()
spec = client.resolve_model('groq-llama3-70b')
print('Test 1 - model registry resolution:', 'PASS' if spec.key == 'groq-llama3-70b' else 'FAIL')

# Test 2: invalid model key
try:
    client.resolve_model('invalid-key')
    print('Test 2 - invalid model key: FAIL')
except ValueError as e:
    print('Test 2 - invalid model key:', 'PASS' if 'Unknown model' in str(e) else 'FAIL')

# Test 3: missing API key detection
import os
os.environ.pop('GROQ_API_KEY', None)
try:
    client.get_api_key(spec)
    print('Test 3 - missing API key: FAIL')
except RuntimeError as e:
    print('Test 3 - missing API key:', 'PASS' if 'GROQ_API_KEY is not set' in str(e) else 'FAIL')

# Test 4: provider payload construction
from llm_client import build_openai_chat_payload, build_gemini_payload
messages = [{'role': 'user', 'content': 'test'}]
payload = build_openai_chat_payload(messages, 'test-model', 0.5, 100)
print('Test 4a - OpenAI payload:', 'PASS' if payload['model'] == 'test-model' else 'FAIL')

payload = build_gemini_payload(messages, 'gemini-test', 0.5, 100)
print('Test 4b - Gemini payload:', 'PASS' if 'generationConfig' in payload else 'FAIL')

# Test 5: response parsing
from llm_client import parse_openai_chat_response, parse_gemini_response
resp = {'choices': [{'message': {'content': 'Hello'}, 'finish_reason': 'stop'}], 'usage': {}}
content, finish, usage = parse_openai_chat_response(resp)
print('Test 5a - OpenAI parsing:', 'PASS' if content == 'Hello' and finish == 'stop' else 'FAIL')

resp = {'candidates': [{'content': {'parts': [{'text': 'Hello'}]}, 'finishReason': 'STOP'}]}
content, finish, usage = parse_gemini_response(resp)
print('Test 5b - Gemini parsing:', 'PASS' if content == 'Hello' and finish == 'STOP' else 'FAIL')

print('\nBasic tests complete!')