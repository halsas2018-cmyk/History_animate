"""
Unit tests for llm_client.py - tests without making real API calls.
"""

import os
import sys
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from pathlib import Path

# Add project to path
sys.path.insert(0, '/root/whiteboard_anime/dynamic_history_video_project')

from llm_client import (
    LLMClient,
    ModelSpec,
    ProviderName,
    PayloadFormat,
    PROVIDER_CONFIGS,
    MODEL_REGISTRY,
    DEFAULT_MODEL_KEY,
    build_openai_chat_payload,
    build_gemini_payload,
    parse_openai_chat_response,
    parse_gemini_response,
    clean_model_output,
    call_llm,
    list_models,
    model_keys,
    available_models,
    load_env_file,
    LLMResponse,
)


class TestModelRegistry:
    """Tests for model registry resolution."""

    def test_default_model_key_exists(self):
        assert DEFAULT_MODEL_KEY in MODEL_REGISTRY
        spec = MODEL_REGISTRY[DEFAULT_MODEL_KEY]
        assert isinstance(spec, ModelSpec)
        assert spec.key == DEFAULT_MODEL_KEY

    def test_all_models_have_required_fields(self):
        for key, spec in MODEL_REGISTRY.items():
            assert spec.key == key
            assert isinstance(spec.provider, ProviderName)
            assert spec.model_id
            assert spec.key_env
            assert spec.description
            assert spec.max_tokens > 0
            assert spec.timeout_multiplier > 0

    def test_resolve_model_valid_key(self):
        client = LLMClient()
        spec = client.resolve_model("groq-llama3-70b")
        assert spec.key == "groq-llama3-70b"
        assert spec.provider == ProviderName.GROQ

    def test_resolve_model_default(self):
        client = LLMClient()
        spec = client.resolve_model(None)
        assert spec.key == DEFAULT_MODEL_KEY

    def test_resolve_model_invalid_key(self):
        client = LLMClient()
        with pytest.raises(ValueError, match="Unknown model"):
            client.resolve_model("nonexistent-model")

    def test_model_keys_sorted(self):
        keys = model_keys()
        assert keys == sorted(keys)
        assert len(keys) == len(MODEL_REGISTRY)


class TestMissingAPIKey:
    """Tests for missing API key detection."""

    def test_get_api_key_missing(self):
        client = LLMClient()
        spec = MODEL_REGISTRY["groq-llama3-70b"]
        # Ensure env var is not set
        os.environ.pop("GROQ_API_KEY", None)

        with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
            client.get_api_key(spec)

    def test_get_api_key_present(self, monkeypatch):
        client = LLMClient()
        spec = MODEL_REGISTRY["groq-llama3-70b"]
        monkeypatch.setenv("GROQ_API_KEY", "test-key-123")

        key = client.get_api_key(spec)
        assert key == "test-key-123"


class TestPayloadConstruction:
    """Tests for provider payload building."""

    def test_build_openai_chat_payload_basic(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        payload = build_openai_chat_payload(
            messages=messages,
            model_id="test-model",
            temperature=0.7,
            max_tokens=100,
        )
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 100
        assert payload["messages"] == messages
        assert "response_format" not in payload

    def test_build_openai_chat_payload_with_json_mode(self):
        messages = [{"role": "user", "content": "Hello"}]
        payload = build_openai_chat_payload(
            messages=messages,
            model_id="test-model",
            temperature=0.5,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        assert payload["response_format"] == {"type": "json_object"}

    def test_build_gemini_payload_basic(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        payload = build_gemini_payload(
            messages=messages,
            model_id="gemini-test",
            temperature=0.5,
            max_tokens=100,
        )
        assert payload["generationConfig"]["temperature"] == 0.5
        assert payload["generationConfig"]["maxOutputTokens"] == 100
        assert "systemInstruction" in payload
        assert len(payload["contents"]) == 2  # user + assistant (system moved to systemInstruction)

    def test_build_gemini_payload_json_mode(self):
        messages = [{"role": "user", "content": "Hello"}]
        payload = build_gemini_payload(
            messages=messages,
            model_id="gemini-test",
            temperature=0.5,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        assert payload["generationConfig"]["response_mime_type"] == "application/json"


class TestResponseParsing:
    """Tests for response parsing."""

    def test_parse_openai_chat_response(self):
        resp_data = {
            "choices": [{
                "message": {"content": "Hello, world!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        content, finish_reason, usage = parse_openai_chat_response(resp_data)
        assert content == "Hello, world!"
        assert finish_reason == "stop"
        assert usage == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_parse_gemini_response(self):
        resp_data = {
            "candidates": [{
                "content": {"parts": [{"text": "Hello, world!"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }
        content, finish_reason, usage = parse_gemini_response(resp_data)
        assert content == "Hello, world!"
        assert finish_reason == "STOP"
        assert usage == {"promptTokenCount": 10, "candidatesTokenCount": 5}

    def test_parse_gemini_response_empty_candidates(self):
        resp_data = {"candidates": []}
        with pytest.raises(RuntimeError, match="no candidates"):
            parse_gemini_response(resp_data)


class TestContentCleaning:
    """Tests for hidden reasoning token stripping."""

    def test_clean_normal_content(self):
        content = "This is normal text."
        assert clean_model_output(content) == "This is normal text."

    def test_clean_removes_reasoning_before_terminator(self):
        content = "Some reasoning...\n\n```\nthe actual answer"
        assert clean_model_output(content) == "the actual answer"

    def test_clean_removes_think_tag(self):
        content = "Reasoning <think> more reasoning the answer"
        assert clean_model_output(content) == "the answer"

    def test_clean_handles_empty_after_strip(self):
        content = "   \n\n  "
        assert clean_model_output(content) == ""


class TestRetryBehavior:
    """Tests for retry logic using mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        client = LLMClient()
        spec = MODEL_REGISTRY["groq-llama3-70b"]

        # Mock the HTTP client
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 100},
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(client, '_get_client', return_value=mock_client):
            result = await client.call(
                [{"role": "user", "content": "test"}],
                model_key="groq-llama3-70b",
            )

        assert isinstance(result, LLMResponse)
        assert result.text == "Test response"
        assert result.model_key == "groq-llama3-70b"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        client = LLMClient(max_retries=3, base_delay=0.01)  # Fast retries for testing
        spec = MODEL_REGISTRY["groq-llama3-70b"]

        # First two calls fail with connection error, third succeeds
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Success!"}, "finish_reason": "stop"}],
        }

        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            mock_response,
        ]
        mock_client.is_closed = False

        with patch.object(client, '_get_client', return_value=mock_client):
            with patch('llm_client.asyncio.sleep', new_callable=AsyncMock):
                result = await client.call(
                    [{"role": "user", "content": "test"}],
                    model_key="groq-llama3-70b",
                )

        assert result.text == "Success!"
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        client = LLMClient(max_retries=3, base_delay=0.01)
        spec = MODEL_REGISTRY["groq-llama3-70b"]

        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Persistent connection error")
        mock_client.is_closed = False

        with patch.object(client, '_get_client', return_value=mock_client):
            with patch('llm_client.asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match="failed after 3 attempts"):
                    await client.call(
                        [{"role": "user", "content": "test"}],
                        model_key="groq-llama3-70b",
                    )

    @pytest.mark.asyncio
    async def test_rate_limit_wait_without_attempt_cost(self):
        client = LLMClient(max_retries=3, base_delay=0.01, max_rate_limit_waits=2)
        spec = MODEL_REGISTRY["groq-llama3-70b"]

        # First call: rate limit with wait time
        # Second call: success
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Success after rate limit"}, "finish_reason": "stop"}],
        }

        rate_limit_error = RuntimeError(
            "groq API error for model 'llama3-70b-8192': Rate limit exceeded. try again in 1.5s"
        )

        mock_client = AsyncMock()
        mock_client.post.side_effect = [rate_limit_error, mock_response]
        mock_client.is_closed = False

        with patch.object(client, '_get_client', return_value=mock_client):
            with patch('llm_client.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                result = await client.call(
                    [{"role": "user", "content": "test"}],
                    model_key="groq-llama3-70b",
                )

        assert result.text == "Success after rate limit"
        # Should have slept for rate limit wait (1.5 + 2 = 3.5s)
        mock_sleep.assert_called()
        # The rate limit wait should not count as a retry attempt
        assert mock_client.post.call_count == 2


class TestEmptyResponseHandling:
    """Tests for empty response handling."""

    @pytest.mark.asyncio
    async def test_empty_content_retries(self):
        client = LLMClient(max_retries=2, base_delay=0.01)
        spec = MODEL_REGISTRY["groq-llama3-70b"]

        # First response: empty content
        mock_response_empty = Mock()
        mock_response_empty.status_code = 200
        mock_response_empty.json.return_value = {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
        }

        # Second response: valid content
        mock_response_valid = Mock()
        mock_response_valid.status_code = 200
        mock_response_valid.json.return_value = {
            "choices": [{"message": {"content": "Valid response"}, "finish_reason": "stop"}],
        }

        mock_client = AsyncMock()
        mock_client.post.side_effect = [mock_response_empty, mock_response_valid]
        mock_client.is_closed = False

        with patch.object(client, '_get_client', return_value=mock_client):
            with patch('llm_client.asyncio.sleep', new_callable=AsyncMock):
                result = await client.call(
                    [{"role": "user", "content": "test"}],
                    model_key="groq-llama3-70b",
                )

        assert result.text == "Valid response"

    @pytest.mark.asyncio
    async def test_finish_reason_length_retries(self):
        client = LLMClient(max_retries=2, base_delay=0.01)
        spec = MODEL_REGISTRY["groq-llama3-70b"]

        mock_response_length = Mock()
        mock_response_length.status_code = 200
        mock_response_length.json.return_value = {
            "choices": [{"message": {"content": "Partial..."}, "finish_reason": "length"}],
        }

        mock_response_valid = Mock()
        mock_response_valid.status_code = 200
        mock_response_valid.json.return_value = {
            "choices": [{"message": {"content": "Complete response"}, "finish_reason": "stop"}],
        }

        mock_client = AsyncMock()
        mock_client.post.side_effect = [mock_response_length, mock_response_valid]
        mock_client.is_closed = False

        with patch.object(client, '_get_client', return_value=mock_client):
            with patch('llm_client.asyncio.sleep', new_callable=AsyncMock):
                result = await client.call(
                    [{"role": "user", "content": "test"}],
                    model_key="groq-llama3-70b",
                )

        assert result.text == "Complete response"


class TestStructuredResponse:
    """Tests for structured response parameter handling."""

    @pytest.mark.asyncio
    async def test_response_format_json_mode(self):
        client = LLMClient()
        spec = MODEL_REGISTRY["groq-llama3-70b"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"key": "value"}'}, "finish_reason": "stop"}],
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(client, '_get_client', return_value=mock_client):
            result = await client.call(
                [{"role": "user", "content": "test"}],
                model_key="groq-llama3-70b",
                response_format={"type": "json_object"},
            )

        # Verify the payload included response_format
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload.get("response_format") == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_gemini_rejects_json_mode_on_unsupported_model(self):
        client = LLMClient()
        spec = MODEL_REGISTRY["gemini-31-flash-lite"]

        with pytest.raises(ValueError, match="does not support structured JSON output"):
            await client.call(
                [{"role": "user", "content": "test"}],
                model_key="gemini-31-flash-lite",
                response_format={"type": "json_object"},
            )

    def test_sync_call_wrapper(self):
        """Test the sync wrapper works."""
        client = LLMClient()

        with patch.object(client, 'call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = LLMResponse(
                text="Sync response",
                model_key="test-model",
                finish_reason="stop",
            )

            result = client.call_sync(
                [{"role": "user", "content": "test"}],
                model_key="test-model",
            )

            assert result.text == "Sync response"
            mock_call.assert_called_once()


class TestConvenienceFunctions:
    """Tests for backward-compat convenience functions."""

    def test_list_models(self):
        models = list_models()
        assert isinstance(models, list)
        assert len(models) == len(MODEL_REGISTRY)
        for m in models:
            assert "key" in m
            assert "provider" in m
            assert "model_id" in m
            assert "description" in m

    def test_available_models_filters_by_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        os.environ.pop("NVIDIA_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)

        available = available_models()
        assert "groq-llama3-70b" in available
        assert "nvidia-llama33" not in available
        assert "gemini-31-flash-lite" not in available

    def test_load_env_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=test-value\n# Comment\nANOTHER_KEY=another-value\n")

        # Ensure keys don't exist
        os.environ.pop("TEST_KEY", None)
        os.environ.pop("ANOTHER_KEY", None)

        load_env_file(env_file)

        assert os.environ.get("TEST_KEY") == "test-value"
        assert os.environ.get("ANOTHER_KEY") == "another-value"


class TestSyncCallLLM:
    """Tests for the backward-compat call_llm function."""

    def test_call_llm_returns_text_only(self):
        with patch('llm_client.LLMClient.call_sync') as mock_call_sync:
            mock_call_sync.return_value = LLMResponse(
                text="Test response",
                model_key="test-model",
                finish_reason="stop",
            )

            result = call_llm(
                [{"role": "user", "content": "test"}],
                model_key="test-model",
                temperature=0.7,
                max_tokens=200,
            )

            assert result == "Test response"
            mock_call_sync.assert_called_once_with(
                [{"role": "user", "content": "test"}],
                model_key="test-model",
                temperature=0.7,
                max_tokens=200,
                response_format=None,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])