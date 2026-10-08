"""
llm_client.py — Provider-agnostic LLM transport for the history-video pipeline.

Owns the MODEL_REGISTRY and a single `call_llm()` that routes to multiple providers.
Uses httpx for async HTTP with sync wrapper. Supports structured JSON output mode
on providers that support it.
"""

import asyncio
import os
import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Provider Configuration
# ═══════════════════════════════════════════════════════════════════════════

class ProviderName(str, Enum):
    GROQ = "groq"
    NVIDIA = "nvidia"
    GEMINI = "gemini"


class PayloadFormat(str, Enum):
    OPENAI_CHAT = "openai_chat"
    GEMINI = "gemini"


@dataclass(frozen=True)
class ProviderConfig:
    name: ProviderName
    base_url: str
    auth_header_template: str  # e.g., "Authorization: Bearer {key}"
    payload_format: PayloadFormat
    supports_json_mode: bool = False


PROVIDER_CONFIGS: dict[ProviderName, ProviderConfig] = {
    ProviderName.GROQ: ProviderConfig(
        name=ProviderName.GROQ,
        base_url="https://api.groq.com/openai/v1/chat/completions",
        auth_header_template="Authorization: Bearer {key}",
        payload_format=PayloadFormat.OPENAI_CHAT,
        supports_json_mode=True,
    ),
    ProviderName.NVIDIA: ProviderConfig(
        name=ProviderName.NVIDIA,
        base_url="https://integrate.api.nvidia.com/v1/chat/completions",
        auth_header_template="Authorization: Bearer {key}",
        payload_format=PayloadFormat.OPENAI_CHAT,
        supports_json_mode=True,
    ),
    ProviderName.GEMINI: ProviderConfig(
        name=ProviderName.GEMINI,
        base_url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        auth_header_template="x-goog-api-key: {key}",
        payload_format=PayloadFormat.GEMINI,
        supports_json_mode=False,  # Gemini uses response_mime_type instead
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: ProviderName
    model_id: str  # Value sent to API in `model` field
    key_env: str  # Environment variable name for API key
    description: str
    max_tokens: int = 8192
    timeout_multiplier: float = 1.0
    supports_json_mode: bool = True  # Can be overridden per-model


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- Groq (free tier, 30 req/min) ---
    "groq-llama3-70b": ModelSpec(
        key="groq-llama3-70b",
        provider=ProviderName.GROQ,
        model_id="llama3-70b-8192",
        key_env="GROQ_API_KEY",
        description="Meta Llama 3 70B on Groq — fast, general purpose",
    ),
    "groq-gpt-oss-120b": ModelSpec(
        key="groq-gpt-oss-120b",
        provider=ProviderName.GROQ,
        model_id="openai/gpt-oss-120b",
        key_env="GROQ_API_KEY",
        description="Groq GPT-OSS 120B — strongest reasoning on Groq",
        timeout_multiplier=3.0,
    ),
    "groq-gpt-oss-20b": ModelSpec(
        key="groq-gpt-oss-20b",
        provider=ProviderName.GROQ,
        model_id="openai/gpt-oss-20b",
        key_env="GROQ_API_KEY",
        description="Groq GPT-OSS 20B — fast, cheap sibling",
    ),
    # --- NVIDIA NIM (free tier; needs NVIDIA_API_KEY from build.nvidia.com) ---
    "nvidia-llama33": ModelSpec(
        key="nvidia-llama33",
        provider=ProviderName.NVIDIA,
        model_id="meta/llama-3.3-70b-instruct",
        key_env="NVIDIA_API_KEY",
        description="Meta Llama 3.3 70B on NVIDIA — fast, general purpose",
    ),
    "nvidia-gpt-oss-120b": ModelSpec(
        key="nvidia-gpt-oss-120b",
        provider=ProviderName.NVIDIA,
        model_id="openai/gpt-oss-120b",
        key_env="NVIDIA_API_KEY",
        description="OpenAI GPT-OSS 120B on NVIDIA — open-weight, strong reasoning",
        timeout_multiplier=3.0,
    ),
    "nvidia-llama-3-1-70b": ModelSpec(
        key="nvidia-llama-3-1-70b",
        provider=ProviderName.NVIDIA,
        model_id="meta/llama-3.1-70b-instruct",
        key_env="NVIDIA_API_KEY",
        description="Meta Llama 3.1 70B Instruct on NVIDIA — solid general purpose",
    ),
    "nvidia-nemotron-ultra": ModelSpec(
        key="nvidia-nemotron-ultra",
        provider=ProviderName.NVIDIA,
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        key_env="NVIDIA_API_KEY",
        description="NVIDIA Nemotron 3 Ultra 550B — biggest reasoning model",
        timeout_multiplier=3.0,
    ),
    # --- Google Gemini (free tier; needs GEMINI_API_KEY from makersuite.google.com) ---
    "gemini-31-flash-lite": ModelSpec(
        key="gemini-31-flash-lite",
        provider=ProviderName.GEMINI,
        model_id="gemini-3.1-flash-lite",
        key_env="GEMINI_API_KEY",
        description="Google Gemini 3.1 Flash Lite — fast, efficient, optimized for cost",
        supports_json_mode=False,
    ),
    "gemini-15-flash": ModelSpec(
        key="gemini-15-flash",
        provider=ProviderName.GEMINI,
        model_id="gemini-1.5-flash",
        key_env="GEMINI_API_KEY",
        description="Google Gemini 1.5 Flash — legacy model version",
        supports_json_mode=False,
    ),
}

DEFAULT_MODEL_KEY = "gemini-31-flash-lite"


# ═══════════════════════════════════════════════════════════════════════════
# Response Types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LLMResponse:
    text: str
    model_key: str
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    raw_response: Optional[dict] = None


# ═══════════════════════════════════════════════════════════════════════════
# Payload Builders
# ═══════════════════════════════════════════════════════════════════════════

def build_openai_chat_payload(
    messages: list[dict],
    model_id: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[dict] = None,
) -> dict:
    """Build OpenAI-compatible chat completions payload."""
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    return payload


def build_gemini_payload(
    messages: list[dict],
    model_id: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[dict] = None,
) -> dict:
    """Build Gemini generateContent payload."""
    system_messages = [m for m in messages if m.get("role") == "system"]
    conversation_messages = [m for m in messages if m.get("role") != "system"]

    contents = []
    for m in conversation_messages:
        role = "model" if m.get("role") == "assistant" else "user"
        contents.append({
            "role": role,
            "parts": [{"text": m.get("content", "")}],
        })

    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    # Gemini uses response_mime_type for JSON mode
    if response_format and response_format.get("type") == "json_object":
        generation_config["response_mime_type"] = "application/json"

    payload = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    if system_messages:
        payload["systemInstruction"] = {
            "parts": [{"text": system_messages[0].get("content", "")}]
        }

    return payload


PAYLOAD_BUILDERS = {
    PayloadFormat.OPENAI_CHAT: build_openai_chat_payload,
    PayloadFormat.GEMINI: build_gemini_payload,
}


# ═══════════════════════════════════════════════════════════════════════════
# Response Parsers
# ═══════════════════════════════════════════════════════════════════════════

def parse_openai_chat_response(resp_data: dict) -> tuple[str, Optional[str], Optional[dict]]:
    """Parse OpenAI-compatible chat completions response."""
    choice = resp_data["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice.get("finish_reason")
    usage = resp_data.get("usage")
    return content, finish_reason, usage


def parse_gemini_response(resp_data: dict) -> tuple[str, Optional[str], Optional[dict]]:
    """Parse Gemini generateContent response."""
    candidates = resp_data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {resp_data}")

    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    content = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict)
    )
    finish_reason = candidate.get("finishReason")
    usage = resp_data.get("usageMetadata")
    return content, finish_reason, usage


RESPONSE_PARSERS = {
    PayloadFormat.OPENAI_CHAT: parse_openai_chat_response,
    PayloadFormat.GEMINI: parse_gemini_response,
}


# ═══════════════════════════════════════════════════════════════════════════
# Content Post-Processing
# ═══════════════════════════════════════════════════════════════════════════

def clean_model_output(content: str) -> str:
    """Strip hidden reasoning tokens that some models emit."""
    # Remove <think> blocks and leading reasoning
    content = content.strip()
    # Strip content before </think> if present
    if "</think>" in content:
        content = content.split("</think>", 1)[-1]
    return content.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Core Client
# ═══════════════════════════════════════════════════════════════════════════

class LLMClient:
    """Async-capable LLM client with retry logic and structured output support."""

    def __init__(
        self,
        model_key: Optional[str] = None,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
        base_delay: float = 5.0,
        max_rate_limit_waits: int = 3,
    ):
        self.default_model_key = model_key or DEFAULT_MODEL_KEY
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_rate_limit_waits = max_rate_limit_waits
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def resolve_model(self, model_key: Optional[str] = None) -> ModelSpec:
        key = model_key or self.default_model_key
        if key not in MODEL_REGISTRY:
            valid = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model '{key}'. Valid model keys: {valid}")
        return MODEL_REGISTRY[key]

    def get_api_key(self, spec: ModelSpec) -> str:
        val = os.environ.get(spec.key_env, "")
        if not val:
            provider_url = {
                ProviderName.GROQ: "https://console.groq.com",
                ProviderName.NVIDIA: "https://build.nvidia.com",
                ProviderName.GEMINI: "https://makersuite.google.com/app/apikey",
            }.get(spec.provider, "")
            raise RuntimeError(
                f"{spec.key_env} is not set — the chosen model '{spec.key}' needs a "
                f"{spec.provider.value.upper()} key. Get one at: {provider_url}\n"
                f"Then export: {spec.key_env}=\"<your-key>\""
            )
        return val

    def _build_payload(
        self,
        spec: ModelSpec,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict],
    ) -> dict:
        provider_config = PROVIDER_CONFIGS[spec.provider]
        builder = PAYLOAD_BUILDERS[provider_config.payload_format]

        # For Gemini, check if this model supports JSON mode
        if provider_config.payload_format == PayloadFormat.GEMINI:
            if response_format and response_format.get("type") == "json_object":
                if not spec.supports_json_mode:
                    raise ValueError(
                        f"Model '{spec.key}' does not support structured JSON output. "
                        f"Use a model with supports_json_mode=True or omit response_format."
                    )

        return builder(
            messages=messages,
            model_id=spec.model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def _get_endpoint(self, spec: ModelSpec) -> str:
        provider_config = PROVIDER_CONFIGS[spec.provider]
        if spec.provider == ProviderName.GEMINI:
            return provider_config.base_url.format(model=spec.model_id)
        return provider_config.base_url

    def _get_headers(self, spec: ModelSpec, api_key: str) -> dict:
        provider_config = PROVIDER_CONFIGS[spec.provider]
        auth_header = provider_config.auth_header_template.format(key=api_key)
        # Parse "Header-Name: value" format
        if ": " in auth_header:
            header_name, header_value = auth_header.split(": ", 1)
            return {
                "Content-Type": "application/json",
                header_name: header_value,
            }
        return {"Content-Type": "application/json"}

    def _parse_response(self, spec: ModelSpec, resp_data: dict) -> tuple[str, Optional[str], Optional[dict]]:
        provider_config = PROVIDER_CONFIGS[spec.provider]
        parser = RESPONSE_PARSERS[provider_config.payload_format]
        return parser(resp_data)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        if isinstance(error, httpx.TimeoutException):
            return True
        if isinstance(error, httpx.ConnectError):
            return True
        if isinstance(error, RuntimeError):
            msg = str(error).lower()
            # Don't retry hard API errors
            if "api error" in msg:
                return False
            # Retry on empty content, length limit, rate limit
            if any(x in msg for x in ["empty content", "finish_reason=length", "rate limit"]):
                return True
        if isinstance(error, (json.JSONDecodeError, KeyError)):
            return True
        return False

    def _extract_rate_limit_wait(self, error: Exception) -> Optional[float]:
        """Extract wait time from rate limit error message."""
        import re
        msg = str(error)
        m = re.search(r"try again in ([\d.]+)s", msg)
        if m and "rate limit" in msg.lower():
            return float(m.group(1)) + 2.0  # Add 2s buffer
        return None

    async def call(
        self,
        messages: list[dict],
        *,
        model_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Main async call. Returns parsed LLMResponse."""
        spec = self.resolve_model(model_key)
        api_key = self.get_api_key(spec)

        # Use model's max_tokens if not specified
        if max_tokens is None:
            max_tokens = spec.max_tokens

        payload = self._build_payload(spec, messages, temperature, max_tokens, response_format)
        endpoint = self._get_endpoint(spec)
        headers = self._get_headers(spec, api_key)

        # Calculate timeout with multiplier for large models
        timeout_multiplier = spec.timeout_multiplier
        request_timeout = max(60.0, max_tokens * 0.3 * timeout_multiplier + 60)

        client = await self._get_client()
        rate_limit_waits = 0
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"LLM call attempt {attempt}/{self.max_retries} for model {spec.key}")
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=request_timeout,
                )

                if response.status_code != 200:
                    # Include response text for debugging
                    error_text = response.text[:500] if response.text else "(empty response)"
                    raise RuntimeError(
                        f"{spec.provider.value} API error for model '{spec.model_id}': "
                        f"HTTP {response.status_code}: {error_text}"
                    )

                resp_data = response.json()

                # Check for API-level error
                if "error" in resp_data:
                    err = resp_data["error"]
                    err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    raise RuntimeError(
                        f"{spec.provider.value} API error for model '{spec.model_id}': {err_msg}"
                    )

                # Parse response
                content, finish_reason, usage = self._parse_response(spec, resp_data)

                if content is None or not content.strip():
                    raise RuntimeError(f"LLM returned empty content for model '{spec.model_id}'")

                if finish_reason == "length":
                    raise RuntimeError(
                        f"model '{spec.model_id}' stopped at the token limit "
                        f"(finish_reason=length) — output truncated"
                    )

                # Clean hidden reasoning tokens
                content = clean_model_output(content)

                if not content:
                    raise RuntimeError(
                        f"model '{spec.model_id}' returned only hidden reasoning, no "
                        f"answer — needs more max_tokens or a different model"
                    )

                return LLMResponse(
                    text=content,
                    model_key=spec.key,
                    finish_reason=finish_reason,
                    usage=usage,
                    raw_response=resp_data,
                )

            except Exception as e:
                last_error = e

                # Check for rate limit
                rate_limit_wait = self._extract_rate_limit_wait(e)
                if rate_limit_wait is not None:
                    if rate_limit_waits >= self.max_rate_limit_waits:
                        raise RuntimeError(
                            f"Rate limit exceeded for {spec.provider.value} model "
                            f"'{spec.model_id}' after {self.max_rate_limit_waits} waits"
                        ) from e
                    rate_limit_waits += 1
                    logger.warning(f"  [rate limit] waiting {rate_limit_wait:.0f}s before retry...")
                    await asyncio.sleep(rate_limit_wait)
                    continue

                # Don't retry hard API errors
                if isinstance(e, RuntimeError) and "api error" in str(e).lower():
                    raise

                # Retry on transient errors
                if not self._is_retryable_error(e):
                    raise

                if attempt < self.max_retries:
                    wait = attempt * self.base_delay
                    logger.warning(f"  [retry {attempt}/{self.max_retries}] waiting {wait}s... ({e})")
                    await asyncio.sleep(wait)
                    continue

                raise RuntimeError(
                    f"LLM call ({spec.key}) failed after {self.max_retries} attempts: {last_error}"
                ) from last_error

        raise RuntimeError(f"LLM call ({spec.key}) failed: {last_error}")

    def call_sync(
        self,
        messages: list[dict],
        *,
        model_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Sync wrapper for non-async callers."""
        return asyncio.run(self.call(
            messages,
            model_key=model_key,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        ))


# ═══════════════════════════════════════════════════════════════════════════
# Convenience Functions (Back-Compat)
# ═══════════════════════════════════════════════════════════════════════════

def call_llm(
    messages: list[dict],
    model_key: str = DEFAULT_MODEL_KEY,
    temperature: float = 0.5,
    max_tokens: int = 1024,
    response_format: Optional[dict] = None,
) -> str:
    """
    Sync, text-only call matching the original video_maker signature.
    Returns just the text content.
    """
    client = LLMClient()
    resp = client.call_sync(
        messages,
        model_key=model_key,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    return resp.text


def list_models() -> list[dict]:
    """Return registry rows with key added for --help / error messages."""
    out = []
    for key, row in MODEL_REGISTRY.items():
        out.append({
            "key": key,
            "provider": row.provider.value,
            "model_id": row.model_id,
            "description": row.description,
        })
    return out


def model_keys() -> list[str]:
    """Sorted list of valid model keys (for error messages)."""
    return sorted(MODEL_REGISTRY)


def available_models() -> list[str]:
    """Return model keys whose API keys are present in environment."""
    avail = []
    for key, row in MODEL_REGISTRY.items():
        if os.environ.get(row.key_env):
            avail.append(key)
    return avail


# ═══════════════════════════════════════════════════════════════════════════
# Config Loading (Explicit, Not at Import Time)
# ═══════════════════════════════════════════════════════════════════════════

def load_env_file(path: Optional[Path] = None) -> None:
    """
    Explicit .env loader. Call this at application startup if you want
    to load a .env file. Does NOT run at module import.
    """
    env_path = path or Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and not os.environ.get(k):
                    os.environ[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# CLI for Testing/Smoke
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="List / smoke-test choosable LLM models.")
    ap.add_argument("--list", action="store_true", help="just print the registry")
    ap.add_argument("--model", default=None, help="model key to smoke-test (needs the key)")
    ap.add_argument("--prompt", default="Say hi in 3 words.")
    ap.add_argument("--bench", action="store_true", help="run every registered model on a story prompt, timed")
    args = ap.parse_args()

    if args.list or (not args.model and not args.bench):
        rows = list_models()
        print(f"{'--model key':<28} {'provider':<8} model_id")
        print("-" * 70)
        for r in rows:
            print(f"{r['key']:<28} {r['provider']:<8} {r['model_id']}")
            print(f"{'':28}          — {r['description']}")
        print(f"\nDefault: {DEFAULT_MODEL_KEY}")
        print("\nSmoke test with: python llm_client.py --model groq-llama3-70b")
        sys.exit(0)

    if args.bench:
        story_prompt = ("Write a simple short story of about 150 words "
                        "about a stray cat that finds a home.")
        print(f"Benchmarking all {len(MODEL_REGISTRY)} models")
        print(f"Prompt: {story_prompt}")
        results = []
        for key in model_keys():
            print(f"\n{'=' * 70}\n--- {key} ---")
            t0 = time.perf_counter()
            try:
                txt = call_llm([{"role": "user", "content": story_prompt}], model_key=key)
                elapsed = time.perf_counter() - t0
                results.append((key, elapsed, txt))
                print(f"time: {elapsed:.1f}s\n")
                print(txt)
            except Exception as e:
                elapsed = time.perf_counter() - t0
                results.append((key, None, ""))
                print(f"FAILED after {elapsed:.1f}s: {e}")
        print(f"\n{'=' * 70}\nRANKING (working models, fastest first)")
        for k, t in sorted(((k, t) for k, t, _ in results if t is not None), key=lambda x: x[1]):
            print(f"{t:6.1f}s  {k}")
        failed = [k for k, t, _ in results if t is None]
        if failed:
            print("failed: " + ", ".join(failed))
        sys.exit(0)

    txt = call_llm([{"role": "user", "content": args.prompt}], model_key=args.model)
    print(f"[{args.model}] -> {txt}")