"""
API client utilities for OR-LLM-Bench evaluation.

Supports three backends:
  1. OpenAI API — set OPENAI_API_KEY env var
  2. Anthropic API — set ANTHROPIC_API_KEY env var
  3. Local vLLM — set VLLM_BASE_URL env var
"""

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from openai import AsyncOpenAI
import tiktoken
from typing import Union


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

OPENAI_MODELS = {"gpt-5", "gpt-5-nano", "o4-mini"}
CLAUDE_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"}


def is_api_model(model: str) -> bool:
    """Check if a model is served via API (not local vLLM)."""
    return model in OPENAI_MODELS or model in CLAUDE_MODELS


def is_claude_model(model: str) -> bool:
    """Check if a model uses the Anthropic API."""
    return model in CLAUDE_MODELS


def _get_model_backend(model: str) -> str:
    """Determine which backend: 'openai', 'claude', or 'vllm'."""
    if model in OPENAI_MODELS:
        return "openai"
    if model in CLAUDE_MODELS:
        return "claude"
    return "vllm"


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

@lru_cache()
def _load_keys() -> dict:
    """Load API keys from api_keys/keys.json if it exists."""
    base_dir = Path(__file__).resolve().parent.parent
    key_path = base_dir / "api_keys" / "keys.json"
    if key_path.exists():
        with open(key_path) as f:
            return json.load(f)
    return {}


def _resolve_key(env_var: str, json_key: str) -> str:
    """Resolve an API key: env var first, then api_keys/keys.json fallback."""
    key = os.getenv(env_var, "").strip()
    if key:
        return key
    keys = _load_keys()
    key = keys.get(json_key, "").strip()
    if key:
        return key
    print(f"Error: Set {env_var} env var or add '{json_key}' to api_keys/keys.json")
    sys.exit(1)


@lru_cache()
def _get_openai_client() -> AsyncOpenAI:
    """Return a shared async OpenAI client."""
    api_key = _resolve_key("OPENAI_API_KEY", "openai_api_key")
    return AsyncOpenAI(api_key=api_key, timeout=1200.0)


@lru_cache()
def _get_claude_client():
    """Return a shared async Anthropic client."""
    from anthropic import AsyncAnthropic

    api_key = _resolve_key("ANTHROPIC_API_KEY", "anthropic_api_key")
    return AsyncAnthropic(api_key=api_key)


@lru_cache()
def _get_vllm_openai_client() -> AsyncOpenAI:
    """Return a shared async OpenAI client for a local vLLM server."""
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    return AsyncOpenAI(api_key="EMPTY", base_url=base_url, timeout=1200.0)


def get_async_openai_client(model: str = "gpt-5"):
    """Return the appropriate async client for a given model.

    For OpenAI/OpenAI-compatible models -> AsyncOpenAI client.
    For Claude models -> ClaudeOpenAIAdapter (same .chat.completions.create interface).
    For vLLM models -> AsyncOpenAI pointed at local vLLM server.
    """
    backend = _get_model_backend(model)
    if backend == "openai":
        return _get_openai_client()
    if backend == "claude":
        return _get_claude_openai_adapter(model)
    return _get_vllm_openai_client()


# ---------------------------------------------------------------------------
# Claude-to-OpenAI adapter
# ---------------------------------------------------------------------------

class _ClaudeOpenAIAdapter:
    """Adapter: Anthropic Claude API -> OpenAI chat.completions interface."""
    def __init__(self, model: str):
        self.model = model
        self._client = None
        self.chat = self
        self.completions = self

    def _ensure_client(self):
        if self._client is None:
            self._client = _get_claude_client()

    async def create(self, model=None, messages=None, **kwargs):
        self._ensure_client()
        model = model or self.model

        system_text = None
        api_messages = []
        for m in (messages or []):
            if m["role"] == "system":
                system_text = m["content"]
            else:
                api_messages.append(m)

        max_tokens = kwargs.pop("max_completion_tokens", kwargs.pop("max_tokens", 16000))
        kwargs.pop("temperature", None)
        kwargs.pop("extra_body", None)

        create_kwargs = dict(model=model, messages=api_messages, max_tokens=max_tokens)
        if system_text:
            create_kwargs["system"] = system_text

        response = await self._client.messages.create(**create_kwargs)
        content = response.content[0].text if response.content else ""
        return _ClaudeResponseShim(
            content, response.usage.input_tokens,
            response.usage.output_tokens, response.model,
        )


class _ClaudeResponseShim:
    """Fake OpenAI ChatCompletion response from Claude data."""
    def __init__(self, content, input_tokens, output_tokens, model):
        self.choices = [type('Choice', (), {
            'message': type('Msg', (), {'content': content})(),
            'finish_reason': 'stop',
        })()]
        self.usage = type('Usage', (), {
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
        })()
        self.model = model


_claude_adapters = {}

def _get_claude_openai_adapter(model: str) -> _ClaudeOpenAIAdapter:
    if model not in _claude_adapters:
        _claude_adapters[model] = _ClaudeOpenAIAdapter(model)
    return _claude_adapters[model]


# ---------------------------------------------------------------------------
# Unified generation
# ---------------------------------------------------------------------------

async def generate_response(
    model: str,
    messages: list,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    **kwargs,
) -> str:
    """Unified async generation that routes to the correct API backend."""
    backend = _get_model_backend(model)

    if backend == "vllm":
        raise ValueError(f"Model '{model}' should be run via local vLLM, not generate_response().")

    if backend == "claude":
        client = _get_claude_client()
        system_text = None
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                api_messages.append(m)

        create_kwargs = dict(model=model, messages=api_messages, max_tokens=max_tokens)
        if system_text:
            create_kwargs["system"] = system_text
        if temperature > 0:
            create_kwargs["temperature"] = temperature

        response = await client.messages.create(**create_kwargs)
        return response.content[0].text

    else:
        client = _get_openai_client()
        create_kwargs = dict(model=model, messages=messages)

        if temperature != 0.0 and not model.startswith(("o3", "o4")):
            create_kwargs["temperature"] = temperature

        if model.startswith(("gpt-", "o3", "o4")):
            create_kwargs["max_completion_tokens"] = max_tokens
        else:
            create_kwargs["max_tokens"] = max_tokens

        create_kwargs.update(kwargs)
        response = await client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_chat_deployment(model: str) -> str:
    """Resolve model name to API deployment name. Identity for direct API access."""
    return model


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
