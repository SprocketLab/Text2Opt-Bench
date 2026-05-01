"""
Central registry of all models used in OR-LLM-Bench.

Backend routing:
  - "openai"    -> OpenAI API (OPENAI_API_KEY env var)
  - "anthropic" -> Anthropic API (ANTHROPIC_API_KEY env var)
  - "vllm"      -> Local vLLM server (VLLM_BASE_URL env var)

Usage:
    from main.model_registry import MAIN_TABLE_MODELS, get_model_id
"""

# ──────────────────────────────────────────────────────────────────────
# Main results table models
# ──────────────────────────────────────────────────────────────────────

MAIN_TABLE_MODELS = {
    # OpenAI models
    "gpt-5": {
        "display_name": "GPT-5",
        "family": "OpenAI",
        "backend": "openai",
        "model_id": "gpt-5",
    },
    "gpt-5-nano": {
        "display_name": "GPT-5-Nano",
        "family": "OpenAI",
        "backend": "openai",
        "model_id": "gpt-5-nano",
    },
    "o4-mini": {
        "display_name": "o4-mini",
        "family": "OpenAI",
        "backend": "openai",
        "model_id": "o4-mini",
    },
    # Anthropic models
    "claude-opus-4-6": {
        "display_name": "Claude Opus 4.6",
        "family": "Anthropic",
        "backend": "anthropic",
        "model_id": "claude-opus-4-6",
    },
    "claude-sonnet-4-6": {
        "display_name": "Claude Sonnet 4.6",
        "family": "Anthropic",
        "backend": "anthropic",
        "model_id": "claude-sonnet-4-6",
    },
    # Open-weight models (vLLM)
    "DeepSeek-R1": {
        "display_name": "DeepSeek-R1",
        "family": "DeepSeek",
        "backend": "vllm",
        "model_id": "DeepSeek-R1",
    },
    "DeepSeek-V3.2": {
        "display_name": "DeepSeek-V3.2",
        "family": "DeepSeek",
        "backend": "vllm",
        "model_id": "DeepSeek-V3.2",
    },
    "Llama-3.3-70B-Instruct": {
        "display_name": "Llama-3.3-70B",
        "family": "Meta",
        "backend": "vllm",
        "model_id": "Llama-3.3-70B-Instruct",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "display_name": "Qwen2.5-7B",
        "family": "Qwen",
        "backend": "vllm",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
    },
}

# ──────────────────────────────────────────────────────────────────────
# Qwen scaling study (RULER binding tasks + resource allocation)
# ──────────────────────────────────────────────────────────────────────

QWEN_SCALING_MODELS = {
    f"Qwen/Qwen2.5-{size}-Instruct": {
        "display_name": f"Qwen2.5-{size}",
        "params": size,
        "backend": "vllm",
    }
    for size in ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
}

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

ALL_MODELS = {**MAIN_TABLE_MODELS, **QWEN_SCALING_MODELS}


def get_model_id(name: str) -> str:
    """Get the API model_id for a given model name."""
    if name in ALL_MODELS:
        return ALL_MODELS[name].get("model_id", name)
    return name


def get_display_name(name: str) -> str:
    """Get the paper-friendly display name for a model."""
    if name in ALL_MODELS:
        return ALL_MODELS[name]["display_name"]
    return name
