from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass
class ProviderConfig:
    """Provider settings; live calls require an explicit opt-in."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    live: bool = False
    request_timeout: float = 30.0
    max_retries: int = 1
    max_output_tokens: int | None = None


def normalize_provider(value: str) -> str:
    """Normalize supported aliases and reject unknown providers early."""
    aliases = {
        "openai": "openai", "open-ai": "openai",
        "custom": "custom", "openai-compatible": "custom",
        "gemini": "gemini", "google": "gemini", "google-genai": "gemini",
        "anthropic": "anthropic", "anthorpic": "anthropic", "claude": "anthropic",
        "ollama": "ollama", "openrouter": "openrouter", "open-router": "openrouter",
    }
    normalized = value.strip().lower().replace("_", "-")
    if normalized not in aliases:
        raise ValueError(f"Unsupported provider: {value!r}. Choose openai, custom, gemini, anthropic, ollama, openrouter.")
    return aliases[normalized]


def build_chat_model(config: ProviderConfig) -> Any:
    """Build a chat model without a request; import only the selected SDK."""
    provider = normalize_provider(config.provider)
    implementations = {
        "openai": ("langchain_openai", "ChatOpenAI"),
        "custom": ("langchain_openai", "ChatOpenAI"),
        "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        "anthropic": ("langchain_anthropic", "ChatAnthropic"),
        "ollama": ("langchain_ollama", "ChatOllama"),
        "openrouter": ("langchain_openrouter", "ChatOpenRouter"),
    }
    if provider == "custom" and not config.base_url:
        raise ValueError("custom provider requires CUSTOM_BASE_URL.")
    if not config.model_name.strip() or config.model_name == "stub":
        raise ValueError("Select a real model name before building a live chat model.")
    module, class_name = implementations[provider]
    try:
        model_class = getattr(import_module(module), class_name)
    except ImportError as exc:
        raise ImportError(f"Install {module.replace('_', '-')} to use provider {provider}.") from exc
    kwargs: dict[str, Any] = {"model": config.model_name, "temperature": config.temperature}
    if provider == "ollama":
        kwargs["client_kwargs"] = {"timeout": config.request_timeout}
        if config.max_output_tokens is not None:
            kwargs["num_predict"] = config.max_output_tokens
    else:
        kwargs.update(timeout=config.request_timeout, max_retries=config.max_retries)
        if config.max_output_tokens is not None:
            kwargs["max_tokens"] = config.max_output_tokens
    if config.api_key and provider != "ollama":
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if provider == "anthropic" and config.max_output_tokens is None:
        kwargs["max_tokens"] = 1024
    return model_class(**kwargs)
