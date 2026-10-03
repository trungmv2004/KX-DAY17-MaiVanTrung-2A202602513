from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider
from memory_store import MemoryPolicy


REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")


@dataclass
class LabConfig:
    """Shared paths, compact settings, and independent model/judge settings."""

    base_dir: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    state_dir: Path = REPO_ROOT / "state"
    compact_threshold_tokens: int = 1200
    compact_keep_messages: int = 4
    model: ProviderConfig = field(default_factory=lambda: ProviderConfig("openai", "stub", 0.0))
    judge_model: ProviderConfig = field(default_factory=lambda: ProviderConfig("openai", "stub", 0.0))
    memory_confidence_threshold: float = 0.8
    memory_decay_half_life_turns: float = 100.0
    memory_decay_min_priority: float = 0.2
    memory_decay_enabled: bool = True

    def __post_init__(self) -> None:
        if self.compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive.")
        if self.compact_keep_messages < 1:
            raise ValueError("compact_keep_messages must be at least 1.")
        self.memory_policy()

    def memory_policy(self) -> MemoryPolicy:
        return MemoryPolicy(self.memory_confidence_threshold, self.memory_decay_half_life_turns, self.memory_decay_min_priority, self.memory_decay_enabled)


def _provider_config(prefix: str, fallback: str = "openai") -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", fallback))
    mode = os.getenv(f"{prefix}_MODE", "offline").strip().lower()
    if mode not in {"offline", "live"}:
        raise ValueError(f"{prefix}_MODE must be offline or live.")
    key_names = {
        "openai": "OPENAI_API_KEY", "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "ollama": "", "openrouter": "OPENROUTER_API_KEY",
    }
    base_names = {
        "openai": "OPENAI_BASE_URL", "custom": "CUSTOM_BASE_URL",
        "gemini": "GEMINI_BASE_URL", "anthropic": "ANTHROPIC_BASE_URL",
        "ollama": "OLLAMA_BASE_URL", "openrouter": "OPENROUTER_BASE_URL",
    }
    model_name = os.getenv(f"{prefix}_MODEL", "stub")
    if mode == "live" and model_name == "stub":
        raise ValueError(f"Set {prefix}_MODEL to a real model name for live mode.")
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
        api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv(key_names[provider]),
        base_url=os.getenv(f"{prefix}_BASE_URL") or os.getenv(base_names[provider]),
        live=mode == "live",
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Read .env without overriding exported values; default to offline mode."""
    root = (base_dir or REPO_ROOT).resolve()
    load_dotenv(root / ".env", override=False)
    model = _provider_config("LLM")
    config = LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=root / "state",
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=_provider_config("JUDGE", model.provider),
        memory_confidence_threshold=float(os.getenv("MEMORY_CONFIDENCE_THRESHOLD", "0.8")),
        memory_decay_half_life_turns=float(os.getenv("MEMORY_DECAY_HALF_LIFE_TURNS", "100")),
        memory_decay_min_priority=float(os.getenv("MEMORY_DECAY_MIN_PRIORITY", "0.2")),
        memory_decay_enabled=_env_bool("MEMORY_DECAY_ENABLED", True),
    )
    config.state_dir.mkdir(parents=True, exist_ok=True)
    return config
