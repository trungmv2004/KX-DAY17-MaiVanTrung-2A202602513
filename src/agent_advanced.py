from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_candidates
from model_provider import build_chat_model
from offline_responses import SYSTEM_PROMPT, model_response, offline_response


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Thread context + persistent user facts + bounded summaries."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles", self.config.memory_policy())
        self.compact_memory = CompactMemoryManager(self.config.compact_threshold_tokens, self.config.compact_keep_messages)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self._thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self._thread_users.setdefault(thread_id, user_id) != user_id:
            raise ValueError("A thread_id cannot be shared by different users.")
        if self.langchain_agent is not None:
            return self._reply(user_id, thread_id, message, live=True)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.total_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply(user_id, thread_id, message, live=False)

    def _reply(self, user_id: str, thread_id: str, message: str, live: bool) -> dict[str, Any]:
        self.profile_store.apply_candidates(user_id, extract_profile_candidates(message))
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = model_response(self.langchain_agent, self._prompt(user_id, thread_id)) if live else self._offline_response(user_id, thread_id, message)
        tokens = estimate_tokens(answer)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {"answer": answer, "agent_tokens": tokens, "prompt_tokens": prompt_tokens}

    def _prompt(self, user_id: str, thread_id: str) -> list[dict[str, str]]:
        context = self.compact_memory.context(thread_id)
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}]
        profile = self.profile_store.active_text(user_id)
        if profile:
            prompt.append({"role": "system", "content": profile})
        if context["summary"]:
            prompt.append({"role": "system", "content": context["summary"]})
        prompt.extend(context["messages"])
        return prompt

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        return sum(estimate_tokens(m["content"]) for m in self._prompt(user_id, thread_id))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        context = self.compact_memory.context(thread_id)
        return offline_response(message, self.profile_store.active_facts(user_id), context["messages"], context["summary"])

    def _maybe_build_langchain_agent(self):
        model = self.config.model
        if self.force_offline or not model.live or model.model_name == "stub":
            return None
        if model.provider != "ollama" and not model.api_key:
            return None
        return build_chat_model(model)
