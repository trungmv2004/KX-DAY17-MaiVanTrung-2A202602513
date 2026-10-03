from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates, merge_response_style
from model_provider import build_chat_model
from offline_responses import SYSTEM_PROMPT, model_response, offline_response


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Full thread history, with no disk profile or cross-session recall."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self._thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self._thread_users.setdefault(thread_id, user_id) != user_id:
            raise ValueError("A thread_id cannot be shared by different users.")
        if self.langchain_agent is not None:
            return self._reply(thread_id, message, live=True)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions[thread_id].token_usage if thread_id in self.sessions else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions[thread_id].prompt_tokens_processed if thread_id in self.sessions else 0

    def memory_file_size(self, user_id: str) -> int:
        return 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply(thread_id, message, live=False)

    def _reply(self, thread_id: str, message: str, live: bool) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + session.messages
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in prompt)
        if live:
            answer = model_response(self.langchain_agent, prompt)
        else:
            facts: dict[str, str] = {}
            for item in session.messages:
                if item["role"] == "user":
                    updates = extract_profile_updates(item["content"])
                    if "response_style" in updates:
                        updates["response_style"] = merge_response_style(facts.get("response_style", ""), updates["response_style"])
                    facts.update(updates)
            answer = offline_response(message, facts, session.messages)
        tokens = estimate_tokens(answer)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"answer": answer, "agent_tokens": tokens, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        model = self.config.model
        if self.force_offline or not model.live or model.model_name == "stub":
            return None
        if model.provider != "ollama" and not model.api_key:
            return None
        return build_chat_model(model)
