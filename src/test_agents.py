from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import format_rows, heuristic_quality, load_conversations, recall_points, run_agent_benchmark, run_suite
from config import LabConfig, load_config
from live_smoke import run_live_checks
from memory_store import CompactMemoryManager, MemoryPolicy, ProfileCandidate, UserProfileStore, estimate_tokens, extract_profile_candidates, extract_profile_updates, summarize_messages
from model_provider import ProviderConfig, build_chat_model, normalize_provider


def make_config(tmp_path: Path) -> LabConfig:
    return LabConfig(
        base_dir=tmp_path,
        data_dir=Path(__file__).resolve().parent.parent / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        model=ProviderConfig("openai", "stub", 0.0),
        judge_model=ProviderConfig("openai", "stub", 0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("user") == ""
    assert store.file_size("user") == 0
    content = "# User\n- location: Huế\n- reference: Huế\n"
    path = store.write_text("user", content)
    assert path.name == "User.md"
    assert store.read_text("user") == content
    assert store.file_size("user") == len(content.encode("utf-8"))
    assert store.edit_text("user", "Huế", "Đà Nẵng")
    assert store.read_text("user").count("Huế") == 1
    assert "- location: Đà Nẵng" in store.read_text("user")
    assert not store.edit_text("user", "absent", "value")
    assert not store.edit_text("user", "Đà Nẵng", "Đà Nẵng")


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    memory = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
    texts = [f"Lượt {index}: " + "ngữ cảnh dài " * 60 for index in range(15)]
    for text in texts:
        memory.append("long", "user", text)
    context = memory.context("long")
    assert memory.compaction_count("long") > 1
    assert [m["content"] for m in context["messages"]] == texts[-2:]
    assert context["summary"]
    assert estimate_tokens(context["summary"]) <= config.compact_threshold_tokens // 3
    assert memory.context("other")["messages"] == []
    assert memory.compaction_count("other") == 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for agent in (baseline, advanced):
        agent.reply("user", "intro", "Mình tên là Lan. Mình ở Huế và đang làm data engineer.")
        assert "Lan" in agent.reply("user", "intro", "Mình tên gì?")["answer"]
    assert "Lan" not in baseline.reply("user", "new", "Mình tên gì?")["answer"]
    assert "Lan" in advanced.reply("user", "new", "Mình tên gì?")["answer"]
    assert baseline.memory_file_size("user") == 0
    assert advanced.memory_file_size("user") > 0
    restarted = AdvancedAgent(config, force_offline=True)
    assert "Lan" in restarted.reply("user", "restart", "Mình tên gì?")["answer"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for index in range(20):
        message = f"Lượt {index}. " + "Đây là ngữ cảnh tạm thời của hội thoại. " * 50
        baseline.reply("user", "long", message)
        advanced.reply("user", "long", message)
    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long") * 0.5
    assert baseline.compaction_count("long") == 0


@pytest.mark.parametrize("user_id", ["../../outside", "..\\..\\outside", "a/b", "a\\b", "CON", "LPT1", ".", "x" * 200])
def test_profile_paths_stay_inside_root(tmp_path: Path, user_id: str) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.path_for(user_id).is_relative_to(store.root_dir.resolve())
    store.write_text(user_id, "profile")
    assert store.read_text(user_id) == "profile"
    assert store.path_for("a/b") != store.path_for("a_b")
    assert store.path_for("Alice") != store.path_for("alice")


@pytest.mark.parametrize("text", [
    "Mình tên gì?", "Mình đang ở đâu?", "Đồ uống yêu thích của mình là gì?",
    "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.",
    "Nhắc lại style trả lời mình thích và đồ uống yêu thích của mình.",
    "Mình đùa là chuyển sang product manager.",
    "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày, không phải nơi ở hiện tại.",
    "Nếu sau này mình ở Hà Nội thì bạn sẽ nhớ thế nào?",
])
def test_extraction_ignores_questions_and_noise(text: str) -> None:
    assert extract_profile_updates(text) == {}


def test_corrections_replace_stale_facts_and_keep_style(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("user", "old", "Mình tên là Minh. Mình ở Huế và đang làm backend engineer. Mình muốn trả lời ngắn gọn thành 3 bullet có ví dụ thực chiến.")
    agent.reply("user", "correction", "Mình đang ở Đà Nẵng. Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    agent.reply("user", "correction", "Mình đùa là chuyển sang product manager. Hà Nội chỉ là nơi đi họp.")
    agent.reply("user", "style", "Mình muốn bạn trả lời ngắn gọn và ưu tiên trade-off.")
    answer = agent.reply("user", "recall", "Nhắc lại tên, nghề nghiệp, nơi ở hiện tại và style trả lời mình thích.")["answer"]
    for fact in ["Minh", "Đà Nẵng", "MLOps engineer", "3 bullet", "trade-off"]:
        assert fact in answer
    assert answer.count("\n- ") == 2
    profile = agent.profile_store.read_text("user")
    assert "Huế" not in profile
    assert "backend engineer" not in profile
    assert profile.count("- location:") == 1
    assert profile.count("- profession:") == 1


def test_users_are_isolated_and_threads_cannot_be_reused(tmp_path: Path) -> None:
    for cls in (BaselineAgent, AdvancedAgent):
        agent = cls(make_config(tmp_path), force_offline=True)
        agent.reply("alice", "alice-thread", "Mình tên là Alice.")
        assert "Alice" not in agent.reply("bob", "bob-thread", "Mình tên gì?")["answer"]
        with pytest.raises(ValueError, match="different users"):
            agent.reply("bob", "alice-thread", "Mình tên gì?")


def test_prompt_and_output_counters_accumulate_turn_deltas(tmp_path: Path) -> None:
    for cls in (BaselineAgent, AdvancedAgent):
        agent = cls(make_config(tmp_path), force_offline=True)
        replies = [agent.reply("user", "one", text) for text in ["Mình tên là Lan.", "Mình tên gì?", "Chào bạn."]]
        assert agent.token_usage("one") == sum(r["agent_tokens"] for r in replies)
        assert agent.prompt_token_usage("one") == sum(r["prompt_tokens"] for r in replies)
        assert all(r["agent_tokens"] == estimate_tokens(r["answer"]) for r in replies)
        assert agent.token_usage("missing") == agent.prompt_token_usage("missing") == 0


def test_temporary_topics_use_thread_context_without_becoming_profile_facts(tmp_path: Path) -> None:
    for cls in (BaselineAgent, AdvancedAgent):
        agent = cls(make_config(tmp_path), force_offline=True)
        agent.reply("user", "topic", "Chủ đề hôm nay: triển khai Redis cho cache.")
        assert "Redis" in agent.reply("user", "topic", "Nhắc lại chủ đề vừa nói.")["answer"]
        assert "Redis" not in agent.reply("user", "fresh", "Nhắc lại chủ đề vừa nói.")["answer"]
        if isinstance(agent, AdvancedAgent):
            assert "Redis" not in agent.profile_store.read_text("user")


def test_force_offline_does_not_build_models(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path)
    config.model = ProviderConfig("openai", "live-model", 0.0, api_key="placeholder", live=True)
    def unexpected(*args):
        raise AssertionError("Offline agent attempted to build a model.")
    monkeypatch.setattr("agent_baseline.build_chat_model", unexpected)
    monkeypatch.setattr("agent_advanced.build_chat_model", unexpected)
    for cls in (BaselineAgent, AdvancedAgent):
        assert cls(config, force_offline=True).langchain_agent is None


def test_live_path_receives_the_correct_memory_context(tmp_path: Path, monkeypatch) -> None:
    calls = []
    class FakeModel:
        def invoke(self, messages):
            calls.append(messages)
            return SimpleNamespace(content="Phản hồi kiểm thử.")
    config = make_config(tmp_path)
    config.model = ProviderConfig("openai", "live-model", 0.0, api_key="placeholder", live=True)
    monkeypatch.setattr("agent_baseline.build_chat_model", lambda config: FakeModel())
    monkeypatch.setattr("agent_advanced.build_chat_model", lambda config: FakeModel())
    for cls in (BaselineAgent, AdvancedAgent):
        agent = cls(config)
        agent.reply("user", "first", "Mình tên là Lan.")
        agent.reply("user", "second", "Mình tên gì?")
        contains_profile = any("- name: Lan" in m["content"] for m in calls[-1])
        assert contains_profile == (cls is AdvancedAgent)


@pytest.mark.parametrize("provider,class_name", [
    ("openai", "ChatOpenAI"), ("custom", "ChatOpenAI"), ("gemini", "ChatGoogleGenerativeAI"),
    ("anthropic", "ChatAnthropic"), ("ollama", "ChatOllama"), ("openrouter", "ChatOpenRouter"),
])
def test_provider_construction_without_api_calls(provider: str, class_name: str) -> None:
    config = ProviderConfig(provider, "smoke-model", 0.0, api_key="placeholder", base_url="http://localhost:1234/v1" if provider == "custom" else None)
    assert type(build_chat_model(config)).__name__ == class_name


def test_provider_aliases_and_invalid_provider() -> None:
    assert normalize_provider(" Anthorpic ") == "anthropic"
    assert normalize_provider("google") == "gemini"
    assert normalize_provider("openai_compatible") == "custom"
    with pytest.raises(ValueError, match="Unsupported provider"):
        normalize_provider("unknown")
    with pytest.raises(ValueError, match="CUSTOM_BASE_URL"):
        build_chat_model(ProviderConfig("custom", "model", 0.0))


def test_load_config_reads_env_and_preserves_exported_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in ["LLM_PROVIDER", "LLM_MODE", "LLM_MODEL", "JUDGE_MODE", "JUDGE_MODEL", "JUDGE_PROVIDER", "COMPACT_THRESHOLD_TOKENS", "COMPACT_KEEP_MESSAGES"]:
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text("LLM_PROVIDER=anthorpic\nCOMPACT_THRESHOLD_TOKENS=300\nCOMPACT_KEEP_MESSAGES=3\n", encoding="utf-8")
    monkeypatch.setenv("COMPACT_THRESHOLD_TOKENS", "200")
    config = load_config(tmp_path)
    assert config.model.provider == "anthropic"
    assert not config.model.live
    assert config.compact_threshold_tokens == 200
    assert config.compact_keep_messages == 3
    assert config.state_dir.is_dir()
    monkeypatch.setenv("COMPACT_THRESHOLD_TOKENS", "0")
    with pytest.raises(ValueError, match="positive"):
        load_config(tmp_path)


def test_token_estimator_and_summary_preserve_current_facts() -> None:
    assert estimate_tokens("") == estimate_tokens("   ") == 0
    assert estimate_tokens("a" * 40) >= estimate_tokens("a" * 4)
    assert estimate_tokens("Huế") == estimate_tokens("Huế")
    first = summarize_messages([{"role": "user", "content": "Mình tên là Lan. Mình ở Huế."}])
    second = summarize_messages([{"role": "system", "content": first}, {"role": "user", "content": "Mình ở Đà Nẵng."}])
    assert "fact.name: Lan" in second
    assert "fact.location: Đà Nẵng" in second
    assert "fact.location: Huế" not in second
    assert summarize_messages([], max_items=0) == ""


def test_benchmark_scoring_and_input_validation(tmp_path: Path) -> None:
    assert recall_points("Lan ở Huế", ["Lan", "Huế"]) == 1
    assert recall_points("Lan", ["Lan", "Huế"]) == 0.5
    assert recall_points("Không biết", ["Lan"]) == 0
    assert recall_points(unicodedata.normalize("NFD", "HUẾ"), ["Huế"]) == 1
    assert recall_points("", []) == 0
    assert heuristic_quality("Lan", ["Lan"]) > heuristic_quality("Không biết", ["Lan"])
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([{"id": "invalid"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="requires"):
        load_conversations(path)


def test_benchmark_uses_new_threads_and_sums_all_turns(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = BaselineAgent(config, force_offline=True)
    conversations = [{"id": "example", "user_id": "user", "turns": ["Mình tên là Lan."], "recall_questions": [{"question": "Mình tên gì?", "expected_contains": ["Lan"]}]}]
    row = run_agent_benchmark("Baseline", agent, conversations, config)
    assert len(agent.sessions) == 2
    assert row.recall_score == 0
    assert row.agent_tokens_only == sum(agent.token_usage(t) for t in agent.sessions)
    assert row.prompt_tokens_processed == sum(agent.prompt_token_usage(t) for t in agent.sessions)
    for column in ["Agent tokens only", "Prompt tokens processed", "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"]:
        assert column in format_rows([row])


def test_real_datasets_repeat_and_compact_ablation(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), compact_threshold_tokens=1200, compact_keep_messages=4)
    standard = run_suite(config.data_dir / "conversations.json", config)
    stress = run_suite(config.data_dir / "advanced_long_context.json", config)
    assert standard[0].recall_score == stress[0].recall_score == 0
    assert standard[1].recall_score == stress[1].recall_score == 1
    assert standard[1].prompt_tokens_processed > standard[0].prompt_tokens_processed
    assert standard[1].compactions == 0
    assert stress[1].prompt_tokens_processed < stress[0].prompt_tokens_processed
    assert stress[1].compactions > 1
    assert stress == run_suite(config.data_dir / "advanced_long_context.json", config)
    no_compact = run_suite(config.data_dir / "advanced_long_context.json", replace(config, compact_threshold_tokens=10**12))
    assert no_compact[1].compactions == 0
    assert no_compact[1].recall_score == 1
    assert no_compact[1].prompt_tokens_processed > stress[1].prompt_tokens_processed


def test_confidence_scores_and_threshold_filtering() -> None:
    assert extract_profile_updates("Bạn của mình tên là Nam.") == {}
    assert extract_profile_updates("Có lẽ mình đang ở Hà Nội.") == {}
    assert extract_profile_updates('Trong ví dụ có câu "mình tên là Nam".') == {}
    uncertain = extract_profile_candidates("Có lẽ mình đang ở Hà Nội.")
    assert uncertain[0].confidence == 0.45
    assert uncertain[0].reason == "uncertain declaration"
    assert extract_profile_updates("Có lẽ mình đang ở Hà Nội.", 0.4) == {"location": "Hà Nội"}
    assert extract_profile_updates("Mình đã chuyển từ Huế sang Hà Nội.") == {"location": "Hà Nội"}


def test_confidence_boundary_and_rejected_fact_does_not_overwrite(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles", MemoryPolicy(confidence_threshold=0.8))
    decision = store.apply_candidates("user", [ProfileCandidate("name", "Lan", 0.8, "boundary")])
    assert decision == {"accepted": 1, "rejected": 0}
    decision = store.apply_candidates("user", [ProfileCandidate("name", "Nam", 0.799, "weak evidence")])
    assert decision == {"accepted": 0, "rejected": 1}
    assert store.facts("user")["name"] == "Lan"
    assert store.metadata("user")["facts"]["name"]["seen_count"] == 1
    assert store.total_size("user") > store.file_size("user")


def test_decay_half_life_expiration_reconfirmation_and_restart(tmp_path: Path) -> None:
    policy = MemoryPolicy(decay_half_life_turns=2, min_priority=0.3)
    store = UserProfileStore(tmp_path / "profiles", policy)
    store.apply_candidates("user", extract_profile_candidates("Mình tên là Lan. Mình ở Huế."))
    store.apply_candidates("user", [])
    store.apply_candidates("user", [])
    assert store.priorities("user")["location"] == pytest.approx(0.49)
    assert store.priorities("user")["name"] == pytest.approx(0.98)
    store.apply_candidates("user", [])
    store.apply_candidates("user", [])
    restarted = UserProfileStore(tmp_path / "profiles", policy)
    assert restarted.active_facts("user") == {"name": "Lan"}
    assert "location" not in restarted.active_text("user")
    assert restarted.facts("user")["location"] == "Huế"  # decay never deletes the source.
    restarted.apply_candidates("user", extract_profile_candidates("Mình ở Huế."))
    assert restarted.priorities("user")["location"] == pytest.approx(0.98)
    assert restarted.active_facts("user")["location"] == "Huế"
    assert restarted.metadata("user")["facts"]["location"]["seen_count"] == 2


def test_uncertain_repetition_and_recall_do_not_refresh_decay(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles", MemoryPolicy(decay_half_life_turns=2))
    store.apply_candidates("user", extract_profile_candidates("Mình ở Huế."))
    store.apply_candidates("user", extract_profile_candidates("Có lẽ mình đang ở Huế."))
    store.apply_candidates("user", extract_profile_candidates("Mình đang ở đâu?"))
    assert store.metadata("user")["facts"]["location"]["last_seen_turn"] == 1
    assert store.priorities("user")["location"] == pytest.approx(0.49)


def test_disabled_decay_keeps_old_facts_and_users_age_independently(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles", MemoryPolicy(decay_half_life_turns=1, decay_enabled=False))
    store.apply_candidates("alice", extract_profile_candidates("Mình ở Huế."))
    for _ in range(8):
        store.apply_candidates("alice", [])
    assert store.priorities("alice")["location"] == pytest.approx(0.98)
    assert store.active_facts("alice")["location"] == "Huế"
    store.apply_candidates("bob", extract_profile_candidates("Mình ở Hà Nội."))
    assert store.metadata("bob")["turn"] == 1


def test_legacy_profile_and_decay_in_advanced_prompt(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), memory_decay_half_life_turns=1, memory_decay_min_priority=0.3)
    agent = AdvancedAgent(config, force_offline=True)
    agent.profile_store.write_text("user", "# User profile\n- name: Lan\n- location: Huế\n")
    assert agent.profile_store.active_facts("user")["location"] == "Huế"
    agent.reply("user", "fresh1", "Xin chào.")
    reply = agent.reply("user", "fresh2", "Mình đang ở đâu?")
    assert "Huế" not in reply["answer"]
    assert "Huế" not in "\n".join(m["content"] for m in agent._prompt("user", "fresh2"))
    assert agent.profile_store.facts("user")["location"] == "Huế"
    agent.reply("user", "correction", "Mình đã chuyển từ Huế sang Hà Nội.")
    assert agent.profile_store.active_facts("user")["location"] == "Hà Nội"


def test_style_retraction_removes_old_bullet_and_short_preferences(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("user", "first", "Mình muốn bạn trả lời ngắn gọn thành 3 bullet.")
    agent.reply("user", "first", "Mình muốn bạn trả lời dài và chi tiết, không dùng bullet nữa.")
    style = agent.profile_store.facts("user")["response_style"]
    assert style == "dài và chi tiết, không dùng bullet"
    answer = agent.reply("user", "new", "Nhắc lại style trả lời mình thích.")["answer"]
    assert not answer.startswith("- ")
    assert "3 bullet" not in answer


def test_memory_policy_env_config_and_validation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", {})
    (tmp_path / ".env").write_text("MEMORY_CONFIDENCE_THRESHOLD=0.9\nMEMORY_DECAY_HALF_LIFE_TURNS=40\nMEMORY_DECAY_MIN_PRIORITY=0.25\nMEMORY_DECAY_ENABLED=false\n", encoding="utf-8")
    config = load_config(tmp_path)
    assert config.memory_policy() == MemoryPolicy(0.9, 40, 0.25, False)
    monkeypatch.setenv("MEMORY_DECAY_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="must be true or false"):
        load_config(tmp_path)
    with pytest.raises(ValueError, match="half-life"):
        MemoryPolicy(decay_half_life_turns=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ProfileCandidate("name", "Lan", float("nan"), "invalid")


def test_live_checks_require_real_model_and_reject_offline_fallback(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(ValueError, match="explicit real model"):
        run_live_checks(config)
    config.model = ProviderConfig("openai", "live-model", 0.0, live=True)
    report = run_live_checks(config)  # Missing key: agent cannot build a model.
    assert report["status"] == "failed"
    assert report["api_calls"] == []
    assert report["checks"] == []
    assert report["error"]["type"] == "ValueError"


def test_live_report_does_not_serialize_provider_error_body(tmp_path: Path, monkeypatch) -> None:
    class FailingModel:
        def invoke(self, messages):
            raise RuntimeError("sensitive-provider-error-body")

    config = make_config(tmp_path)
    config.model = ProviderConfig("openai", "live-model", 0.0, api_key="placeholder", live=True)
    monkeypatch.setattr("agent_baseline.build_chat_model", lambda config: FailingModel())
    report = run_live_checks(config)
    assert report["status"] == "failed"
    assert report["error"] == {"type": "RuntimeError", "status_code": None}
    assert "sensitive-provider-error-body" not in json.dumps(report)
