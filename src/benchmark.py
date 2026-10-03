from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    conversations = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(conversations, list):
        raise ValueError("Dataset must contain a list of conversations.")
    ids = set()
    for item in conversations:
        if not isinstance(item, dict) or not {"id", "user_id", "turns", "recall_questions"} <= item.keys():
            raise ValueError("Conversation requires id, user_id, turns, recall_questions.")
        if not isinstance(item["id"], str) or not isinstance(item["user_id"], str):
            raise ValueError("Conversation id and user_id must be strings.")
        if item["id"] in ids:
            raise ValueError(f"Duplicate conversation id: {item['id']}.")
        ids.add(item["id"])
        if not isinstance(item["turns"], list) or not all(isinstance(turn, str) for turn in item["turns"]):
            raise ValueError("turns must contain strings.")
        if not isinstance(item["recall_questions"], list):
            raise ValueError("recall_questions must be a list.")
        for recall in item["recall_questions"]:
            if not isinstance(recall, dict) or not isinstance(recall.get("question"), str):
                raise ValueError("Recall questions must contain a question string.")
            expected = recall.get("expected_contains")
            if not isinstance(expected, list) or not expected or not all(isinstance(value, str) and value for value in expected):
                raise ValueError("expected_contains must be a nonempty list of nonempty strings.")
    return conversations


def _coverage(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    normalized = unicodedata.normalize("NFC", answer).casefold()
    return sum(unicodedata.normalize("NFC", value).casefold() in normalized for value in expected) / len(expected)


def recall_points(answer: str, expected: list[str]) -> float:
    coverage = _coverage(answer, expected)
    return 1.0 if coverage == 1 else 0.5 if coverage > 0 else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """80% literal fact coverage + 20% concise, nonempty response; no LLM judge."""
    concise = 1.0 if answer.strip() and len(answer) <= 600 else 0.0
    return 0.8 * _coverage(answer, expected) + 0.2 * concise


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config: LabConfig) -> BenchmarkRow:
    """Score immediately after each conversation, before later corrections arrive."""
    if not agent.force_offline:
        raise ValueError("Use force_offline=True for reproducible benchmark runs.")
    users = {conversation["user_id"] for conversation in conversations}
    initial_bytes = sum(agent.memory_file_size(user) for user in users)
    tokens = prompt_tokens = 0
    recall_scores: list[float] = []
    qualities: list[float] = []
    threads: set[str] = set()
    for conversation in conversations:
        user = conversation["user_id"]
        thread = f"conversation:{conversation['id']}"
        threads.add(thread)
        for message in conversation["turns"]:
            reply = agent.reply(user, thread, message)
            tokens += reply["agent_tokens"]
            prompt_tokens += reply["prompt_tokens"]
        for index, recall in enumerate(conversation["recall_questions"]):
            # Each recall question gets a clean thread, identically for both agents.
            recall_thread = f"recall:{conversation['id']}:{index}"
            threads.add(recall_thread)
            reply = agent.reply(user, recall_thread, recall["question"])
            tokens += reply["agent_tokens"]
            prompt_tokens += reply["prompt_tokens"]
            recall_scores.append(recall_points(reply["answer"], recall["expected_contains"]))
            qualities.append(heuristic_quality(reply["answer"], recall["expected_contains"]))
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=mean(recall_scores) if recall_scores else 0.0,
        response_quality=mean(qualities) if qualities else 0.0,
        memory_growth_bytes=sum(agent.memory_file_size(user) for user in users) - initial_bytes,
        compactions=sum(agent.compaction_count(thread) for thread in threads),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"]
    values = [[row.agent_name, row.agent_tokens_only, row.prompt_tokens_processed, f"{row.recall_score:.3f}", f"{row.response_quality:.3f}", row.memory_growth_bytes, row.compactions] for row in rows]
    return tabulate(values, headers=headers, tablefmt="github", disable_numparse=True)


def run_suite(dataset: Path, config: LabConfig) -> list[BenchmarkRow]:
    """Reset only this suite's benchmark workspace; keep personal profiles intact."""
    suite_dir = config.state_dir / "benchmarks" / dataset.stem
    root = config.state_dir.resolve()
    resolved = suite_dir.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("Benchmark workspace must remain inside state_dir.")
    conversations = load_conversations(dataset)
    if suite_dir.exists():
        shutil.rmtree(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    isolated = replace(config, state_dir=suite_dir)
    return [
        run_agent_benchmark("Baseline", BaselineAgent(isolated, force_offline=True), conversations, isolated),
        run_agent_benchmark("Advanced", AdvancedAgent(isolated, force_offline=True), conversations, isolated),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic, offline memory benchmark.")
    parser.add_argument("--output", type=Path, help="Write the two Markdown comparison tables.")
    parser.add_argument("--json", type=Path, help="Write raw metrics for analysis.")
    parser.add_argument("--no-compact", action="store_true", help="Ablation: effectively disable compaction.")
    args = parser.parse_args()
    config = load_config()
    if args.no_compact:
        config = replace(config, compact_threshold_tokens=10**12)
    suites = {}
    for title, filename in [("Standard Benchmark", "conversations.json"), ("Long-Context Stress Benchmark", "advanced_long_context.json")]:
        suites[title] = run_suite(config.data_dir / filename, config)
    report = "\n\n".join(f"## {title}\n\n{format_rows(rows)}" for title, rows in suites.items()) + "\n"
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "settings": {
                "compact_threshold_tokens": config.compact_threshold_tokens,
                "compact_keep_messages": config.compact_keep_messages,
                "mode": "offline",
                "memory_policy": asdict(config.memory_policy()),
                "memory_growth_files": ["User.md", "Memory.json"],
            },
            "suites": {title: [asdict(row) for row in rows] for title, rows in suites.items()},
        }
        args.json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
