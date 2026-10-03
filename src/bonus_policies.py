"""Synthetic ablations for confidence filtering and memory decay; no API calls."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_advanced import AdvancedAgent
from benchmark import format_rows, run_agent_benchmark
from config import load_config


def main() -> None:
    config = load_config()
    experiments = [
        (
            "Confidence threshold",
            [{"id": "confidence", "user_id": "confidence_sample", "turns": [
                "Mình tên là Lan. Mình ở Huế.",
                "Bạn của mình tên là Nam.",
                "Có lẽ mình đang ở Hà Nội.",
            ], "recall_questions": [{"question": "Nhắc lại tên và nơi ở hiện tại của mình.", "expected_contains": ["Lan", "Huế"]}]}],
            [
                ("Threshold 0", replace(config, memory_confidence_threshold=0, memory_decay_enabled=False, memory_decay_min_priority=0)),
                ("Threshold 0.8", replace(config, memory_confidence_threshold=0.8, memory_decay_enabled=False, memory_decay_min_priority=0)),
            ],
        ),
        (
            "Memory decay",
            [{"id": "decay", "user_id": "decay_sample", "turns": [
                "Mình tên là LanDecay. Mình ở Huế. Mình thích Python.",
                *[f"Lượt {i}: Nội dung tạm thời của cuộc trao đổi." for i in range(10)],
            ], "recall_questions": [{"question": "Nhắc lại tên và nơi ở hiện tại của mình.", "expected_contains": ["LanDecay", "Huế"]}]}],
            [
                ("Decay off", replace(config, memory_decay_enabled=False, memory_decay_half_life_turns=2, memory_decay_min_priority=0.3)),
                ("Half-life 2 turns", replace(config, memory_decay_enabled=True, memory_decay_half_life_turns=2, memory_decay_min_priority=0.3)),
            ],
        ),
    ]
    reports = {}
    sections = []
    for title, conversations, variants in experiments:
        rows, settings = [], []
        for name, variant in variants:
            with TemporaryDirectory(prefix="policy-bonus-", dir=config.state_dir) as directory:
                state = Path(directory).resolve()
                if not state.is_relative_to(config.state_dir.resolve()):
                    raise ValueError("Policy experiment workspace must remain inside state_dir.")
                isolated = replace(variant, state_dir=state, compact_threshold_tokens=10**12)
                agent = AdvancedAgent(isolated, force_offline=True)
                rows.append(run_agent_benchmark(name, agent, conversations, isolated))
                user = conversations[0]["user_id"]
                settings.append({"policy": asdict(isolated.memory_policy()), "stored_facts": len(agent.profile_store.facts(user)), "active_facts": len(agent.profile_store.active_facts(user))})
        reports[title] = {"rows": [asdict(row) for row in rows], "variants": settings}
        sections.append(f"## {title}\n\n{format_rows(rows)}")
    report = "\n\n".join(sections) + "\n"
    target = config.base_dir / "results"
    target.mkdir(parents=True, exist_ok=True)
    (target / "bonus_policies.md").write_text(report, encoding="utf-8", newline="\n")
    (target / "bonus_policies.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(report)


if __name__ == "__main__":
    main()
