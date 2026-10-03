"""Ablation measuring the effect of accepting corrections to persistent facts."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_advanced import AdvancedAgent
from benchmark import format_rows, load_conversations, run_agent_benchmark
from config import load_config
from memory_store import UserProfileStore


class FirstFactWinsStore(UserProfileStore):
    """Experimental control: ignore corrections to location and profession."""

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        if key in {"location", "profession"} and key in self.facts(user_id):
            return False
        return super().upsert_fact(user_id, key, value)


def main() -> None:
    config = load_config()
    results = {}
    for title, filename in [("Standard Benchmark", "conversations.json"), ("Long-Context Stress Benchmark", "advanced_long_context.json")]:
        conversations = load_conversations(config.data_dir / filename)
        rows = []
        for accepts_corrections in [False, True]:
            with TemporaryDirectory(prefix="bonus-", dir=config.state_dir) as directory:
                path = Path(directory).resolve()
                if not path.is_relative_to(config.state_dir.resolve()):
                    raise ValueError("Bonus workspace must remain inside state_dir.")
                isolated = replace(config, state_dir=path)
                agent = AdvancedAgent(isolated, force_offline=True)
                if not accepts_corrections:
                    agent.profile_store = FirstFactWinsStore(path / "profiles", isolated.memory_policy())
                name = "Advanced (first fact wins)" if not accepts_corrections else "Advanced (corrections)"
                rows.append(run_agent_benchmark(name, agent, conversations, isolated))
        results[title] = rows
    output = "\n\n".join(f"## {title}\n\n{format_rows(rows)}" for title, rows in results.items()) + "\n"
    target = config.base_dir / "results"
    target.mkdir(parents=True, exist_ok=True)
    (target / "bonus_conflicts.md").write_text(output, encoding="utf-8", newline="\n")
    (target / "bonus_conflicts.json").write_text(json.dumps({title: [asdict(row) for row in rows] for title, rows in results.items()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
