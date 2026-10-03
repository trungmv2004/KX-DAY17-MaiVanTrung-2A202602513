"""Explicit, bounded live integration checks using synthetic user data only."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from model_provider import ProviderConfig, normalize_provider


def run_live_checks(config) -> dict:
    if not config.model.live or config.model.model_name == "stub":
        raise ValueError("Live checks require an explicit real model.")
    report = {"provider": config.model.provider, "model": config.model.model_name, "mode": "live", "checks": [], "api_calls": [], "status": "running"}

    class RecordingModel:
        def __init__(self, model):
            if model is None:
                raise ValueError("Agent did not build a real model; offline fallback is forbidden here.")
            self.model = model

        def invoke(self, messages):
            started = time.monotonic()
            response = self.model.invoke(messages)
            report["api_calls"].append({
                "duration_seconds": round(time.monotonic() - started, 3),
                "usage": getattr(response, "usage_metadata", None),
            })
            return response

    def make_agent(cls, cfg):
        agent = cls(cfg, force_offline=False)
        agent.langchain_agent = RecordingModel(agent.langchain_agent)
        return agent

    def check(name, agent, user, thread, message, expected=(), forbidden=()):
        before = len(report["api_calls"])
        answer = agent.reply(user, thread, message)["answer"]
        passed = bool(answer.strip()) and len(report["api_calls"]) == before + 1
        passed = passed and all(value.casefold() in answer.casefold() for value in expected)
        passed = passed and all(value.casefold() not in answer.casefold() for value in forbidden)
        report["checks"].append({"name": name, "passed": passed, "answer": answer})
        if not passed:
            raise AssertionError(f"Live check failed: {name}")

    try:
        baseline = make_agent(BaselineAgent, config)
        check("baseline_introduction", baseline, "live_sample", "baseline_intro", "Mình tên là LanTest. Mình đang ở Huế.")
        check("baseline_within_thread", baseline, "live_sample", "baseline_intro", "Mình tên gì?", expected=("LanTest",))
        check("baseline_forgets_new_thread", baseline, "live_sample", "baseline_new", "Mình tên gì?", forbidden=("LanTest",))
        advanced = make_agent(AdvancedAgent, config)
        check("advanced_introduction", advanced, "live_sample", "advanced_intro", "Mình tên là LanTest. Mình đang ở Huế.")
        check("advanced_correction", advanced, "live_sample", "advanced_intro", "Mình đã chuyển từ Huế sang Hà Nội.")
        restarted = make_agent(AdvancedAgent, config)
        check("advanced_recall_after_restart", restarted, "live_sample", "advanced_new", "Nhắc lại tên và nơi ở hiện tại của mình.", expected=("LanTest", "Hà Nội"), forbidden=("Huế",))
        check("confidence_ignores_third_party", restarted, "live_sample", "advanced_new", "Bạn của mình tên là Nam.")
        check("confidence_keeps_user_name", restarted, "live_sample", "advanced_other", "Mình tên gì?", expected=("LanTest",), forbidden=("Nam",))
        decay_config = replace(config, memory_decay_half_life_turns=1, memory_decay_min_priority=0.3)
        decay_agent = make_agent(AdvancedAgent, decay_config)
        check("decay_introduction", decay_agent, "decay_sample", "decay_intro", "Mình tên là BinhDecay. Mình đang ở Hà Nội.")
        decay_agent.profile_store.apply_candidates("decay_sample", [])
        check("decay_omits_old_location_new_thread", decay_agent, "decay_sample", "decay_new", "Nhắc lại tên và nơi ở hiện tại của mình.", expected=("BinhDecay",), forbidden=("Hà Nội",))
        report["status"] = "passed"
    except Exception as exc:
        # Never serialize provider exception bodies: they may contain sensitive URLs.
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "status_code": getattr(exc, "status_code", getattr(exc, "code", None))}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Real model name; overrides LLM_MODEL for this run only.")
    parser.add_argument("--provider", help="Provider; defaults to LLM_PROVIDER.")
    parser.add_argument("--base-url", help="Override endpoint for this run only.")
    parser.add_argument("--api-key-env", help="Read the key from this named environment variable; never pass a key as an argument.")
    parser.add_argument("--output", type=Path, default=Path("results/live_smoke.json"))
    args = parser.parse_args()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    config = load_config()
    provider = normalize_provider(args.provider or config.model.provider)
    model_name = args.model or config.model.model_name
    if model_name == "stub":
        parser.error("Specify --model with a real model name; .env remains unchanged.")
    key_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY", "custom": "CUSTOM_API_KEY", "ollama": ""}
    key = os.getenv(args.api_key_env) if args.api_key_env else config.model.api_key if provider == config.model.provider else os.getenv(key_names[provider])
    if provider != "ollama" and not key:
        parser.error("No API key for the selected provider. Configure the corresponding key variable.")
    model = ProviderConfig(provider, model_name, 0.0, api_key=key, base_url=args.base_url or (config.model.base_url if provider == config.model.provider else None), live=True, request_timeout=20, max_retries=0, max_output_tokens=256)
    with TemporaryDirectory(prefix="live-smoke-", dir=config.state_dir) as directory:
        path = Path(directory).resolve()
        if not path.is_relative_to(config.state_dir.resolve()):
            raise ValueError("Live workspace must remain inside state_dir.")
        report = run_live_checks(replace(config, model=model, state_dir=path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"provider": provider, "model": model_name, "status": report["status"], "checks_passed": sum(c["passed"] for c in report["checks"]), "api_calls": len(report["api_calls"]), "error": report.get("error"), "report": str(args.output)}, ensure_ascii=False))
    sys.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
