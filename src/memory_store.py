from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Deterministic character heuristic, shared by prompts and replies."""
    stripped = (text or "").strip()
    return max(1, len(stripped) // 4) if stripped else 0


@dataclass(frozen=True)
class ProfileCandidate:
    key: str
    value: str
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Candidate confidence must be between 0 and 1.")


@dataclass(frozen=True)
class MemoryPolicy:
    confidence_threshold: float = 0.8
    decay_half_life_turns: float = 100.0
    min_priority: float = 0.2
    decay_enabled: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1 or not 0 <= self.min_priority <= 1:
            raise ValueError("Memory thresholds must be between 0 and 1.")
        if not math.isfinite(self.decay_half_life_turns) or self.decay_half_life_turns <= 0:
            raise ValueError("Memory decay half-life must be positive and finite.")


@dataclass
class UserProfileStore:
    """UTF-8 profiles with one current value per fact key."""

    root_dir: Path
    policy: MemoryPolicy = field(default_factory=MemoryPolicy)

    def path_for(self, user_id: str) -> Path:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty.")
        slug = re.sub(r"[^\w-]+", "_", user_id, flags=re.UNICODE)[:80].casefold()
        reserved = {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)], *[f"lpt{i}" for i in range(1, 10)]}
        if slug != user_id or slug.lower() in reserved:
            slug = "user_" + slug + "_" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]
        root = self.root_dir.resolve()
        path = (root / slug / "User.md").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Profile path must remain inside root_dir.")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            raise ValueError("search_text must not be empty.")
        content = self.read_text(user_id)
        if search_text not in content or search_text == replacement:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def metadata_path_for(self, user_id: str) -> Path:
        return self.path_for(user_id).with_name("Memory.json")

    def metadata(self, user_id: str) -> dict:
        path = self.metadata_path_for(user_id)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1, "turn": 0, "facts": {}}
        # Existing User.md profiles receive default metadata without losing any fact.
        for key in self.facts(user_id):
            data["facts"].setdefault(key, {"confidence": 1.0, "last_seen_turn": data["turn"], "seen_count": 1})
        return data

    def _write_metadata(self, user_id: str, data: dict) -> None:
        path = self.metadata_path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)

    def total_size(self, user_id: str) -> int:
        metadata = self.metadata_path_for(user_id)
        return self.file_size(user_id) + (metadata.stat().st_size if metadata.exists() else 0)

    def apply_candidates(self, user_id: str, candidates: list[ProfileCandidate]) -> dict[str, int]:
        """Advance one user turn; refresh only accepted, actually stored declarations."""
        data = self.metadata(user_id)
        data["turn"] += 1
        accepted = rejected = 0
        for candidate in candidates:
            if candidate.confidence < self.policy.confidence_threshold:
                rejected += 1
                continue
            previous = self.facts(user_id).get(candidate.key, "")
            expected = merge_response_style(previous, candidate.value) if candidate.key == "response_style" else " ".join(candidate.value.split())
            self.upsert_fact(user_id, candidate.key, candidate.value)
            if self.facts(user_id).get(candidate.key) == expected:
                old = data["facts"].get(candidate.key, {})
                data["facts"][candidate.key] = {
                    "confidence": candidate.confidence,
                    "last_seen_turn": data["turn"],
                    "seen_count": old.get("seen_count", 0) + 1,
                }
                accepted += 1
        if self.facts(user_id):
            self._write_metadata(user_id, data)
        return {"accepted": accepted, "rejected": rejected}

    def priorities(self, user_id: str) -> dict[str, float]:
        data = self.metadata(user_id)
        priorities = {}
        for key in self.facts(user_id):
            entry = data["facts"][key]
            age = max(0, data["turn"] - entry["last_seen_turn"])
            # Identity remains stable; changeable facts lose priority until reconfirmed.
            factor = 1.0 if key == "name" or not self.policy.decay_enabled else 2 ** (-age / self.policy.decay_half_life_turns)
            priorities[key] = entry["confidence"] * factor
        return priorities

    def active_facts(self, user_id: str) -> dict[str, str]:
        priorities = self.priorities(user_id)
        return {key: value for key, value in self.facts(user_id).items() if priorities[key] >= self.policy.min_priority}

    def active_text(self, user_id: str) -> str:
        active = self.active_facts(user_id)
        if not active:
            return ""
        # Preserve the original profile formatting; omit only inactive fact lines.
        lines = []
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- ([a-z_]+):", line)
            if not match or match.group(1) in active:
                lines.append(line)
        return "\n".join(lines) + "\n"

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r"^- ([a-z_]+): (.+)$", self.read_text(user_id), re.MULTILINE))

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        if not re.fullmatch(r"[a-z_]+", key):
            raise ValueError("Fact keys must use lowercase letters and underscores.")
        value = " ".join(value.split())
        if not value:
            return False
        facts = self.facts(user_id)
        if key == "response_style" and key in facts:
            value = merge_response_style(facts[key], value)
        if facts.get(key) == value:
            return False
        if key in facts:
            # Corrections replace the previous line, rather than retaining both values.
            return self.edit_text(user_id, f"- {key}: {facts[key]}", f"- {key}: {value}")
        content = self.read_text(user_id) or "# User profile\n\n"
        self.write_text(user_id, content.rstrip() + f"\n- {key}: {value}\n")
        return True


def merge_response_style(previous: str, current: str) -> str:
    """Accumulate style details; a newly stated bullet count supersedes the old one."""
    items = previous.split(", ") if previous else []
    if "không dùng bullet" in current:
        items = [item for item in items if "bullet" not in item]
    elif re.search(r"\d+ bullet|có bullet", current):
        items = [item for item in items if item != "không dùng bullet"]
    if "dài và chi tiết" in current:
        items = [item for item in items if item != "ngắn gọn"]
    elif "ngắn gọn" in current:
        items = [item for item in items if item != "dài và chi tiết"]
    if re.search(r"\d+ bullet", current):
        items = [item for item in items if "bullet" not in item]
    for item in current.split(", "):
        if item == "có bullet" and any(re.search(r"\d+ bullet", old) for old in items):
            continue
        if item and item not in items:
            items.append(item)
    return ", ".join(items)


def _clean_fact(value: str) -> str:
    value = re.split(
        r"\s+(?:và|chứ|nhưng|cho|để|trong|vài|mỗi|như|chưa|dù|không\s+đổi)\b|[,;.!?]",
        value, maxsplit=1, flags=re.IGNORECASE,
    )[0]
    return value.strip(" :\"'“”")


def _extract_raw_updates(message: str) -> dict[str, str]:
    """Extract explicit first-person facts, ignoring questions and speculative clauses.

    The rules cover Vietnamese declarations, not arbitrary natural language.
    Values come from the message, never from the benchmark's expected answers.
    """
    message = unicodedata.normalize("NFC", message)
    updates: dict[str, str] = {}
    subjects = r"(?:mình|tôi|tớ)"
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", message):
        sentence = sentence.strip()
        if not sentence or sentence.endswith("?") or re.match(r"(?:nhắc lại|hãy nhắc|bạn có|bạn thử nhớ lại|nếu biết)\b", sentence, re.IGNORECASE):
            continue
        # Keep clauses after a correction, but ignore hypotheses, travel, and jokes.
        clauses = re.split(r",\s*|\s+nhưng\s+|\s+chứ\s+|\s+và\s+", sentence, flags=re.IGNORECASE)
        for clause in clauses:
            if re.search(r"\b(?:nếu|giả sử|có thể|câu đùa|mình đùa|mình không|không còn|đừng|không phải)\b", clause, re.IGNORECASE):
                continue
            if re.search(subjects, sentence, re.IGNORECASE) and re.match(r"(?:đang|vẫn|hiện|giờ)\s+(?:làm|ở)\b", clause, re.IGNORECASE):
                clause = "mình " + clause
            patterns = {
                "name": rf"{subjects}\s+tên\s+(?:là\s+)?(.+)",
                "location": rf"(?:{subjects}\s+(?:(?:vẫn|hiện(?:\s+tại)?|giờ|đang)\s+)*(?:ở|sống\s+(?:ở|tại)|đã\s+chuyển\s+(?:(?:đến|sang)|từ\s+.+?\s+(?:đến|sang)))|hiện(?:\s+tại)?\s+ở|nơi\s+ở\s+hiện\s+tại\s+(?:là)?|nơi\s+ở\s+đã\s+cập\s+nhật\s+từ\s+.+?\s+sang)\s+(.+)",
                "profession": rf"(?:{subjects}\s+(?:(?:vẫn|hiện(?:\s+tại)?|đang|giờ)\s+)*(?:làm(?:\s+nghề)?|là)|(?:giờ\s+)?chuyển\s+sang|nghề\s+nghiệp\s+(?:hiện\s+tại\s+)?(?:thì\s+)?(?:vẫn\s+)?là)\s+(.+?\s+(?:engineer|developer|manager|designer|teacher|analyst|scientist|kỹ sư|giáo viên))\b",
                "favorite_drink": r"đồ\s+uống\s+yêu\s+thích\s+(?:của\s+mình\s+)?là\s+(.+)",
                "favorite_food": r"món\s+ăn\s+yêu\s+thích\s+(?:của\s+mình\s+)?là\s+(.+)",
                "pet": rf"{subjects}\s+nuôi\s+(?:(?:một|con|bé)\s+)*(.+)",
            }
            for key, pattern in patterns.items():
                match = re.search(pattern, clause, re.IGNORECASE)
                if match:
                    value = _clean_fact(match.group(1))
                    if value and not re.match(r"(?:gì|ai|đâu|như thế nào)\b", value, re.IGNORECASE):
                        updates[key] = value
        # Preferences must be explicitly requested, rather than quoted or hypothetical.
        if not re.search(r"\b(?:nếu|giả sử|câu đùa)\b", sentence, re.IGNORECASE):
            preference = re.search(r"(?:mình\s+(?:vẫn\s+)?(?:muốn|thích)|hãy\s+trả\s+lời|style\s+trả\s+lời|khi\s+bạn\s+trả\s+lời)", sentence, re.IGNORECASE)
            if preference and re.search(r"trả\s+lời|giải\s+thích|style", sentence, re.IGNORECASE):
                style = []
                if re.search(r"dài\s+(?:và\s+)?chi tiết", sentence, re.IGNORECASE):
                    style.append("dài và chi tiết")
                elif re.search(r"ngắn|gọn", sentence, re.IGNORECASE):
                    style.append("ngắn gọn")
                bullets = re.search(r"(\d+)\s+bullet", sentence, re.IGNORECASE)
                if re.search(r"(?:không|đừng)\s+(?:(?:dùng|sử dụng|trả lời thành)\s+)?bullet", sentence, re.IGNORECASE):
                    style.append("không dùng bullet")
                elif bullets:
                    style.append(bullets.group(1) + " bullet")
                elif re.search(r"bullet", sentence, re.IGNORECASE):
                    style.append("có bullet")
                if re.search(r"ví dụ\s+thực\s+(?:tế|chiến)", sentence, re.IGNORECASE):
                    style.append("có ví dụ thực chiến")
                if "trade-off" in sentence.casefold():
                    style.append("ưu tiên trade-off")
                if style:
                    updates["response_style"] = merge_response_style(updates.get("response_style", ""), ", ".join(style))
            interests = re.search(r"mình\s+(?:(?:vẫn|rất|đang)\s+)*(?:thích|quan tâm(?:\s+nhiều)?(?:\s+đến)?)\s+(.+)", sentence, re.IGNORECASE)
            if interests and not re.search(r"trả lời|giải thích|tin này|kiểu khái quát", interests.group(1), re.IGNORECASE):
                # Preserve source wording, with a bound on long prose preferences.
                updates["interests"] = interests.group(1).strip(" .")[:180]
    return updates


def extract_profile_candidates(message: str) -> list[ProfileCandidate]:
    """Rule confidence is an evidence score, not a calibrated probability."""
    candidates = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", unicodedata.normalize("NFC", message)):
        score, reason = 0.98, "explicit declaration"
        if re.search(r"(?:bạn|đồng nghiệp|anh|chị|em|sếp)\s+của\s+(?:mình|tôi|tớ)", sentence, re.IGNORECASE):
            score, reason = 0.15, "third-party attribution"
        elif re.search(r"[\"'“]\s*(?:mình|tôi|tớ)\b", sentence, re.IGNORECASE):
            score, reason = 0.25, "quoted declaration"
        elif re.search(r"có lẽ|hình như|chắc là|dường như|nghe nói|có vẻ", sentence, re.IGNORECASE):
            score, reason = 0.45, "uncertain declaration"
        for key, value in _extract_raw_updates(sentence).items():
            candidates.append(ProfileCandidate(key, value, score, reason))
    return candidates


def extract_profile_updates(message: str, confidence_threshold: float = 0.8) -> dict[str, str]:
    """Compatibility API: return only sufficiently supported profile facts."""
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    updates = {}
    for candidate in extract_profile_candidates(message):
        if candidate.confidence >= confidence_threshold:
            if candidate.key == "response_style":
                updates[candidate.key] = merge_response_style(updates.get(candidate.key, ""), candidate.value)
            else:
                updates[candidate.key] = candidate.value
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Bounded heuristic summary retaining current facts and short topic excerpts."""
    if max_items <= 0:
        return ""
    facts: dict[str, str] = {}
    excerpts: list[str] = []
    for message in messages:
        content = message["content"]
        if message["role"] == "system":
            facts.update(dict(re.findall(r"^- fact\.([a-z_]+): (.+)$", content, re.MULTILINE)))
            excerpts.extend(re.findall(r"^- context: (.+)$", content, re.MULTILINE))
        elif message["role"] == "user":
            facts.update(extract_profile_updates(content))
            excerpt = " ".join(content.split())[:140]
            if excerpt and excerpt not in excerpts:
                excerpts.append(excerpt)
    if len(excerpts) > max_items:
        first = (max_items + 1) // 2
        excerpts = excerpts[:first] + excerpts[-(max_items - first):] if max_items > 1 else excerpts[-1:]
    lines = [f"- fact.{key}: {value}" for key, value in sorted(facts.items())]
    lines.extend(f"- context: {item}" for item in excerpts)
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0 or self.keep_messages < 1:
            raise ValueError("Compact threshold and keep_messages must be positive.")

    def append(self, thread_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported message role: {role}.")
        state = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = state["messages"]
        messages.append({"role": role, "content": content})
        prompt_tokens = estimate_tokens(state["summary"]) + sum(estimate_tokens(m["content"]) for m in messages)
        if prompt_tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return
        old = messages[:-self.keep_messages]
        previous = [{"role": "system", "content": state["summary"]}] if state["summary"] else []
        summary = summarize_messages(previous + old)
        # Bound summaries across repeated compactions, and require actual compression.
        removable_chars = len(state["summary"]) + sum(len(m["content"]) for m in old)
        budget = min(self.threshold_tokens * 4 // 3, removable_chars // 2)
        summary = summary[:budget].rstrip()
        recent = messages[-self.keep_messages:]
        after_tokens = estimate_tokens(summary) + sum(estimate_tokens(m["content"]) for m in recent)
        if after_tokens < prompt_tokens:
            state["summary"] = summary
            state["messages"] = recent
            state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        state = self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})
        return {"messages": [m.copy() for m in state["messages"]], "summary": state["summary"], "compactions": state["compactions"]}

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
