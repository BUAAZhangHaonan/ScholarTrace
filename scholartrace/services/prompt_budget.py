"""Conservative per-request prompt budgeting helpers for GLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any


@dataclass(frozen=True)
class PromptBudget:
    """Estimate prompt size conservatively and keep one request under budget."""

    model_context_tokens: int = 128_000
    response_headroom_tokens: int = 16_000
    tool_headroom_tokens: int = 8_000
    message_overhead_tokens: int = 24

    @property
    def max_input_tokens(self) -> int:
        return max(
            1,
            self.model_context_tokens
            - self.response_headroom_tokens
            - self.tool_headroom_tokens,
        )

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, ceil(len(text.encode("utf-8")) / 2))

    def estimate_messages(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            total += self._message_noncontent_cost(message)
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.estimate_text(content)
            else:
                total += self.estimate_text(str(content))
        return total

    def _message_noncontent_cost(self, message: dict[str, Any]) -> int:
        return self.message_overhead_tokens + self.estimate_text(
            str(message.get("role", ""))
        )

    def truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if self.estimate_text(text) <= max_tokens:
            return text

        truncated = text
        while truncated and self.estimate_text(truncated) > max_tokens:
            new_length = max(1, int(len(truncated) * 0.8))
            if new_length >= len(truncated):
                new_length = len(truncated) - 1
            truncated = truncated[:new_length]
        return truncated

    def pack_items(
        self,
        items: list[str],
        *,
        fixed_messages: list[dict[str, Any]] | None = None,
        prefix: str = "",
        separator: str = "\n",
        suffix: str = "",
    ) -> list[list[str]]:
        """Greedily pack item strings into request-sized batches."""
        fixed_messages = fixed_messages or []
        fixed_cost = self.estimate_messages(fixed_messages)
        prefix_cost = self.estimate_text(prefix)
        suffix_cost = self.estimate_text(suffix)
        if fixed_cost + prefix_cost + suffix_cost >= self.max_input_tokens:
            raise ValueError("Fixed prompt content exceeds the input budget")

        separator_cost = self.estimate_text(separator)
        available_for_items = (
            self.max_input_tokens - fixed_cost - prefix_cost - suffix_cost
        )

        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for item in items:
            allowed_item_tokens = max(1, available_for_items - separator_cost)
            candidate_item = self.truncate_text(item, allowed_item_tokens)
            candidate_cost = self.estimate_text(candidate_item)
            if current_batch:
                candidate_cost += separator_cost

            if current_batch and current_tokens + candidate_cost > available_for_items:
                batches.append(current_batch)
                current_batch = [candidate_item]
                current_tokens = self.estimate_text(candidate_item)
            else:
                current_batch.append(candidate_item)
                current_tokens += candidate_cost

        if current_batch:
            batches.append(current_batch)
        return batches

    def trim_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        preserve: int = 1,
    ) -> list[dict[str, Any]]:
        """Keep the newest conversation messages while staying under budget."""
        if preserve >= len(messages):
            return messages

        fixed_messages = messages[:preserve]
        rolling_messages = messages[preserve:]
        trimmed = fixed_messages + rolling_messages

        while (
            len(trimmed) > preserve + 1
            and self.estimate_messages(trimmed) > self.max_input_tokens
        ):
            trimmed = trimmed[:preserve] + trimmed[preserve + 1 :]

        if self.estimate_messages(trimmed) <= self.max_input_tokens:
            return trimmed

        mutable = [dict(message) for message in trimmed]
        for index in range(len(mutable) - 1, -1, -1):
            if self.estimate_messages(mutable) <= self.max_input_tokens:
                break
            reserved = sum(
                self._message_noncontent_cost(message)
                + (
                    self.estimate_text(str(message.get("content", "")))
                    if idx != index
                    else 0
                )
                for idx, message in enumerate(mutable)
            )
            allowed = max(0, self.max_input_tokens - reserved)
            mutable[index]["content"] = self.truncate_text(
                str(mutable[index].get("content", "")),
                allowed,
            )

        while len(mutable) > preserve and self.estimate_messages(mutable) > self.max_input_tokens:
            mutable = mutable[:preserve] + mutable[preserve + 1 :]

        if self.estimate_messages(mutable) <= self.max_input_tokens:
            return mutable

        for index in range(len(mutable) - 1, -1, -1):
            mutable[index]["content"] = ""
            if self.estimate_messages(mutable) <= self.max_input_tokens:
                return mutable

        return mutable[:1] if mutable else []


DEFAULT_PROMPT_BUDGET = PromptBudget()
