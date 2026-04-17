from __future__ import annotations

from scholartrace.services.prompt_budget import PromptBudget


def test_trim_messages_hard_case_stays_under_budget():
    budget = PromptBudget(
        model_context_tokens=500,
        response_headroom_tokens=70,
        tool_headroom_tokens=70,
        message_overhead_tokens=40,
    )
    messages = [
        {"role": "system", "content": "S" * 120},
        {"role": "user", "content": "U" * 400},
        {"role": "assistant", "content": "A" * 500},
    ]

    trimmed = budget.trim_messages(messages, preserve=1)

    assert trimmed[0]["role"] == "system"
    assert len(trimmed) == 2
    assert budget.estimate_messages(trimmed) <= budget.max_input_tokens
