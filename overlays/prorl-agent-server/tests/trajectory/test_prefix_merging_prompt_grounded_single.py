from __future__ import annotations

import sys

import pytest

if sys.version_info < (3, 10):
    pytest.skip("trajectory pydantic models require Python >= 3.10", allow_module_level=True)

import asyncio

from polar.trajectory.builder.prefix_merging import PrefixMergingBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession


def _completion(
    completion_id: str,
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    messages: list[dict[str, str]],
    assistant_content: str,
) -> CompletionRecord:
    return CompletionRecord(
        completion_id=completion_id,
        request={"messages": messages},
        response={
            "choices": [
                {
                    "input_token_ids": prompt_ids,
                    "token_ids": response_ids,
                    "message": {"role": "assistant", "content": assistant_content},
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {"token_id": token_id, "logprob": -0.1}
                            for token_id in response_ids
                        ]
                    },
                }
            ]
        },
    )


def test_prompt_grounded_single_preserves_interstitial_messages_for_num_turns(monkeypatch) -> None:
    monkeypatch.setenv("POLAR_PREFIX_MERGING_MODE", "prompt_grounded_single")
    first_messages = [{"role": "user", "content": "q"}]
    second_messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "search"},
        {"role": "tool", "content": "result"},
    ]
    session = CompletionSession(
        session_id="session-1",
        completions=[
            _completion(
                "c1",
                prompt_ids=[10, 11],
                response_ids=[20, 21],
                messages=first_messages,
                assistant_content="search",
            ),
            _completion(
                "c2",
                # Prompt suffix contains the previous assistant response plus
                # tool/user interstitial tokens.  prompt_grounded_single keeps the token
                # stream prompt-grounded and masks this suffix as context.
                prompt_ids=[10, 11, 20, 21, 30, 31],
                response_ids=[40, 41],
                messages=second_messages,
                assistant_content="answer",
            ),
        ],
    )

    trajectory = asyncio.run(PrefixMergingBuilder().build(session))

    assert trajectory.status == "COMPLETED"
    assert len(trajectory.traces) == 1
    trace = trajectory.traces[0]
    assert trace.response_ids == [20, 21, 30, 31, 40, 41]
    assert trace.loss_mask == [1, 1, 0, 0, 1, 1]
    assert [m["role"] for m in trace.response_messages] == ["assistant", "tool", "assistant"]
