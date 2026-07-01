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


def test_prompt_grounded_single_groups_by_merge_group_without_adapter_stitch(monkeypatch) -> None:
    monkeypatch.setenv("POLAR_PREFIX_MERGING_MODE", "prompt_grounded_single")
    session = CompletionSession(
        session_id="session-1",
        completions=[
            _completion(
                "main-1",
                prompt_ids=[1, 2],
                response_ids=[10],
                messages=[{"role": "user", "content": "q"}],
                assistant_content="main search",
            ).model_copy(update={"metadata": {"merge_group_id": "sid:main", "segment_kind": "final"}}),
            _completion(
                "sub-1",
                prompt_ids=[7, 8],
                response_ids=[20],
                messages=[{"role": "user", "content": "sub q"}],
                assistant_content="sub search",
            ).model_copy(update={"metadata": {"merge_group_id": "sid:subagent:0", "segment_kind": "subagent"}}),
            _completion(
                "sub-2",
                prompt_ids=[7, 8, 20, 90],
                response_ids=[21],
                messages=[
                    {"role": "user", "content": "sub q"},
                    {"role": "assistant", "content": "sub search"},
                    {"role": "tool", "content": "sub result"},
                ],
                assistant_content="sub answer",
            ).model_copy(update={"metadata": {"merge_group_id": "sid:subagent:0", "segment_kind": "subagent"}}),
            _completion(
                "main-2",
                prompt_ids=[1, 2, 10, 91],
                response_ids=[11],
                messages=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "main search"},
                    {"role": "tool", "content": "main result"},
                ],
                assistant_content="main answer",
            ).model_copy(update={"metadata": {"merge_group_id": "sid:main", "segment_kind": "final"}}),
        ],
    )

    trajectory = asyncio.run(PrefixMergingBuilder().build(session))

    assert trajectory.status == "COMPLETED"
    assert trajectory.metadata["builder"] == "prefix_merging_prompt_grounded_single"
    assert trajectory.metadata["trace_count"] == 2
    assert trajectory.metadata["prompt_grounded_single_segment_grouping"] == 1
    by_group = {trace.metadata["merge_group_id"]: trace for trace in trajectory.traces}
    assert by_group["sid:main"].response_ids == [10, 91, 11]
    assert by_group["sid:main"].loss_mask == [1, 0, 1]
    assert by_group["sid:subagent:0"].response_ids == [20, 90, 21]
    assert by_group["sid:subagent:0"].loss_mask == [1, 0, 1]


def test_prompt_grounded_single_wipe_groups_emit_one_trace_per_segment(monkeypatch) -> None:
    monkeypatch.setenv("POLAR_PREFIX_MERGING_MODE", "prompt_grounded_single")
    session = CompletionSession(
        session_id="session-1",
        completions=[
            _completion(
                "wipe-0-a",
                prompt_ids=[1, 2],
                response_ids=[10],
                messages=[{"role": "user", "content": "q"}],
                assistant_content="search",
            ).model_copy(update={"metadata": {"merge_group_id": "sid:main:wipe:0", "segment_kind": "wipe"}}),
            _completion(
                "wipe-0-b",
                prompt_ids=[1, 2, 10, 90],
                response_ids=[11],
                messages=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "search"},
                    {"role": "tool", "content": "result"},
                ],
                assistant_content="more",
            ).model_copy(update={"metadata": {"merge_group_id": "sid:main:wipe:0", "segment_kind": "wipe"}}),
            _completion(
                "wipe-1-a",
                prompt_ids=[1, 2, 99],
                response_ids=[30],
                messages=[
                    {"role": "user", "content": "q"},
                    {"role": "user", "content": "[context compacted]"},
                ],
                assistant_content="answer",
            ).model_copy(update={"metadata": {"merge_group_id": "sid:main:wipe:1", "segment_kind": "wipe"}}),
        ],
    )

    trajectory = asyncio.run(PrefixMergingBuilder().build(session))

    assert trajectory.status == "COMPLETED"
    assert trajectory.metadata["trace_count"] == 2
    by_group = {trace.metadata["merge_group_id"]: trace for trace in trajectory.traces}
    assert by_group["sid:main:wipe:0"].response_ids == [10, 90, 11]
    assert by_group["sid:main:wipe:0"].loss_mask == [1, 0, 1]
    assert by_group["sid:main:wipe:1"].response_ids == [30]
    assert by_group["sid:main:wipe:1"].loss_mask == [1]
