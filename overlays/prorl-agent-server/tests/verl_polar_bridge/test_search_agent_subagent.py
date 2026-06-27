import sys
from types import SimpleNamespace

from verl_polar_bridge.adapter import session_result_to_verl_samples
from verl_polar_bridge.artifacts import full_trajectory_payloads
from verl_polar_bridge.search_agent.driver import (
    _compact_messages,
    _extract_tool_calls,
    _safe_to_compact_messages,
    _segment_boundary_reasons,
)


def test_extract_tool_calls_accepts_subagent_without_query():
    text = '<tool_call>{"name":"subagent","arguments":{"task":"find evidence","context":"hypothesis"}}</tool_call>'
    calls = _extract_tool_calls(text)
    assert calls == [
        {
            "name": "subagent",
            "subagent": {"task": "find evidence", "context": "hypothesis"},
        }
    ]


class _Timing:
    def model_dump(self, mode="python"):
        return {}


def _trace(group, kind, prompt, response, reward=1.0, **metadata):
    return SimpleNamespace(
        prompt_ids=prompt,
        response_ids=response,
        loss_mask=[1] * len(response),
        response_logprobs=[{"token_id": token, "logprob": -0.1} for token in response],
        prompt_messages=[],
        response_messages=[{"role": "assistant", "content": kind}],
        metadata={"merge_group_id": group, "segment_kind": kind, **metadata},
        reward=reward,
        finish_reason="stop",
    )


def test_adapter_group_stitch_splits_subagent_and_final(monkeypatch):
    monkeypatch.setenv("POLAR_STITCH_BY_MERGE_GROUP", "1")
    monkeypatch.setenv("POLAR_SEGMENT_REWARD_MODE", "prompt_grounded_split")
    # final group is append-only: second prompt is first prompt + first response + tool/report token.
    final0 = _trace("sid:main", "final", [1, 2], [10])
    final1 = _trace("sid:main", "final", [1, 2, 10, 90], [11])
    sub0 = _trace("sid:subagent:0", "subagent", [3, 4], [20])
    sub1 = _trace("sid:subagent:0", "subagent", [3, 4, 20, 91], [21])
    result = SimpleNamespace(
        trajectory=SimpleNamespace(
            traces=[final0, sub0, sub1, final1],
            metadata={"source_uid": "row-1"},
            status="COMPLETED",
            error=None,
        ),
        metadata={"source_uid": "row-1"},
        node_id="node",
        error=None,
        session_id="sid",
        status="COMPLETED",
        task_id="task",
        timing=_Timing(),
    )
    samples = session_result_to_verl_samples(result, group_index=0, trajectory_index=0, uid="row-1")
    kinds = sorted((s.metadata["polar"].get("segment_kind"), s.reward) for s in samples)
    assert len(samples) == 2
    assert [kind for kind, _ in kinds] == ["final", "subagent"]
    assert all(abs(reward - 0.5) < 1e-6 for _, reward in kinds)
    assert {s.uid for s in samples} == {"row-1"}
    assert {s.metadata["polar"]["num_segments"] for s in samples} == {2}


def test_full_trajectory_payload_groups_subagent_with_parent(monkeypatch):
    monkeypatch.setenv("POLAR_STITCH_BY_MERGE_GROUP", "1")
    monkeypatch.setenv("POLAR_SEGMENT_REWARD_MODE", "prompt_grounded_split")
    final0 = _trace("sid:main", "final", [1, 2], [10])
    final1 = _trace("sid:main", "final", [1, 2, 10, 90], [11])
    sub0 = _trace(
        "sid:subagent:0",
        "subagent",
        [3, 4],
        [20],
        parent_merge_group_id="sid:main",
        dispatch_index=0,
    )
    sub1 = _trace(
        "sid:subagent:0",
        "subagent",
        [3, 4, 20, 91],
        [21],
        parent_merge_group_id="sid:main",
        dispatch_index=0,
    )
    result = SimpleNamespace(
        trajectory=SimpleNamespace(
            traces=[final0, sub0, sub1, final1],
            metadata={"source_uid": "row-1"},
            status="COMPLETED",
            error=None,
        ),
        metadata={"source_uid": "row-1"},
        node_id="node",
        error=None,
        session_id="sid",
        status="COMPLETED",
        task_id="task",
        timing=_Timing(),
    )
    samples = session_result_to_verl_samples(result, group_index=0, trajectory_index=0, uid="row-1")
    rows = full_trajectory_payloads(samples, global_steps=1, require_subagent=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["has_subagent"] is True
    assert row["sample_count"] == 2
    assert [seg["segment_kind"] for seg in row["segments"]] == ["subagent", "final"]
    assert row["segments"][0]["parent_merge_group_id"] == "sid:main"
    assert row["segments"][1]["merge_group_id"] == "sid:main"


def test_evaluator_prefers_final_traces_for_answer_decode():
    if sys.version_info < (3, 10):
        import pytest
        pytest.skip("polar trajectory models use py310 union annotations")
    from verl_polar_bridge.search_agent import evaluator as ev

    old = ev._reward_tokenizer
    old_flag = ev._env_flag
    try:
        class Tok:
            def decode(self, ids, skip_special_tokens=True):
                return {1: "<answer>wrong</answer>", 2: "<answer>right</answer>"}.get(ids[0], "")

        ev._reward_tokenizer = lambda: Tok()
        ev._env_flag = lambda name, default=False: True if name == "POLAR_SEARCH_REWARD_SCORE_TRUNCATED" else default
        traj = SimpleNamespace(
            traces=[
                SimpleNamespace(response_ids=[1], metadata={"segment_kind": "subagent"}, response_messages=[]),
                SimpleNamespace(response_ids=[2], metadata={"segment_kind": "final"}, response_messages=[]),
            ]
        )
        text, source = ev._prediction_text_with_source(traj, {})
        assert source == "trajectory_response_ids_truncated"
        assert text == "<answer>right</answer>"
    finally:
        ev._reward_tokenizer = old
        ev._env_flag = old_flag


def test_wipe_boundary_reasons_by_turn_and_context():
    assert _segment_boundary_reasons(
        enabled=True,
        turn=3,
        max_turns=4,
        prompt_tokens=900,
        max_model_len=1000,
        ratio=0.8,
    ) == ["turn", "context"]
    assert _segment_boundary_reasons(
        enabled=True,
        turn=0,
        max_turns=4,
        prompt_tokens=100,
        max_model_len=1000,
        ratio=0.8,
    ) == []
    assert _segment_boundary_reasons(
        enabled=False,
        turn=3,
        max_turns=4,
        prompt_tokens=900,
        max_model_len=1000,
        ratio=0.8,
    ) == []


def test_wipe_safe_tail_rejects_tool_tail():
    assert _safe_to_compact_messages([{"role": "assistant", "content": "x"}]) is True
    assert _safe_to_compact_messages([{"role": "tool", "content": "result"}]) is False


def test_compact_messages_can_preserve_assistant_tail():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "old"},
        {"role": "tool", "content": "old result"},
        {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
    ]
    compacted = _compact_messages(messages, preserve_tail_messages=1)
    assert compacted[:2] == messages[:2]
    assert compacted[-1] == messages[-1]
    assert compacted[-1]["role"] == "assistant"
    assert any(m.get("content", "").startswith("[context compacted]") for m in compacted)


def test_adapter_normalizes_last_wipe_segment_to_final(monkeypatch):
    monkeypatch.setenv("POLAR_STITCH_BY_MERGE_GROUP", "1")
    monkeypatch.setenv("POLAR_SEGMENT_REWARD_MODE", "none")
    wipe0 = _trace(
        "sid:main:wipe:0",
        "wipe",
        [1, 2],
        [10],
        parent_merge_group_id="sid:main",
        merge_group_index=0,
    )
    wipe1 = _trace(
        "sid:main:wipe:1",
        "wipe",
        [1, 2, 99],
        [11],
        parent_merge_group_id="sid:main",
        merge_group_index=1,
    )
    result = SimpleNamespace(
        trajectory=SimpleNamespace(
            traces=[wipe0, wipe1],
            metadata={"source_uid": "row-1"},
            status="COMPLETED",
            error=None,
        ),
        metadata={"source_uid": "row-1"},
        node_id="node",
        error=None,
        session_id="sid",
        status="COMPLETED",
        task_id="task",
        timing=_Timing(),
    )
    samples = session_result_to_verl_samples(result, group_index=0, trajectory_index=0, uid="row-1")
    kinds = [s.metadata["polar"].get("segment_kind") for s in samples]
    assert kinds == ["wipe", "final"]
    assert samples[-1].metadata["polar"].get("is_final_segment") is True
