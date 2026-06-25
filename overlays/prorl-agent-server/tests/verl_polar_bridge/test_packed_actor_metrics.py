from verl_polar_bridge.packed_actor import (
    PackedActorSample,
    compute_group_grpo_advantages,
    packed_training_payload_metrics,
)


def _sample(idx, *, group="g", parent="p", segment="s", tokens=2, reward=1.0):
    return PackedActorSample(
        sample_index=idx,
        sample_uid=f"rollout-{idx}",
        group_uid=group,
        rollout_uid=f"rollout-{idx}",
        source_uid=f"source-{idx}",
        parent_sample_uid=parent,
        segment_group_id=segment,
        segment_kind="subagent",
        segment_weight=0.5,
        input_ids=[1] * (tokens + 1),
        loss_mask_full=[0] + [1] * tokens,
        rollout_log_probs_full=[0.0] * (tokens + 1),
        reward=reward,
        token_length=tokens + 1,
        prompt_length=1,
        response_length=tokens,
        num_turns=1,
        trainable_tokens=tokens,
        parent_sample_trainable_tokens=4,
    )


def test_packed_training_metrics_report_parent_weight_conservation():
    samples = [
        _sample(0, parent="p0", segment="p0:a", tokens=2, reward=1.0),
        _sample(1, parent="p0", segment="p0:b", tokens=2, reward=1.0),
        _sample(2, parent="p1", segment="p1:a", tokens=4, reward=0.0),
    ]
    advantages = compute_group_grpo_advantages(samples, parent_level=True)
    metrics = packed_training_payload_metrics(samples, advantages_by_sample=advantages, prefix="x")

    assert metrics["x/packed_parent/loss_weight_bad_count"] == 0.0
    assert metrics["x/packed_parent/loss_weight_sum_mean"] == 1.0
    assert metrics["x/packed_parent/advantage_mismatch_count"] == 0.0
    assert metrics["x/packed_parent/group_uid_mismatch_count"] == 0.0
    assert metrics["x/packed_grpo/group_size_max"] == 2.0
