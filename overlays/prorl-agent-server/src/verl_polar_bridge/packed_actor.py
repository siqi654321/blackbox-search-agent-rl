"""Helpers for packed-variable actor dry-run/update paths.

The trainer patch imports this module only when packed-variable dry-run/update
is enabled.  It converts the JSON-like payload emitted by
``variable_pack.py`` into jagged TensorDicts consumed by the VERL engine worker
path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class PackedActorSample:
    sample_index: int
    sample_uid: str
    group_uid: str
    rollout_uid: str
    source_uid: str
    parent_sample_uid: str
    segment_group_id: str
    segment_kind: str | None
    segment_weight: float
    input_ids: list[int]
    loss_mask_full: list[int]
    rollout_log_probs_full: list[float]
    reward: float
    token_length: int
    prompt_length: int
    response_length: int
    num_turns: int
    trainable_tokens: int
    parent_sample_trainable_tokens: int
    parent_slot_padding: bool = False


def packed_variable_payload_to_actor_samples(payload: dict[str, Any]) -> list[PackedActorSample]:
    records = payload.get("records") or []
    samples: list[PackedActorSample] = []
    for idx, record in enumerate(records):
        input_ids = [int(v) for v in (record.get("input_ids") or [])]
        loss_mask = [1 if int(v) else 0 for v in (record.get("loss_mask_full") or [])]
        rollout_log_probs = [float(v) for v in (record.get("rollout_log_probs_full") or [])]
        if not input_ids:
            continue
        if len(loss_mask) != len(input_ids):
            raise ValueError(f"packed record {idx} loss_mask_full length mismatch")
        if len(rollout_log_probs) != len(input_ids):
            raise ValueError(f"packed record {idx} rollout_log_probs_full length mismatch")
        prompt_length = int(record.get("prompt_length", 0) or 0)
        response_length = int(record.get("response_length", 0) or 0)
        if prompt_length < 0 or response_length < 0:
            raise ValueError(f"packed record {idx} has negative prompt/response length")
        if prompt_length + response_length != len(input_ids):
            # Backward-compatible fallback for older payloads that did not
            # carry explicit prompt/response lengths.  The first trainable token
            # is in the response, but some response tokens may be masked; use
            # the loss-mask boundary as a best-effort split only for such old
            # payloads.  New payloads from variable_pack.py carry exact lengths.
            first_trainable = next((i for i, v in enumerate(loss_mask) if int(v) != 0), len(input_ids))
            prompt_length = first_trainable
            response_length = len(input_ids) - prompt_length
        samples.append(
            PackedActorSample(
                sample_index=int(record.get("sample_index", idx)),
                sample_uid=str(record.get("sample_uid", idx)),
                group_uid=str(record.get("group_uid", record.get("sample_uid", idx))),
                rollout_uid=str(record.get("rollout_uid", record.get("sample_uid", idx))),
                source_uid=str(record.get("source_uid", record.get("sample_uid", idx))),
                parent_sample_uid=str(
                    record.get(
                        "parent_sample_uid",
                        f"{record.get('group_uid', record.get('sample_uid', idx))}:trajectory:{record.get('trajectory_index', idx)}",
                    )
                ),
                segment_group_id=str(record.get("segment_group_id", record.get("sample_uid", idx))),
                segment_kind=record.get("segment_kind"),
                segment_weight=float(record.get("segment_weight", 1.0) or 1.0),
                input_ids=input_ids,
                loss_mask_full=loss_mask,
                rollout_log_probs_full=rollout_log_probs,
                reward=float(record.get("reward", 0.0) or 0.0),
                token_length=len(input_ids),
                prompt_length=prompt_length,
                response_length=response_length,
                num_turns=int(record.get("num_turns", 0) or 0),
                trainable_tokens=int(record.get("trainable_tokens", sum(loss_mask)) or 0),
                parent_sample_trainable_tokens=int(
                    record.get("parent_sample_trainable_tokens", record.get("trainable_tokens", sum(loss_mask))) or 0
                ),
                parent_slot_padding=bool(record.get("parent_slot_padding", False)),
            )
        )
    if not samples:
        raise ValueError("packed-variable payload has no actor samples")
    return samples


def partition_actor_samples_by_tokens(samples: list[PackedActorSample], *, dp_size: int) -> list[list[int]]:
    """Greedy longest-first token balancing over DP ranks."""

    dp_size = max(1, int(dp_size or 1))
    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    loads = [0 for _ in range(dp_size)]
    for idx, sample in sorted(enumerate(samples), key=lambda item: item[1].token_length, reverse=True):
        rank = min(range(dp_size), key=lambda r: loads[r])
        partitions[rank].append(idx)
        loads[rank] += int(sample.token_length)

    # Real GRPO long runs should have at least one sample per DP rank.  Avoid
    # duplicating samples silently because that would change gradients/rewards.
    if any(not part for part in partitions):
        raise ValueError(
            f"packed-variable batch has fewer samples than actor DP ranks: "
            f"samples={len(samples)}, dp_size={dp_size}"
        )
    return partitions


def pad_actor_samples_to_divisor(
    samples: list[PackedActorSample],
    *,
    divisor: int,
) -> tuple[list[PackedActorSample], int]:
    """Pad actor samples by duplicating front rows, matching DataProto padding.

    ``verl.protocol.pad_dataproto_to_divisor`` appends duplicates from the
    beginning of the batch.  The packed update path intentionally mirrors that
    historical true-long behavior for now, so duplicated rows still participate
    in GRPO statistics and PPO loss unless a future explicit neutralization
    switch is added.
    """

    divisor = max(1, int(divisor or 1))
    if not samples:
        raise ValueError("cannot pad empty packed actor sample list")
    remainder = len(samples) % divisor
    if remainder == 0:
        return list(samples), 0
    pad_size = divisor - remainder
    repeats = [samples[i % len(samples)] for i in range(pad_size)]
    return list(samples) + repeats, pad_size


def partition_actor_samples_equal_count_row_order(
    samples: list[PackedActorSample],
    *,
    dp_size: int,
) -> list[list[int]]:
    """Partition padded samples by original row order with equal DP counts.

    This matches VERL's fixed DataProto update semantics more closely than
    token-balanced assignment: after pad_dataproto_to_divisor, worker ranks see
    contiguous row-order shards and then split those rows into PPO mini-batches.
    Keep token balancing as an opt-in performance mode because PPO/KL metrics
    can differ when the mini-batch order changes during an update step.
    """

    dp_size = max(1, int(dp_size or 1))
    if not samples:
        raise ValueError("cannot partition empty packed actor sample list")
    if len(samples) % dp_size != 0:
        raise ValueError(
            f"packed-variable row-order partition requires samples % dp_size == 0: "
            f"samples={len(samples)}, dp_size={dp_size}"
        )
    target_count = len(samples) // dp_size
    if target_count <= 0:
        raise ValueError(
            f"packed-variable batch has fewer samples than actor DP ranks: "
            f"samples={len(samples)}, dp_size={dp_size}"
        )
    return [
        list(range(rank * target_count, (rank + 1) * target_count))
        for rank in range(dp_size)
    ]


def partition_actor_samples_equal_count_by_tokens(
    samples: list[PackedActorSample],
    *,
    dp_size: int,
) -> list[list[int]]:
    """Partition samples with equal row counts, token balance as secondary.

    Distributed packed actor update executes one PPO mini-batch loop per DP
    rank.  Unequal local row counts can make one rank run a different number of
    iterations/collectives, which eventually manifests as an NCCL timeout.  This
    helper therefore makes equal sample count a hard invariant and greedily
    balances token load only within that constraint.
    """

    dp_size = max(1, int(dp_size or 1))
    if not samples:
        raise ValueError("cannot partition empty packed actor sample list")
    if len(samples) % dp_size != 0:
        raise ValueError(
            f"packed-variable equal-count partition requires samples % dp_size == 0: "
            f"samples={len(samples)}, dp_size={dp_size}"
        )
    target_count = len(samples) // dp_size
    if target_count <= 0:
        raise ValueError(
            f"packed-variable batch has fewer samples than actor DP ranks: "
            f"samples={len(samples)}, dp_size={dp_size}"
        )

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    loads = [0 for _ in range(dp_size)]
    for idx, sample in sorted(enumerate(samples), key=lambda item: item[1].token_length, reverse=True):
        candidates = [rank for rank in range(dp_size) if len(partitions[rank]) < target_count]
        if not candidates:
            raise RuntimeError("packed-variable partition internal error: no rank has remaining capacity")
        rank = min(candidates, key=lambda r: (loads[r], len(partitions[r]), r))
        partitions[rank].append(idx)
        loads[rank] += int(sample.token_length)

    if any(len(part) != target_count for part in partitions):
        raise RuntimeError(
            "packed-variable equal-count partition failed: "
            f"target={target_count}, counts={[len(part) for part in partitions]}"
        )
    return partitions


def group_actor_samples_by_parent(samples: list[PackedActorSample]) -> list[list[PackedActorSample]]:
    """Group contiguous segment rows that belong to the same parent rollout.

    Polar emits fanout segments for a parent rollout contiguously.  Preserve
    row order so GRPO groups remain aligned with the padded DataProto path.
    """

    groups: list[list[PackedActorSample]] = []
    by_parent: dict[str, list[PackedActorSample]] = {}
    for sample in samples:
        key = str(sample.parent_sample_uid)
        if key not in by_parent:
            by_parent[key] = []
            groups.append(by_parent[key])
        by_parent[key].append(sample)
    return groups


def pad_parent_sample_groups_to_divisor(
    parent_groups: list[list[PackedActorSample]],
    *,
    divisor: int,
) -> tuple[list[list[PackedActorSample]], int]:
    """Pad by duplicating whole parent rollouts, not individual segments."""

    divisor = max(1, int(divisor or 1))
    if not parent_groups:
        raise ValueError("cannot pad empty packed parent sample group list")
    remainder = len(parent_groups) % divisor
    if remainder == 0:
        return [list(group) for group in parent_groups], 0
    pad_size = divisor - remainder
    out = [list(group) for group in parent_groups]
    for pad_idx in range(pad_size):
        source_group = parent_groups[pad_idx % len(parent_groups)]
        parent_uid = f"{source_group[0].parent_sample_uid}::__parent_pad_{pad_idx}"
        duplicated = [
            replace(sample, parent_sample_uid=parent_uid, parent_slot_padding=True)
            for sample in source_group
        ]
        out.append(duplicated)
    return out, pad_size


def partition_parent_sample_groups_row_order(
    parent_groups: list[list[PackedActorSample]],
    *,
    dp_size: int,
) -> list[list[list[PackedActorSample]]]:
    """Partition parent rollout slots contiguously with equal parent counts."""

    dp_size = max(1, int(dp_size or 1))
    if not parent_groups:
        raise ValueError("cannot partition empty packed parent sample group list")
    if len(parent_groups) % dp_size != 0:
        raise ValueError(
            f"packed-variable parent row-order partition requires parent_groups % dp_size == 0: "
            f"parent_groups={len(parent_groups)}, dp_size={dp_size}"
        )
    target_count = len(parent_groups) // dp_size
    if target_count <= 0:
        raise ValueError(
            f"packed-variable parent batch has fewer parent samples than actor DP ranks: "
            f"parent_groups={len(parent_groups)}, dp_size={dp_size}"
        )
    return [
        [list(group) for group in parent_groups[rank * target_count : (rank + 1) * target_count]]
        for rank in range(dp_size)
    ]


def flatten_parent_sample_groups(parent_groups: list[list[PackedActorSample]]) -> list[PackedActorSample]:
    return [sample for group in parent_groups for sample in group]


def pad_parent_minibatches_to_equal_row_count(
    shard_parent_minibatches: list[list[list[list[PackedActorSample]]]],
) -> tuple[list[list[list[PackedActorSample]]], int, int]:
    """Flatten parent mini-batches and pad segment rows to equal row count.

    VERL's worker DataLoader splits TensorDicts by row count.  Parent fanout
    makes each parent mini-batch contain a variable number of segment rows, so
    append zero-loss duplicate rows until every local parent mini-batch has the
    same row count.  These padding rows are only for row-boundary alignment;
    their loss mask is all zero and they are not included in rollout metrics.
    """

    flat: list[list[list[PackedActorSample]]] = [
        [flatten_parent_sample_groups(parent_minibatch) for parent_minibatch in shard]
        for shard in shard_parent_minibatches
    ]
    max_rows = max((len(minibatch) for shard in flat for minibatch in shard), default=0)
    if max_rows <= 0:
        raise ValueError("cannot pad empty parent-aware mini-batches")
    row_pad_count = 0
    next_pad_index = -1
    for shard in flat:
        for minibatch in shard:
            if not minibatch:
                raise ValueError("empty parent-aware mini-batch")
            while len(minibatch) < max_rows:
                minibatch.append(_zero_loss_padding_sample(minibatch[0], next_pad_index))
                next_pad_index -= 1
                row_pad_count += 1
    return flat, max_rows, row_pad_count


def _zero_loss_padding_sample(source: PackedActorSample, sample_index: int) -> PackedActorSample:
    return replace(
        source,
        sample_index=int(sample_index),
        sample_uid=f"{source.sample_uid}::__row_pad_{abs(sample_index)}",
        rollout_uid=f"{source.rollout_uid}::__row_pad_{abs(sample_index)}",
        source_uid=f"{source.source_uid}::__row_pad_{abs(sample_index)}",
        parent_sample_uid=f"{source.parent_sample_uid}::__row_pad_{abs(sample_index)}",
        segment_group_id=f"{source.segment_group_id}::__row_pad_{abs(sample_index)}",
        segment_kind="row_pad",
        segment_weight=0.0,
        loss_mask_full=[0 for _ in source.loss_mask_full],
        rollout_log_probs_full=[0.0 for _ in source.rollout_log_probs_full],
        reward=0.0,
        trainable_tokens=0,
        parent_sample_trainable_tokens=0,
        parent_slot_padding=True,
    )


def actor_samples_to_nested_tensordict(
    samples: list[PackedActorSample],
    *,
    temperature: float,
    calculate_entropy: bool = False,
) -> Any:
    """Build a padding-free TensorDict for actor/ref logprob inference."""

    td = _base_nested_tensordict(samples)
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(
        td,
        temperature=float(temperature),
        compute_loss=False,
        calculate_entropy=bool(calculate_entropy),
        # Actor parity shards are token-balanced, so per-DP sample counts are not
        # guaranteed to be divisible by rollout.log_prob_micro_batch_size_per_gpu.
        # Force micro-batch size 1 for this diagnostics-only forward to avoid
        # padding/duplication while keeping the path padding-free.
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
    )
    return td


def actor_samples_to_training_tensordict(
    samples: list[PackedActorSample],
    *,
    temperature: float,
    advantages_by_sample: dict[int, float],
    calculate_entropy: bool,
    compute_loss: bool = True,
    legacy_loss_scale: bool = False,
    global_mini_batch_size: int | None = None,
    parent_sample_loss: bool = False,
) -> Any:
    """Build a padding-free TensorDict for real PPO update."""

    import torch
    from verl.utils import tensordict_utils as tu

    td = _base_nested_tensordict(samples)
    old_rows: list[torch.Tensor] = []
    adv_rows: list[torch.Tensor] = []
    weight_rows: list[torch.Tensor] = []
    loss_scale_rows: list[torch.Tensor] = []
    parent_sample_loss = bool(parent_sample_loss)
    use_segment_weight_loss = _env_flag("POLAR_PACKED_SEGMENT_WEIGHT_LOSS", default=True) and not parent_sample_loss
    denom = max(1, int(global_mini_batch_size or len(samples) or 1))
    for sample in samples:
        old_rows.append(torch.tensor(sample.rollout_log_probs_full, dtype=torch.float32))
        effective_weight = float(sample.segment_weight) if use_segment_weight_loss else 1.0
        adv = float(advantages_by_sample.get(sample.sample_index, 0.0)) * effective_weight
        adv_rows.append(torch.full((sample.token_length,), adv, dtype=torch.float32))
        weight_rows.append(torch.full((sample.token_length,), effective_weight, dtype=torch.float32))
        # New-engine packed PPO normally computes a global token-mean loss for
        # each mini-batch.  The legacy padded DataProto path scales each
        # dynamic micro-batch by row count / ppo_mini_batch_size, which makes
        # the PG objective much closer to a per-sample mean.  Store the
        # row-local inverse trainable-token count here; ppo_loss multiplies it
        # by the current global mini-batch token count / global mini-batch rows
        # before calling the native policy-loss function, so the native
        # token-mean aggregator yields a per-sample mean while preserving
        # rollout-IS weights.
        trainable_tokens = max(1, int(sample.trainable_tokens or 0))
        loss_denom_tokens = max(
            1,
            int(sample.parent_sample_trainable_tokens or trainable_tokens) if parent_sample_loss else trainable_tokens,
        )
        scale = (1.0 / float(loss_denom_tokens)) if legacy_loss_scale else 1.0
        loss_scale_rows.append(torch.full((sample.token_length,), scale, dtype=torch.float32))
    td["old_log_probs"] = torch.nested.as_nested_tensor(old_rows, layout=torch.jagged).contiguous()
    td["advantages"] = torch.nested.as_nested_tensor(adv_rows, layout=torch.jagged).contiguous()
    td["rollout_log_probs"] = td["old_log_probs"].clone()
    td["rollout_is_weights"] = torch.nested.as_nested_tensor(weight_rows, layout=torch.jagged).contiguous()
    td["polar_packed_loss_scale"] = torch.nested.as_nested_tensor(loss_scale_rows, layout=torch.jagged).contiguous()
    tu.assign_non_tensor(
        td,
        temperature=float(temperature),
        compute_loss=bool(compute_loss),
        calculate_entropy=bool(calculate_entropy),
        global_batch_size=len(samples),
        polar_packed_legacy_loss_scale=bool(legacy_loss_scale),
        polar_packed_global_mini_batch_size=int(denom),
        polar_packed_parent_sample_loss=bool(parent_sample_loss),
    )
    return td


def assign_old_log_probs_from_nested_output(td: Any, log_probs: Any) -> int:
    """Assign recomputed actor log-probs as the packed PPO old-policy anchor.

    The worker returns full-sequence model-position log-probs: row position
    ``p`` predicts token ``p + 1``.  The fixed padded DataProto path stores
    ``old_log_probs`` token-aligned with response tokens.  Keep the same
    contract for packed data by shifting the full-sequence rows right:
    token position ``t`` stores the log-prob produced at model position
    ``t - 1``.  The first slot is ignored by the packed response mask.
    """

    _validate_nested_row_lengths(td, log_probs, label="old_log_probs")
    td["old_log_probs"] = _model_log_probs_to_token_aligned_full_rows(log_probs).contiguous()
    return _trainable_token_count(td)


def assign_ref_log_probs_from_nested_output(td: Any, log_probs: Any) -> int:
    """Assign padding-free nested ref log-probs for actor KL loss.

    Match fixed DataProto semantics: ``ref_log_prob`` is token-aligned with the
    action tokens, not model-position aligned.
    """

    _validate_nested_row_lengths(td, log_probs, label="ref_log_prob")
    td["ref_log_prob"] = _model_log_probs_to_token_aligned_full_rows(log_probs).contiguous()
    return _trainable_token_count(td)


def compute_group_grpo_advantages(
    samples: list[PackedActorSample],
    *,
    norm_by_std: bool = True,
    parent_level: bool = False,
) -> dict[int, float]:
    """Compute GRPO-style advantages, optionally at parent-rollout level.

    With subagent fanout, one parent rollout can produce multiple segment rows
    that share ``group_uid``.  ``parent_level=True`` first collapses those
    sibling segments into a single logical rollout for GRPO statistics, then
    broadcasts the parent advantage back to all segment rows.
    """

    from collections import defaultdict

    grouped: dict[str, list[PackedActorSample]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.parent_sample_uid if parent_level else sample.group_uid)].append(sample)

    out: dict[int, float] = {}
    if not parent_level:
        for group in grouped.values():
            rewards = [float(sample.reward) for sample in group]
            mean = sum(rewards) / max(len(rewards), 1)
            if norm_by_std and len(rewards) > 1:
                # Match VERL's GRPO path, which uses torch.std(...), i.e. unbiased
                # sample std by default, before adding epsilon in the denominator.
                var = sum((r - mean) ** 2 for r in rewards) / max(len(rewards) - 1, 1)
                std = var ** 0.5
            else:
                std = 1.0
            denom = std + 1e-6 if norm_by_std else 1.0
            for sample in group:
                out[int(sample.sample_index)] = (float(sample.reward) - mean) / denom
        return out

    parent_rows: list[tuple[list[PackedActorSample], float]] = []
    for _parent_uid, group in grouped.items():
        if not group:
            continue
        # True fanout has multiple segment ids under one parent rollout and is
        # one logical GRPO sample.  Pure prompt_grounded_single pad duplicates can share
        # the same parent uid and segment id; keep those as duplicate logical
        # rows to match the padded DataProto path.
        if len({str(sample.segment_group_id) for sample in group}) <= 1:
            for sample in group:
                parent_rows.append(([sample], float(sample.reward)))
        else:
            parent_rows.append((group, float(group[0].reward)))

    grpo_groups: dict[str, list[tuple[list[PackedActorSample], float]]] = defaultdict(list)
    for group, reward in parent_rows:
        if not group:
            continue
        grpo_groups[str(group[0].group_uid)].append((group, reward))

    for group in grpo_groups.values():
        rewards = [reward for _segments, reward in group]
        mean = sum(rewards) / max(len(rewards), 1)
        if norm_by_std and len(rewards) > 1:
            var = sum((r - mean) ** 2 for r in rewards) / max(len(rewards) - 1, 1)
            std = var ** 0.5
        else:
            std = 1.0
        denom = std + 1e-6 if norm_by_std else 1.0
        for segments, reward in group:
            adv = (float(reward) - mean) / denom
            for sample in segments:
                out[int(sample.sample_index)] = adv
    return out


def parent_sample_loss_denominators(samples: list[PackedActorSample]) -> dict[int, int]:
    """Return trainable-token denominators for parent-rollout token means."""

    return {
        int(sample.sample_index): max(
            1,
            int(sample.parent_sample_trainable_tokens or sample.trainable_tokens or 0),
        )
        for sample in samples
    }


def parent_sample_count(samples: list[PackedActorSample]) -> int:
    return len({str(sample.parent_sample_uid) for sample in samples})


def packed_training_payload_metrics(
    samples: list[PackedActorSample],
    *,
    advantages_by_sample: dict[int, float],
    prefix: str,
) -> dict[str, float]:
    advantages = [float(advantages_by_sample.get(sample.sample_index, 0.0)) for sample in samples]
    trainable_tokens = sum(int(sample.trainable_tokens) for sample in samples)
    token_count = sum(int(sample.token_length) for sample in samples)
    group_count = len({sample.group_uid for sample in samples})
    segment_group_count = len({sample.segment_group_id for sample in samples})
    parent_count = parent_sample_count(samples)
    from collections import defaultdict

    parent_segment_ids: dict[str, set[str]] = defaultdict(set)
    parent_samples: dict[str, list[PackedActorSample]] = defaultdict(list)
    grpo_group_sizes: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        parent_segment_ids[str(sample.parent_sample_uid)].add(str(sample.segment_group_id))
        if not sample.parent_slot_padding:
            parent_samples[str(sample.parent_sample_uid)].append(sample)
            grpo_group_sizes[str(sample.group_uid)].add(str(sample.parent_sample_uid))
    fanout_parent_count = sum(1 for segment_ids in parent_segment_ids.values() if len(segment_ids) > 1)

    loss_weight_sums: list[float] = []
    fanout_counts: list[float] = []
    parent_trainable_sums: list[float] = []
    parent_declared_denoms: list[float] = []
    advantage_unique_counts: list[float] = []
    group_uid_unique_counts: list[float] = []
    for group in parent_samples.values():
        if not group:
            continue
        declared = max(1, max(int(sample.parent_sample_trainable_tokens or 0) for sample in group))
        actual = sum(int(sample.trainable_tokens or 0) for sample in group)
        loss_weight_sums.append(float(actual) / float(declared))
        fanout_counts.append(float(len({str(sample.segment_group_id) for sample in group})))
        parent_trainable_sums.append(float(actual))
        parent_declared_denoms.append(float(declared))
        advantage_unique_counts.append(
            float(len({round(float(advantages_by_sample.get(sample.sample_index, 0.0)), 12) for sample in group}))
        )
        group_uid_unique_counts.append(float(len({str(sample.group_uid) for sample in group})))
    group_sizes = [float(len(values)) for values in grpo_group_sizes.values()]

    out = {
        f"{prefix}/num_samples": float(len(samples)),
        f"{prefix}/num_groups": float(group_count),
        f"{prefix}/num_segment_groups": float(segment_group_count),
        f"{prefix}/num_parent_samples": float(parent_count),
        f"{prefix}/num_fanout_parent_samples": float(fanout_parent_count),
        f"{prefix}/token_count": float(token_count),
        f"{prefix}/trainable_tokens": float(trainable_tokens),
        f"{prefix}/advantage_mean": float(sum(advantages) / max(len(advantages), 1)),
        f"{prefix}/advantage_abs_mean": float(sum(abs(v) for v in advantages) / max(len(advantages), 1)),
        f"{prefix}/segment_weight_mean": float(sum(sample.segment_weight for sample in samples) / max(len(samples), 1)),
        f"{prefix}/packed_parent/fanout_count_mean": _mean(fanout_counts),
        f"{prefix}/packed_parent/fanout_count_max": max(fanout_counts, default=0.0),
        f"{prefix}/packed_parent/trainable_tokens_sum_mean": _mean(parent_trainable_sums),
        f"{prefix}/packed_parent/trainable_tokens_sum_max": max(parent_trainable_sums, default=0.0),
        f"{prefix}/packed_parent/declared_loss_denom_mean": _mean(parent_declared_denoms),
        f"{prefix}/packed_parent/loss_weight_sum_mean": _mean(loss_weight_sums),
        f"{prefix}/packed_parent/loss_weight_sum_min": min(loss_weight_sums, default=0.0),
        f"{prefix}/packed_parent/loss_weight_sum_max": max(loss_weight_sums, default=0.0),
        f"{prefix}/packed_parent/loss_weight_bad_count": float(
            sum(1 for value in loss_weight_sums if abs(value - 1.0) > 1e-3)
        ),
        f"{prefix}/packed_parent/advantage_unique_count_max": max(advantage_unique_counts, default=0.0),
        f"{prefix}/packed_parent/advantage_mismatch_count": float(
            sum(1 for value in advantage_unique_counts if value > 1.0)
        ),
        f"{prefix}/packed_parent/group_uid_unique_count_max": max(group_uid_unique_counts, default=0.0),
        f"{prefix}/packed_parent/group_uid_mismatch_count": float(
            sum(1 for value in group_uid_unique_counts if value > 1.0)
        ),
        f"{prefix}/packed_grpo/group_size_mean": _mean(group_sizes),
        f"{prefix}/packed_grpo/group_size_min": min(group_sizes, default=0.0),
        f"{prefix}/packed_grpo/group_size_max": max(group_sizes, default=0.0),
        f"{prefix}/packed_grpo/singleton_group_count": float(sum(1 for value in group_sizes if value <= 1.0)),
    }
    return out


def packed_actor_data_metrics(
    samples: list[PackedActorSample],
    *,
    prefix: str,
    max_prompt_length: int | None = None,
    max_response_length: int | None = None,
    advantages_by_sample: dict[int, float] | None = None,
) -> dict[str, float]:
    """Compute packed-native score/reward/length metrics.

    These metrics intentionally use the packed update samples after any
    pad-compatible row duplication.  With ``pad_compat_original_pad_path=1``,
    duplicated pad rows therefore affect the means exactly like the historical
    fixed DataProto pad path.
    """

    if not samples:
        return {f"{prefix}/data_metrics_native": 0.0}

    rewards = [float(sample.reward) for sample in samples]
    prompt_lengths = [float(max(0, int(sample.prompt_length))) for sample in samples]
    response_lengths = [float(max(0, int(sample.response_length))) for sample in samples]
    non_aborted_response_lengths = [v for v in response_lengths if v > 0.0]
    configured_max_response_length = float(
        max(0, int(max_response_length)) if max_response_length is not None else max(response_lengths, default=0.0)
    )
    configured_max_prompt_length = float(
        max(0, int(max_prompt_length)) if max_prompt_length is not None else max(prompt_lengths, default=0.0)
    )
    advantages_by_sample = advantages_by_sample or {}
    valid_advantages: list[float] = []
    valid_returns: list[float] = []
    use_segment_weight_loss = _env_flag("POLAR_PACKED_SEGMENT_WEIGHT_LOSS", default=True) and not _env_flag(
        "POLAR_PACKED_PARENT_SAMPLE_LOSS", default=False
    )
    for sample in samples:
        effective_weight = float(sample.segment_weight) if use_segment_weight_loss else 1.0
        advantage = float(advantages_by_sample.get(int(sample.sample_index), 0.0)) * effective_weight
        # Match fixed compute_data_metrics: masked_select over the response mask
        # repeats the scalar advantage once per trainable token.
        valid_advantages.extend([advantage] * max(0, int(sample.trainable_tokens)))
        valid_returns.extend([advantage] * max(0, int(sample.trainable_tokens)))

    out: dict[str, float] = {
        f"{prefix}/data_metrics_native": 1.0,
        f"{prefix}/data_metrics_sample_count": float(len(samples)),
        # SearchR1/GRPO outcome reward path has score == reward when
        # algorithm.use_kl_in_reward=False, which is the true-long alignment
        # configuration used by this packed update path.
        "critic/score/mean": _mean(rewards),
        "critic/score/max": max(rewards),
        "critic/score/min": min(rewards),
        "critic/rewards/mean": _mean(rewards),
        "critic/rewards/max": max(rewards),
        "critic/rewards/min": min(rewards),
        "response_length/mean": _mean(response_lengths),
        "response_length/max": max(response_lengths),
        "response_length/min": min(response_lengths),
        "response_length/clip_ratio": (
            _mean([1.0 if v == configured_max_response_length else 0.0 for v in response_lengths])
            if configured_max_response_length > 0.0
            else 0.0
        ),
        "response/aborted_ratio": _mean([1.0 if v == 0.0 else 0.0 for v in response_lengths]),
        "prompt_length/mean": _mean(prompt_lengths),
        "prompt_length/max": max(prompt_lengths),
        "prompt_length/min": min(prompt_lengths),
        "prompt_length/clip_ratio": (
            _mean([1.0 if v == configured_max_prompt_length else 0.0 for v in prompt_lengths])
            if configured_max_prompt_length > 0.0
            else 0.0
        ),
    }
    if valid_advantages:
        out.update(
            {
                "critic/advantages/mean": _mean(valid_advantages),
                "critic/advantages/max": max(valid_advantages),
                "critic/advantages/min": min(valid_advantages),
                "critic/returns/mean": _mean(valid_returns),
                "critic/returns/max": max(valid_returns),
                "critic/returns/min": min(valid_returns),
            }
        )
    if non_aborted_response_lengths:
        out.update(
            {
                "response_length_non_aborted/mean": _mean(non_aborted_response_lengths),
                "response_length_non_aborted/max": max(non_aborted_response_lengths),
                "response_length_non_aborted/min": min(non_aborted_response_lengths),
                "response_length_non_aborted/clip_ratio": (
                    _mean([1.0 if v == configured_max_response_length else 0.0 for v in non_aborted_response_lengths])
                    if configured_max_response_length > 0.0
                    else 0.0
                ),
            }
        )
    else:
        out.update(
            {
                "response_length_non_aborted/mean": 0.0,
                "response_length_non_aborted/max": 0.0,
                "response_length_non_aborted/min": 0.0,
                "response_length_non_aborted/clip_ratio": 0.0,
            }
        )
    num_turns = [float(sample.num_turns) for sample in samples]
    if num_turns:
        out.update(
            {
                "num_turns/min": min(num_turns),
                "num_turns/max": max(num_turns),
                "num_turns/mean": _mean(num_turns),
            }
        )
    return out


def packed_actor_logprob_parity_metrics(
    samples: list[PackedActorSample],
    log_probs: Any,
    *,
    prefix: str,
) -> dict[str, float]:
    """Compare full-sequence nested actor logprobs with rollout logprobs."""

    import torch

    rows = list(log_probs.unbind()) if isinstance(log_probs, torch.Tensor) and log_probs.is_nested else list(log_probs)
    if len(rows) != len(samples):
        raise ValueError(f"packed logprob row mismatch: {len(rows)} != {len(samples)}")
    actor_vals: list[torch.Tensor] = []
    rollout_vals: list[torch.Tensor] = []
    for sample, row in zip(samples, rows, strict=True):
        row = row.detach().to(torch.float32)
        if int(row.numel()) != int(sample.token_length):
            raise ValueError(f"packed logprob token mismatch sample={sample.sample_index}")
        mask = torch.tensor(sample.loss_mask_full, dtype=torch.bool, device=row.device)
        if int(mask.sum().item()) <= 0:
            continue
        # model output at position p predicts token p+1; trainable token at
        # index t is therefore compared to log_probs[t-1].  The first token is
        # never trainable because prompt mask is zero.
        shifted_mask = mask[1:]
        actor_vals.append(row[:-1][shifted_mask])
        rollout = torch.tensor(sample.rollout_log_probs_full, dtype=torch.float32, device=row.device)
        rollout_vals.append(rollout[1:][shifted_mask])
    if not actor_vals:
        return {f"{prefix}/valid": 0.0, f"{prefix}/tokens": 0.0}
    actor = torch.cat(actor_vals)
    rollout = torch.cat(rollout_vals)
    diff = actor - rollout
    actor_probs = torch.exp(actor)
    rollout_probs = torch.exp(rollout)
    if actor.numel() > 1 and torch.std(actor_probs) > 0 and torch.std(rollout_probs) > 0:
        corr = torch.corrcoef(torch.stack([actor_probs, rollout_probs], dim=0))[0][1]
        corr_value = float(corr.detach().item())
    else:
        corr_value = 1.0
    ratio = torch.exp(diff).clamp(min=1e-30, max=1e30)
    out = {
        f"{prefix}/valid": 1.0,
        f"{prefix}/tokens": float(actor.numel()),
        f"{prefix}/logprob_abs_diff_mean": float(diff.abs().mean().detach().item()),
        f"{prefix}/logprob_abs_diff_max": float(diff.abs().max().detach().item()),
        f"{prefix}/actor_probs_pearson_corr": corr_value,
        f"{prefix}/rollout_corr/k3_kl": float(((ratio - 1.0) - torch.log(ratio)).mean().detach().item()),
        f"{prefix}/rollout_corr/chi2_token": float(((ratio - 1.0) ** 2).mean().detach().item()),
    }
    out.update(_segment_logprob_breakdown(samples, rows, prefix=prefix))
    return out


def _segment_logprob_breakdown(samples: list[PackedActorSample], rows: list[Any], *, prefix: str) -> dict[str, float]:
    import torch

    grouped: dict[str, list[torch.Tensor]] = {}
    for sample, row in zip(samples, rows, strict=True):
        row = row.detach().to(torch.float32)
        mask = torch.tensor(sample.loss_mask_full, dtype=torch.bool, device=row.device)
        if int(row.numel()) <= 1 or int(mask.sum().item()) <= 0:
            continue
        actor = row[:-1][mask[1:]]
        rollout = torch.tensor(sample.rollout_log_probs_full, dtype=torch.float32, device=row.device)[1:][mask[1:]]
        if int(actor.numel()) <= 0:
            continue
        diff = actor - rollout
        kind = _normalized_segment_kind(sample)
        grouped.setdefault(kind, []).append(diff)
        trainable_positions = [idx for idx, value in enumerate(sample.loss_mask_full) if int(value)]
        if len(trainable_positions) == int(diff.numel()):
            pos_tensor = torch.tensor(trainable_positions, dtype=torch.long, device=diff.device)
            first = int(trainable_positions[0])
            last = int(trainable_positions[-1])
            buckets = {
                "near_start": pos_tensor < first + 64,
                "near_end": (last - pos_tensor) < 64,
            }
            middle = ~(buckets["near_start"] | buckets["near_end"])
            buckets["middle"] = middle
            for bucket, bucket_mask in buckets.items():
                if bool(bucket_mask.any().item()):
                    grouped.setdefault(f"{kind}/{bucket}", []).append(diff[bucket_mask])
    return _ratio_breakdown_metrics(grouped, prefix=prefix)


def _ratio_breakdown_metrics(grouped: dict[str, list[Any]], *, prefix: str) -> dict[str, float]:
    import torch

    out: dict[str, float] = {}
    for key, chunks in grouped.items():
        if not chunks:
            continue
        diff = torch.cat(chunks)
        ratio = torch.exp(diff).clamp(min=1e-30, max=1e30)
        out[f"{prefix}/rollout_corr/{key}/tokens"] = float(diff.numel())
        out[f"{prefix}/rollout_corr/{key}/k3_kl"] = float(((ratio - 1.0) - torch.log(ratio)).mean().detach().item())
        out[f"{prefix}/rollout_corr/{key}/chi2_token"] = float(((ratio - 1.0) ** 2).mean().detach().item())
        out[f"{prefix}/rollout_corr/{key}/ratio_fraction_high"] = float(
            (ratio > 2.0).to(torch.float32).mean().detach().item()
        )
        out[f"{prefix}/rollout_corr/{key}/ratio_fraction_low"] = float(
            (ratio < 0.5).to(torch.float32).mean().detach().item()
        )
    return out


def _normalized_segment_kind(sample: PackedActorSample) -> str:
    kind = str(sample.segment_kind or "unknown").strip().lower() or "unknown"
    if kind not in {"final", "subagent", "wipe", "main", "row_pad"}:
        kind = "other"
    return kind


def apply_smoke_reward_diversity(
    samples: list[PackedActorSample],
    *,
    prefix: str,
) -> tuple[list[PackedActorSample], dict[str, float]]:
    """Optional smoke-only reward perturbation for update-path debugging."""

    mutated: list[PackedActorSample] = []
    for idx, sample in enumerate(samples):
        mutated.append(
            PackedActorSample(
                **{
                    **sample.__dict__,
                    "reward": sample.reward + (0.01 if idx % 2 else -0.01),
                }
            )
        )
    return mutated, {f"{prefix}/smoke_reward_diversity_applied": 1.0}


def _base_nested_tensordict(samples: list[PackedActorSample]) -> Any:
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    input_rows = [torch.tensor(sample.input_ids, dtype=torch.long) for sample in samples]
    position_rows = [torch.arange(sample.token_length, dtype=torch.long) for sample in samples]
    mask_rows = [torch.ones(sample.token_length, dtype=torch.long) for sample in samples]
    loss_rows = [torch.tensor(sample.loss_mask_full, dtype=torch.long) for sample in samples]
    td = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(input_rows, layout=torch.jagged).contiguous(),
            "position_ids": torch.nested.as_nested_tensor(position_rows, layout=torch.jagged).contiguous(),
            "attention_mask": torch.nested.as_nested_tensor(mask_rows, layout=torch.jagged).contiguous(),
            "loss_mask": torch.nested.as_nested_tensor(loss_rows, layout=torch.jagged).contiguous(),
        },
        batch_size=[len(samples)],
    )
    tu.assign_non_tensor(td, pad_mode=DatasetPadMode.NO_PADDING)
    return td


def concat_training_tensordicts(tds: list[Any]) -> Any:
    """Concatenate per-mini-batch packed TensorDicts along row dimension."""

    if not tds:
        raise ValueError("cannot concatenate empty packed TensorDict list")
    if len(tds) == 1:
        return tds[0]

    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    tensor_keys = [
        "input_ids",
        "position_ids",
        "attention_mask",
        "loss_mask",
        "old_log_probs",
        "advantages",
        "rollout_log_probs",
        "rollout_is_weights",
        "polar_packed_loss_scale",
    ]
    rows_by_key: dict[str, list[torch.Tensor]] = {key: [] for key in tensor_keys if key in tds[0].keys()}
    for td in tds:
        for key in rows_by_key:
            rows_by_key[key].extend(list(td[key].unbind()))
    merged = TensorDict(
        {
            key: torch.nested.as_nested_tensor(rows, layout=torch.jagged).contiguous()
            for key, rows in rows_by_key.items()
        },
        batch_size=[len(rows_by_key["input_ids"])],
    )
    tu.assign_non_tensor(merged, pad_mode=DatasetPadMode.NO_PADDING)
    return merged


def _validate_nested_row_lengths(td: Any, values: Any, *, label: str) -> None:
    import torch

    input_ids = td["input_ids"]
    if not isinstance(values, torch.Tensor) or not values.is_nested:
        raise ValueError(f"packed {label} must be a nested/jagged torch.Tensor")
    input_rows = list(input_ids.unbind())
    value_rows = list(values.unbind())
    if len(input_rows) != len(value_rows):
        raise ValueError(f"packed {label} row mismatch: {len(value_rows)} != {len(input_rows)}")
    for idx, (input_row, value_row) in enumerate(zip(input_rows, value_rows, strict=True)):
        if int(input_row.numel()) != int(value_row.numel()):
            raise ValueError(
                f"packed {label} token length mismatch at row {idx}: "
                f"{int(value_row.numel())} != {int(input_row.numel())}"
            )


def _trainable_token_count(td: Any) -> int:
    loss_mask = td["loss_mask"]
    values = loss_mask.values() if getattr(loss_mask, "is_nested", False) else loss_mask
    return int(values.sum().detach().item())


def _model_log_probs_to_token_aligned_full_rows(log_probs: Any) -> Any:
    import torch

    rows = []
    for row in log_probs.unbind():
        aligned = torch.zeros_like(row)
        if int(row.numel()) > 1:
            aligned[1:] = row[:-1]
        rows.append(aligned)
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
