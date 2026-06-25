"""Variable-length / packed handoff for Polar -> VERL training.

This module is intentionally independent from VERL.  The rollout manager can
attach the returned payload to ``DataProto.meta_info`` while the current fixed
DataProto path remains available as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from verl_polar_bridge.adapter import VerlPolarSample


@dataclass(frozen=True)
class PackedVariableConfig:
    """Small env-backed config for packed-variable handoff."""

    enabled: bool
    max_pack_tokens: int = 65536


def resolve_packed_variable_config() -> PackedVariableConfig:
    """Resolve packed-variable settings from environment variables.

    The payload is emitted when explicitly enabled or when either packed dry-run
    or packed actor update is requested.  This keeps default long runs unchanged.
    """

    enabled = (
        _env_flag("POLAR_PACKED_VARIABLE_ENABLE")
        or _env_flag("POLAR_PACKED_VARIABLE_ACTOR_DRY_RUN")
        or _env_flag("POLAR_PACKED_VARIABLE_DRY_RUN")
        or _env_flag("POLAR_PACKED_VARIABLE_ACTOR_UPDATE")
    )
    return PackedVariableConfig(
        enabled=enabled,
        max_pack_tokens=max(1, _env_int("POLAR_PACKED_VARIABLE_PACK_MAX_TOKENS", 65536)),
    )


def build_packed_variable_payload(
    samples: list[VerlPolarSample],
    *,
    prompt_length: int,
    response_length: int,
    max_pack_tokens: int = 65536,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build a variable-length packed payload plus logging metrics."""

    records = [_sample_to_record(idx, sample) for idx, sample in enumerate(samples)]
    packs = _greedy_pack(records, max_pack_tokens=max_pack_tokens)
    total_tokens = sum(int(record["token_length"]) for record in records)
    trainable_tokens = sum(int(record["trainable_tokens"]) for record in records)
    fixed_tokens = int(len(records) * (int(prompt_length) + int(response_length)))
    pad_tokens_avoided = max(0, fixed_tokens - total_tokens)
    max_seq_len = max((int(record["token_length"]) for record in records), default=0)
    max_trainable = max((int(record["trainable_tokens"]) for record in records), default=0)

    payload = {
        "version": 1,
        "packing": {
            "strategy": "greedy_ordered",
            "max_pack_tokens": int(max_pack_tokens),
        },
        "fixed_shape": {
            "prompt_length": int(prompt_length),
            "response_length": int(response_length),
            "row_tokens": int(prompt_length) + int(response_length),
        },
        "records": records,
        "packs": packs,
    }
    metrics = {
        "polar/packed_variable/enabled": 1.0,
        "polar/packed_variable/sample_count": float(len(records)),
        "polar/packed_variable/pack_count": float(len(packs)),
        "polar/packed_variable/total_tokens": float(total_tokens),
        "polar/packed_variable/trainable_tokens": float(trainable_tokens),
        "polar/packed_variable/max_seq_len": float(max_seq_len),
        "polar/packed_variable/max_trainable_tokens": float(max_trainable),
        "polar/packed_variable/fixed_tokens": float(fixed_tokens),
        "polar/packed_variable/pad_tokens_avoided": float(pad_tokens_avoided),
        "polar/packed_variable/compression_ratio_vs_fixed": float(total_tokens / fixed_tokens) if fixed_tokens > 0 else 0.0,
        "polar/packed_variable/max_pack_tokens": float(max_pack_tokens),
    }
    return payload, metrics


def compact_samples_for_fixed_output(samples: list[VerlPolarSample]) -> list[VerlPolarSample]:
    """Return one representative fixed DataProto row per source uid.

    Packed actor update consumes all trainable samples from the side-channel
    payload.  The fixed DataProto output is still needed for trainer plumbing,
    but it does not need to fan out to every segment.  Keeping one row per
    source uid prevents fixed padding from multiplying by segment count.
    """

    selected: list[VerlPolarSample] = []
    seen: set[str] = set()
    for sample in samples:
        key = _sample_uid(sample, _source_uid(sample))
        if key in seen:
            continue
        selected.append(_fixed_placeholder_copy(sample))
        seen.add(key)
    return selected or samples[:1]


def _fixed_placeholder_copy(sample: VerlPolarSample) -> VerlPolarSample:
    """A tiny non-trainable fixed row preserving uid/provenance for plumbing."""

    metadata = sample.metadata
    try:
        from copy import deepcopy

        metadata = deepcopy(sample.metadata)
        polar = metadata.setdefault("polar", {})
        if isinstance(polar, dict):
            polar["packed_variable_fixed_placeholder"] = True
            polar["raw_prompt_len_before_packed_placeholder"] = len(sample.prompt_ids)
            polar["raw_response_len_before_packed_placeholder"] = len(sample.response_ids)
    except Exception:
        pass
    source_uid = _source_uid(sample)
    fixed_uid = _sample_uid(sample, source_uid)
    if isinstance(metadata, dict):
        polar = metadata.setdefault("polar", {})
        if isinstance(polar, dict):
            polar["sample_uid"] = fixed_uid
            polar["source_uid"] = fixed_uid
    return VerlPolarSample(
        uid=sample.uid,
        group_index=sample.group_index,
        trajectory_index=sample.trajectory_index,
        trace_index=sample.trace_index,
        prompt_ids=[int(sample.prompt_ids[0]) if sample.prompt_ids else 0],
        response_ids=[int(sample.response_ids[0]) if sample.response_ids else 0],
        response_mask=[0],
        rollout_log_probs=[0.0],
        reward=float(sample.reward),
        status=sample.status,
        prompt=sample.prompt,
        response="",
        metadata=metadata,
        remove_sample=False,
    )


def _sample_to_record(sample_index: int, sample: VerlPolarSample) -> dict[str, Any]:
    prompt_ids = [int(v) for v in (sample.prompt_ids or [])]
    response_ids = [int(v) for v in (sample.response_ids or [])]
    response_mask = [1 if int(v) else 0 for v in (sample.response_mask or [])]
    rollout_log_probs = [float(v) for v in (sample.rollout_log_probs or [])]
    if len(response_mask) != len(response_ids):
        raise ValueError(
            f"sample {sample_index} response_mask length mismatch: "
            f"{len(response_mask)} != {len(response_ids)}"
        )
    if len(rollout_log_probs) != len(response_ids):
        raise ValueError(
            f"sample {sample_index} rollout_log_probs length mismatch: "
            f"{len(rollout_log_probs)} != {len(response_ids)}"
        )

    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    polar = polar if isinstance(polar, dict) else {}
    trace_meta = polar.get("trace_metadata") if isinstance(polar.get("trace_metadata"), dict) else {}
    segment_weight = _float_value(
        polar.get("segment_weight", trace_meta.get("segment_weight", 1.0)),
        default=1.0,
    )
    segment_group_id = (
        polar.get("merge_group_id")
        or trace_meta.get("merge_group_id")
        or polar.get("segment_group_id")
        or trace_meta.get("segment_group_id")
        or f"{_source_uid(sample)}:0"
    )
    segment_kind = polar.get("segment_kind") or trace_meta.get("segment_kind") or trace_meta.get("segment_type")
    source_uid = _source_uid(sample)
    rollout_uid = _sample_uid(sample, source_uid)
    group_uid = _grpo_group_uid(sample, rollout_uid)
    input_ids = prompt_ids + response_ids
    loss_mask_full = [0] * len(prompt_ids) + response_mask
    rollout_log_probs_full = [0.0] * len(prompt_ids) + rollout_log_probs
    trainable_tokens = int(sum(response_mask))
    parent_sample_trainable_tokens = int(
        _float_value(polar.get("parent_sample_trainable_tokens", trainable_tokens), default=float(trainable_tokens))
    )
    if parent_sample_trainable_tokens <= 0:
        parent_sample_trainable_tokens = trainable_tokens
    parent_sample_uid = polar.get("parent_sample_uid") or polar.get("session_id") or f"{group_uid}:trajectory:{sample.trajectory_index}"
    return {
        "sample_index": int(sample_index),
        "sample_uid": str(rollout_uid),
        "source_uid": source_uid,
        "parent_sample_uid": str(parent_sample_uid),
        "group_uid": str(group_uid),
        "rollout_uid": str(rollout_uid),
        "segment_group_id": str(segment_group_id),
        "segment_kind": None if segment_kind is None else str(segment_kind),
        "segment_weight": float(segment_weight),
        "group_index": int(sample.group_index),
        "trajectory_index": int(sample.trajectory_index),
        "trace_index": int(sample.trace_index),
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "input_ids": input_ids,
        "loss_mask_full": loss_mask_full,
        "rollout_log_probs_full": rollout_log_probs_full,
        "reward": float(sample.reward),
        "status": str(getattr(sample.status, "value", sample.status)),
        "remove_sample": bool(sample.remove_sample),
        "token_length": int(len(input_ids)),
        "prompt_length": int(len(prompt_ids)),
        "response_length": int(len(response_ids)),
        "trainable_tokens": trainable_tokens,
        "parent_sample_trainable_tokens": int(parent_sample_trainable_tokens),
        "num_turns": _packed_num_turns(sample),
        "metadata": sample.metadata,
    }


def _greedy_pack(records: list[dict[str, Any]], *, max_pack_tokens: int) -> list[dict[str, Any]]:
    packs: list[dict[str, Any]] = []
    current: list[int] = []
    current_tokens = 0
    for idx, record in enumerate(records):
        length = int(record["token_length"])
        if current and current_tokens + length > max_pack_tokens:
            packs.append(
                {
                    "pack_index": len(packs),
                    "sample_indices": current,
                    "token_count": int(current_tokens),
                }
            )
            current = []
            current_tokens = 0
        current.append(idx)
        current_tokens += length
    if current:
        packs.append(
            {
                "pack_index": len(packs),
                "sample_indices": current,
                "token_count": int(current_tokens),
            }
        )
    return packs


def _sample_uid(sample: VerlPolarSample, source_uid: str) -> str:
    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    if isinstance(polar, dict):
        value = polar.get("sample_uid") or polar.get("group_id")
        if value is not None:
            return str(value)
    return str(source_uid)


def _grpo_group_uid(sample: VerlPolarSample, rollout_uid: str) -> str:
    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    if isinstance(polar, dict):
        value = polar.get("grpo_group_uid")
        if value is not None:
            return str(value)
    return str(rollout_uid)


def _source_uid(sample: VerlPolarSample) -> str:
    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    if isinstance(polar, dict):
        value = polar.get("source_uid")
        if value is not None:
            return str(value)
    return str(sample.uid)


def _float_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _packed_num_turns(sample: VerlPolarSample) -> int:
    """Match dataproto._num_turns for packed-native metrics."""

    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    driver_num_turns = _packed_driver_num_turns(polar)
    if driver_num_turns is not None:
        return driver_num_turns
    trace_debug = polar.get("trace_debug", {}) if isinstance(polar, dict) else {}
    response_messages = trace_debug.get("response_messages", []) if isinstance(trace_debug, dict) else []
    if isinstance(response_messages, list):
        assistant_turns = 0
        tool_or_user_blocks = 0
        in_tool_or_user_block = False
        for msg in response_messages:
            role = str(msg.get("role", "")) if isinstance(msg, dict) else ""
            if role == "assistant":
                assistant_turns += 1
                in_tool_or_user_block = False
            elif role in {"tool", "user"}:
                if not in_tool_or_user_block:
                    tool_or_user_blocks += 1
                in_tool_or_user_block = True
            else:
                in_tool_or_user_block = False
        if assistant_turns or tool_or_user_blocks:
            return assistant_turns + tool_or_user_blocks + 1
        return 1 + len(response_messages)
    return 0


def _packed_driver_num_turns(polar: dict[str, Any]) -> int | None:
    if not isinstance(polar, dict):
        return None
    kind = str(polar.get("segment_kind") or polar.get("segment_type") or "").strip().lower()
    if kind in {"subagent", "wipe"}:
        return None
    evaluation = (polar.get("trajectory_metadata") or {}).get("evaluation") or {}
    if not isinstance(evaluation, dict):
        return None
    raw = evaluation.get("driver_num_turns")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)
