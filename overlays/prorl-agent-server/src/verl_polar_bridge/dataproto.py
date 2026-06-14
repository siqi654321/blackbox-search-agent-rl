"""Build VERL ``DataProto`` batches from Polar rollout samples."""

from __future__ import annotations

import os
from typing import Any

from verl_polar_bridge.adapter import VerlPolarSample
from verl_polar_bridge.debug_utils import debug_print, env_flag, env_int, messages_summary, token_preview


def samples_to_tensor_dicts(
    samples: list[VerlPolarSample],
    *,
    prompt_length: int,
    response_length: int,
    pad_token_id: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return ``(tensors, non_tensors, meta_info)`` ready for ``DataProto``.

    This helper intentionally has no hard dependency on VERL itself, which keeps
    unit tests light.  ``samples_to_dataproto`` wraps it with VERL's
    ``DataProto.from_dict`` when running inside a VERL environment.
    """
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if response_length <= 0:
        raise ValueError("response_length must be positive")
    if not samples:
        raise ValueError("samples must be non-empty")

    import numpy as np

    prompt_rows: list[list[int]] = []
    response_rows: list[list[int]] = []
    response_attention_rows: list[list[int]] = []
    response_mask_rows: list[list[int]] = []
    rollout_logprob_rows: list[list[float]] = []
    rm_score_rows: list[list[float]] = []

    for sample in samples:
        prompt_ids = _fit_left(sample.prompt_ids, prompt_length, pad_token_id)
        response_ids, actual_response_len = _fit_right_with_length(sample.response_ids, response_length, pad_token_id)
        loss_mask = _fit_right(sample.response_mask, response_length, 0)[:actual_response_len] + [0] * (
            response_length - actual_response_len
        )
        logprobs = _fit_right(sample.rollout_log_probs, response_length, 0.0)[:actual_response_len] + [0.0] * (
            response_length - actual_response_len
        )
        response_attention = [1] * actual_response_len + [0] * (response_length - actual_response_len)
        rm_scores = [0.0] * response_length
        if actual_response_len > 0 and any(loss_mask):
            rm_scores[actual_response_len - 1] = float(sample.reward)

        prompt_rows.append(prompt_ids)
        response_rows.append(response_ids)
        response_attention_rows.append(response_attention)
        response_mask_rows.append(loss_mask)
        rollout_logprob_rows.append(logprobs)
        rm_score_rows.append(rm_scores)

    if env_flag("POLAR_DATAPROTO_DEBUG", default=False):
        debug_print(
            "POLAR_DATAPROTO_DEBUG",
            {
                "event": "samples_to_tensor_dicts",
                "sample_count": len(samples),
                "configured_prompt_length": prompt_length,
                "configured_response_length": response_length,
                "raw_prompt_lens": [len(sample.prompt_ids) for sample in samples],
                "raw_response_lens": [len(sample.response_ids) for sample in samples],
                "raw_loss_tokens": [sum(int(v) for v in sample.response_mask) for sample in samples],
                "raw_logprob_lens": [len(sample.rollout_log_probs) for sample in samples],
                "raw_rewards": [float(sample.reward) for sample in samples],
                "raw_status": [str(getattr(sample.status, "value", sample.status)) for sample in samples],
                "fitted_prompt_lens": [sum(1 for token in row if token != int(pad_token_id)) for row in prompt_rows],
                "fitted_response_lens": [sum(row) for row in response_attention_rows],
                "fitted_loss_tokens": [sum(int(v) for v in row) for row in response_mask_rows],
                "first_sample": _debug_sample_summary(samples[0]) if samples else None,
                "samples": _debug_sample_rows(samples, prompt_rows, response_rows, response_attention_rows, response_mask_rows, rollout_logprob_rows),
            },
        )

    prompt_attention_rows = [[1 if token != int(pad_token_id) else 0 for token in row] for row in prompt_rows]
    attention_rows = [p + r for p, r in zip(prompt_attention_rows, response_attention_rows)]
    input_rows = [p + r for p, r in zip(prompt_rows, response_rows)]
    position_rows = _compute_position_rows(attention_rows)

    tensors = _tensorize(
        {
            "prompts": prompt_rows,
            "responses": response_rows,
            "response_mask": response_mask_rows,
            "input_ids": input_rows,
            "attention_mask": attention_rows,
            "position_ids": position_rows,
            "rollout_log_probs": rollout_logprob_rows,
            "rm_scores": rm_score_rows,
        }
    )
    # For trainer-side dynamic-history alignment, rollout rows must carry the
    # original VERL row uid.  Some Polar internals also expose a dataset-level
    # ``source_uid`` in metadata; keep that only in polar_metadata/provenance.
    source_uids = [str(sample.uid) for sample in samples]
    sample_uids = [_sample_uid(sample, source_uid) for sample, source_uid in zip(samples, source_uids)]
    non_tensors = {
        # Advantage grouping should follow the original VERL rollout row, not a
        # per-trace Polar id.  Store trace/session provenance in polar_metadata.
        "uid": np.array(sample_uids, dtype=object),
        "source_uid": np.array(source_uids, dtype=object),
        "__num_turns__": np.array([_num_turns(sample) for sample in samples], dtype=np.int32),
        "polar_metadata": np.array([sample.metadata.get("polar", {}) for sample in samples], dtype=object),
        "polar_status": np.array([getattr(sample.status, "value", str(sample.status)) for sample in samples], dtype=object),
        "polar_group_index": np.array([sample.group_index for sample in samples], dtype=np.int32),
        "polar_trajectory_index": np.array([sample.trajectory_index for sample in samples], dtype=np.int32),
        "polar_trace_index": np.array([sample.trace_index for sample in samples], dtype=np.int32),
        "polar_remove_sample": np.array([bool(sample.remove_sample) for sample in samples], dtype=object),
        "raw_prompt": np.array([sample.prompt for sample in samples], dtype=object),
        "polar_response_text": np.array([sample.response for sample in samples], dtype=object),
        # VERL's RayPPOTrainer unconditionally iterates this key when building
        # image_seqlens, even for text-only batches. Native AgentLoopManager only
        # adds it for multimodal samples, but that currently trips a KeyError in
        # this trainer branch. Provide empty dicts for text-only Polar rollouts.
        "multi_modal_inputs": np.array([{} for _ in samples], dtype=object),
    }
    meta_info = {
        # Keep provenance in non_tensor_batch for debugging/artifacts, but do
        # not expose it as reward_extra_info. VERL validation metrics reduce
        # every reward_extra_key with numpy mean/std unless it is a string;
        # dict-valued polar_metadata therefore crashes val_only. Numeric reward
        # details should be added explicitly by reward code, not by the rollout
        # adapter.
        "reward_extra_keys": [],
        "metrics": [],
        "polar_dynamic_history": True,
    }
    return tensors, non_tensors, meta_info


def _debug_sample_rows(
    samples: list[VerlPolarSample],
    prompt_rows: list[list[int]],
    response_rows: list[list[int]],
    response_attention_rows: list[list[int]],
    response_mask_rows: list[list[int]],
    rollout_logprob_rows: list[list[float]],
) -> list[dict[str, Any]]:
    limit = env_int("POLAR_DATAPROTO_DEBUG_LIMIT", 8)
    if limit <= 0:
        return []
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples[:limit]):
        mask = [int(v) for v in response_mask_rows[idx]]
        trainable_positions = [pos for pos, value in enumerate(mask) if value]
        polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
        trace_meta = polar.get("trace_metadata", {}) if isinstance(polar, dict) else {}
        rows.append(
            {
                "row": idx,
                "uid": sample.uid,
                "source_uid": polar.get("source_uid") if isinstance(polar, dict) else None,
                "group_index": sample.group_index,
                "trace_index": sample.trace_index,
                "status": str(getattr(sample.status, "value", sample.status)),
                "reward": float(sample.reward),
                "raw_prompt": token_preview(sample.prompt_ids),
                "raw_response": token_preview(sample.response_ids),
                "fitted_prompt": token_preview(prompt_rows[idx]),
                "fitted_response": token_preview(response_rows[idx]),
                "response_attention_tokens": int(sum(int(v) for v in response_attention_rows[idx])),
                "trainable_tokens": int(sum(mask)),
                "trainable_pos_head": trainable_positions[:16],
                "trainable_pos_tail": trainable_positions[-16:] if len(trainable_positions) > 16 else trainable_positions,
                "rollout_lp_masked": [float(rollout_logprob_rows[idx][pos]) for pos in trainable_positions[:16]],
                "trace_meta": {
                    key: trace_meta.get(key)
                    for key in (
                        "builder",
                        "request_id",
                        "completion_id",
                        "completion_ids",
                        "completion_count",
                        "native_prompt_len",
                        "native_response_len",
                        "native_logprob_len",
                        "native_prompt_lens",
                        "native_response_lens",
                        "native_logprob_lens",
                    )
                    if isinstance(trace_meta, dict) and key in trace_meta
                },
            }
        )
    return rows


def _debug_sample_summary(sample: VerlPolarSample) -> dict[str, Any]:
    return {
        "uid": sample.uid,
        "group_index": sample.group_index,
        "trajectory_index": sample.trajectory_index,
        "trace_index": sample.trace_index,
        "prompt_ids": token_preview(sample.prompt_ids),
        "response_ids": token_preview(sample.response_ids),
        "loss_tokens": sum(int(v) for v in sample.response_mask),
        "logprob_len": len(sample.rollout_log_probs),
        "reward": float(sample.reward),
        "status": str(getattr(sample.status, "value", sample.status)),
        "prompt_messages": messages_summary(sample.prompt) if isinstance(sample.prompt, list) else {"type": type(sample.prompt).__name__},
        "polar_metadata_keys": sorted(str(k) for k in ((sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}) or {}).keys()),
    }


def samples_to_dataproto(
    samples: list[VerlPolarSample],
    *,
    prompt_length: int,
    response_length: int,
    pad_token_id: int = 0,
) -> Any:
    """Build a VERL ``DataProto`` from Polar samples."""
    tensors, non_tensors, meta_info = samples_to_tensor_dicts(
        samples,
        prompt_length=prompt_length,
        response_length=response_length,
        pad_token_id=pad_token_id,
    )
    try:
        import torch
        from verl.protocol import DataProto
    except ImportError as exc:
        raise ImportError(
            "VERL is required for samples_to_dataproto. Add VERL to PYTHONPATH or "
            "use samples_to_tensor_dicts in tests."
        ) from exc
    torch_tensors = {
        key: value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        for key, value in tensors.items()
    }
    return DataProto.from_dict(tensors=torch_tensors, non_tensors=non_tensors, meta_info=meta_info)


def prompt_rows_to_samples(prompts: Any) -> list[Any]:
    """Extract sample-like rows from a VERL input ``DataProto``.

    The bridge prefers ``raw_prompt`` because Polar task templates need the
    original text/chat prompt.  Other non-tensor fields are copied into
    ``metadata``/``extra_info`` so templates can reference them.
    """
    rows: list[Any] = []
    non_tensor_batch = getattr(prompts, "non_tensor_batch", {}) or {}
    size = len(prompts)
    for i in range(size):
        raw_prompt = _row_value(non_tensor_batch, "raw_prompt", i, default=None)
        if raw_prompt is None:
            raw_prompt = _row_value(non_tensor_batch, "prompt", i, default="")
        uid = _row_value(non_tensor_batch, "uid", i, default=str(i))
        metadata = {}
        extra_info = {}
        reward_model = {}
        for key, values in non_tensor_batch.items():
            value = _row_value(non_tensor_batch, key, i, default=None)
            if key == "extra_info" and isinstance(value, dict):
                extra_info.update(value)
            elif key == "reward_model" and isinstance(value, dict):
                reward_model.update(value)
            elif key not in {"raw_prompt", "prompt", "uid"}:
                metadata[key] = value
        rows.append(
            _RowSample(
                prompt=raw_prompt,
                uid=str(uid),
                metadata=metadata,
                extra_info=extra_info,
                reward_model=reward_model,
                index=i,
                group_index=i,
            )
        )
    return rows


class _RowSample:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _row_value(batch: dict[str, Any], key: str, index: int, default: Any = None) -> Any:
    if key not in batch:
        return default
    values = batch[key]
    try:
        return values[index]
    except Exception:
        return default


def _fit_left(values: list[int], length: int, pad_value: int) -> list[int]:
    values = list(values)[-length:]
    return [pad_value] * (length - len(values)) + values


def _fit_right(values: list[Any], length: int, pad_value: Any) -> list[Any]:
    values = list(values)[:length]
    return values + [pad_value] * (length - len(values))


def _fit_right_with_length(values: list[int], length: int, pad_value: int) -> tuple[list[int], int]:
    values = list(values)[:length]
    actual_len = len(values)
    return values + [pad_value] * (length - actual_len), actual_len


def _compute_position_rows(attention_rows: list[list[int]]) -> list[list[int]]:
    rows: list[list[int]] = []
    for row in attention_rows:
        current = -1
        out: list[int] = []
        for value in row:
            if int(value):
                current += 1
                out.append(current)
            else:
                out.append(max(current, 0))
        rows.append(out)
    return rows


def _tensorize(values: dict[str, list[list[Any]]]) -> dict[str, Any]:
    try:
        import torch

        tensors = {}
        for key, value in values.items():
            dtype = torch.float32 if key in {"rollout_log_probs", "rm_scores"} else torch.long
            tensors[key] = torch.tensor(value, dtype=dtype)
        return tensors
    except ImportError:
        import numpy as np

        tensors = {}
        for key, value in values.items():
            dtype = np.float32 if key in {"rollout_log_probs", "rm_scores"} else np.int64
            tensors[key] = np.array(value, dtype=dtype)
        return tensors


def _num_turns(sample: VerlPolarSample) -> int:
    polar = sample.metadata.get("polar", {})
    trace_debug = polar.get("trace_debug", {}) if isinstance(polar, dict) else {}
    response_messages = trace_debug.get("response_messages", []) if isinstance(trace_debug, dict) else []
    if isinstance(response_messages, list):
        # Match VERL's native ToolAgentLoop accounting:
        #
        #   num_turns = user_turns + assistant_turns + 1
        #
        # ``sample.prompt`` is the initial raw chat prompt and usually contains
        # both system and user messages.  Counting all of it makes Polar's
        # metric one larger than native SearchR1 when the prompt is
        # [system, user].  Count response-side assistant messages plus
        # contiguous tool/user response blocks instead, and add the same native
        # initial-prompt offset of one.
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



def _sample_uid(sample: VerlPolarSample, source_uid: str) -> str:
    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    if isinstance(polar, dict):
        value = polar.get("sample_uid") or polar.get("group_id")
        if value is not None:
            return str(value)
    return str(source_uid)

def _source_uid(sample: VerlPolarSample) -> str:
    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    if isinstance(polar, dict):
        value = polar.get("source_uid")
        if value is not None:
            return str(value)
    return str(sample.uid)
