"""Convert Polar rollout results into VERL-style rollout samples."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import os
from collections import OrderedDict
from typing import Any, TYPE_CHECKING

from verl_polar_bridge._messages import messages_to_text
from verl_polar_bridge.debug_utils import debug_print

if TYPE_CHECKING:
    from polar.rollout.models import SessionResult
    from polar.trajectory.models import Trace

logger = logging.getLogger(__name__)

_STITCH_DEBUG_EMITTED = 0
_PROMPT_ALIGNMENT_AUDIT_EMITTED = 0



class RolloutLogprobError(ValueError):
    """Raised when a trainable Polar trace lacks aligned rollout logprobs."""


class VerlPolarStatus(str, Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"
    TRUNCATED = "truncated"


@dataclass
class VerlPolarSample:
    """One trainable VERL row produced from one Polar trace."""

    uid: str
    group_index: int
    trajectory_index: int
    trace_index: int
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    rollout_log_probs: list[float]
    reward: float
    status: VerlPolarStatus
    prompt: Any = ""
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    remove_sample: bool = False

    @property
    def has_trainable_tokens(self) -> bool:
        return has_trainable_tokens(self)


def session_result_to_verl_samples(
    result: "SessionResult",
    group_index: int,
    *,
    trajectory_index: int,
    uid: str | None = None,
    reward_key: str = "score",
    max_tokens: int | None = None,
    overflow_policy: str = "drop",
    stitch_traces: bool = True,
) -> list[VerlPolarSample]:
    """Convert one Polar session result into VERL samples.

    For ``prefix_merging_prompt_grounded_single`` trajectories, the Polar
    builder is the single source of truth for token-level merging.  The adapter
    must not stitch again: doing so hides builder bugs and can merge across
    segment boundaries.  With SearchR1 subagent/wipe, the builder emits one
    already-merged trace per ``merge_group_id``/segment, and this adapter only
    converts those traces to VERL rows plus segment metadata.

    The legacy adapter stitch remains available for non-prompt-grounded builders
    as a compatibility fallback.
    """
    traces = list(result.trajectory.traces)
    source_uid = (
        (getattr(result, "metadata", {}) or {}).get("source_uid")
        if isinstance(getattr(result, "metadata", None), dict)
        else None
    )
    if stitch_traces and not _trajectory_is_prompt_grounded_single(result):
        if _env_flag("POLAR_STITCH_BY_MERGE_GROUP", default=False):
            grouped_traces = _try_stitch_trace_groups(
                traces,
                session_id=getattr(result, "session_id", None),
                task_id=getattr(result, "task_id", None),
                source_uid=source_uid,
            )
            if grouped_traces:
                grouped_samples: list[VerlPolarSample] = []
                for trace_index, trace in enumerate(grouped_traces):
                    sample = _build_sample(
                        result=result,
                        trace=trace,
                        trace_index=trace_index,
                        group_index=group_index,
                        trajectory_index=trajectory_index,
                        uid=uid,
                        reward_key=reward_key,
                        max_tokens=max_tokens,
                        overflow_policy=overflow_policy,
                    )
                    if sample is not None:
                        grouped_samples.append(sample)
                if grouped_samples:
                    return _annotate_segment_samples(grouped_samples)

        merged_trace = _try_stitch_traces(
            traces,
            session_id=getattr(result, "session_id", None),
            task_id=getattr(result, "task_id", None),
            source_uid=source_uid,
        )
        if merged_trace is not None:
            sample = _build_sample(
                result=result,
                trace=merged_trace,
                trace_index=0,
                group_index=group_index,
                trajectory_index=trajectory_index,
                uid=uid,
                reward_key=reward_key,
                max_tokens=max_tokens,
                overflow_policy=overflow_policy,
            )
            if sample is not None:
                return _annotate_segment_samples([sample])
    samples: list[VerlPolarSample] = []
    for trace_index, trace in enumerate(traces):
        sample = _build_sample(
            result=result,
            trace=trace,
            trace_index=trace_index,
            group_index=group_index,
            trajectory_index=trajectory_index,
            uid=uid,
            reward_key=reward_key,
            max_tokens=max_tokens,
            overflow_policy=overflow_policy,
        )
        if sample is not None:
            samples.append(sample)

    if samples:
        return _annotate_segment_samples(samples)

    logger.warning(
        "Session %s: no usable trace (traces=%d, max_tokens=%s); emitting placeholder",
        result.session_id,
        len(traces),
        max_tokens,
    )
    return [
        _build_placeholder_sample(
            result=result,
            group_index=group_index,
            trajectory_index=trajectory_index,
            uid=uid,
            reward_key=reward_key,
        )
    ]


def _trajectory_is_prompt_grounded_single(result: Any) -> bool:
    trajectory = getattr(result, "trajectory", None)
    metadata = getattr(trajectory, "metadata", {}) if trajectory is not None else {}
    if isinstance(metadata, dict) and metadata.get("builder") == "prefix_merging_prompt_grounded_single":
        return True
    for trace in list(getattr(trajectory, "traces", []) or []):
        trace_meta = getattr(trace, "metadata", {}) or {}
        if isinstance(trace_meta, dict) and trace_meta.get("builder") == "prefix_merging_prompt_grounded_single":
            return True
    return False


def task_result_to_verl_samples(
    task_result: Any,
    *,
    group_index: int,
    uid: str | None = None,
    reward_key: str = "score",
    max_tokens: int | None = None,
    overflow_policy: str = "drop",
    stitch_traces: bool = True,
) -> list[VerlPolarSample]:
    samples: list[VerlPolarSample] = []
    for trajectory_index, result in enumerate(getattr(task_result, "results", []) or []):
        samples.extend(
            session_result_to_verl_samples(
                result,
                group_index=group_index,
                trajectory_index=trajectory_index,
                uid=uid,
                reward_key=reward_key,
                max_tokens=max_tokens,
                overflow_policy=overflow_policy,
                stitch_traces=stitch_traces,
            )
        )
    return samples


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _try_stitch_trace_groups(
    traces: list[Any],
    *,
    session_id: str | None = None,
    task_id: str | None = None,
    source_uid: str | None = None,
) -> list[Any] | None:
    if not traces:
        return None
    groups: "OrderedDict[str, list[Any]]" = OrderedDict()
    for idx, trace in enumerate(traces):
        group_id = _trace_merge_group_id(trace) or f"__trace_{idx}"
        groups.setdefault(str(group_id), []).append(trace)
    if len(groups) <= 1:
        return None

    merged: list[Any] = []
    for group_id, group_traces in groups.items():
        if len(group_traces) == 1:
            merged.append(group_traces[0])
            continue
        stitched = _try_stitch_traces(
            group_traces,
            session_id=session_id,
            task_id=task_id,
            source_uid=source_uid,
        )
        if stitched is None:
            logger.warning(
                "Session %s: merge_group_id=%s could not be stitched; falling back to %d per-turn traces",
                session_id,
                group_id,
                len(group_traces),
            )
            merged.extend(group_traces)
        else:
            merged.append(stitched)
    return merged


def _annotate_segment_samples(samples: list[VerlPolarSample]) -> list[VerlPolarSample]:
    if not samples:
        return samples
    k = len(samples)
    parent_trainable_tokens = sum(int(sum(int(v) for v in (sample.response_mask or []))) for sample in samples)
    final_wipe_index = _max_segment_kind_index(samples, kind="wipe")
    reward_mode = os.environ.get("POLAR_SEGMENT_REWARD_MODE", "none").strip().lower()
    split_reward = reward_mode in {"prompt_grounded_split", "split", "r_over_k", "reward_split"}
    for idx, sample in enumerate(samples):
        polar = sample.metadata.setdefault("polar", {})
        if not isinstance(polar, dict):
            sample.metadata["polar"] = polar = {}
        kind = str(polar.get("segment_kind") or "").strip().lower()
        try:
            merge_group_index = int(polar.get("merge_group_index") or 0)
        except (TypeError, ValueError):
            merge_group_index = 0
        is_last_main_wipe = kind == "wipe" and final_wipe_index is not None and merge_group_index == final_wipe_index
        if is_last_main_wipe:
            polar["segment_kind"] = "final"
            polar["is_final_segment"] = True
        original_reward = float(sample.reward)
        if split_reward:
            sample.reward = original_reward / max(1, k)
        polar.setdefault("sample_uid", sample.uid)
        polar["segment_idx"] = int(idx)
        polar["num_segments"] = int(k)
        polar["prompt_grounded_segment_count"] = int(k)
        polar["segment_weight"] = 1.0 / max(1, k)
        polar["prompt_grounded_segment_reward_split"] = bool(split_reward)
        polar["original_reward"] = original_reward
        polar["segment_reward"] = float(sample.reward)
        polar["parent_sample_trainable_tokens"] = int(parent_trainable_tokens)
    return samples


def _max_segment_kind_index(samples: list[VerlPolarSample], *, kind: str) -> int | None:
    indices: list[int] = []
    target = str(kind).strip().lower()
    for sample in samples:
        polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
        if not isinstance(polar, dict):
            continue
        if str(polar.get("segment_kind") or "").strip().lower() != target:
            continue
        try:
            indices.append(int(polar.get("merge_group_index") or 0))
        except (TypeError, ValueError):
            indices.append(0)
    return max(indices) if indices else None

def _try_stitch_traces(
    traces: list[Any],
    *,
    session_id: str | None = None,
    task_id: str | None = None,
    source_uid: str | None = None,
) -> Any | None:
    """Merge per-turn traces into one full-trajectory trace using token ids only."""
    debug_context = {"session_id": session_id, "task_id": task_id, "source_uid": source_uid, "trace_count": len(traces)}

    def fail(reason: str, **extra: Any) -> None:
        _log_stitch_debug("fail", reason=reason, context=debug_context, traces=traces, extra=extra)
        return None

    if len(traces) <= 1:
        _log_stitch_debug("skip", reason="single_or_no_trace", context=debug_context, traces=traces, extra={})
        return None
    merge_group_ids = [_trace_merge_group_id(trace) for trace in traces]
    explicit_merge_group_ids = [value for value in merge_group_ids if value is not None]
    if explicit_merge_group_ids and len(set(explicit_merge_group_ids)) > 1:
        _log_stitch_debug(
            "skip",
            reason="multiple_merge_groups",
            context=debug_context,
            traces=traces,
            extra={"merge_group_ids": merge_group_ids},
        )
        return None
    first = traces[0]
    prompt_ids = list(getattr(first, "prompt_ids", []) or [])
    first_response = list(getattr(first, "response_ids", []) or []) or _response_ids_from_logprobs(first)
    if not prompt_ids or not first_response:
        fail("first_trace_missing_tokens", first_prompt_len=len(prompt_ids), first_response_len=len(first_response))
        return None

    response_ids: list[int] = []
    loss_mask: list[int] = []
    response_logprobs: list[dict[str, Any]] = []
    response_messages: list[dict[str, Any]] = []
    prompt_alignment_spans: list[dict[str, Any]] = []

    prev_prompt_ids = prompt_ids
    prev_response_ids: list[int] | None = None
    merged_count = 0
    for idx, trace in enumerate(traces):
        current_prompt_ids = list(getattr(trace, "prompt_ids", []) or [])
        current_response_ids = list(getattr(trace, "response_ids", []) or []) or _response_ids_from_logprobs(trace)
        if not current_prompt_ids or not current_response_ids:
            fail("trace_missing_tokens", trace_index=idx, prompt_len=len(current_prompt_ids), response_len=len(current_response_ids))
            return None
        if idx == 0:
            interstitial: list[int] = []
        else:
            prefix_len = _continuation_prefix_len(prev_prompt_ids, current_prompt_ids)
            if prefix_len is None:
                common_prefix_len = _common_prefix_len(current_prompt_ids, prev_prompt_ids)
                fail(
                    "prompt_not_append_only",
                    trace_index=idx,
                    prev_prompt_len=len(prev_prompt_ids),
                    current_prompt_len=len(current_prompt_ids),
                    common_prefix_len=common_prefix_len,
                    prev_prompt_suffix_len=len(prev_prompt_ids) - common_prefix_len,
                    prev_prompt_head=_token_snippet(prev_prompt_ids),
                    current_prompt_head=_token_snippet(current_prompt_ids),
                )
                return None
            canonical_tail = current_prompt_ids[prefix_len:]
            assert prev_response_ids is not None
            interstitial = _canonical_interstitial_after_response(canonical_tail, prev_response_ids)
            if interstitial is None:
                fail(
                    "interstitial_split_failed",
                    trace_index=idx,
                    diagnostics=_interstitial_diagnostics(canonical_tail, prev_response_ids),
                )
                return None
        if interstitial:
            response_ids.extend(interstitial)
            loss_mask.extend([0] * len(interstitial))
            response_logprobs.extend({"token_id": int(token_id), "logprob": 0.0} for token_id in interstitial)

        alignment_span = _adapter_prompt_alignment_span(
            source="adapter_stitch",
            trace_index=idx,
            base_prompt_ids=prompt_ids,
            stitched_prefix_ids=prompt_ids + response_ids,
            current_prompt_ids=current_prompt_ids,
            response_len=len(current_response_ids),
            interstitial_len=len(interstitial or []),
            trace=trace,
        )
        prompt_alignment_spans.append(alignment_span)
        if (not alignment_span["prompt_aligned"]) or _prompt_alignment_audit_all():
            _log_prompt_alignment_audit({"event": "span", **alignment_span, "context": debug_context})

        trace_loss_mask = list(getattr(trace, "loss_mask", []) or [1] * len(current_response_ids))
        if len(trace_loss_mask) != len(current_response_ids):
            fail("loss_mask_length_mismatch", trace_index=idx, loss_mask_len=len(trace_loss_mask), response_len=len(current_response_ids))
            return None
        trace_logprobs = getattr(trace, "response_logprobs", None) or []
        if trace_logprobs and len(trace_logprobs) != len(current_response_ids):
            fail("logprob_length_mismatch", trace_index=idx, logprob_len=len(trace_logprobs), response_len=len(current_response_ids))
            return None
        response_ids.extend(current_response_ids)
        loss_mask.extend([1 if int(value) else 0 for value in trace_loss_mask])
        if trace_logprobs:
            response_logprobs.extend(deepcopy(item) if isinstance(item, dict) else {"token_id": int(current_response_ids[pos]), "logprob": 0.0} for pos, item in enumerate(trace_logprobs))
        else:
            response_logprobs.extend({"token_id": int(token_id), "logprob": 0.0} for token_id in current_response_ids)
        response_messages.extend(deepcopy(m) for m in (getattr(trace, "response_messages", []) or []))
        prev_prompt_ids = current_prompt_ids
        prev_response_ids = current_response_ids
        merged_count += 1

    _log_stitch_debug("success", reason="stitched", context=debug_context, traces=traces, extra={"merged_count": merged_count, "stitched_response_len": len(response_ids), "trainable_tokens": int(sum(loss_mask))})

    last = traces[merged_count - 1]
    metadata = deepcopy(getattr(last, "metadata", {}) or {})
    metadata["adapter_stitched_trace_count"] = merged_count
    metadata["adapter_stitched_from_trace_count"] = len(traces)
    metadata["adapter_prompt_alignment_mismatch_count"] = sum(
        1 for span in prompt_alignment_spans if not span.get("prompt_aligned")
    )
    metadata["adapter_prompt_alignment_prompt_grounded_drift_tokens_sum"] = sum(
        int(span.get("prompt_grounded_drift_tokens") or 0) for span in prompt_alignment_spans
    )
    metadata["adapter_prompt_alignment_span_count"] = len(prompt_alignment_spans)
    if _prompt_alignment_audit_enabled():
        metadata["adapter_prompt_alignment_spans"] = prompt_alignment_spans
    try:
        return last.model_copy(
            update={
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "loss_mask": loss_mask,
                "prompt_messages": deepcopy(getattr(first, "prompt_messages", []) or []),
                "response_messages": response_messages,
                "response_logprobs": response_logprobs,
                "metadata": metadata,
            }
        )
    except Exception:
        class _MergedTrace:
            pass

        merged = _MergedTrace()
        merged.prompt_ids = prompt_ids
        merged.response_ids = response_ids
        merged.loss_mask = loss_mask
        merged.prompt_messages = deepcopy(getattr(first, "prompt_messages", []) or [])
        merged.response_messages = response_messages
        merged.tools = deepcopy(getattr(first, "tools", None))
        merged.finish_reason = getattr(last, "finish_reason", None)
        merged.response_logprobs = response_logprobs
        merged.metadata = metadata
        merged.reward = getattr(last, "reward", None)
        return merged




def _continuation_prefix_len(prev_prompt_ids: list[int], current_prompt_ids: list[int]) -> int | None:
    """Return prefix length to cut before the new canonical tail.

    Most turns are strictly append-only: current prompt starts with the complete
    previous prompt.  Qwen/tool chat templates, however, may render the previous
    prompt with a short assistant-generation suffix at the end, then render the
    next prompt by replacing that suffix with the sampled assistant message,
    tool/user glue, and a fresh generation suffix.  In GPU logs this shows up as
    ``common_prefix_len == len(prev_prompt_ids) - 4``.

    Allow only a small suffix replacement so we do not silently merge unrelated
    prompts.  The subsequent interstitial splitter must still find/remove the
    previous response from the canonical tail.
    """
    if len(current_prompt_ids) >= len(prev_prompt_ids) and current_prompt_ids[: len(prev_prompt_ids)] == prev_prompt_ids:
        return len(prev_prompt_ids)

    common_prefix_len = _common_prefix_len(current_prompt_ids, prev_prompt_ids)
    replaced_suffix_len = len(prev_prompt_ids) - common_prefix_len
    if 0 < replaced_suffix_len <= _max_prompt_suffix_replacement_tokens():
        return common_prefix_len
    return None


def _max_prompt_suffix_replacement_tokens() -> int:
    try:
        return max(0, int(os.getenv("VERL_POLAR_STITCH_MAX_PROMPT_SUFFIX_REPLACE", "16")))
    except ValueError:
        return 16

def _stitch_debug_enabled() -> bool:
    value = os.getenv("VERL_POLAR_DEBUG_STITCH", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _stitch_debug_limit() -> int:
    try:
        return max(0, int(os.getenv("VERL_POLAR_DEBUG_STITCH_LIMIT", "20")))
    except ValueError:
        return 20


def _prompt_alignment_audit_enabled() -> bool:
    value = os.getenv("POLAR_PROMPT_ALIGNMENT_AUDIT", "0")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _prompt_alignment_audit_all() -> bool:
    value = os.getenv("POLAR_PROMPT_ALIGNMENT_AUDIT_ALL", "0")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _prompt_alignment_audit_limit() -> int:
    try:
        return max(0, int(os.getenv("POLAR_PROMPT_ALIGNMENT_AUDIT_LIMIT", "200")))
    except ValueError:
        return 200


def _token_window(values: list[int], center: int, *, radius: int = 16) -> dict[str, Any]:
    values = list(values or [])
    center = max(0, min(int(center), len(values)))
    start = max(0, center - radius)
    end = min(len(values), center + radius)
    return {
        "len": len(values),
        "center": center,
        "start": start,
        "end": end,
        "ids": values[start:end],
    }


def _adapter_prompt_alignment_span(
    *,
    source: str,
    trace_index: int,
    base_prompt_ids: list[int],
    stitched_prefix_ids: list[int],
    current_prompt_ids: list[int],
    response_len: int,
    interstitial_len: int,
    trace: Any,
) -> dict[str, Any]:
    prompt_lcp = _common_prefix_len(stitched_prefix_ids, current_prompt_ids)
    prompt_aligned = (
        len(stitched_prefix_ids) == len(current_prompt_ids)
        and prompt_lcp == len(current_prompt_ids)
    )
    base_lcp = _common_prefix_len(base_prompt_ids, current_prompt_ids)
    if base_lcp == len(base_prompt_ids):
        prompt_suffix = current_prompt_ids[len(base_prompt_ids):]
        response_so_far = stitched_prefix_ids[len(base_prompt_ids):]
        prompt_grounded_matched_len = _common_prefix_len(response_so_far, prompt_suffix)
    else:
        prompt_suffix = []
        response_so_far = []
        prompt_grounded_matched_len = 0
    metadata = getattr(trace, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    response_start = max(0, len(stitched_prefix_ids) - len(base_prompt_ids))
    span: dict[str, Any] = {
        "source": source,
        "trace_index": trace_index,
        "completion_id": metadata.get("completion_id"),
        "request_id": metadata.get("request_id"),
        "response_start": response_start,
        "response_end": response_start + int(response_len),
        "prompt_aligned": bool(prompt_aligned),
        "prompt_lcp": int(prompt_lcp),
        "stitched_prefix_len": len(stitched_prefix_ids),
        "actual_prompt_len": len(current_prompt_ids),
        "prefix_len_delta": len(stitched_prefix_ids) - len(current_prompt_ids),
        "base_prompt_len": len(base_prompt_ids),
        "base_prompt_lcp": int(base_lcp),
        "prompt_suffix_len": len(prompt_suffix),
        "response_so_far_len": len(response_so_far),
        "prompt_grounded_matched_len": int(prompt_grounded_matched_len),
        "prompt_grounded_drift_tokens": max(0, len(response_so_far) - int(prompt_grounded_matched_len)),
        "interstitial_len": int(interstitial_len),
    }
    if not prompt_aligned:
        span["stitched_mismatch_window"] = _token_window(stitched_prefix_ids, prompt_lcp)
        span["actual_prompt_mismatch_window"] = _token_window(current_prompt_ids, prompt_lcp)
    return span


def _log_prompt_alignment_audit(payload: dict[str, Any]) -> None:
    if not _prompt_alignment_audit_enabled():
        return
    global _PROMPT_ALIGNMENT_AUDIT_EMITTED
    limit = _prompt_alignment_audit_limit()
    if limit and _PROMPT_ALIGNMENT_AUDIT_EMITTED >= limit:
        return
    _PROMPT_ALIGNMENT_AUDIT_EMITTED += 1
    debug_print("POLAR_PROMPT_ALIGNMENT_AUDIT", payload, stream="stderr")


def _token_snippet(values: list[int], *, limit: int = 12) -> dict[str, Any]:
    values = list(values or [])
    if len(values) <= limit * 2:
        return {"len": len(values), "ids": values}
    return {"len": len(values), "head": values[:limit], "tail": values[-limit:]}


def _interstitial_diagnostics(canonical_tail: list[int], prev_response_ids: list[int]) -> dict[str, Any]:
    lcp = _common_prefix_len(canonical_tail, prev_response_ids)
    subseq_start = _find_subsequence(canonical_tail, prev_response_ids)
    eot_positions = [idx for idx, token_id in enumerate(canonical_tail) if token_id == 151645][:8]
    return {
        "canonical_tail_len": len(canonical_tail),
        "prev_response_len": len(prev_response_ids),
        "lcp": lcp,
        "subsequence_start": subseq_start,
        "qwen_eot_positions_first8": eot_positions,
        "prev_response_endswith_eot": bool(prev_response_ids and prev_response_ids[-1] == 151645),
        "canonical_tail": _token_snippet(canonical_tail),
        "prev_response": _token_snippet(prev_response_ids),
    }


def _trace_debug_summary(trace: Any, index: int) -> dict[str, Any]:
    prompt_ids = list(getattr(trace, "prompt_ids", []) or [])
    response_ids = list(getattr(trace, "response_ids", []) or []) or _response_ids_from_logprobs(trace)
    loss_mask = list(getattr(trace, "loss_mask", []) or [])
    logprobs = getattr(trace, "response_logprobs", None) or []
    metadata = getattr(trace, "metadata", {}) or {}
    metadata_summary = {}
    if isinstance(metadata, dict):
        for key in (
            "trace_id",
            "request_id",
            "completion_index",
            "turn_index",
            "adapter_stitched_trace_count",
            "builder",
        ):
            if key in metadata:
                metadata_summary[key] = metadata[key]
    return {
        "index": index,
        "prompt_len": len(prompt_ids),
        "response_len": len(response_ids),
        "loss_mask_len": len(loss_mask),
        "trainable_tokens": int(sum(int(v) for v in loss_mask)) if loss_mask else None,
        "logprob_len": len(logprobs),
        "finish_reason": getattr(trace, "finish_reason", None),
        "prompt_head": _token_snippet(prompt_ids, limit=6),
        "response": _token_snippet(response_ids, limit=8),
        "metadata": metadata_summary,
    }


def _log_stitch_debug(
    event: str,
    *,
    reason: str,
    context: dict[str, Any],
    traces: list[Any],
    extra: dict[str, Any] | None = None,
) -> None:
    if not _stitch_debug_enabled():
        return
    global _STITCH_DEBUG_EMITTED
    limit = _stitch_debug_limit()
    if limit and _STITCH_DEBUG_EMITTED >= limit:
        return
    _STITCH_DEBUG_EMITTED += 1
    payload = {
        "event": event,
        "reason": reason,
        "context": context,
        "extra": extra or {},
        "traces": [_trace_debug_summary(trace, idx) for idx, trace in enumerate(traces)],
    }
    logger.warning("verl_polar_stitch_debug %s", payload)

def _canonical_interstitial_after_response(canonical_tail: list[int], prev_response_ids: list[int]) -> list[int] | None:
    """Return canonical tail after the previous raw response token sequence.

    ``canonical_tail`` comes from the next request's server-side prompt ids.
    The previous assistant response is already represented by raw sampled ids in
    the stitched stream, so we remove one canonical copy before appending tool /
    template interstitial tokens.  Prefer exact token matching.  If the serving
    stack trims/adds a stop token or slightly canonicalizes trailing whitespace,
    fall back to a conservative longest-common-prefix split.  This still never
    decodes/re-encodes text; it only decides how many canonical prompt tokens are
    non-trainable interstitial.
    """
    if not prev_response_ids:
        return canonical_tail

    # Ideal case: the next prompt tail contains an exact canonical copy of the
    # previous sampled response.  This covers most append-only ChatML/Qwen turns.
    start = _find_subsequence(canonical_tail, prev_response_ids)
    if start is not None:
        return canonical_tail[start + len(prev_response_ids) :]

    # ChatML/Qwen prompt tails often include the canonical assistant message
    # wrapper before the previous response body, e.g.
    #   <|im_start|>assistant\n + canonical(previous assistant) + <|im_end|> + tool/user glue
    # If the canonical body is not byte-for-byte the sampled tokens (trimmed stop
    # token, whitespace normalization, etc.), exact subsequence and prefix checks
    # both fail because the tail starts with the assistant header.  In that case
    # split at the first end-of-turn marker and keep only the non-trainable glue.
    # Token 151645 is <|im_end|> for the Qwen ChatML models used by SearchR1.
    for eot_id in (151645,):
        try:
            pos = canonical_tail.index(eot_id)
        except ValueError:
            continue
        if prev_response_ids and prev_response_ids[-1] == eot_id:
            return canonical_tail[pos + 1 :]
        return canonical_tail[pos:]

    # Common case when the rollout engine excluded a stop token or the template
    # canonicalized final whitespace and the canonical tail starts directly with
    # the assistant body.  Drop the matched prefix and keep the rest as
    # loss_mask=0 interstitial.  Require a substantial overlap to avoid silently
    # duplicating an unrelated assistant response.
    lcp = _common_prefix_len(canonical_tail, prev_response_ids)
    if _is_substantial_overlap(lcp, canonical_tail, prev_response_ids):
        return canonical_tail[lcp:]

    return None


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


def _is_substantial_overlap(overlap: int, canonical_tail: list[int], prev_response_ids: list[int]) -> bool:
    if overlap <= 0:
        return False
    shorter = min(len(canonical_tail), len(prev_response_ids))
    if shorter <= 0:
        return False
    return overlap >= min(32, max(8, int(shorter * 0.8)))


def _find_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle:
        return 0
    if len(needle) > len(values):
        return None
    first = needle[0]
    limit = len(values) - len(needle) + 1
    for idx in range(limit):
        if values[idx] == first and values[idx : idx + len(needle)] == needle:
            return idx
    return None


def _alignment_debug_enabled() -> bool:
    value = os.getenv("POLAR_ALIGNMENT_DEBUG", os.getenv("POLAR_ADAPTER_ALIGNMENT_DEBUG", "0"))
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _alignment_debug_limit() -> int:
    try:
        return max(0, int(os.getenv("POLAR_ALIGNMENT_DEBUG_LIMIT", "8")))
    except Exception:
        return 8


def _hash_values(values: Any) -> str:
    try:
        data = json.dumps(list(values or []), sort_keys=True, default=str).encode("utf-8")
    except Exception:
        data = repr(values).encode("utf-8", errors="replace")
    return hashlib.sha1(data).hexdigest()[:16]


def _log_sample_alignment_debug(*, session_id: str | None, trace_index: int, prompt_ids: list[int], response_ids: list[int], loss_mask: list[int], response_log_probs: list[float], trace: Any, metadata: dict[str, Any]) -> None:
    if not _alignment_debug_enabled():
        return
    try:
        global _STITCH_DEBUG_EMITTED
        limit = _alignment_debug_limit()
        if limit and _STITCH_DEBUG_EMITTED >= limit:
            return
        _STITCH_DEBUG_EMITTED += 1
        train_pos = [idx for idx, value in enumerate(loss_mask) if int(value)]
        trace_logprobs = getattr(trace, "response_logprobs", None) or []
        raw_logprob_token_ids = [item.get("token_id") for item in trace_logprobs[:16] if isinstance(item, dict)]
        payload = {
            "session_id": session_id,
            "trace_index": int(trace_index),
            "prompt": _token_snippet(prompt_ids, limit=12),
            "response": _token_snippet(response_ids, limit=16),
            "response_hash": _hash_values(response_ids),
            "loss_mask_len": len(loss_mask),
            "trainable_tokens": int(sum(int(v) for v in loss_mask)),
            "trainable_pos_head": train_pos[:16],
            "trainable_pos_tail": train_pos[-16:] if len(train_pos) > 16 else train_pos,
            "rollout_log_probs_len": len(response_log_probs),
            "rollout_log_probs_head": [float(v) for v in response_log_probs[:16]],
            "rollout_log_probs_masked_head": [float(response_log_probs[pos]) for pos in train_pos[:16] if pos < len(response_log_probs)],
            "raw_logprob_len": len(trace_logprobs),
            "raw_logprob_token_ids_head": raw_logprob_token_ids,
            "raw_logprob_token_match_head": raw_logprob_token_ids == response_ids[: len(raw_logprob_token_ids)],
            "finish_reason": getattr(trace, "finish_reason", None),
            "metadata": {
                key: metadata.get(key)
                for key in (
                    "session_id",
                    "task_id",
                    "source_uid",
                    "sample_uid",
                    "raw_response_len",
                    "raw_total_len",
                    "overflow_policy",
                )
                if key in metadata
            },
            "trace_metadata": {
                key: (getattr(trace, "metadata", {}) or {}).get(key)
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
                if isinstance(getattr(trace, "metadata", {}) or {}, dict) and key in (getattr(trace, "metadata", {}) or {})
            },
        }
        logger.warning("POLAR_ADAPTER_ALIGNMENT_DEBUG %s", json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        logger.exception("POLAR_ADAPTER_ALIGNMENT_DEBUG failed")


def _build_sample(
    *,
    result: "SessionResult",
    trace: "Trace",
    trace_index: int,
    group_index: int,
    trajectory_index: int,
    uid: str | None,
    reward_key: str,
    max_tokens: int | None = None,
    overflow_policy: str = "drop",
) -> VerlPolarSample | None:
    prompt_ids = list(trace.prompt_ids)
    response_ids = list(trace.response_ids) or _response_ids_from_logprobs(trace)
    if not prompt_ids or not response_ids:
        logger.warning(
            "Dropping trace %d from session %s: missing tokens (prompt=%d, response=%d)",
            trace_index,
            result.session_id,
            len(prompt_ids),
            len(response_ids),
        )
        return None

    normalized_overflow_policy = str(overflow_policy or "drop").strip().lower()
    if normalized_overflow_policy not in {"drop", "verl_truncate", "none"}:
        raise ValueError("overflow_policy must be one of: drop, verl_truncate, none")

    total_len = len(prompt_ids) + len(response_ids)
    if max_tokens is not None and total_len > max_tokens and normalized_overflow_policy == "drop":
        logger.warning(
            "Dropping trace %d from session %s: total_len=%d > max_tokens=%d",
            trace_index,
            result.session_id,
            total_len,
            max_tokens,
        )
        return None

    status = _sample_status(result, trace)
    trainable = status not in (VerlPolarStatus.ABORTED, VerlPolarStatus.FAILED)
    loss_mask = _loss_mask_from_trace(
        trace,
        len(response_ids),
        require_loss_mask=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )
    if not trainable:
        loss_mask = [0] * len(response_ids)

    response_log_probs = _extract_rollout_log_probs(
        trace,
        response_len=len(response_ids),
        loss_mask=loss_mask,
        require_trainable_logprobs=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )

    prompt_messages = deepcopy(trace.prompt_messages)
    response_messages = deepcopy(trace.response_messages)
    metadata = _polar_metadata(result, trace, trace_index)
    metadata["reward_key"] = reward_key
    metadata["overflow_policy"] = normalized_overflow_policy
    metadata["raw_response_len"] = len(response_ids)
    metadata["raw_total_len"] = total_len
    if max_tokens is not None and total_len > max_tokens:
        metadata["overflowed_max_tokens"] = int(max_tokens)

    _log_sample_alignment_debug(
        session_id=result.session_id,
        trace_index=trace_index,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=loss_mask,
        response_log_probs=response_log_probs,
        trace=trace,
        metadata=metadata,
    )

    resolved_uid = _resolve_uid(uid, result, trace)
    # Preserve one GRPO/PPO group per original VERL input row.  When the Polar
    # trajectory builder emits multiple trainable traces for a single multi-turn
    # session, each trace still corresponds to the same rollout/sample.  Giving
    # each trace a distinct row uid makes VERL treat them as extra rollout rows,
    # which can produce non-divisible batch sizes (e.g. 17 % dp_size) and changes
    # advantage grouping semantics.  Keep the row uid stable and use
    # (trajectory_index, trace_index) only as provenance metadata.
    metadata.setdefault("sample_uid", resolved_uid)
    return VerlPolarSample(
        uid=resolved_uid,
        group_index=group_index,
        trajectory_index=trajectory_index,
        trace_index=trace_index,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=loss_mask,
        rollout_log_probs=response_log_probs,
        reward=_reward_value(trace),
        status=status,
        prompt=prompt_messages if prompt_messages else "",
        response=messages_to_text(response_messages),
        metadata={"polar": metadata},
    )


def _build_placeholder_sample(
    *,
    result: "SessionResult",
    group_index: int,
    trajectory_index: int,
    uid: str | None,
    reward_key: str,
) -> VerlPolarSample:
    metadata = _polar_metadata(result, None, -1)
    metadata.update({"placeholder": True, "reward_key": reward_key})
    return VerlPolarSample(
        uid=_resolve_uid(uid, result, None),
        group_index=group_index,
        trajectory_index=trajectory_index,
        trace_index=-1,
        prompt_ids=[0],
        response_ids=[0],
        response_mask=[0],
        rollout_log_probs=[0.0],
        reward=0.0,
        status=VerlPolarStatus.ABORTED,
        metadata={"polar": metadata},
        remove_sample=True,
    )


def _polar_metadata(result: "SessionResult", trace: "Trace | None", trace_index: int) -> dict[str, Any]:
    trajectory = result.trajectory
    metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": str(result.status),
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trace_metadata": deepcopy(getattr(trace, "metadata", {}) or {}) if trace is not None else {},
        "trajectory_error": trajectory.error,
        "trajectory_metadata": deepcopy(trajectory.metadata),
        "trajectory_status": trajectory.status,
    }
    if trace is not None:
        metadata["trace_debug"] = {
            "finish_reason": trace.finish_reason,
            "response_messages": deepcopy(trace.response_messages),
        }
        metadata.update(_segment_metadata(trace))
    metadata.update(_scheduler_metadata(result, trace))
    return metadata


def _trace_merge_group_id(trace: Any) -> str | None:
    metadata = getattr(trace, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("merge_group_id") or metadata.get("segment_group_id")
    return None if value is None else str(value)


def _segment_metadata(trace: "Trace") -> dict[str, Any]:
    trace_meta = getattr(trace, "metadata", {}) or {}
    if not isinstance(trace_meta, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "merge_group_id",
        "merge_group_index",
        "num_merge_groups",
        "segment_kind",
        "is_final_segment",
        "segment_weight",
        "segment_group_id",
        "parent_merge_group_id",
        "dispatch_index",
        "segment_index",
        "harness_event",
        "harness_mode",
    ):
        if key in trace_meta:
            out[key] = deepcopy(trace_meta[key])
    if "segment_kind" not in out and "segment_type" in trace_meta:
        out["segment_kind"] = deepcopy(trace_meta["segment_type"])
    if "merge_group_id" not in out and "segment_group_id" in out:
        out["merge_group_id"] = out["segment_group_id"]
    return out


def _scheduler_metadata(result: "SessionResult", trace: "Trace | None") -> dict[str, Any]:
    keys = {
        "group_id",
        "policy_version",
        "rollout_step",
        "source_uid",
        "polar_task_id",
        "polar_session_id",
        "polar_scheduler_group_id",
        "polar_policy_version",
        "polar_policy_staleness",
        "accepted_rollout_id",
    }
    merged: dict[str, Any] = {}
    for source in (
        getattr(result, "metadata", None),
        getattr(result.trajectory, "metadata", None),
        getattr(trace, "metadata", None) if trace is not None else None,
    ):
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                merged[key] = source[key]
    return merged


def _resolve_uid(uid: str | None, result: "SessionResult", trace: "Trace | None") -> str:
    if uid is not None:
        return str(uid)
    for source in (
        getattr(trace, "metadata", None) if trace is not None else None,
        getattr(result.trajectory, "metadata", None),
        getattr(result, "metadata", None),
    ):
        if isinstance(source, dict):
            for key in ("source_uid", "uid"):
                if source.get(key) is not None:
                    return str(source[key])
    return str(result.task_id)


def _reward_value(trace: "Trace") -> float:
    return float(trace.reward) if trace.reward is not None else 0.0


def _sample_status(result: "SessionResult", trace: "Trace") -> VerlPolarStatus:
    trajectory_status = result.trajectory.status
    result_status = str(result.status)
    if trajectory_status == "TIMEOUT" or result_status == "TIMEOUT":
        return VerlPolarStatus.ABORTED
    if trajectory_status == "ERROR" or result_status == "ERROR" or result.error or result.trajectory.error:
        return VerlPolarStatus.FAILED
    if trace.finish_reason == "length":
        return VerlPolarStatus.TRUNCATED
    return VerlPolarStatus.COMPLETED


def _extract_rollout_log_probs(
    trace: "Trace",
    *,
    response_len: int,
    loss_mask: list[int],
    require_trainable_logprobs: bool,
    session_id: str,
    trace_index: int,
) -> list[float]:
    logprobs = trace.response_logprobs
    if not logprobs:
        if require_trainable_logprobs and any(loss_mask):
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing rollout_log_probs "
                "for trainable response tokens"
            )
        return [0.0] * response_len
    if len(logprobs) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: rollout_log_probs length "
            f"{len(logprobs)} != response length {response_len}"
        )

    values: list[float] = []
    for pos, (entry, mask_value) in enumerate(zip(logprobs, loss_mask)):
        if not isinstance(entry, dict):
            if mask_value:
                raise RolloutLogprobError(
                    f"Session {session_id} trace {trace_index}: logprob entry {pos} is not a mapping"
                )
            values.append(0.0)
            continue
        if mask_value and "logprob" not in entry:
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: trainable token {pos} is missing logprob"
            )
        values.append(float(entry.get("logprob", 0.0)))
    return values


def _loss_mask_from_trace(
    trace: "Trace",
    response_len: int,
    *,
    require_loss_mask: bool,
    session_id: str,
    trace_index: int,
) -> list[int]:
    mask = list(trace.loss_mask)
    if not mask:
        if require_loss_mask:
            raise RolloutLogprobError(f"Session {session_id} trace {trace_index}: missing loss_mask")
        return [0] * response_len
    if len(mask) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: loss_mask length {len(mask)} != response length {response_len}"
        )
    return [1 if int(value) else 0 for value in mask]


def _response_ids_from_logprobs(trace: "Trace") -> list[int]:
    if not trace.response_logprobs:
        return []
    return [
        int(item["token_id"])
        for item in trace.response_logprobs
        if isinstance(item, dict) and item.get("token_id") is not None
    ]


def has_trainable_tokens(sample: VerlPolarSample) -> bool:
    return (not sample.remove_sample) and any(int(value) for value in sample.response_mask)
