"""Helpers for converting completion records into trajectory traces."""

from __future__ import annotations

from copy import deepcopy
import os
from typing import Any

from polar.trajectory.models import CompletionRecord, Trace
from verl_polar_bridge.debug_utils import debug_print, messages_summary, stable_hash, token_preview


def _record_debug_enabled() -> bool:
    return os.environ.get("POLAR_RECORD_UTILS_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _record_debug(event: str, payload: dict[str, Any]) -> None:
    if _record_debug_enabled():
        debug_print("POLAR_RECORD_UTILS_DEBUG", {"event": event, **payload}, stream="stderr")


def _extract_response_ids(response: dict[str, Any], choice: dict[str, Any]) -> list[int]:
    token_ids = choice.get("token_ids", response.get("token_ids"))
    if isinstance(token_ids, list):
        return list(token_ids)

    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        content = logprobs.get("content")
        if isinstance(content, list):
            extracted = [
                int(item["token_id"])
                for item in content
                if isinstance(item, dict) and item.get("token_id") is not None
            ]
            if extracted:
                return extracted
    return []


def _extract_response_logprobs(choice: dict[str, Any]) -> list[dict[str, Any]] | None:
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        content = logprobs.get("content")
        if isinstance(content, list):
            return [deepcopy(item) for item in content if isinstance(item, dict)]
    return None


def _extract_sglang_meta(response: dict[str, Any], choice: dict[str, Any]) -> dict[str, Any]:
    meta = choice.get("meta_info")
    if isinstance(meta, dict):
        return meta
    meta = response.get("meta_info")
    return meta if isinstance(meta, dict) else {}


def _extract_prompt_ids(response: dict[str, Any], choice: dict[str, Any]) -> list[int]:
    prompt_ids = choice.get("input_token_ids") or response.get("prompt_token_ids")
    if isinstance(prompt_ids, list):
        return list(prompt_ids)
    meta = _extract_sglang_meta(response, choice)
    prompt_ids = meta.get("input_token_ids")
    if isinstance(prompt_ids, list):
        return list(prompt_ids)
    return []


def _extract_sglang_response_ids(response: dict[str, Any], choice: dict[str, Any]) -> list[int]:
    meta = _extract_sglang_meta(response, choice)
    for key in ("output_token_ids", "completion_token_ids", "token_ids"):
        token_ids = meta.get(key)
        if isinstance(token_ids, list):
            return list(token_ids)
    return []


def _extract_sglang_response_logprobs(response: dict[str, Any], choice: dict[str, Any]) -> list[dict[str, Any]] | None:
    meta = _extract_sglang_meta(response, choice)
    token_ids = _extract_sglang_response_ids(response, choice)
    logprobs = meta.get("output_token_logprobs") or meta.get("token_logprobs")
    if not isinstance(logprobs, list) or not token_ids:
        return None
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(logprobs[: len(token_ids)]):
        token_id = token_ids[idx]
        logprob = None
        token = None
        if isinstance(item, (int, float)):
            logprob = float(item)
        elif isinstance(item, (list, tuple)):
            if item:
                logprob = float(item[0]) if isinstance(item[0], (int, float)) else None
            if len(item) > 1:
                token_id = item[1] if isinstance(item[1], int) else token_id
            if len(item) > 2:
                token = item[2]
        elif isinstance(item, dict):
            raw_lp = item.get("logprob", item.get("log_prob"))
            logprob = float(raw_lp) if isinstance(raw_lp, (int, float)) else None
            token_id = item.get("token_id", token_id)
            token = item.get("token")
        normalized.append({"token_id": int(token_id), "logprob": 0.0 if logprob is None else logprob, "token": token})
    return normalized or None


def _extract_prompt_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    return [deepcopy(message) for message in messages if isinstance(message, dict)]


def _extract_tools(request: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    extracted = [deepcopy(tool) for tool in tools if isinstance(tool, dict)]
    return extracted or None


def _request_marker(request: dict[str, Any]) -> dict[str, Any]:
    """Return trajectory-control marker fields from a request.

    The bridge sends internal accounting/probe calls through the same gateway
    as real model turns.  Those calls are marked in ``extra_body`` so builders
    can ignore them without guessing from text or token ids.  Check top-level
    keys as a compatibility fallback for older/local callers.
    """

    marker: dict[str, Any] = {}
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict):
        marker.update(extra_body)
    for key in ("polar_skip_trajectory", "polar_internal", "purpose"):
        if key in request:
            marker[key] = request[key]
    return marker


def is_internal_completion_record(completion: CompletionRecord) -> bool:
    """Whether a completion should be excluded from training trajectories.

    This is intentionally based on explicit request markers, not on decoded
    text.  As a narrow compatibility fallback, also skip the Search bridge's
    historical tool-token-count probe shape (tool-only prompt, max_tokens=0,
    add_generation_prompt=False), which produces an empty assistant response
    and polluted prefix-merging with a bogus extra trace.
    """

    requests = [
        request
        for request in (completion.original_request, completion.request)
        if isinstance(request, dict)
    ]
    for request in requests:
        marker = _request_marker(request)
        if marker.get("polar_skip_trajectory") is True:
            return True
        if marker.get("polar_internal") is True and marker.get("purpose") == "tool_response_token_count":
            return True
        if marker.get("polar_internal") is True and marker.get("purpose") == "prompt_token_count":
            return True

    for request in requests:
        messages = request.get("messages")
        if (
            request.get("max_tokens") == 0
            and request.get("add_generation_prompt") is False
            and isinstance(messages, list)
            and len(messages) == 1
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "tool"
        ):
            return True
    return False


def build_trace_from_completion(completion: CompletionRecord) -> Trace:
    """Normalize one stored completion record into a trajectory trace."""

    request = completion.request if isinstance(completion.request, dict) else {}
    original_request = completion.original_request if isinstance(completion.original_request, dict) else {}
    response = completion.response if isinstance(completion.response, dict) else {}
    choices = response.get("choices")
    first_choice = (
        choices[0]
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else {}
    )
    prompt_ids = _extract_prompt_ids(response, first_choice)
    response_message = first_choice.get("message")
    finish_reason = first_choice.get("finish_reason")

    response_ids = _extract_response_ids(response, first_choice) or _extract_sglang_response_ids(response, first_choice)
    response_logprobs = _extract_response_logprobs(first_choice) or _extract_sglang_response_logprobs(response, first_choice)
    trace_metadata = deepcopy(completion.metadata)
    trace_metadata.setdefault("completion_id", completion.completion_id)
    trace_metadata.setdefault("native_prompt_len", len(prompt_ids))
    trace_metadata.setdefault("native_response_len", len(response_ids))
    trace_metadata.setdefault("native_logprob_len", len(response_logprobs or []))
    trace_metadata.setdefault("native_loss_mask_len", len(response_ids))
    request_id = (
        trace_metadata.get("request_id")
        or request.get("request_id")
        or original_request.get("request_id")
    )
    extra_body = request.get("extra_body") if isinstance(request.get("extra_body"), dict) else {}
    original_extra_body = original_request.get("extra_body") if isinstance(original_request.get("extra_body"), dict) else {}
    for source in (extra_body, original_extra_body):
        raw = source.get("polar_trace_metadata") or source.get("trace_metadata")
        if isinstance(raw, dict) and raw.get("request_id") is not None:
            request_id = raw.get("request_id")
            break
    if request_id is not None:
        trace_metadata.setdefault("request_id", str(request_id))
    trace_metadata.setdefault("builder_source", "completion_record")

    _record_debug(
        "build_trace",
        {
            "completion_id": completion.completion_id,
            "request_id": trace_metadata.get("request_id"),
            "prompt_ids": token_preview(prompt_ids),
            "response_ids": token_preview(response_ids),
            "response_logprob_len": len(response_logprobs or []),
            "finish_reason": finish_reason,
            "request_messages": messages_summary(request.get("messages")),
            "original_messages": messages_summary(original_request.get("messages")),
            "tools_hash": stable_hash(_extract_tools(request)),
            "metadata_keys": sorted(str(k) for k in trace_metadata.keys()),
        },
    )

    return Trace(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=[1] * len(response_ids),
        prompt_messages=_extract_prompt_messages(request),
        response_messages=[deepcopy(response_message)] if isinstance(response_message, dict) else [],
        tools=_extract_tools(request),
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        response_logprobs=response_logprobs,
        metadata=trace_metadata,
    )
