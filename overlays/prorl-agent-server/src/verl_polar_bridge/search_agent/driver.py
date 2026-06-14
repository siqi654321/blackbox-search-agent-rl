"""Standalone SearchR1-like agent driver used by the Polar shell harness.

This driver intentionally depends only on stdlib + requests so it can run inside
simple Polar runtimes. It implements the Search-R1/Qwen-style protocol seen in
existing VERL configs:

- tool call: ``<tool_call>{"name":"search","arguments":{"query_list":[...]}}</tool_call>``
- optional shorthand: ``<query>...</query>`` when explicitly enabled with
  ``POLAR_SEARCH_ENABLE_QUERY_SHORTHAND=1``.  It is disabled by default because
  VERL's baseline HermesToolParser only recognizes ``<tool_call>``.
- final answer: ``<answer>...</answer>``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from verl_polar_bridge.debug_utils import messages_summary, stable_hash, text_preview

import requests

_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_QUERY_RE = re.compile(r"<query>(.*?)</query>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def main() -> None:
    timing = _new_timing()
    timing_start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction-file", required=True)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--retrieval-url", required=True)
    ap.add_argument("--model", default=os.environ.get("SEARCH_MODEL", "qwen3-search-policy"))
    ap.add_argument("--max-turns", type=int, default=int(os.environ.get("SEARCH_MAX_TURNS", "100")))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("SEARCH_MAX_TOKENS", "1024")))
    ap.add_argument("--prompt-length", type=int, default=int(os.environ.get("SEARCH_PROMPT_LENGTH", "4096")))
    ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("SEARCH_MAX_MODEL_LEN", "40960")))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("SEARCH_TEMPERATURE", "1.0")))
    ap.add_argument("--top-p", type=float, default=float(os.environ.get("SEARCH_TOP_P", "1.0")))
    ap.add_argument("--top-k", type=int, default=int(os.environ.get("SEARCH_TOP_K", os.environ.get("SEARCH_TOPK_SAMPLING", "-1"))))
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(os.environ.get("SEARCH_REPETITION_PENALTY", "1.0")),
    )
    ap.add_argument(
        "--do-sample",
        default=os.environ.get("SEARCH_DO_SAMPLE", os.environ.get("SMOKE_ROLLOUT_DO_SAMPLE", "true")),
        help="Whether to sample during policy rollout. false/0/no/off maps to greedy decoding.",
    )
    ap.add_argument("--topk", type=int, default=int(os.environ.get("SEARCH_TOPK") or _configured_topk(10)))
    ap.add_argument("--max-tool-response-length", type=int, default=int(os.environ.get("SEARCH_MAX_TOOL_RESPONSE_LENGTH", "2048")))
    ap.add_argument("--tool-response-truncate-side", choices=["left", "right", "middle"], default=os.environ.get("SEARCH_TOOL_RESPONSE_TRUNCATE_SIDE", "middle"))
    args = ap.parse_args()
    do_sample = _str_to_bool(args.do_sample, default=True)

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "polar")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is required")

    instruction = open(args.instruction_file, encoding="utf-8").read()
    messages: list[dict[str, Any]] = _instruction_to_messages(instruction)
    transcript: list[dict[str, Any]] = []
    final_text = ""
    debug_enabled = _env_flag("POLAR_SEARCH_DRIVER_DEBUG", default=False)
    debug_limit = _env_int("POLAR_SEARCH_DRIVER_DEBUG_LIMIT", 32)
    # Budget accounting is intentionally kept separate from OpenAI
    # ``usage.prompt_tokens``.  A multi-turn Polar harness re-renders the full
    # chat prompt on every completion request, so
    # ``current_prompt_tokens - initial_prompt_tokens + completion_tokens``
    # counts repeated chat-template/tool-schema overhead that VERL's native
    # ToolAgentLoop does not put into ``response_mask``.  The baseline loop
    # lets each assistant turn generate, then truncates the final concatenated
    # response stream to ``response_length``.  For baseline comparison, use the
    # sampled assistant tokens as the scheduling budget and rely on
    # ``polar.overflow_policy=verl_truncate`` / DataProto packing for the final
    # prefix truncation.
    budget_mode = os.environ.get("POLAR_SEARCH_BUDGET_MODE", "verl_response_mask").strip().lower()
    if budget_mode not in {"verl_response_mask", "assistant_only", "prompt_delta", "none"}:
        budget_mode = "verl_response_mask"
    if budget_mode == "verl_response_mask":
        # Backward-compatible name used by earlier debugging runs.
        budget_mode = "assistant_only"
    per_turn_max_tokens = _env_int_optional("POLAR_SEARCH_PER_TURN_MAX_TOKENS")
    bridge_side_scheduling = _env_flag("POLAR_SEARCH_BRIDGE_MAX_TOKENS", default=True)
    initial_prompt_tokens: int | None = None
    legacy_prompt_delta_budget = 0
    consumed_response_budget = 0
    cumulative_completion_tokens = 0

    session_request_id = _request_id("searchr1")
    if debug_enabled:
        _debug_print({
            "event": "start",
            "session_request_id": session_request_id,
            "instruction": text_preview(instruction),
            "messages": messages_summary(messages),
            "tool_schema_names": [_schema_tool_name(schema) for schema in _search_tool_schema()],
            "tool_config_path": _search_tool_config_path(),
            "topk": args.topk,
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "do_sample": do_sample,
            "budget_mode": budget_mode,
            "bridge_side_scheduling": bridge_side_scheduling,
        })
    completion_request_index = 0
    for turn in range(args.max_turns):
        if budget_mode == "prompt_delta":
            sent_max_tokens = max(1, args.max_tokens - consumed_response_budget)
        elif per_turn_max_tokens is not None:
            sent_max_tokens = max(1, per_turn_max_tokens)
        elif bridge_side_scheduling:
            prompt_token_count = None
            # The native bridge renders the exact prompt ids and applies the
            # same VERL async SGLang per-turn cap from that rendered length.
            # This model-window value is only a safety upper bound for any
            # non-bridge endpoint; the bridge overwrites it before /generate.
            sent_max_tokens = max(1, args.max_model_len)
        else:
            prompt_token_count = _prompt_token_count(
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                messages=messages,
                tools=_search_tool_schema(),
                timing=timing,
            )
            if prompt_token_count is None:
                # Fallback for non-bridge endpoints.  Native VERL schedules
                # assistant generation against the whole
                # prompt_length+response_length window, not against the
                # remaining response_length budget.  If the exact rendered
                # prompt length cannot be probed, prefer a generous cap over
                # the old response_length cap; the final DataProto path still
                # truncates to response_length exactly like VERL.
                prompt_token_count = (
                    initial_prompt_tokens + consumed_response_budget
                    if initial_prompt_tokens is not None
                    else 0
                )
            sent_max_tokens = _native_turn_max_tokens(
                prompt_tokens=prompt_token_count,
                prompt_length=args.prompt_length,
                response_length=args.max_tokens,
                max_model_len=args.max_model_len,
            )
        if debug_enabled and turn < debug_limit:
            _debug_print({
                "event": "before_completion",
                "turn": turn,
                "session_request_id": session_request_id,
                "message_count": len(messages),
                "messages": messages_summary(messages),
                "sent_max_tokens": sent_max_tokens,
                "native_prompt_tokens": prompt_token_count if budget_mode != "prompt_delta" and per_turn_max_tokens is None else None,
                "bridge_side_scheduling": bridge_side_scheduling,
            })
        assistant, response = _chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=args.model,
            messages=messages,
            max_tokens=sent_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            do_sample=do_sample,
            tools=_search_tool_schema(),
            trace_metadata={
                "request_id": session_request_id,
                "segment_type": "assistant",
                "segment_index": completion_request_index,
                "harness_mode": "searchr1",
                "harness_event": "assistant_turn",
                "turn": turn,
            },
            bridge_schedule_max_tokens=bridge_side_scheduling and budget_mode != "prompt_delta" and per_turn_max_tokens is None,
            prompt_length=args.prompt_length,
            response_length=args.max_tokens,
            max_model_len=args.max_model_len,
            timing=timing,
        )
        completion_request_index += 1
        final_text = assistant
        messages.append({"role": "assistant", "content": assistant})
        usage = response.get("usage") if isinstance(response, dict) else {}
        prompt_tokens = _usage_int(usage, "prompt_tokens")
        completion_tokens = _usage_int(usage, "completion_tokens")
        if initial_prompt_tokens is None and prompt_tokens is not None:
            initial_prompt_tokens = prompt_tokens
        prompt_delta = max(0, prompt_tokens - initial_prompt_tokens) if prompt_tokens is not None and initial_prompt_tokens is not None else 0
        if completion_tokens is not None:
            cumulative_completion_tokens += completion_tokens
            legacy_prompt_delta_budget = prompt_delta + completion_tokens
            if budget_mode == "prompt_delta":
                consumed_response_budget = legacy_prompt_delta_budget
            elif budget_mode == "assistant_only":
                consumed_response_budget = cumulative_completion_tokens
            else:
                consumed_response_budget = 0
        calls = _extract_tool_calls(assistant, response=response)
        has_answer = bool(_ANSWER_RE.search(assistant))
        budget_exhausted = budget_mode != "none" and consumed_response_budget >= args.max_tokens
        finish_reason = _choice_finish_reason(response)
        transcript.append({
            "turn": turn,
            "assistant": assistant,
            "tool_calls": calls,
            "usage": usage,
            "finish_reason": finish_reason,
            "budget_mode": budget_mode,
            "native_prompt_tokens": prompt_token_count if budget_mode != "prompt_delta" and per_turn_max_tokens is None else None,
            "response_budget_used": consumed_response_budget,
            "legacy_prompt_delta_budget_used": legacy_prompt_delta_budget,
            "cumulative_completion_tokens": cumulative_completion_tokens,
            "sent_max_tokens": sent_max_tokens,
            "remaining_tokens": max(0, args.max_tokens - consumed_response_budget),
            "has_answer": has_answer,
            "budget_exhausted": budget_exhausted,
        })
        if debug_enabled and turn < debug_limit:
            _debug_print({
                "event": "turn",
                "turn": turn,
                "budget_mode": budget_mode,
                "sent_max_tokens": sent_max_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "initial_prompt_tokens": initial_prompt_tokens,
                "prompt_delta": prompt_delta,
                "response_budget_used": consumed_response_budget,
                "legacy_prompt_delta_budget_used": legacy_prompt_delta_budget,
                "cumulative_completion_tokens": cumulative_completion_tokens,
                "budget": args.max_tokens,
                "finish_reason": finish_reason,
                "has_answer": has_answer,
                "tool_call_count": len(calls),
                "tool_call_names": [c.get("name") for c in calls],
                "assistant_len_chars": len(assistant or ""),
                "assistant_hash": stable_hash(assistant or ""),
                "assistant_head": (assistant or "")[:240],
                "assistant_tail": (assistant or "")[-400:],
                "response_choice_keys": sorted(((response.get("choices") or [{}])[0]).keys()) if isinstance(response, dict) and isinstance(response.get("choices"), list) and response.get("choices") else [],
                "message_tool_calls_count": len((((response.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [])) if isinstance(response, dict) and isinstance(response.get("choices"), list) and response.get("choices") else 0,
            })
        if has_answer or not calls or budget_exhausted:
            if debug_enabled:
                _debug_print({
                    "event": "terminate",
                    "turn": turn,
                    "reason": "answer" if has_answer else "no_tool_calls" if not calls else "budget_exhausted",
                    "response_budget_used": consumed_response_budget,
                    "budget": args.max_tokens,
                    "finish_reason": finish_reason,
                })
            break
        allowed_tool_names = _search_tool_names()
        for call in calls:
            if call.get("name") not in allowed_tool_names:
                tool_text = f"Unsupported tool: {call.get('name')}"
            else:
                query_list = call.get("query_list") or []
                tool_text = _call_retrieval(args.retrieval_url, query_list=query_list, topk=args.topk, timing=timing)
                tool_text = _truncate_text(
                    tool_text,
                    max_chars=args.max_tool_response_length,
                    side=args.tool_response_truncate_side,
                )
            messages.append({"role": "tool", "content": tool_text})
            if bridge_side_scheduling and budget_mode == "assistant_only":
                tool_response_tokens = 0
            else:
                tool_response_tokens = _tool_response_token_len(
                    base_url=base_url,
                    api_key=api_key,
                    model=args.model,
                    tool_text=tool_text,
                    timing=timing,
                )
            if budget_mode == "assistant_only":
                consumed_response_budget += tool_response_tokens
            budget_exhausted = budget_mode != "none" and consumed_response_budget >= args.max_tokens
            transcript.append({
                "turn": turn,
                "tool": call,
                "tool_result": tool_text,
                "tool_response_tokens": tool_response_tokens,
                "response_budget_used": consumed_response_budget,
                "budget_exhausted": budget_exhausted,
            })
            if debug_enabled and turn < debug_limit:
                _debug_print({
                    "event": "tool",
                    "turn": turn,
                    "name": call.get("name"),
                    "query_count": len(call.get("query_list") or []),
                    "tool_text_len_chars": len(tool_text or ""),
                    "tool_response_tokens": tool_response_tokens,
                    "response_budget_used": consumed_response_budget,
                    "budget_exhausted": budget_exhausted,
                    "tool_text_head": (tool_text or "")[:240],
                    "tool_text_tail": (tool_text or "")[-240:],
                })
            if budget_exhausted:
                if debug_enabled:
                    _debug_print({
                        "event": "terminate",
                        "turn": turn,
                        "reason": "response_length_after_tool",
                        "response_budget_used": consumed_response_budget,
                        "tool_response_tokens": tool_response_tokens,
                        "budget": args.max_tokens,
                    })
                break
        if budget_exhausted:
            break

    if debug_enabled:
        _debug_print({
            "event": "final",
            "final": text_preview(final_text),
            "message_count": len(messages),
            "messages": messages_summary(messages),
            "transcript_len": len(transcript),
            "response_budget_used": consumed_response_budget,
            "legacy_prompt_delta_budget_used": legacy_prompt_delta_budget,
            "cumulative_completion_tokens": cumulative_completion_tokens,
        })
    timing["driver_total_s"] = time.perf_counter() - timing_start
    timing["completion_overhead_s"] = max(
        0.0,
        timing.get("completion_s", 0.0) - timing.get("bridge_total_s", 0.0),
    )
    timing["bridge_overhead_s"] = max(
        0.0,
        timing.get("bridge_total_s", 0.0) - timing.get("upstream_generate_s", 0.0),
    )
    timing["non_completion_overhead_s"] = max(
        0.0,
        timing["driver_total_s"]
        - timing.get("completion_s", 0.0)
        - timing.get("prompt_probe_s", 0.0)
        - timing.get("tool_token_probe_s", 0.0)
        - timing.get("retrieval_s", 0.0),
    )
    output = {
        "final": final_text,
        "instruction": instruction,
        "messages": messages,
        "transcript": transcript,
        "response_budget_used": consumed_response_budget,
        "legacy_prompt_delta_budget_used": legacy_prompt_delta_budget,
        "cumulative_completion_tokens": cumulative_completion_tokens,
        "budget_mode": budget_mode,
        "max_response_budget": args.max_tokens,
        "timing": timing,
    }
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    _mirror_debug_artifact(output, session_request_id=session_request_id)
    if _env_flag("POLAR_SEARCH_PERF_DEBUG", default=False):
        _debug_print({"event": "perf", "timing": timing})
    print(final_text)


def _instruction_to_messages(instruction: str) -> list[dict[str, Any]]:
    parsed = _parse_json_maybe_nested(instruction)
    if isinstance(parsed, list):
        messages: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            message = dict(item)
            message["role"] = str(message.get("role", "user"))
            if "content" not in message:
                message["content"] = ""
            messages.append(message)
        if messages:
            return messages
    return [{"role": "user", "content": instruction}]


def _parse_json_maybe_nested(text: str, *, max_depth: int = 2) -> Any:
    value: Any = text
    for _ in range(max_depth):
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return value
        try:
            value = json.loads(stripped)
        except Exception:
            return value
    return value


def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    repetition_penalty: float | None,
    do_sample: bool,
    tools: list[dict[str, Any]] | None = None,
    trace_metadata: dict[str, Any] | None = None,
    bridge_schedule_max_tokens: bool = False,
    prompt_length: int | None = None,
    response_length: int | None = None,
    max_model_len: int | None = None,
    timing: dict[str, float] | None = None,
) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "max_tokens": max_tokens,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 1,
    }
    extra_body: dict[str, Any] = {}
    if trace_metadata:
        extra_body["polar_trace_metadata"] = trace_metadata
    if bridge_schedule_max_tokens:
        extra_body.update(
            {
                "polar_bridge_schedule_max_tokens": True,
                "polar_prompt_length": int(prompt_length or 0),
                "polar_response_length": int(response_length or max_tokens),
                "polar_max_model_len": int(max_model_len or 0),
            }
        )
    if extra_body:
        payload["extra_body"] = extra_body
    if tools:
        # Standalone VERL ToolAgentLoop passes the OpenAI tool schema into
        # apply_chat_template(..., tools=self.tool_schemas). Send the same
        # schema through the gateway so native_openai_server can do likewise.
        payload["tools"] = tools
    t0 = time.perf_counter()
    try:
        response = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=900)
        response.raise_for_status()
    finally:
        _timing_add(timing, "completion_s", time.perf_counter() - t0)
        _timing_inc(timing, "completion_count")
    data = response.json()
    metadata = data.get("metadata") if isinstance(data, dict) else None
    bridge_timing = metadata.get("bridge_timing") if isinstance(metadata, dict) else None
    if isinstance(bridge_timing, dict):
        for key, value in bridge_timing.items():
            if isinstance(value, (int, float)):
                _timing_add(timing, str(key), float(value))
    return data["choices"][0]["message"].get("content") or "", data


def _tool_response_token_len(
    *,
    base_url: str,
    api_key: str,
    model: str,
    tool_text: str,
    timing: dict[str, float] | None = None,
) -> int:
    mode = os.environ.get("POLAR_SEARCH_TOOL_TOKEN_COUNT_MODE", "bridge_prompt_delta").strip().lower()
    if mode in {"0", "off", "none", "disabled"}:
        return 0
    t0 = time.perf_counter()
    _timing_inc(timing, "tool_token_probe_count")
    try:
        url = base_url.rstrip("/") + "/chat/completions"
        # Match VERL ToolAgentLoop._handle_processing_tools_state:
        # apply_chat_template(add_messages=[{"role":"tool",...}],
        # remove_system_prompt=True).  The native bridge supports returning
        # input_token_ids, and max_tokens=0 avoids sampling new assistant text.
        payload = {
            "model": model,
            "messages": [{"role": "tool", "content": tool_text or ""}],
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": 0,
            "stream": False,
            "add_generation_prompt": False,
            "chat_template_kwargs": {"remove_system_prompt": True},
            # This is an internal prompt-token-count probe used only to match
            # VERL's tool-response budgeting.  It is not an agent generation
            # turn and must not become a Polar trajectory completion; otherwise
            # prefix_merging sees a tool-only/max_tokens=0 request as a second
            # chain and validation can fan out one input row into two samples.
            "extra_body": {
                "polar_skip_trajectory": True,
                "polar_internal": True,
                "purpose": "tool_response_token_count",
            },
        }
        response = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=900)
        response.raise_for_status()
        data = response.json()
        metadata = data.get("metadata") if isinstance(data, dict) else None
        bridge_timing = metadata.get("bridge_timing") if isinstance(metadata, dict) else None
        if isinstance(bridge_timing, dict):
            for key, value in bridge_timing.items():
                if isinstance(value, (int, float)):
                    _timing_add(timing, f"tool_token_probe_{key}", float(value))
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            ids = choices[0].get("input_token_ids") if isinstance(choices[0], dict) else None
            if isinstance(ids, list):
                return len(ids)
        usage = data.get("usage") if isinstance(data, dict) else None
        value = _usage_int(usage, "prompt_tokens")
        return int(value or 0)
    except Exception:
        # Scheduling/debug-only estimate. If the bridge cannot render the
        # tool-only template, do not perturb training token provenance.
        return 0
    finally:
        _timing_add(timing, "tool_token_probe_s", time.perf_counter() - t0)


def _prompt_token_count(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    timing: dict[str, float] | None = None,
) -> int | None:
    """Return the exact chat-template prompt token count for the next turn.

    Standalone VERL does not cap each assistant turn at ``response_length``.
    Its SGLang agent-loop server computes:

        max_new_tokens = prompt_length + response_length - len(prompt_ids)

    where ``prompt_ids`` is the fully rendered chat prompt for that turn.  The
    Polar driver has only OpenAI-style messages, so ask the native bridge to
    render the same prompt with ``max_tokens=0``.  The request is explicitly
    marked internal so trajectory builders ignore it; newer native bridges
    short-circuit this path without calling SGLang.
    """

    if not _env_flag("POLAR_SEARCH_NATIVE_PROMPT_PROBE", default=True):
        return None
    t0 = time.perf_counter()
    _timing_inc(timing, "prompt_probe_count")
    try:
        url = base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": 0,
            "stream": False,
            "add_generation_prompt": True,
            "extra_body": {
                "polar_skip_trajectory": True,
                "polar_internal": True,
                "purpose": "prompt_token_count",
            },
        }
        if tools:
            payload["tools"] = tools
        response = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=900)
        response.raise_for_status()
        data = response.json()
        metadata = data.get("metadata") if isinstance(data, dict) else None
        bridge_timing = metadata.get("bridge_timing") if isinstance(metadata, dict) else None
        if isinstance(bridge_timing, dict):
            for key, value in bridge_timing.items():
                if isinstance(value, (int, float)):
                    _timing_add(timing, f"prompt_probe_{key}", float(value))
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            ids = choice.get("input_token_ids")
            if isinstance(ids, list):
                return len(ids)
        usage = data.get("usage") if isinstance(data, dict) else None
        return _usage_int(usage, "prompt_tokens")
    except Exception as exc:
        if _env_flag("POLAR_SEARCH_DRIVER_DEBUG", default=False):
            _debug_print({"event": "prompt_token_count_failed", "error": repr(exc)})
        return None
    finally:
        _timing_add(timing, "prompt_probe_s", time.perf_counter() - t0)


def _native_turn_max_tokens(
    *,
    prompt_tokens: int,
    prompt_length: int,
    response_length: int,
    max_model_len: int,
) -> int:
    """Match VERL async SGLang server's default per-turn max_new_tokens."""

    prompt_tokens = max(0, int(prompt_tokens))
    train_window = max(1, int(prompt_length) + int(response_length) - prompt_tokens)
    model_window = max(1, int(max_model_len) - prompt_tokens)
    return max(1, min(train_window, model_window))


def _request_id(prefix: str) -> str:
    session_id = os.environ.get("SESSION_ID") or os.environ.get("POLAR_SESSION_ID") or ""
    if session_id:
        return f"{prefix}-{session_id}"
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _usage_int(usage: Any, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _choice_finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    return str(choice.get("finish_reason")) if isinstance(choice, dict) and choice.get("finish_reason") is not None else None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_int_optional(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _debug_print(payload: dict[str, Any]) -> None:
    print("POLAR_SEARCH_DRIVER_DEBUG " + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


def _new_timing() -> dict[str, float]:
    return {
        "completion_s": 0.0,
        "completion_count": 0.0,
        "completion_overhead_s": 0.0,
        "bridge_overhead_s": 0.0,
        "prompt_probe_s": 0.0,
        "prompt_probe_count": 0.0,
        "prompt_probe_bridge_total_s": 0.0,
        "prompt_probe_prompt_render_s": 0.0,
        "bridge_total_s": 0.0,
        "upstream_generate_s": 0.0,
        "prompt_render_s": 0.0,
        "response_json_s": 0.0,
        "extract_logprobs_s": 0.0,
        "decode_text_s": 0.0,
        "logprob_content_s": 0.0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "meta_output_token_logprobs_len": 0.0,
        "tool_token_probe_s": 0.0,
        "tool_token_probe_count": 0.0,
        "tool_token_probe_bridge_total_s": 0.0,
        "tool_token_probe_prompt_render_s": 0.0,
        "retrieval_s": 0.0,
        "retrieval_count": 0.0,
    }


def _timing_add(timing: dict[str, float] | None, key: str, value: float) -> None:
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0)) + float(value)


def _timing_inc(timing: dict[str, float] | None, key: str, value: float = 1.0) -> None:
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0)) + float(value)


def _mirror_debug_artifact(payload: dict[str, Any], *, session_request_id: str) -> None:
    """Persist compact driver diagnostics outside per-session temp dirs.

    Gateway session directories are intentionally deleted after result
    delivery, so stderr and ``search_agent_output.json`` are often gone by the
    time a compare run finishes.  For baseline alignment debugging, mirror a
    compact copy into ``POLAR_SEARCH_DEBUG_DIR`` or, by default,
    ``$LOG_DIR/artifacts/search_driver_debug``.
    """

    target_dir = (
        os.environ.get("POLAR_SEARCH_DEBUG_DIR")
        or (
            os.path.join(os.environ["LOG_DIR"], "artifacts", "search_driver_debug")
            if os.environ.get("LOG_DIR")
            else ""
        )
    )
    if not target_dir:
        return
    try:
        os.makedirs(target_dir, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_request_id)[-160:] or "session"
        path = os.path.join(target_dir, f"{safe_name}.json")
        turns = [item for item in payload.get("transcript", []) if isinstance(item, dict) and "assistant" in item]
        compact = {
            "session_request_id": session_request_id,
            "budget_mode": payload.get("budget_mode"),
            "max_response_budget": payload.get("max_response_budget"),
            "response_budget_used": payload.get("response_budget_used"),
            "cumulative_completion_tokens": payload.get("cumulative_completion_tokens"),
            "timing": payload.get("timing"),
            "message_count": len(payload.get("messages") or []),
            "assistant_turns": len(turns),
            "turns": [
                {
                    "turn": item.get("turn"),
                    "sent_max_tokens": item.get("sent_max_tokens"),
                    "native_prompt_tokens": item.get("native_prompt_tokens"),
                    "finish_reason": item.get("finish_reason"),
                    "completion_tokens": (item.get("usage") or {}).get("completion_tokens") if isinstance(item.get("usage"), dict) else None,
                    "prompt_tokens": (item.get("usage") or {}).get("prompt_tokens") if isinstance(item.get("usage"), dict) else None,
                    "response_budget_used": item.get("response_budget_used"),
                    "has_answer": item.get("has_answer"),
                    "budget_exhausted": item.get("budget_exhausted"),
                    "tool_call_count": len(item.get("tool_calls") or []),
                    "tool_call_names": [call.get("name") for call in (item.get("tool_calls") or []) if isinstance(call, dict)],
                    "assistant_hash": stable_hash(item.get("assistant") or ""),
                    "assistant_tail": str(item.get("assistant") or "")[-400:],
                }
                for item in turns
            ],
            "tool_response_tokens": [
                item.get("tool_response_tokens")
                for item in payload.get("transcript", [])
                if isinstance(item, dict) and "tool_response_tokens" in item
            ],
            "final_hash": stable_hash(payload.get("final") or ""),
            "final_tail": str(payload.get("final") or "")[-600:],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as exc:
        if _env_flag("POLAR_SEARCH_DRIVER_DEBUG", default=False):
            _debug_print({"event": "mirror_debug_artifact_failed", "error": repr(exc)})


def _search_tool_config_path() -> str:
    return str(
        os.environ.get("POLAR_SEARCH_TOOL_CONFIG_PATH")
        or os.environ.get("STANDALONE_TOOL_CONFIG_PATH")
        or ""
    )


@lru_cache(maxsize=8)
def _load_tool_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        if _env_flag("POLAR_SEARCH_DRIVER_DEBUG", default=False):
            _debug_print({"event": "tool_config_load_failed", "path": path, "error": repr(exc)})
        return {}
    return data if isinstance(data, dict) else {}


def _configured_topk(default: int = 10) -> int:
    data = _load_tool_config(_search_tool_config_path())
    tools = data.get("tools") if isinstance(data, dict) else None
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        cfg = item.get("config")
        if isinstance(cfg, dict) and cfg.get("topk") is not None:
            try:
                return int(cfg.get("topk"))
            except (TypeError, ValueError):
                pass
    return default


@lru_cache(maxsize=8)
def _load_tool_schemas_from_config(path: str) -> tuple[dict[str, Any], ...]:
    """Load OpenAI tool schemas from the same YAML used by VERL standalone.

    Keeping the exact schema text aligned matters: Qwen/Hermes chat templates
    render tool names, descriptions, and argument JSON schema into the prompt.
    A hand-written Polar schema can therefore change prompt_length and rollout
    behavior even when the underlying retrieval function is identical.
    """

    data = _load_tool_config(path)
    if not data:
        return ()
    tools = data.get("tools") if isinstance(data, dict) else None
    if not isinstance(tools, list):
        return ()
    schemas: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        schema = item.get("tool_schema")
        if isinstance(schema, dict):
            schemas.append(schema)
    return tuple(schemas)


def _default_search_tool_schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Search Wikipedia via dense retrieval server and summarize results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to retrieve relevant documents",
                    }
                },
                "required": ["query"],
            },
        },
    }


def _schema_tool_name(schema: dict[str, Any]) -> str | None:
    function = schema.get("function") if isinstance(schema, dict) else None
    if isinstance(function, dict) and function.get("name") is not None:
        return str(function.get("name"))
    return None


def _search_tool_name() -> str:
    # Explicit env wins.  Otherwise, when compare runs provide the standalone
    # tool YAML, derive the tool name from that exact schema.  Fall back to the
    # historical Polar SearchR1Harness name.
    if os.environ.get("POLAR_SEARCH_TOOL_NAME"):
        return str(os.environ["POLAR_SEARCH_TOOL_NAME"])
    schemas = _load_tool_schemas_from_config(_search_tool_config_path())
    for schema in schemas:
        name = _schema_tool_name(schema)
        if name:
            return name
    return "local_search"


def _search_tool_schema() -> list[dict[str, Any]]:
    schemas = list(_load_tool_schemas_from_config(_search_tool_config_path()))
    if schemas:
        return schemas
    return [_default_search_tool_schema(_search_tool_name())]


def _search_tool_names() -> set[str]:
    # Keep search/local_search as compatibility aliases so older checkpoints or
    # hand-written prompts still execute, but render only the configured schema.
    names = {"search", "local_search", _search_tool_name()}
    for schema in _search_tool_schema():
        name = _schema_tool_name(schema)
        if name:
            names.add(name)
    return names


def _query_list_from_args(args: Any) -> list[str]:
    if not isinstance(args, dict):
        return []
    query = args.get("query")
    if isinstance(query, str) and query.strip():
        return [query.strip()]
    query_list = args.get("query_list")
    if isinstance(query_list, list):
        return [str(q) for q in query_list if str(q).strip()]
    if isinstance(query_list, str) and query_list.strip():
        return [query_list.strip()]
    return []


def _extract_tool_calls(text: str, *, response: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    choices = (response or {}).get("choices") if isinstance(response, dict) else None
    message = ((choices or [{}])[0].get("message") or {}) if isinstance(choices, list) and choices else {}
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        fn = tool_call.get("function") or {}
        name = fn.get("name") or tool_call.get("name")
        args = fn.get("arguments") or tool_call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"query": args}
        query_list = _query_list_from_args(args)
        if name and query_list:
            calls.append({"name": name, "query_list": query_list, "raw": tool_call})
    for raw in _TOOL_RE.findall(text or ""):
        try:
            payload = json.loads(raw)
            args = payload.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            calls.append({"name": payload.get("name"), "query_list": _query_list_from_args(args)})
        except Exception:
            continue

    # Baseline alignment: VERL's HermesToolParser only extracts <tool_call> JSON.
    # Keep the legacy <query> shorthand behind an explicit switch for non-baseline
    # experiments; enabling it changes turn counts and response length/reward stats.
    if _env_flag("POLAR_SEARCH_ENABLE_QUERY_SHORTHAND", default=False):
        for raw in _QUERY_RE.findall(text or ""):
            query = raw.strip()
            if query:
                calls.append({"name": _search_tool_name(), "query_list": [query]})
    return calls


def _call_retrieval(
    base_url: str,
    *,
    query_list: list[str],
    topk: int,
    timing: dict[str, float] | None = None,
) -> str:
    if not query_list:
        return "No query provided."
    url = _retrieval_endpoint(base_url)
    timeout = float(os.environ.get("SEARCH_RETRIEVAL_TIMEOUT", "6000"))
    payload = {"queries": query_list, "topk": topk, "return_scores": True}
    t0 = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 422:
            response = requests.post(url, json={"query_list": query_list, "topk": topk, "return_scores": True}, timeout=timeout)
        response.raise_for_status()
    finally:
        _timing_add(timing, "retrieval_s", time.perf_counter() - t0)
        _timing_inc(timing, "retrieval_count")
    data = response.json()
    return _format_retrieval(data)


def _retrieval_endpoint(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/retrieve") or stripped.endswith("/retrieve_summarize_compat"):
        return stripped
    path = os.environ.get("SEARCH_RETRIEVAL_PATH", "retrieve_summarize_compat").strip("/")
    return stripped + "/" + path


def _format_retrieval(data: Any) -> str:
    if isinstance(data, str):
        return data
    raw_results = data.get("result") if isinstance(data, dict) else None
    if isinstance(raw_results, list):
        pretty_results: list[str] = []
        for retrieval in raw_results:
            pretty_results.append(_passages_to_string(retrieval))
        formatted = "\n---\n".join(item for item in pretty_results if item)
        if formatted:
            return json.dumps({"result": formatted}, ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False)


def _passages_to_string(retrieval_result: Any) -> str:
    """Match VERL SearchTool's formatted retrieval observation where possible."""
    if not isinstance(retrieval_result, list):
        return json.dumps(retrieval_result, ensure_ascii=False)
    chunks: list[str] = []
    for idx, doc_item in enumerate(retrieval_result):
        if not isinstance(doc_item, dict):
            chunks.append(str(doc_item))
            continue
        document = doc_item.get("document") if isinstance(doc_item.get("document"), dict) else {}
        content = str(document.get("contents", ""))
        title, _, text = content.partition("\n")
        if not title:
            title = str(document.get("title", f"Doc {idx + 1}"))
        body = text if text else content
        chunks.append(f"Doc {idx + 1} (Title: {title})\n{body}\n")
    return "\n".join(chunks).strip()


def _truncate_text(text: str, *, max_chars: int, side: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    # Match VERL ToolAgentLoop._call_tool truncation exactly, including the
    # marker text and the historical left/right side semantics.
    if side == "left":
        return text[:max_chars] + "...(truncated)"
    if side == "right":
        return "(truncated)..." + text[-max_chars:]
    length = max_chars // 2
    return text[:length] + "...(truncated)..." + text[-length:]


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _str_to_bool(raw, default=default)


def _str_to_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


if __name__ == "__main__":
    main()
