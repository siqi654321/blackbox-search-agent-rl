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
import html
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
    compaction_enabled = _env_flag_any(
        ("POLAR_SEARCH_WIPE_ENABLE", "POLAR_SEARCH_COMPACTION_ENABLE"),
        default=False,
    )
    compaction_max_turns = _env_int_first(
        ("POLAR_SEARCH_WIPE_MAX_TURNS", "POLAR_SEARCH_COMPACTION_MAX_TURNS"),
        0,
    )
    compaction_context_ratio = _env_float_first(
        ("POLAR_SEARCH_WIPE_CONTEXT_RATIO", "POLAR_SEARCH_COMPACTION_CONTEXT_RATIO"),
        0.0,
    )
    current_merge_group_index = 0
    num_merge_groups_estimate = 1
    deferred_wipe_reasons: list[str] = []
    deferred_wipe_prompt_tokens: int | None = None
    subagent_enabled = _env_flag("POLAR_SEARCH_SUBAGENT_ENABLE", default=False)
    max_subagents = _env_int("POLAR_SEARCH_MAX_SUBAGENTS", 1)
    subagent_count = 0
    subagent_reports: list[dict[str, Any]] = []
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
    main_merge_group_id = f"{session_request_id}:main"
    if debug_enabled:
        _debug_print({
            "event": "start",
            "session_request_id": session_request_id,
            "instruction": text_preview(instruction),
            "messages": messages_summary(messages),
            "tool_schema_names": [_schema_tool_name(schema) for schema in _main_tool_schema(subagent_enabled=subagent_enabled)],
            "subagent_enabled": subagent_enabled,
            "max_subagents": max_subagents,
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
            "wipe_enabled": compaction_enabled,
            "wipe_max_turns": compaction_max_turns,
            "wipe_context_ratio": compaction_context_ratio,
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
                tools=_main_tool_schema(subagent_enabled=subagent_enabled),
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
        current_segment_group_id = _main_segment_group_id(
            main_merge_group_id,
            current_merge_group_index,
            segmented=compaction_enabled,
        )
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
            tools=_main_tool_schema(subagent_enabled=subagent_enabled),
            trace_metadata={
                "request_id": session_request_id,
                "segment_type": "final" if not compaction_enabled else "wipe",
                "segment_kind": "final" if not compaction_enabled else "wipe",
                "is_final_segment": True if not compaction_enabled else False,
                "merge_group_id": current_segment_group_id,
                "segment_group_id": current_segment_group_id,
                "parent_merge_group_id": main_merge_group_id,
                "merge_group_index": current_merge_group_index if compaction_enabled else 0,
                "num_merge_groups": num_merge_groups_estimate,
                "segment_weight": 1.0,
                "segment_index": completion_request_index,
                "harness_mode": "searchr1",
                "harness_event": "main_turn",
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
        boundary_reasons = _segment_boundary_reasons(
            enabled=compaction_enabled,
            turn=turn,
            max_turns=compaction_max_turns,
            prompt_tokens=prompt_tokens,
            max_model_len=args.max_model_len,
            ratio=compaction_context_ratio,
        )
        if boundary_reasons:
            _record_wipe_candidate(timing, boundary_reasons, prompt_tokens=prompt_tokens)

        if has_answer or not calls or budget_exhausted:
            if boundary_reasons or deferred_wipe_reasons:
                _timing_inc(timing, "wipe_terminal_skip_count")
                if deferred_wipe_reasons:
                    _timing_inc(timing, "wipe_deferred_dropped_count")
                if debug_enabled:
                    _debug_print({
                        "event": "wipe_skip_terminal",
                        "turn": turn,
                        "reasons": list(deferred_wipe_reasons or boundary_reasons),
                        "prompt_tokens": deferred_wipe_prompt_tokens if deferred_wipe_reasons else prompt_tokens,
                        "has_answer": has_answer,
                        "tool_call_count": len(calls),
                        "budget_exhausted": budget_exhausted,
                    })
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

        apply_reasons_before_tools = list(deferred_wipe_reasons or boundary_reasons)
        apply_prompt_tokens_before_tools = deferred_wipe_prompt_tokens if deferred_wipe_reasons else prompt_tokens
        if apply_reasons_before_tools:
            if _safe_to_compact_messages(messages):
                if deferred_wipe_reasons:
                    _timing_inc(timing, "wipe_deferred_applied_count")
                messages, current_merge_group_index, num_merge_groups_estimate = _apply_wipe_compaction(
                    messages,
                    timing=timing,
                    transcript=transcript,
                    turn=turn,
                    reasons=apply_reasons_before_tools,
                    prompt_tokens=apply_prompt_tokens_before_tools,
                    max_model_len=args.max_model_len,
                    context_ratio=compaction_context_ratio,
                    current_merge_group_index=current_merge_group_index,
                    num_merge_groups_estimate=num_merge_groups_estimate,
                    debug_enabled=debug_enabled,
                    preserve_tail_messages=1,
                    apply_point="before_tools",
                )
                deferred_wipe_reasons = []
                deferred_wipe_prompt_tokens = None
            else:
                if not deferred_wipe_reasons:
                    deferred_wipe_reasons = list(boundary_reasons)
                    deferred_wipe_prompt_tokens = prompt_tokens
                    _timing_inc(timing, "wipe_unsafe_tail_deferred_count")
                    if debug_enabled:
                        _debug_print({
                            "event": "wipe_defer_unsafe_tail",
                            "turn": turn,
                            "reasons": boundary_reasons,
                            "prompt_tokens": prompt_tokens,
                            "tail_role": _message_tail_role(messages),
                            "apply_point": "before_tools",
                        })
        allowed_tool_names = _search_tool_names()
        subagent_executed_this_turn = False
        for call in calls:
            is_subagent_call = _is_subagent_tool(call.get("name"))
            if is_subagent_call:
                _timing_inc(timing, "subagent_requested_count")
                if not subagent_enabled:
                    _timing_inc(timing, "subagent_ignored_count")
                    tool_text = "Subagent tool is disabled."
                elif subagent_count >= max_subagents:
                    _timing_inc(timing, "subagent_ignored_count")
                    tool_text = "Subagent ignored: maximum subagent calls reached."
                else:
                    sub_args = call.get("subagent") or {}
                    subagent_count += 1
                    dispatch_index = subagent_count - 1
                    _timing_inc(timing, "subagent_applied_count")
                    subagent_result = _run_subagent(
                        base_url=base_url,
                        api_key=api_key,
                        model=args.model,
                        retrieval_url=args.retrieval_url,
                        parent_instruction=instruction,
                        task=str(sub_args.get("task") or ""),
                        context=str(sub_args.get("context") or ""),
                        requested_max_turns=sub_args.get("max_turns"),
                        dispatch_index=dispatch_index,
                        session_request_id=session_request_id,
                        main_merge_group_id=main_merge_group_id,
                        args=args,
                        do_sample=do_sample,
                        bridge_side_scheduling=bridge_side_scheduling,
                        budget_mode=budget_mode,
                        per_turn_max_tokens=per_turn_max_tokens,
                        timing=timing,
                    )
                    report = str(subagent_result.get("report") or "")
                    tool_text = _format_subagent_report(report, task=str(sub_args.get("task") or ""))
                    subagent_reports.append(
                        {
                            "dispatch_index": dispatch_index,
                            "task": sub_args.get("task"),
                            "context": sub_args.get("context"),
                            "report": report,
                            "messages": subagent_result.get("messages") or [],
                            "transcript": subagent_result.get("transcript") or [],
                        }
                    )
                subagent_executed_this_turn = True
            elif call.get("name") not in allowed_tool_names:
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
                    "is_subagent": bool(is_subagent_call),
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
            if subagent_executed_this_turn:
                # Keep first implementation deterministic: one subagent dispatch per main turn.
                break
        if budget_exhausted:
            if deferred_wipe_reasons:
                _timing_inc(timing, "wipe_deferred_dropped_count")
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
        "subagent_reports": subagent_reports,
        "timing": timing,
    }
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    _mirror_debug_artifact(output, session_request_id=session_request_id)
    _maybe_dump_interaction_artifact(output, session_request_id=session_request_id)
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


def _run_subagent(
    *,
    base_url: str,
    api_key: str,
    model: str,
    retrieval_url: str,
    parent_instruction: str,
    task: str,
    context: str,
    requested_max_turns: Any,
    dispatch_index: int,
    session_request_id: str,
    main_merge_group_id: str,
    args: argparse.Namespace,
    do_sample: bool,
    bridge_side_scheduling: bool,
    budget_mode: str,
    per_turn_max_tokens: int | None,
    timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run a focused search-only subagent and return its report plus transcript.

    Subagent completions use their own merge group so adapter-side stitching
    produces a trainable ``segment_kind=subagent`` sample.  The report returned
    to the main loop becomes a non-trainable tool observation in the final/main
    segment.  The transcript/messages are only for human-readable interaction
    artifacts and do not affect training.
    """

    t0 = time.perf_counter()
    default_turns = _env_int("POLAR_SEARCH_SUBAGENT_MAX_TURNS", 3)
    try:
        max_turns = int(requested_max_turns) if requested_max_turns is not None else default_turns
    except (TypeError, ValueError):
        max_turns = default_turns
    max_turns = max(1, min(max_turns, default_turns if default_turns > 0 else max_turns))
    max_tokens = max(1, _env_int("POLAR_SEARCH_SUBAGENT_MAX_TOKENS", min(int(args.max_tokens), 4096)))
    report_max_chars = max(1, _env_int("POLAR_SEARCH_SUBAGENT_REPORT_MAX_CHARS", 4096))
    report_format = os.environ.get("POLAR_SEARCH_SUBAGENT_REPORT_FORMAT", "sections").strip().lower()
    if report_format not in {"sections", "freeform"}:
        report_format = "sections"
    expected_output = _truncate_text(
        str(context or ""),
        max_chars=max(1, _env_int("POLAR_SEARCH_SUBAGENT_CONTEXT_MAX_CHARS", 2048)),
        side="middle",
    )
    sub_merge_group_id = f"{session_request_id}:subagent:{int(dispatch_index)}"
    sub_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a focused research sub-agent. Use the search tool to gather evidence when useful. "
                "Investigate only the delegated task and return a concise evidence-based report for the main agent. "
                "Do not call subagents. Do not answer the original task directly unless the delegated task asks for it. "
                "Prefer independent verification when evidence is ambiguous, contradictory, or multi-hop. "
                + (
                    "Return exactly these sections: Findings, Evidence, Uncertainty, Recommendation to main agent."
                    if report_format == "sections"
                    else "Return a concise report for the main agent."
                )
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question:\n{parent_instruction}\n\n"
                f"Main-agent context:\n{expected_output}\n\n"
                f"Sub-agent task:\n{task}\n\n"
                + (
                    "Investigate this task and return a report with:\n"
                    "Findings:\n- concise conclusions\n"
                    "Evidence:\n- search-backed facts or observations\n"
                    "Uncertainty:\n- caveats, conflicts, or missing evidence\n"
                    "Recommendation to main agent:\n- how the main agent should use this report\n"
                    if report_format == "sections"
                    else "Investigate this task and return a concise report."
                )
            ),
        },
    ]
    sub_transcript: list[dict[str, Any]] = []
    last_assistant = ""
    completion_index = 0
    for sub_turn in range(max_turns):
        if per_turn_max_tokens is not None:
            sent_max_tokens = max(1, min(per_turn_max_tokens, max_tokens))
        elif bridge_side_scheduling and budget_mode != "prompt_delta":
            sent_max_tokens = max(1, args.max_model_len)
        else:
            prompt_tokens = _prompt_token_count(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=sub_messages,
                tools=_search_tool_schema(),
                timing=timing,
            )
            if prompt_tokens is None:
                prompt_tokens = 0
            sent_max_tokens = min(
                max_tokens,
                _native_turn_max_tokens(
                    prompt_tokens=prompt_tokens,
                    prompt_length=int(args.prompt_length),
                    response_length=max_tokens,
                    max_model_len=int(args.max_model_len),
                ),
            )
        assistant, response = _chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=sub_messages,
            max_tokens=sent_max_tokens,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            repetition_penalty=float(args.repetition_penalty),
            do_sample=do_sample,
            tools=_search_tool_schema(),
            trace_metadata={
                "request_id": session_request_id,
                "segment_type": "subagent",
                "segment_kind": "subagent",
                "merge_group_id": sub_merge_group_id,
                "segment_group_id": sub_merge_group_id,
                "merge_group_index": int(dispatch_index) + 1,
                "parent_merge_group_id": main_merge_group_id,
                "dispatch_index": int(dispatch_index),
                "segment_index": completion_index,
                "harness_mode": "searchr1",
                "harness_event": "subagent_turn",
                "subagent_turn": sub_turn,
            },
            bridge_schedule_max_tokens=bridge_side_scheduling and budget_mode != "prompt_delta" and per_turn_max_tokens is None,
            prompt_length=int(args.prompt_length),
            response_length=max_tokens,
            max_model_len=int(args.max_model_len),
            timing=timing,
        )
        completion_index += 1
        last_assistant = assistant
        sub_messages.append({"role": "assistant", "content": assistant})
        usage = response.get("usage") if isinstance(response, dict) else {}
        if (value := _usage_int(usage, "prompt_tokens")) is not None:
            _timing_inc(timing, "subagent_prompt_tokens", float(value))
        if (value := _usage_int(usage, "completion_tokens")) is not None:
            _timing_inc(timing, "subagent_completion_tokens", float(value))
        _timing_inc(timing, "subagent_turns")
        calls = [call for call in _extract_tool_calls(assistant, response=response) if not _is_subagent_tool(call.get("name"))]
        sub_transcript.append(
            {
                "turn": sub_turn,
                "assistant": assistant,
                "tool_calls": calls,
                "usage": usage,
                "finish_reason": _choice_finish_reason(response),
                "sent_max_tokens": sent_max_tokens,
                "has_answer": bool(_ANSWER_RE.search(assistant)),
            }
        )
        if _ANSWER_RE.search(assistant) or not calls:
            break
        allowed_tool_names = _search_tool_names()
        for call in calls:
            if call.get("name") not in allowed_tool_names:
                tool_text = f"Unsupported tool: {call.get('name')}"
            else:
                query_list = call.get("query_list") or []
                tool_text = _call_retrieval(retrieval_url, query_list=query_list, topk=int(args.topk), timing=timing)
                tool_text = _truncate_text(
                    tool_text,
                    max_chars=int(args.max_tool_response_length),
                    side=str(args.tool_response_truncate_side),
                )
            sub_messages.append({"role": "tool", "content": tool_text})
            sub_transcript.append(
                {
                    "turn": sub_turn,
                    "tool": call,
                    "tool_result": tool_text,
                }
            )
            _timing_inc(timing, "subagent_tool_turns")
    report = _normalize_subagent_report(
        last_assistant or "Subagent produced no report.",
        report_format=report_format,
    )
    report = _truncate_text(report, max_chars=report_max_chars, side="middle")
    _timing_inc(timing, "subagent_report_chars", float(len(report)))
    _timing_add(timing, "subagent_s", time.perf_counter() - t0)
    return {
        "dispatch_index": int(dispatch_index),
        "task": task,
        "context": context,
        "report": report,
        "messages": sub_messages,
        "transcript": sub_transcript,
        "merge_group_id": sub_merge_group_id,
    }


def _format_subagent_report(report: str, *, task: str) -> str:
    guidance = (
        "Use this report as non-authoritative evidence. Continue reasoning and verify if needed."
    )
    return (
        "Sub-agent report"
        + (f" for task: {task}" if task else "")
        + "\n\n"
        + guidance
        + "\n\n"
        + (report or "")
    )


def _normalize_subagent_report(report: str, *, report_format: str) -> str:
    report = str(report or "").strip()
    if report_format != "sections":
        return report
    section_names = ("Findings:", "Evidence:", "Uncertainty:", "Recommendation to main agent:")
    if all(name in report for name in section_names):
        return report
    return (
        "Findings:\n"
        f"{report or 'No findings reported.'}\n\n"
        "Evidence:\n"
        "- See the sub-agent text above; no explicit evidence section was produced.\n\n"
        "Uncertainty:\n"
        "- Not specified by the sub-agent.\n\n"
        "Recommendation to main agent:\n"
        "- Treat the findings as supporting context and verify against the original question."
    )


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


def _env_int_first(names: tuple[str, ...], default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            return int(default)
    return int(default)


def _env_int_optional(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_float_first(names: tuple[str, ...], default: float) -> float:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return float(raw)
        except ValueError:
            return float(default)
    return float(default)


def _main_segment_group_id(main_merge_group_id: str, merge_group_index: int, *, segmented: bool) -> str:
    if not segmented:
        return main_merge_group_id
    return f"{main_merge_group_id}:wipe:{int(merge_group_index)}"


def _segment_boundary_reasons(
    *,
    enabled: bool,
    turn: int,
    max_turns: int,
    prompt_tokens: int | None,
    max_model_len: int,
    ratio: float,
) -> list[str]:
    if not enabled:
        return []
    by_turn = max_turns > 0 and (turn + 1) % max_turns == 0
    by_context = (
        ratio > 0.0
        and prompt_tokens is not None
        and max_model_len > 0
        and float(prompt_tokens) >= float(max_model_len) * float(ratio)
    )
    reasons: list[str] = []
    if by_turn:
        reasons.append("turn")
    if by_context:
        reasons.append("context")
    return reasons


def _should_start_new_segment(
    *,
    enabled: bool,
    turn: int,
    max_turns: int,
    prompt_tokens: int | None,
    max_model_len: int,
    ratio: float,
) -> bool:
    return bool(
        _segment_boundary_reasons(
            enabled=enabled,
            turn=turn,
            max_turns=max_turns,
            prompt_tokens=prompt_tokens,
            max_model_len=max_model_len,
            ratio=ratio,
        )
    )


def _record_wipe_candidate(
    timing: dict[str, float] | None,
    reasons: list[str],
    *,
    prompt_tokens: int | None,
) -> None:
    if not reasons:
        return
    _timing_inc(timing, "wipe_candidate_count")
    if "turn" in reasons:
        _timing_inc(timing, "wipe_turn_candidate_count")
    if "context" in reasons:
        _timing_inc(timing, "wipe_context_candidate_count")
    if prompt_tokens is not None:
        _timing_inc(timing, "wipe_candidate_prompt_tokens", float(prompt_tokens))


def _message_tail_role(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    return str(messages[-1].get("role") or "")


def _safe_to_compact_messages(messages: list[dict[str, Any]]) -> bool:
    return _message_tail_role(messages) != "tool"


def _apply_wipe_compaction(
    messages: list[dict[str, Any]],
    *,
    timing: dict[str, float] | None,
    transcript: list[dict[str, Any]],
    turn: int,
    reasons: list[str],
    prompt_tokens: int | None,
    max_model_len: int,
    context_ratio: float,
    current_merge_group_index: int,
    num_merge_groups_estimate: int,
    debug_enabled: bool,
    preserve_tail_messages: int = 0,
    apply_point: str = "after_tools",
) -> tuple[list[dict[str, Any]], int, int]:
    message_count_before = len(messages)
    messages = _compact_messages(messages, preserve_tail_messages=preserve_tail_messages)
    message_count_after = len(messages)
    current_merge_group_index += 1
    num_merge_groups_estimate = max(num_merge_groups_estimate, current_merge_group_index + 1)
    _timing_inc(timing, "wipe_count")
    if apply_point == "before_tools":
        _timing_inc(timing, "wipe_before_tools_count")
    if "turn" in reasons:
        _timing_inc(timing, "wipe_by_turn_count")
    if "context" in reasons:
        _timing_inc(timing, "wipe_by_context_count")
    if prompt_tokens is not None:
        _timing_inc(timing, "wipe_prompt_tokens", float(prompt_tokens))
    _timing_inc(timing, "wipe_message_count_before", float(message_count_before))
    _timing_inc(timing, "wipe_message_count_after", float(message_count_after))
    transcript.append(
        {
            "turn": turn,
            "wipe": {
                "reasons": list(reasons),
                "prompt_tokens": prompt_tokens,
                "max_model_len": max_model_len,
                "context_ratio": context_ratio,
                "message_count_before": message_count_before,
                "message_count_after": message_count_after,
                "next_merge_group_index": current_merge_group_index,
                "preserve_tail_messages": int(preserve_tail_messages),
                "apply_point": str(apply_point),
            },
        }
    )
    if debug_enabled:
        _debug_print(
            {
                "event": "wipe_segment_boundary",
                "legacy_event": "compaction_segment_boundary",
                "turn": turn,
                "reasons": reasons,
                "merge_group_index": current_merge_group_index,
                "num_merge_groups_estimate": num_merge_groups_estimate,
                "message_count_before_wipe": message_count_before,
                "message_count_after_wipe": message_count_after,
                "preserve_tail_messages": int(preserve_tail_messages),
                "apply_point": str(apply_point),
            }
        )
    return messages, current_merge_group_index, num_merge_groups_estimate


def _compact_messages(messages: list[dict[str, Any]], *, preserve_tail_messages: int = 0) -> list[dict[str, Any]]:
    """Lightweight local wipe/compaction placeholder.

    This creates a hard merge boundary for training while avoiding an extra
    summarizer dependency in the first implementation.  It keeps the initial
    system/user context and appends a compact transcript digest as a user
    message.  The next assistant turn therefore starts a new merge group.
    """

    if len(messages) <= 2:
        return messages
    preserve_tail_messages = max(0, int(preserve_tail_messages))
    if preserve_tail_messages > 0 and len(messages) <= 2 + preserve_tail_messages:
        return messages
    head = messages[:2]
    preserved_tail = messages[-preserve_tail_messages:] if preserve_tail_messages > 0 else []
    body_end = len(messages) - preserve_tail_messages if preserve_tail_messages > 0 else len(messages)
    body = messages[2:body_end]
    if len(body) > 8:
        compact_source = body[:-8]
        tail = body[-8:] + preserved_tail
    else:
        compact_source = body
        tail = preserved_tail
    digest_items = []
    for msg in compact_source:
        role = str(msg.get("role", ""))
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        digest_items.append(f"{role}: {content[:300]}")
    digest = "\n".join(digest_items)
    compact_msg = {
        "role": "user",
        "content": "[context compacted]\n" + digest[-4000:],
    }
    return head + [compact_msg] + tail


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
        "subagent_s": 0.0,
        "subagent_requested_count": 0.0,
        "subagent_applied_count": 0.0,
        "subagent_ignored_count": 0.0,
        "subagent_turns": 0.0,
        "subagent_tool_turns": 0.0,
        "subagent_prompt_tokens": 0.0,
        "subagent_completion_tokens": 0.0,
        "subagent_report_chars": 0.0,
        "wipe_candidate_count": 0.0,
        "wipe_turn_candidate_count": 0.0,
        "wipe_context_candidate_count": 0.0,
        "wipe_terminal_skip_count": 0.0,
        "wipe_unsafe_tail_deferred_count": 0.0,
        "wipe_deferred_applied_count": 0.0,
        "wipe_deferred_dropped_count": 0.0,
        "wipe_before_tools_count": 0.0,
        "wipe_count": 0.0,
        "wipe_by_turn_count": 0.0,
        "wipe_by_context_count": 0.0,
        "wipe_prompt_tokens": 0.0,
        "wipe_message_count_before": 0.0,
        "wipe_message_count_after": 0.0,
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
            "wipe_enabled": _env_flag_any(
                ("POLAR_SEARCH_WIPE_ENABLE", "POLAR_SEARCH_COMPACTION_ENABLE"),
                default=False,
            ),
            "wipe_max_turns": _env_int_first(
                ("POLAR_SEARCH_WIPE_MAX_TURNS", "POLAR_SEARCH_COMPACTION_MAX_TURNS"),
                0,
            ),
            "wipe_context_ratio": _env_float_first(
                ("POLAR_SEARCH_WIPE_CONTEXT_RATIO", "POLAR_SEARCH_COMPACTION_CONTEXT_RATIO"),
                0.0,
            ),
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
            "wipe_events": [
                item.get("wipe")
                for item in payload.get("transcript", [])
                if isinstance(item, dict) and isinstance(item.get("wipe"), dict)
            ],
            "final_hash": stable_hash(payload.get("final") or ""),
            "final_tail": str(payload.get("final") or "")[-600:],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as exc:
        if _env_flag("POLAR_SEARCH_DRIVER_DEBUG", default=False):
            _debug_print({"event": "mirror_debug_artifact_failed", "error": repr(exc)})


def _maybe_dump_interaction_artifact(payload: dict[str, Any], *, session_request_id: str) -> None:
    """Dump a human-readable HTML transcript for sessions with subagent + wipe.

    This artifact is intentionally driver-side rather than training-sample-side:
    it preserves the chronological interaction view (main turns, tool results,
    wipe boundary, nested subagent turns, and final answer) and is filtered to
    only sessions where both a subagent was applied and wipe actually happened.
    """

    if not _env_flag("POLAR_SUBAGENT_WIPE_INTERACTION_ARTIFACT", default=False):
        return
    timing = payload.get("timing") if isinstance(payload.get("timing"), dict) else {}
    has_subagent = float((timing or {}).get("subagent_applied_count") or 0.0) > 0.0
    has_wipe = float((timing or {}).get("wipe_count") or 0.0) > 0.0
    if not (has_subagent and has_wipe):
        return

    target_dir = (
        os.environ.get("POLAR_SUBAGENT_WIPE_INTERACTION_DIR")
        or (
            os.path.join(os.environ["LOG_DIR"], "artifacts", "subagent_wipe_interactions")
            if os.environ.get("LOG_DIR")
            else ""
        )
    )
    if not target_dir:
        return
    try:
        os.makedirs(target_dir, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_request_id)[-160:] or "session"
        json_path = os.path.join(target_dir, f"{safe_name}.json")
        html_path = os.path.join(target_dir, f"{safe_name}.html")
        index_path = os.path.join(target_dir, "index.html")

        data = _interaction_payload(payload, session_request_id=session_request_id)
        fmt = os.environ.get("POLAR_SUBAGENT_WIPE_INTERACTION_FORMAT", "html").strip().lower()
        write_json = fmt in {"json", "both", "html,json", "json,html"}
        write_html = fmt in {"", "html", "both", "html,json", "json,html"}
        if write_json:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        if write_html:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(_render_interaction_html(data))
            _write_interaction_index(target_dir, index_path=index_path)
    except Exception as exc:
        if _env_flag("POLAR_SEARCH_DRIVER_DEBUG", default=False):
            _debug_print({"event": "interaction_artifact_failed", "error": repr(exc)})


def _interaction_payload(payload: dict[str, Any], *, session_request_id: str) -> dict[str, Any]:
    timing = payload.get("timing") if isinstance(payload.get("timing"), dict) else {}
    transcript = payload.get("transcript") if isinstance(payload.get("transcript"), list) else []
    return {
        "session_request_id": session_request_id,
        "instruction": payload.get("instruction") or "",
        "prompt": _readable_prompt(payload.get("instruction") or ""),
        "final": payload.get("final") or "",
        "answer": _extract_answer(payload.get("final") or ""),
        "timing": timing or {},
        "summary": {
            "main_turns": len([item for item in transcript if isinstance(item, dict) and "assistant" in item]),
            "subagent_requested": float((timing or {}).get("subagent_requested_count") or 0.0),
            "subagent_applied": float((timing or {}).get("subagent_applied_count") or 0.0),
            "subagent_ignored": float((timing or {}).get("subagent_ignored_count") or 0.0),
            "wipe_count": float((timing or {}).get("wipe_count") or 0.0),
            "wipe_by_turn": float((timing or {}).get("wipe_by_turn_count") or 0.0),
            "wipe_by_context": float((timing or {}).get("wipe_by_context_count") or 0.0),
        },
        "wipe_config": {
            "enabled": _env_flag_any(
                ("POLAR_SEARCH_WIPE_ENABLE", "POLAR_SEARCH_COMPACTION_ENABLE"),
                default=False,
            ),
            "max_turns": _env_int_first(
                ("POLAR_SEARCH_WIPE_MAX_TURNS", "POLAR_SEARCH_COMPACTION_MAX_TURNS"),
                0,
            ),
            "context_ratio": _env_float_first(
                ("POLAR_SEARCH_WIPE_CONTEXT_RATIO", "POLAR_SEARCH_COMPACTION_CONTEXT_RATIO"),
                0.0,
            ),
        },
        "subagent_config": {
            "enabled": _env_flag("POLAR_SEARCH_SUBAGENT_ENABLE", default=False),
            "max_subagents": _env_int("POLAR_SEARCH_MAX_SUBAGENTS", 1),
            "max_turns": _env_int("POLAR_SEARCH_SUBAGENT_MAX_TURNS", 3),
            "max_tokens": _env_int("POLAR_SEARCH_SUBAGENT_MAX_TOKENS", 4096),
        },
        "transcript": transcript,
        "subagent_reports": payload.get("subagent_reports") or [],
        "messages": payload.get("messages") or [],
        "response_budget_used": payload.get("response_budget_used"),
        "max_response_budget": payload.get("max_response_budget"),
    }


def _write_interaction_index(target_dir: str, *, index_path: str) -> None:
    entries: list[dict[str, str]] = []
    for path in sorted(Path(target_dir).glob("*.html")):
        if path.name == "index.html":
            continue
        title = path.stem
        entries.append({"file": path.name, "title": title})
    items = "\n".join(
        f'<tr><td><a href="{html.escape(entry["file"])}">{html.escape(entry["title"])}</a></td></tr>'
        for entry in entries
    )
    body = items or '<tr><td class="muted">No interaction artifacts yet.</td></tr>'
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Subagent + Wipe Interactions</title>
  <style>{_interaction_css()}</style>
</head>
<body>
  <main class="index">
    <h1>Subagent + Wipe Interactions</h1>
    <p class="muted">Sessions are dumped only when both a subagent was applied and wipe happened.</p>
    <table><thead><tr><th>Session</th></tr></thead><tbody>{body}</tbody></table>
  </main>
</body>
</html>
"""
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def _render_interaction_html(data: dict[str, Any]) -> str:
    session = str(data.get("session_request_id") or "")
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    final = str(data.get("final") or "")
    answer = str(data.get("answer") or "")
    prompt = str(data.get("prompt") or "")
    timeline_html = _render_interaction_timeline(data)
    toc_html = _render_interaction_toc(data)
    final_answer = answer or _extract_answer(final) or "(not found)"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Subagent + Wipe Interaction: {html.escape(session)}</title>
  <style>{_interaction_css()}</style>
</head>
<body>
  <header class="top">
    <div>
      <h1>Subagent + Wipe Interaction</h1>
      <div class="session">{html.escape(session)}</div>
    </div>
    <div class="badges">
      <span class="badge good">subagent ✓</span>
      <span class="badge wipe">wipe ✓</span>
      <span class="badge answer">answer ✓</span>
    </div>
  </header>
  <section class="summary">
    {_metric_card("Main turns", summary.get("main_turns"))}
    {_metric_card("Subagent req/app/ign", f'{summary.get("subagent_requested", 0):g} / {summary.get("subagent_applied", 0):g} / {summary.get("subagent_ignored", 0):g}')}
    {_metric_card("Wipe count", summary.get("wipe_count"))}
    {_metric_card("Final answer", final_answer, wide=True)}
  </section>
  <section class="layout">
    <aside class="toc">
      <h2>Timeline</h2>
      {toc_html}
    </aside>
    <main class="content">
      <section id="prompt" class="event prompt">
        <h2>🧑 Original Prompt</h2>
        {_pre(prompt)}
      </section>
      {timeline_html}
      <section id="final-answer" class="event final">
        <h2>✅ Final Answer</h2>
        {_pre(final)}
      </section>
    </main>
  </section>
</body>
</html>
"""


def _render_interaction_toc(data: dict[str, Any]) -> str:
    links = ['<a href="#prompt">🧑 Prompt</a>']
    for item in data.get("transcript") or []:
        if not isinstance(item, dict):
            continue
        turn = item.get("turn")
        if "assistant" in item:
            calls = item.get("tool_calls") if isinstance(item.get("tool_calls"), list) else []
            label = f"🤖 Main turn {turn}"
            if any(_is_subagent_tool(call.get("name")) for call in calls if isinstance(call, dict)):
                label += " · subagent"
            if item.get("has_answer"):
                label += " · answer"
            links.append(f'<a href="#main-turn-{html.escape(str(turn))}">{html.escape(label)}</a>')
        elif isinstance(item.get("wipe"), dict):
            links.append(f'<a class="wipe-link" href="#wipe-turn-{html.escape(str(turn))}">🧹 Wipe after turn {html.escape(str(turn))}</a>')
        elif "tool" in item:
            tool = item.get("tool") if isinstance(item.get("tool"), dict) else {}
            name = str(tool.get("name") or "tool")
            links.append(f'<a href="#tool-turn-{html.escape(str(turn))}-{len(links)}">🔧 {html.escape(name)} result</a>')
    links.append('<a href="#final-answer">✅ Final answer</a>')
    return "\n".join(links)


def _render_interaction_timeline(data: dict[str, Any]) -> str:
    out: list[str] = []
    subagent_reports = [
        item for item in (data.get("subagent_reports") or []) if isinstance(item, dict)
    ]
    applied_subagent_index = 0
    tool_index = 0
    for item in data.get("transcript") or []:
        if not isinstance(item, dict):
            continue
        turn = item.get("turn")
        if "assistant" in item:
            calls = [call for call in (item.get("tool_calls") or []) if isinstance(call, dict)]
            call_html = "".join(_render_tool_call(call, idx=i) for i, call in enumerate(calls))
            out.append(
                f"""
<section id="main-turn-{html.escape(str(turn))}" class="event main">
  <h2>🤖 Main Turn {html.escape(str(turn))}</h2>
  <div class="meta">{_usage_line(item)}</div>
  <h3>Assistant</h3>
  {_pre(str(item.get("assistant") or ""))}
  {call_html}
</section>
"""
            )
        elif isinstance(item.get("wipe"), dict):
            wipe = item.get("wipe") or {}
            out.append(_render_wipe_event(wipe, turn=turn))
        elif "tool" in item:
            tool = item.get("tool") if isinstance(item.get("tool"), dict) else {}
            name = str(tool.get("name") or "tool")
            is_subagent = _is_subagent_tool(name)
            sub_html = ""
            if is_subagent:
                report = subagent_reports[applied_subagent_index] if applied_subagent_index < len(subagent_reports) else {}
                if report:
                    sub_html = _render_subagent_report(report)
                    applied_subagent_index += 1
            out.append(
                f"""
<section id="tool-turn-{html.escape(str(turn))}-{tool_index}" class="event tool {'subagent-tool' if is_subagent else ''}">
  <h2>{'🕵️' if is_subagent else '🔧'} Tool Result to Main: {html.escape(name)}</h2>
  <div class="meta">turn={html.escape(str(turn))} · response_budget_used={html.escape(str(item.get("response_budget_used")))} · tool_response_tokens={html.escape(str(item.get("tool_response_tokens")))}</div>
  {sub_html}
  <details {'open' if is_subagent else ''}>
    <summary>Main received tool result</summary>
    {_pre(_artifact_text(str(item.get("tool_result") or "")))}
  </details>
</section>
"""
            )
            tool_index += 1
    return "\n".join(out)


def _render_tool_call(call: dict[str, Any], *, idx: int) -> str:
    name = str(call.get("name") or "tool")
    if _is_subagent_tool(name):
        args = call.get("subagent") if isinstance(call.get("subagent"), dict) else {}
        body = (
            "<h4>Task</h4>"
            + _pre(str(args.get("task") or ""))
            + "<h4>Context</h4>"
            + _pre(str(args.get("context") or ""))
        )
        return f"""
<div class="tool-call subagent-call">
  <h3>🕵️ Subagent Call #{idx}: {html.escape(name)}</h3>
  {body}
</div>
"""
    query_list = call.get("query_list") if isinstance(call.get("query_list"), list) else []
    body = "\n".join(f"- {query}" for query in query_list)
    return f"""
<div class="tool-call">
  <h3>🔧 Tool Call #{idx}: {html.escape(name)}</h3>
  {_pre(body or json.dumps(call, ensure_ascii=False, indent=2))}
</div>
"""


def _render_wipe_event(wipe: dict[str, Any], *, turn: Any) -> str:
    reasons = ", ".join(str(item) for item in wipe.get("reasons") or [])
    return f"""
<section id="wipe-turn-{html.escape(str(turn))}" class="event wipe-event">
  <h2>🧹 Wipe after Main Turn {html.escape(str(turn))}</h2>
  <div class="wipe-grid">
    {_metric_card("reason", reasons or "-")}
    {_metric_card("apply point", wipe.get("apply_point"))}
    {_metric_card("prompt tokens", wipe.get("prompt_tokens"))}
    {_metric_card("messages", f'{wipe.get("message_count_before")} → {wipe.get("message_count_after")}')}
    {_metric_card("preserve tail", wipe.get("preserve_tail_messages"))}
    {_metric_card("next merge group", wipe.get("next_merge_group_index"))}
  </div>
</section>
"""


def _render_subagent_report(report: dict[str, Any]) -> str:
    dispatch = report.get("dispatch_index")
    transcript = [item for item in (report.get("transcript") or []) if isinstance(item, dict)]
    pieces = [
        f"""
<div class="subagent-box">
  <h3>🕵️ Subagent #{html.escape(str(dispatch))} Transcript</h3>
  <h4>Delegated task</h4>
  {_pre(str(report.get("task") or ""))}
  <h4>Delegated context</h4>
  {_pre(str(report.get("context") or ""))}
"""
    ]
    tool_i = 0
    for item in transcript:
        turn = item.get("turn")
        if "assistant" in item:
            calls = [call for call in (item.get("tool_calls") or []) if isinstance(call, dict)]
            pieces.append(
                f"""
  <section class="sub-turn">
    <h4>Subagent Turn {html.escape(str(turn))}</h4>
    <div class="meta">{_usage_line(item)}</div>
    {_pre(str(item.get("assistant") or ""))}
    {''.join(_render_tool_call(call, idx=i) for i, call in enumerate(calls))}
  </section>
"""
            )
        elif "tool" in item:
            tool = item.get("tool") if isinstance(item.get("tool"), dict) else {}
            name = str(tool.get("name") or "tool")
            pieces.append(
                f"""
  <details class="sub-tool">
    <summary>🔧 Subagent Tool Result {tool_i}: {html.escape(name)} · turn {html.escape(str(turn))}</summary>
    {_pre(_artifact_text(str(item.get("tool_result") or "")))}
  </details>
"""
            )
            tool_i += 1
    pieces.append(
        f"""
  <h4>Report returned to main</h4>
  {_pre(str(report.get("report") or ""))}
</div>
"""
    )
    return "\n".join(pieces)


def _usage_line(item: dict[str, Any]) -> str:
    usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
    return " · ".join(
        [
            f"prompt_tokens={html.escape(str(usage.get('prompt_tokens')))}",
            f"completion_tokens={html.escape(str(usage.get('completion_tokens')))}",
            f"finish={html.escape(str(item.get('finish_reason')))}",
            f"has_answer={html.escape(str(item.get('has_answer')))}",
        ]
    )


def _metric_card(label: str, value: Any, *, wide: bool = False) -> str:
    cls = "card wide" if wide else "card"
    return f'<div class="{cls}"><div class="label">{html.escape(str(label))}</div><div class="value">{html.escape(str(value))}</div></div>'


def _pre(text: str) -> str:
    return f"<pre>{html.escape(str(text or ''))}</pre>"


def _artifact_text(text: str) -> str:
    max_chars = _env_int("POLAR_SUBAGENT_WIPE_INTERACTION_TOOL_MAX_CHARS", 20000)
    return _truncate_text(text, max_chars=max_chars, side="middle")


def _readable_prompt(instruction: str) -> str:
    messages = _instruction_to_messages(str(instruction or ""))
    user_messages = [
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if user_messages:
        return "\n\n".join(user_messages)
    return str(instruction or "")


def _extract_answer(text: str) -> str:
    matches = _ANSWER_RE.findall(text or "")
    return str(matches[-1]).strip() if matches else ""


def _interaction_css() -> str:
    return """
:root { --bg:#0f172a; --panel:#111827; --muted:#94a3b8; --text:#e5e7eb; --line:#334155; --accent:#60a5fa; --wipe:#f59e0b; --sub:#a78bfa; --tool:#34d399; --final:#22c55e; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
a { color:inherit; }
.top { position:sticky; top:0; z-index:3; display:flex; justify-content:space-between; gap:16px; padding:18px 24px; border-bottom:1px solid var(--line); background:rgba(15,23,42,.96); backdrop-filter:blur(8px); }
h1 { margin:0; font-size:22px; }
.session,.muted,.meta,.label { color:var(--muted); }
.badges { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.badge { padding:4px 10px; border-radius:999px; font-weight:700; color:#020617; }
.badge.good,.badge.answer { background:var(--final); }
.badge.wipe { background:var(--wipe); }
.summary { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; padding:16px 24px; }
.card { border:1px solid var(--line); border-radius:12px; padding:12px; background:var(--panel); min-width:0; }
.card.wide { grid-column:span 1; }
.value { font-size:16px; font-weight:700; word-break:break-word; }
.layout { display:grid; grid-template-columns:300px minmax(0,1fr); gap:0; align-items:start; }
.toc { position:sticky; top:86px; max-height:calc(100vh - 86px); overflow:auto; padding:16px; border-right:1px solid var(--line); }
.toc h2 { font-size:15px; margin:0 0 10px; }
.toc a { display:block; padding:7px 9px; margin:4px 0; text-decoration:none; border-radius:8px; color:#cbd5e1; }
.toc a:hover { background:#1f2937; }
.toc .wipe-link { color:#fde68a; }
.content { padding:16px 24px 64px; min-width:0; }
.event { border:1px solid var(--line); border-radius:14px; padding:18px; margin:0 0 16px; background:rgba(17,24,39,.9); box-shadow:0 1px 0 rgba(255,255,255,.03) inset; }
.event h2 { margin:0 0 12px; font-size:20px; }
.event h3 { margin:16px 0 8px; font-size:16px; }
.event h4 { margin:14px 0 6px; font-size:14px; color:#cbd5e1; }
.main { border-left:4px solid var(--accent); }
.tool { border-left:4px solid var(--tool); }
.subagent-tool,.subagent-call,.subagent-box { border-color:var(--sub); }
.wipe-event { border-left:4px solid var(--wipe); background:rgba(120,53,15,.24); }
.final { border-left:4px solid var(--final); }
.tool-call,.subagent-box { margin-top:12px; border:1px solid var(--line); border-radius:12px; padding:12px; background:#0b1220; }
.subagent-call { background:rgba(88,28,135,.22); }
.sub-turn { margin:12px 0; border-top:1px solid var(--line); padding-top:10px; }
.wipe-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
pre { white-space:pre-wrap; overflow:auto; max-height:70vh; background:#020617; color:#e2e8f0; border:1px solid #1e293b; border-radius:10px; padding:12px; font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
details { margin-top:10px; border:1px solid var(--line); border-radius:10px; padding:10px; background:#0b1220; }
summary { cursor:pointer; color:#bfdbfe; font-weight:700; }
table { border-collapse:collapse; width:100%; background:var(--panel); border:1px solid var(--line); }
th,td { border-bottom:1px solid var(--line); padding:10px; text-align:left; }
.index { max-width:1000px; margin:40px auto; padding:0 20px; }
@media (max-width: 900px) { .layout { grid-template-columns:1fr; } .toc { position:static; max-height:none; border-right:0; border-bottom:1px solid var(--line); } .summary { grid-template-columns:1fr 1fr; } .wipe-grid { grid-template-columns:1fr; } }
"""


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


def _main_tool_schema(*, subagent_enabled: bool) -> list[dict[str, Any]]:
    schemas = list(_search_tool_schema())
    if subagent_enabled:
        existing = {_schema_tool_name(schema) for schema in schemas}
        if "subagent" not in existing:
            schemas.append(_subagent_tool_schema())
    return schemas


def _subagent_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "subagent",
            "description": (
                "Delegate focused independent research to a sub-agent when evidence is complex, "
                "ambiguous, contradictory, multi-hop, or requires verification before final answering. "
                "The sub-agent can search and returns a concise evidence-based report. Do not use "
                "for trivial single-hop lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Focused, concrete research task for the sub-agent.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional current reasoning context or hypothesis.",
                    },
                    "expected_output": {
                        "type": "string",
                        "description": (
                            "Optional guidance for what the report should contain, such as "
                            "evidence to verify, uncertainty to resolve, or format expectations."
                        ),
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "Optional maximum assistant/search turns for the sub-agent.",
                    },
                },
                "required": ["task"],
            },
        },
    }


def _search_tool_names() -> set[str]:
    # Keep search/local_search as compatibility aliases so older checkpoints or
    # hand-written prompts still execute, but render only the configured schema.
    names = {"search", "local_search", _search_tool_name()}
    for schema in _search_tool_schema():
        name = _schema_tool_name(schema)
        if name:
            names.add(name)
    return names


def _subagent_tool_names() -> set[str]:
    return {"subagent", "agent", "investigate", "research_agent"}


def _is_subagent_tool(name: object) -> bool:
    return str(name or "").strip() in _subagent_tool_names()


def _subagent_args_from_args(args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    task = args.get("task") or args.get("instruction") or args.get("query")
    if not isinstance(task, str) or not task.strip():
        return {}
    out: dict[str, Any] = {
        "task": task.strip(),
        "context": str(args.get("context") or ""),
    }
    if args.get("expected_output") is not None:
        out["context"] = (out["context"] + "\n\nExpected output:\n" + str(args.get("expected_output"))).strip()
    if args.get("max_turns") is not None:
        out["max_turns"] = args.get("max_turns")
    return out


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
        subagent_args = _subagent_args_from_args(args) if name and _is_subagent_tool(name) else {}
        if name and subagent_args:
            calls.append({"name": name, "subagent": subagent_args, "raw": tool_call})
        elif name and query_list:
            calls.append({"name": name, "query_list": query_list, "raw": tool_call})
    for raw in _TOOL_RE.findall(text or ""):
        try:
            payload = json.loads(raw)
            args = payload.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            name = payload.get("name")
            subagent_args = _subagent_args_from_args(args) if _is_subagent_tool(name) else {}
            if subagent_args:
                calls.append({"name": name, "subagent": subagent_args})
            else:
                calls.append({"name": name, "query_list": _query_list_from_args(args)})
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


def _env_flag_any(names: tuple[str, ...], *, default: bool = False) -> bool:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None:
            return _str_to_bool(raw, default=default)
    return default


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
