"""Prefix-merging trajectory builder.

Reconstructs a single token-level training trace out of the many independent
LLM completions an agent emits during one rollout.  A harness (claude_code,
codex, pi, ...) drives the agent and each turn hits the gateway as a separate
completion request; this builder stitches those completions back into the
``prompt + response_1 + interstitial + response_2 + ...`` stream that an RL
trainer needs, without introducing tokenization drift.

Design in two stages:

1. **Grouping** — detect which completions belong to the same append-only
   agent chain.  A cheap message-level key is used as an O(1) index, and a
   strict token-prefix check (``C_{k+1}.prompt_ids`` must start with
   ``C_k.prompt_ids``) is the final arbiter.  Completions whose tokens
   diverge start a fresh chain instead of silently polluting an existing one.

2. **Finalization** — walk each chain and build a merged token stream:

   - Assistant bodies come from the **raw** ``response_ids`` actually sampled
     by the model.  Their logprobs are real and we never decode→re-encode,
     so BPE non-canonicality cannot bite.
   - Interstitials (tool results, chat-template glue, intermediate user
     turns) come from ``C_{i+1}.prompt_ids`` — the server's **canonical**
     tokenization.  The boundary between "canonical copy of the previous
     assistant body" and the actual interstitial is the first end-of-turn
     token (``<|im_end|>`` on Qwen / ChatML; auto-detected or configurable).
   - Interstitial slots get synthesized logprobs and a zero ``loss_mask``;
     sampled assistant slots keep their real logprobs and a one ``loss_mask``.

See ``docs/prefix_merging_algorithm.md`` for a full walkthrough with
examples, invariants, and edge cases.
"""

from __future__ import annotations

import os
import logging
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from typing import Any

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.builder.record_utils import build_trace_from_completion, is_internal_completion_record
from polar.trajectory.models import CompletionRecord, CompletionSession, Trace, Trajectory
from verl_polar_bridge.debug_utils import debug_print, messages_summary, stable_hash, token_preview

logger = logging.getLogger(__name__)

# finish_reasons where the model emitted the natural end-of-turn token itself.
_NATURAL_STOP_REASONS = frozenset({"stop", "tool_calls", "stop_sequence"})
_PROMPT_ALIGNMENT_AUDIT_EMITTED = 0


def _prefix_debug_enabled() -> bool:
    return os.environ.get("POLAR_PREFIX_MERGING_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _prefix_debug(event: str, payload: dict[str, Any]) -> None:
    if _prefix_debug_enabled():
        debug_print("POLAR_PREFIX_MERGING_DEBUG", {"event": event, **payload}, stream="stderr")


def _prompt_alignment_audit_enabled() -> bool:
    return os.environ.get("POLAR_PROMPT_ALIGNMENT_AUDIT", "0").strip().lower() in {"1", "true", "yes", "on"}


def _prompt_grounded_single_merge_enabled() -> bool:
    mode = os.environ.get("POLAR_PREFIX_MERGING_MODE", "").strip().lower()
    if mode in {"prompt_grounded", "prompt_grounded_single", "prompt-grounded", "prompt-grounded-single"}:
        return True
    return os.environ.get("POLAR_PROMPT_GROUNDED_SINGLE_MERGE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _prompt_grounded_single_segment_grouping_enabled() -> bool:
    raw = os.environ.get("POLAR_PROMPT_GROUNDED_SINGLE_SEGMENT_GROUPING")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _trace_merge_group_id(trace: Trace) -> str | None:
    metadata = getattr(trace, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("merge_group_id") or metadata.get("segment_group_id")
    return None if value is None else str(value)


def _prompt_alignment_audit_all() -> bool:
    return os.environ.get("POLAR_PROMPT_ALIGNMENT_AUDIT_ALL", "0").strip().lower() in {"1", "true", "yes", "on"}


def _prompt_alignment_audit_limit() -> int:
    try:
        return max(0, int(os.environ.get("POLAR_PROMPT_ALIGNMENT_AUDIT_LIMIT", "200")))
    except ValueError:
        return 200


def _log_prompt_alignment_audit(payload: dict[str, Any]) -> None:
    if not _prompt_alignment_audit_enabled():
        return
    global _PROMPT_ALIGNMENT_AUDIT_EMITTED
    limit = _prompt_alignment_audit_limit()
    if limit and _PROMPT_ALIGNMENT_AUDIT_EMITTED >= limit:
        return
    _PROMPT_ALIGNMENT_AUDIT_EMITTED += 1
    debug_print("POLAR_PROMPT_ALIGNMENT_AUDIT", payload, stream="stderr")


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


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


def _prompt_alignment_span(
    *,
    source: str,
    completion_index: int,
    completion_id: str | None,
    request_id: Any,
    base_prompt_ids: list[int],
    stitched_prefix_ids: list[int],
    current_prompt_ids: list[int],
    response_len: int,
    interstitial_len: int,
    canonical_tail_len: int,
    prev_raw_response_len: int,
) -> dict[str, Any]:
    """Describe whether the training prefix before a response equals its rollout prompt."""
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
    response_start = max(0, len(stitched_prefix_ids) - len(base_prompt_ids))
    span = {
        "source": source,
        "completion_index": completion_index,
        "completion_id": completion_id,
        "request_id": request_id,
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
        "canonical_tail_len": int(canonical_tail_len),
        "prev_raw_response_len": int(prev_raw_response_len),
    }
    if not prompt_aligned:
        span["stitched_mismatch_window"] = _token_window(stitched_prefix_ids, prompt_lcp)
        span["actual_prompt_mismatch_window"] = _token_window(current_prompt_ids, prompt_lcp)
        if _is_chatml_generation_marker_extra(stitched_prefix_ids, current_prompt_ids, prompt_lcp):
            span["mismatch_kind"] = "extra_chatml_generation_prompt"
    return span


def _is_chatml_generation_marker_extra(
    stitched_prefix_ids: list[int],
    current_prompt_ids: list[int],
    prompt_lcp: int,
) -> bool:
    """True when stitched prefix has an extra ChatML assistant generation suffix.

    Qwen ChatML generation prompts include ``<|im_start|>assistant\n<think>\n``.
    The next turn's concrete assistant message should not contain that generation
    suffix; it starts with the sampled assistant body instead.  This helper
    identifies the exact drift pattern without hard-failing other tokenizers.
    """
    extra4 = [151667, 198, 151668, 271]
    if stitched_prefix_ids[prompt_lcp : prompt_lcp + 4] == extra4:
        return True
    extra8 = [151644, 77091, 198, 151667, 198, 151668, 271, 151657]
    if stitched_prefix_ids[prompt_lcp - 3 : prompt_lcp + 5] == extra8[:8]:
        return True
    return False


def _zero_logprob_slot(token_id: int) -> dict[str, Any]:
    return {"token_id": int(token_id), "logprob": 0.0}


def _completion_debug_summary(completion: CompletionRecord, trace: Trace | None = None) -> dict[str, Any]:
    trace = trace or build_trace_from_completion(completion)
    return {
        "completion_id": completion.completion_id,
        "request_id": (trace.metadata or {}).get("request_id"),
        "prompt_ids": token_preview(trace.prompt_ids),
        "response_ids": token_preview(trace.response_ids),
        "loss_mask_len": len(trace.loss_mask or []),
        "logprob_len": len(trace.response_logprobs or []),
        "finish_reason": trace.finish_reason,
        "prompt_messages": messages_summary(trace.prompt_messages),
        "response_messages": messages_summary(trace.response_messages),
        "tools_hash": stable_hash(trace.tools),
        "metadata_keys": sorted(str(k) for k in (trace.metadata or {}).keys()),
    }



# ---------------------------------------------------------------------------
# Message-level grouping helpers — used to detect which completions belong
# to the same agentic chain (C_{i+1}'s prompt == C_i's prompt + response).
# ---------------------------------------------------------------------------


def _flatten_message_content(content: Any) -> str:
    """Extract text from a message content field (string or content-part array)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content) if content is not None else ""


def _expand_messages_for_grouping(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    if role != "assistant" or not message.get("tool_calls"):
        return [message]

    expanded: list[dict[str, Any]] = []
    content = message.get("content")
    if content not in (None, "", []):
        expanded.append({"role": role, "content": content})
    expanded.append(
        {"role": role, "content": None, "tool_calls": message.get("tool_calls")}
    )
    return expanded


def _is_grouping_noise_message(message: dict[str, Any]) -> bool:
    role = message.get("role")
    if role in ["tool"]:
        return True
    if role == "assistant" and message.get("tool_calls"):
        return False
    content = _flatten_message_content(message.get("content")).strip()
    if role == "assistant" and not content and not message.get("tool_calls"):
        return True
    return False


def _normalize_messages(messages: list[dict[str, Any]]) -> str:
    """Flatten a message list into a deterministic key string.

    Format: ``role:content<SEP>role:content<SEP>...``
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant" and msg.get("tool_calls"):
            content = ""
        else:
            content = _flatten_message_content(msg.get("content"))
        parts.append(f"{role}:{content}")
    return "<SEP>".join(parts)


def _grouping_key(messages: list[dict[str, Any]]) -> str:
    """Normalize the structural conversation context used for chaining.

    Tool-result messages are omitted because they are harness artifacts that
    appear between assistant turns in the next request prompt.
    """
    return _normalize_messages(
        [
            expanded_message
            for message in messages
            for expanded_message in _expand_messages_for_grouping(message)
            if not _is_grouping_noise_message(expanded_message)
        ],
    )


class PrefixMergingBuilder(BaseTrajectoryBuilder):
    """Rebuild a chain's merged token stream using raw + canonical-interstitial.

    Parameters
    ----------
    end_of_turn_token_id:
        Explicit end-of-turn (EOT) token id used to locate the
        canonical-tail split between the prior assistant body and the
        interstitial.  When None (default), the builder auto-detects it
        from the last token of the first completion with a natural stop
        reason.  For Qwen / ChatML templates this is the
        ``<|im_end|>`` token id.
    """

    def __init__(
        self,
        *,
        end_of_turn_token_id: int | None = None,
    ) -> None:
        self._configured_eot_id = end_of_turn_token_id

    async def build(self, session: CompletionSession) -> Trajectory:
        raw_completion_count = len(session.completions)
        completions = [
            completion
            for completion in session.completions
            if not is_internal_completion_record(completion)
        ]
        skipped_internal_count = raw_completion_count - len(completions)
        _prefix_debug(
            "build_start",
            {
                "session_id": session.session_id,
                "task_id": session.task_id,
                "raw_completion_count": raw_completion_count,
                "completion_count": len(completions),
                "skipped_internal_count": skipped_internal_count,
                "session_metadata_keys": sorted(str(k) for k in (session.metadata or {}).keys()),
            },
        )

        if not completions:
            return Trajectory(
                status="ERROR",
                metadata={
                    "builder": "prefix_merging",
                    "session_id": session.session_id,
                    "task_metadata": dict(session.metadata),
                    "record_count": 0,
                    "record_count_raw": raw_completion_count,
                    "record_count_skipped_internal": skipped_internal_count,
                    **_top_level_scheduler_metadata(session.metadata),
                },
                traces=[],
                error="no completions" if raw_completion_count == 0 else "no non-internal completions",
            )

        if _prompt_grounded_single_merge_enabled() and _prompt_grounded_single_segment_grouping_enabled():
            return self._build_prompt_grounded_single_grouped_trajectory(
                session,
                completions,
                raw_completion_count=raw_completion_count,
                skipped_internal_count=skipped_internal_count,
            )

        chains: list[list[CompletionRecord]] = []
        waiting_chains: dict[str, deque[int]] = defaultdict(deque)

        for completion in completions:
            trace = build_trace_from_completion(completion)
            prompt_key = _grouping_key(trace.prompt_messages)
            chain_idx = self._pop_compatible_chain(
                prompt_key=prompt_key,
                prompt_ids=trace.prompt_ids,
                chains=chains,
                waiting_chains=waiting_chains,
            )

            if chain_idx is not None:
                chains[chain_idx].append(completion)
            else:
                chain_idx = len(chains)
                chains.append([completion])

            next_key = _grouping_key(trace.prompt_messages + trace.response_messages)
            waiting_chains[next_key].append(chain_idx)
            _prefix_debug(
                "chain_assign",
                {
                    "session_id": session.session_id,
                    "completion": _completion_debug_summary(completion, trace),
                    "prompt_key_hash": stable_hash(prompt_key),
                    "next_key_hash": stable_hash(next_key),
                    "chain_idx": chain_idx,
                    "chain_len": len(chains[chain_idx]),
                },
            )

        stats: dict[str, int] = {
            "chains_total": len(chains),
            "chains_reconstructed_full": 0,
            "chains_reconstructed_truncated": 0,
            "completions_total": len(completions),
            "completions_total_raw": raw_completion_count,
            "completions_skipped_internal": skipped_internal_count,
            "completions_merged": 0,
        }
        final_traces = [self._finalize_chain(chain, stats) for chain in chains]
        builder_name = "prefix_merging_prompt_grounded_single" if _prompt_grounded_single_merge_enabled() else "prefix_merging"
        _prefix_debug(
            "build_final",
            {
                "session_id": session.session_id,
                "chain_count": len(chains),
                "chain_lengths": [len(chain) for chain in chains],
                "stats": dict(stats),
                "trace_prompt_lens": [len(trace.prompt_ids) for trace in final_traces],
                "trace_response_lens": [len(trace.response_ids) for trace in final_traces],
                "trace_loss_tokens": [sum(int(v) for v in (trace.loss_mask or [])) for trace in final_traces],
            },
        )

        return Trajectory(
            status="COMPLETED",
            metadata={
                "builder": builder_name,
                "session_id": session.session_id,
                "task_id": session.task_id,
                "api_type": session.api_type,
                "model_requested": session.model_requested,
                "model_used": session.model_used,
                "record_count": len(completions),
                "record_count_raw": raw_completion_count,
                "record_count_skipped_internal": skipped_internal_count,
                "task_metadata": dict(session.metadata),
                "trace_count": len(chains),
                "reconstruction_stats": stats,
                **_top_level_scheduler_metadata(session.metadata),
            },
            traces=final_traces,
        )

    def _build_prompt_grounded_single_grouped_trajectory(
        self,
        session: CompletionSession,
        completions: list[CompletionRecord],
        *,
        raw_completion_count: int,
        skipped_internal_count: int,
    ) -> Trajectory:
        """Build one prompt-grounded trace per explicit segment/merge group.

        The legacy prefix-merging chain assignment is deliberately strict: a
        later completion may join a chain only when its prompt is an append-only
        token prefix of the previous prompt.  That is useful for the legacy
        canonical-tail stitcher, but it prevents prompt-grounded recovery from
        seeing the cases it is designed to handle, such as prompt drift,
        wipe/compact boundaries, or Search subagent interleaving.

        In prompt-grounded-single mode, the harness-provided ``merge_group_id``
        is the hard segment boundary.  Within each group, prompts are reconciled
        by ``_finalize_chain_prompt_grounded_single`` using the actual rollout
        prompt for every completion.  This makes the Polar builder itself emit:

        * one main trace for normal SearchR1;
        * one trace per main wipe segment when wipe is enabled;
        * one trace per independent subagent segment when subagent is enabled.

        Adapter-side token stitch should therefore be unnecessary for
        ``prefix_merging_prompt_grounded_single`` trajectories.
        """

        grouped: "OrderedDict[str, list[CompletionRecord]]" = OrderedDict()
        explicit_group_count = 0
        for idx, completion in enumerate(completions):
            trace = build_trace_from_completion(completion)
            group_id = _trace_merge_group_id(trace)
            if group_id is None:
                group_id = "__default__"
            else:
                explicit_group_count += 1
            grouped.setdefault(str(group_id), []).append(completion)
            _prefix_debug(
                "prompt_grounded_single_group_assign",
                {
                    "session_id": session.session_id,
                    "completion": _completion_debug_summary(completion, trace),
                    "group_id": group_id,
                    "completion_index": idx,
                    "explicit_group": group_id != "__default__",
                },
            )

        stats: dict[str, int] = {
            "chains_total": len(grouped),
            "chains_reconstructed_full": 0,
            "chains_reconstructed_truncated": 0,
            "completions_total": len(completions),
            "completions_total_raw": raw_completion_count,
            "completions_skipped_internal": skipped_internal_count,
            "completions_merged": 0,
            "prompt_grounded_single_segment_grouping": 1,
            "prompt_grounded_single_explicit_group_count": explicit_group_count,
        }
        final_traces = [self._finalize_chain_prompt_grounded_single(chain, stats) for chain in grouped.values()]
        _prefix_debug(
            "prompt_grounded_single_grouped_build_final",
            {
                "session_id": session.session_id,
                "group_ids": list(grouped.keys()),
                "group_lengths": [len(chain) for chain in grouped.values()],
                "stats": dict(stats),
                "trace_prompt_lens": [len(trace.prompt_ids) for trace in final_traces],
                "trace_response_lens": [len(trace.response_ids) for trace in final_traces],
                "trace_loss_tokens": [sum(int(v) for v in (trace.loss_mask or [])) for trace in final_traces],
            },
        )

        return Trajectory(
            status="COMPLETED",
            metadata={
                "builder": "prefix_merging_prompt_grounded_single",
                "session_id": session.session_id,
                "task_id": session.task_id,
                "api_type": session.api_type,
                "model_requested": session.model_requested,
                "model_used": session.model_used,
                "record_count": len(completions),
                "record_count_raw": raw_completion_count,
                "record_count_skipped_internal": skipped_internal_count,
                "task_metadata": dict(session.metadata),
                "trace_count": len(final_traces),
                "reconstruction_stats": stats,
                "prompt_grounded_single_segment_grouping": 1,
                "prompt_grounded_single_group_ids": list(grouped.keys()),
                **_top_level_scheduler_metadata(session.metadata),
            },
            traces=final_traces,
        )

    # ------------------------------------------------------------------
    # Chain finalization
    # ------------------------------------------------------------------

    def _finalize_chain(
        self,
        chain: list[CompletionRecord],
        stats: dict[str, int],
    ) -> Trace:
        if _prompt_grounded_single_merge_enabled():
            return self._finalize_chain_prompt_grounded_single(chain, stats)

        # Everything in C_1.prompt_ids is the non-trainable
        # prompt; C_1.response_ids plus every subsequent raw response +
        # canonical interstitial becomes the trainable response.  No role-shape
        # constraint on the initial conversation — a harness preamble like
        # codex's [system, user, user, assistant, tool, ...] is treated as
        # static context.
        first_trace = build_trace_from_completion(chain[0])
        eot_id = self._resolve_eot_id(chain)

        prompt_ids = list(first_trace.prompt_ids)
        stream_ids: list[int] = list(prompt_ids)
        response_slots: list[dict[str, Any] | None] = []
        loss_mask: list[int] = []
        response_messages: list[dict[str, Any]] = []
        prompt_alignment_spans: list[dict[str, Any]] = []

        # Track the canonical prompt_ids of the most recently merged
        # completion — used for the canonical-vs-canonical prefix check.
        prev_prompt_ids: list[int] = list(first_trace.prompt_ids)
        prev_raw_response: list[int] = list(first_trace.response_ids)

        # Running count of messages consumed = prompt_messages + all response_messages emitted.
        msg_acc = len(first_trace.prompt_messages)

        first_span = _prompt_alignment_span(
            source="prefix_merging",
            completion_index=0,
            completion_id=str(chain[0].completion_id),
            request_id=(first_trace.metadata or {}).get("request_id"),
            base_prompt_ids=prompt_ids,
            stitched_prefix_ids=list(stream_ids),
            current_prompt_ids=prompt_ids,
            response_len=len(first_trace.response_ids),
            interstitial_len=0,
            canonical_tail_len=0,
            prev_raw_response_len=0,
        )
        prompt_alignment_spans.append(first_span)
        if _prompt_alignment_audit_all():
            _log_prompt_alignment_audit({"event": "span", **first_span})

        self._append_response_tokens(first_trace, stream_ids, response_slots, loss_mask)
        response_messages.extend(deepcopy(m) for m in first_trace.response_messages)
        msg_acc += len(first_trace.response_messages)
        kept = 1

        for i in range(1, len(chain)):
            Ci_trace = build_trace_from_completion(chain[i])
            Ci_prompt_ids = list(Ci_trace.prompt_ids)

            # Canonical-vs-canonical prefix check: both sides are server-side
            # tokenizations of the same message prefix — matches reliably
            # unless the harness rewrote prior messages.
            if (
                len(Ci_prompt_ids) < len(prev_prompt_ids)
                or Ci_prompt_ids[: len(prev_prompt_ids)] != prev_prompt_ids
            ):
                logger.debug(
                    "prefix_merging: canonical prefix break at step %d/%d",
                    i,
                    len(chain),
                )
                _prefix_debug(
                    "finalize_break_prefix",
                    {
                        "chain_len": len(chain),
                        "step": i,
                        "prev_completion": _completion_debug_summary(chain[i - 1]),
                        "current_completion": _completion_debug_summary(chain[i], Ci_trace),
                        "prev_prompt_ids": token_preview(prev_prompt_ids),
                        "current_prompt_ids": token_preview(Ci_prompt_ids),
                    },
                )
                break

            # canonical_tail = canonical tokens for [prev assistant msg + new interstitials].
            canonical_tail = Ci_prompt_ids[len(prev_prompt_ids):]
            interstitial = self._slice_interstitial(
                canonical_tail=canonical_tail,
                prev_raw_response=prev_raw_response,
                eot_id=eot_id,
            )
            if interstitial is None:
                logger.debug(
                    "prefix_merging: interstitial split failed at step %d/%d "
                    "(eot_id=%r, tail_len=%d)",
                    i,
                    len(chain),
                    eot_id,
                    len(canonical_tail),
                )
                _prefix_debug(
                    "finalize_break_interstitial",
                    {
                        "chain_len": len(chain),
                        "step": i,
                        "eot_id": eot_id,
                        "canonical_tail": token_preview(canonical_tail),
                        "prev_raw_response": token_preview(prev_raw_response),
                        "current_completion": _completion_debug_summary(chain[i], Ci_trace),
                    },
                )
                break

            if interstitial:
                _prefix_debug(
                    "interstitial_append",
                    {
                        "chain_len": len(chain),
                        "step": i,
                        "eot_id": eot_id,
                        "canonical_tail": token_preview(canonical_tail),
                        "prev_raw_response": token_preview(prev_raw_response),
                        "interstitial": token_preview(interstitial),
                        "prev_raw_ends_eot": bool(prev_raw_response and prev_raw_response[-1] == eot_id),
                    },
                )
                stream_ids.extend(interstitial)
                response_slots.extend([None] * len(interstitial))
                loss_mask.extend([0] * len(interstitial))

            alignment_span = _prompt_alignment_span(
                source="prefix_merging",
                completion_index=i,
                completion_id=str(chain[i].completion_id),
                request_id=(Ci_trace.metadata or {}).get("request_id"),
                base_prompt_ids=prompt_ids,
                stitched_prefix_ids=list(stream_ids),
                current_prompt_ids=Ci_prompt_ids,
                response_len=len(Ci_trace.response_ids),
                interstitial_len=len(interstitial or []),
                canonical_tail_len=len(canonical_tail),
                prev_raw_response_len=len(prev_raw_response),
            )
            prompt_alignment_spans.append(alignment_span)
            if (not alignment_span["prompt_aligned"]) or _prompt_alignment_audit_all():
                _log_prompt_alignment_audit({"event": "span", **alignment_span})

            # Message-level interstitial bookkeeping.
            if len(Ci_trace.prompt_messages) > msg_acc:
                interstitial_msgs = Ci_trace.prompt_messages[msg_acc:]
                response_messages.extend(deepcopy(m) for m in interstitial_msgs)
                msg_acc += len(interstitial_msgs)

            self._append_response_tokens(Ci_trace, stream_ids, response_slots, loss_mask)
            response_messages.extend(deepcopy(m) for m in Ci_trace.response_messages)
            msg_acc += len(Ci_trace.response_messages)

            prev_prompt_ids = Ci_prompt_ids
            prev_raw_response = list(Ci_trace.response_ids)
            kept += 1

        stats["completions_merged"] += kept
        if kept == len(chain):
            stats["chains_reconstructed_full"] += 1
        else:
            stats["chains_reconstructed_truncated"] += 1
        _prefix_debug(
            "finalize_chain",
            {
                "chain_len": len(chain),
                "kept": kept,
                "eot_id": eot_id,
                "prompt_ids": token_preview(prompt_ids),
                "response_len": len(stream_ids) - len(prompt_ids),
                "loss_tokens": sum(int(v) for v in loss_mask),
            },
        )

        response_ids = stream_ids[len(prompt_ids):]
        response_logprobs = self._finalize_logprobs(response_slots, response_ids)
        _prefix_debug(
            "finalize_alignment",
            {
                "chain_len": len(chain),
                "kept": kept,
                "prompt_ids": token_preview(prompt_ids),
                "response_ids": token_preview(response_ids),
                "loss_mask_len": len(loss_mask),
                "loss_tokens": sum(int(v) for v in loss_mask),
                "response_logprob_len": len(response_logprobs or []),
                "response_slot_real_count": sum(1 for slot in response_slots if slot is not None),
                "response_slot_synth_count": sum(1 for slot in response_slots if slot is None),
                "native_response_lens": [len(build_trace_from_completion(c).response_ids) for c in chain[:kept]],
                "native_logprob_lens": [len(build_trace_from_completion(c).response_logprobs or []) for c in chain[:kept]],
            },
        )
        last_kept_trace = build_trace_from_completion(chain[kept - 1])
        metadata = self._chain_metadata(chain[:kept])
        metadata["prompt_alignment_mismatch_count"] = sum(
            1 for span in prompt_alignment_spans if not span.get("prompt_aligned")
        )
        metadata["prompt_alignment_prompt_grounded_drift_tokens_sum"] = sum(
            int(span.get("prompt_grounded_drift_tokens") or 0) for span in prompt_alignment_spans
        )
        metadata["prompt_alignment_span_count"] = len(prompt_alignment_spans)
        if _prompt_alignment_audit_enabled():
            metadata["prompt_alignment_spans"] = prompt_alignment_spans

        return Trace(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            loss_mask=loss_mask,
            prompt_messages=[deepcopy(m) for m in first_trace.prompt_messages],
            response_messages=response_messages,
            tools=deepcopy(first_trace.tools),
            finish_reason=last_kept_trace.finish_reason,
            response_logprobs=response_logprobs,
            metadata=metadata,
        )

    def _finalize_chain_prompt_grounded_single(
        self,
        chain: list[CompletionRecord],
        stats: dict[str, int],
    ) -> Trace:
        """Finalize one append-only chain using the reference implementation's prompt-grounded merge.

        Unlike the legacy canonical-tail stitcher, this path treats each
        completion's ``prompt_ids`` as the source of truth for the prefix that
        conditioned its sampled ``response_ids``.  For every later turn, the
        already-assembled response stream is reconciled against the current
        actual prompt suffix:

        * base prompt changed -> reset the in-segment state to the current turn
          prompt, matching the reference implementation's defensive behavior inside ``merge_turns``;
        * response-so-far drifted -> truncate to the matched prefix;
        * truncation cuts through a historical output span -> mask the retained
          partial prefix of that output span;
        * current prompt context tail -> append with ``loss_mask=0``;
        * current sampled output -> append with its original mask/logprobs.
        """
        first_trace = build_trace_from_completion(chain[0])
        prompt_ids = list(first_trace.prompt_ids)
        response_ids: list[int] = []
        response_slots: list[dict[str, Any] | None] = []
        loss_mask: list[int] = []
        output_spans: list[tuple[int, int]] = []
        response_messages: list[dict[str, Any]] = []
        prompt_alignment_spans: list[dict[str, Any]] = []
        kept = 0
        msg_acc = len(first_trace.prompt_messages)

        resets = 0
        truncations = 0
        truncated_tokens = 0
        partial_masked_tokens = 0
        context_tail_tokens = 0

        for i, completion in enumerate(chain):
            trace = build_trace_from_completion(completion)
            current_prompt_ids = list(trace.prompt_ids)
            current_response_ids = list(trace.response_ids)

            if i > 0:
                if current_prompt_ids[: len(prompt_ids)] != prompt_ids:
                    resets += 1
                    _prefix_debug(
                        "prompt_grounded_single_reset",
                        {
                            "chain_len": len(chain),
                            "step": i,
                            "base_prompt_len": len(prompt_ids),
                            "current_prompt_len": len(current_prompt_ids),
                            "base_prompt_lcp": _common_prefix_len(prompt_ids, current_prompt_ids),
                            "current_completion": _completion_debug_summary(completion, trace),
                        },
                    )
                    prompt_ids = list(current_prompt_ids)
                    response_ids = []
                    response_slots = []
                    loss_mask = []
                    output_spans = []
                    response_messages = []
                    msg_acc = len(trace.prompt_messages)
                else:
                    prompt_suffix = current_prompt_ids[len(prompt_ids) :]
                    matched_len = _common_prefix_len(response_ids, prompt_suffix)

                    if matched_len < len(response_ids):
                        drift = len(response_ids) - matched_len
                        truncations += 1
                        truncated_tokens += drift
                        _prefix_debug(
                            "prompt_grounded_single_truncate",
                            {
                                "chain_len": len(chain),
                                "step": i,
                                "matched_len": matched_len,
                                "response_so_far_len": len(response_ids),
                                "drift_tokens": drift,
                                "prompt_suffix_len": len(prompt_suffix),
                                "current_completion": _completion_debug_summary(completion, trace),
                            },
                        )
                        for start, end in output_spans:
                            if start < matched_len < end:
                                masked = matched_len - start
                                if masked > 0:
                                    partial_masked_tokens += masked
                                    loss_mask[start:matched_len] = [0] * masked
                                    for pos in range(start, matched_len):
                                        response_slots[pos] = _zero_logprob_slot(response_ids[pos])

                        response_ids = response_ids[:matched_len]
                        response_slots = response_slots[:matched_len]
                        loss_mask = loss_mask[:matched_len]
                        output_spans = [
                            (start, min(end, matched_len))
                            for start, end in output_spans
                            if start < matched_len
                        ]

                    context_tail = prompt_suffix[matched_len:]
                    if context_tail:
                        context_tail_tokens += len(context_tail)
                        response_ids.extend(context_tail)
                        response_slots.extend([None] * len(context_tail))
                        loss_mask.extend([0] * len(context_tail))

                    # Message-level bookkeeping mirrors the legacy prefix
                    # merger without touching token/loss streams.  The
                    # prompt_grounded-single token path already injects tool/user
                    # observations through ``context_tail`` with loss_mask=0;
                    # appending the corresponding prompt messages here keeps
                    # metadata-derived metrics such as ``num_turns`` aligned
                    # with VERL standalone's ``user_turns + assistant_turns
                    # + 1`` accounting.
                    if len(trace.prompt_messages) > msg_acc:
                        interstitial_msgs = trace.prompt_messages[msg_acc:]
                        response_messages.extend(deepcopy(m) for m in interstitial_msgs)
                        msg_acc += len(interstitial_msgs)

            alignment_span = _prompt_alignment_span(
                source="prefix_merging_prompt_grounded_single",
                completion_index=i,
                completion_id=str(completion.completion_id),
                request_id=(trace.metadata or {}).get("request_id"),
                base_prompt_ids=prompt_ids,
                stitched_prefix_ids=prompt_ids + response_ids,
                current_prompt_ids=current_prompt_ids,
                response_len=len(current_response_ids),
                interstitial_len=0,
                canonical_tail_len=0,
                prev_raw_response_len=0,
            )
            prompt_alignment_spans.append(alignment_span)
            if (not alignment_span["prompt_aligned"]) or _prompt_alignment_audit_all():
                _log_prompt_alignment_audit({"event": "span", **alignment_span})

            output_start = len(response_ids)
            self._append_response_to_response_stream(trace, response_ids, response_slots, loss_mask)
            output_spans.append((output_start, len(response_ids)))
            response_messages.extend(deepcopy(m) for m in trace.response_messages)
            msg_acc += len(trace.response_messages)
            kept += 1

        stats["completions_merged"] += kept
        if kept == len(chain):
            stats["chains_reconstructed_full"] += 1
        else:
            stats["chains_reconstructed_truncated"] += 1

        for pos, mask_value in enumerate(loss_mask):
            if not int(mask_value):
                response_slots[pos] = _zero_logprob_slot(response_ids[pos])

        response_logprobs = self._finalize_logprobs(response_slots, response_ids)
        last_kept_trace = build_trace_from_completion(chain[kept - 1])
        metadata = self._chain_metadata(chain[:kept])
        metadata["builder"] = "prefix_merging_prompt_grounded_single"
        metadata["prompt_grounded_single_merge_enabled"] = 1
        metadata["prompt_grounded_single_reset_count"] = resets
        metadata["prompt_grounded_single_truncate_count"] = truncations
        metadata["prompt_grounded_single_truncated_tokens"] = truncated_tokens
        metadata["prompt_grounded_single_partial_masked_tokens"] = partial_masked_tokens
        metadata["prompt_grounded_single_context_tail_tokens"] = context_tail_tokens
        metadata["prompt_alignment_mismatch_count"] = sum(
            1 for span in prompt_alignment_spans if not span.get("prompt_aligned")
        )
        metadata["prompt_alignment_prompt_grounded_drift_tokens_sum"] = sum(
            int(span.get("prompt_grounded_drift_tokens") or 0) for span in prompt_alignment_spans
        )
        metadata["prompt_alignment_span_count"] = len(prompt_alignment_spans)
        if _prompt_alignment_audit_enabled():
            metadata["prompt_alignment_spans"] = prompt_alignment_spans

        _prefix_debug(
            "prompt_grounded_single_finalize_chain",
            {
                "chain_len": len(chain),
                "kept": kept,
                "prompt_ids": token_preview(prompt_ids),
                "response_ids": token_preview(response_ids),
                "response_len": len(response_ids),
                "loss_tokens": sum(int(v) for v in loss_mask),
                "resets": resets,
                "truncations": truncations,
                "truncated_tokens": truncated_tokens,
                "partial_masked_tokens": partial_masked_tokens,
                "context_tail_tokens": context_tail_tokens,
            },
        )

        return Trace(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            loss_mask=loss_mask,
            prompt_messages=[deepcopy(m) for m in first_trace.prompt_messages],
            response_messages=response_messages,
            tools=deepcopy(first_trace.tools),
            finish_reason=last_kept_trace.finish_reason,
            response_logprobs=response_logprobs,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_eot_id(self, chain: list[CompletionRecord]) -> int | None:
        """Return configured EOT id, else auto-detect from the chain.

        Auto-detection uses the last token of the first completion whose
        ``finish_reason`` indicates the model emitted the natural stop
        marker itself (stop / tool_calls / stop_sequence).
        """
        if self._configured_eot_id is not None:
            return self._configured_eot_id
        for completion in chain:
            trace = build_trace_from_completion(completion)
            if (
                trace.finish_reason in _NATURAL_STOP_REASONS
                and trace.response_ids
            ):
                return trace.response_ids[-1]
        return None

    @staticmethod
    def _slice_interstitial(
        *,
        canonical_tail: list[int],
        prev_raw_response: list[int],
        eot_id: int | None,
    ) -> list[int] | None:
        """Extract the canonical interstitial from C_{i+1}'s prompt tail.

        ``canonical_tail`` = canonical tokens for [prev assistant msg +
        harness-inserted messages + generation-prompt glue].  The first
        occurrence of ``eot_id`` marks the end of the prev assistant
        body; everything after is interstitial.

        If ``prev_raw_response`` already ends with ``eot_id`` (natural
        stop / tool_calls), skip it in the canonical tail to avoid
        duplication; otherwise (truncation) include it so the stream
        still closes the assistant turn.

        Returns None if ``eot_id`` is unknown or not present — caller
        should treat this as a break.
        """
        if eot_id is None:
            return None
        try:
            k = canonical_tail.index(eot_id)
        except ValueError:
            return None
        if prev_raw_response and prev_raw_response[-1] == eot_id:
            return canonical_tail[k + 1 :]
        return canonical_tail[k:]

    @staticmethod
    def _append_response_tokens(
        trace: Trace,
        stream_ids: list[int],
        response_slots: list[dict[str, Any] | None],
        loss_mask: list[int],
    ) -> None:
        """Append a completion's response_ids and parallel logprob slots."""
        response_ids = list(trace.response_ids)
        stream_ids.extend(response_ids)
        trace_loss_mask = list(trace.loss_mask) or [1] * len(response_ids)
        if len(trace_loss_mask) != len(response_ids):
            raise ValueError("trace loss_mask length must match response_ids length")
        loss_mask.extend(trace_loss_mask)
        logprobs = trace.response_logprobs or []
        for pos in range(len(response_ids)):
            entry = logprobs[pos] if pos < len(logprobs) else None
            response_slots.append(deepcopy(entry) if isinstance(entry, dict) else None)

    @staticmethod
    def _append_response_to_response_stream(
        trace: Trace,
        response_ids: list[int],
        response_slots: list[dict[str, Any] | None],
        loss_mask: list[int],
    ) -> None:
        """Append a completion response to an already response-only stream."""
        trace_response_ids = list(trace.response_ids)
        response_ids.extend(trace_response_ids)
        trace_loss_mask = list(trace.loss_mask) or [1] * len(trace_response_ids)
        if len(trace_loss_mask) != len(trace_response_ids):
            raise ValueError("trace loss_mask length must match response_ids length")
        loss_mask.extend(trace_loss_mask)
        logprobs = trace.response_logprobs or []
        for pos in range(len(trace_response_ids)):
            entry = logprobs[pos] if pos < len(logprobs) else None
            response_slots.append(deepcopy(entry) if isinstance(entry, dict) else None)

    @staticmethod
    def _finalize_logprobs(
        slots: list[dict[str, Any] | None],
        response_ids: list[int],
    ) -> list[dict[str, Any]] | None:
        if not any(slot is not None for slot in slots):
            return None
        return [
            slot if slot is not None else {"token_id": response_ids[i], "logprob": 0.0}
            for i, slot in enumerate(slots)
        ]

    @staticmethod
    def _chain_metadata(chain: list[CompletionRecord]) -> dict[str, Any]:
        completion_metadata = [dict(completion.metadata) for completion in chain]
        merged = dict(completion_metadata[0]) if completion_metadata else {}
        completion_ids = [str(completion.completion_id) for completion in chain]
        merged.setdefault("builder", "prefix_merging")
        merged.setdefault("chain_id", "prefix_chain:" + ":".join(completion_ids))
        merged["completion_ids"] = completion_ids
        merged["completion_count"] = len(chain)
        merged["completion_metadata"] = completion_metadata
        merged["native_prompt_lens"] = []
        merged["native_response_lens"] = []
        merged["native_logprob_lens"] = []
        for completion in chain:
            trace = build_trace_from_completion(completion)
            meta = dict(getattr(trace, "metadata", {}) or {})
            merged["native_prompt_lens"].append(int(meta.get("native_prompt_len", len(trace.prompt_ids))))
            merged["native_response_lens"].append(int(meta.get("native_response_len", len(trace.response_ids))))
            merged["native_logprob_lens"].append(int(meta.get("native_logprob_len", len(trace.response_logprobs or []))))
        return merged

    @staticmethod
    def _pop_compatible_chain(
        *,
        prompt_key: str,
        prompt_ids: list[int],
        chains: list[list[CompletionRecord]],
        waiting_chains: dict[str, deque[int]],
    ) -> int | None:
        """Pop a waiting chain that matches both at message-key and token levels.

        The message-level key (produced by ``_grouping_key``) is only a
        *necessary* condition for joining a chain.  Its normalization drops
        tool messages and empty/``<think>`` assistants — both of which can
        hide genuine token-level divergence (cache-control shifts, tools
        schema rewrites, ``<system-reminder>`` injections).

        The *sufficient* condition is the strict append-only token-prefix
        invariant: ``C_{k+1}.prompt_ids`` must start with ``C_k.prompt_ids``.
        Enforcing this at chain-join time means a completion whose raw
        tokenization diverges from the waiting chain's tail starts its own
        new chain, instead of being silently appended (only to be dropped
        later in finalization).

        Scans candidates in FIFO order; returns the first compatible index
        and pops it.  Returns None if no candidate passes the token check.
        """
        queue = waiting_chains.get(prompt_key)
        if not queue:
            return None
        for pos, chain_idx in enumerate(queue):
            last_trace = build_trace_from_completion(chains[chain_idx][-1])
            last_pids = last_trace.prompt_ids
            if (
                not prompt_ids
                or not last_pids
                or len(prompt_ids) < len(last_pids)
                or prompt_ids[: len(last_pids)] != last_pids
            ):
                continue
            del queue[pos]
            if not queue:
                waiting_chains.pop(prompt_key, None)
            return chain_idx
        return None


def _top_level_scheduler_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = {"group_id", "policy_version", "rollout_step"}
    return {key: metadata[key] for key in keys if key in metadata}
