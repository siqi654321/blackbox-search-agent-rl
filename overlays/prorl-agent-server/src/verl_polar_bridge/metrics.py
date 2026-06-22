"""Metric helpers for VERL + Polar rollout integration."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from verl_polar_bridge.adapter import VerlPolarSample


def summarize_samples(samples: list[VerlPolarSample]) -> dict[str, float]:
    rewards = [sample.reward for sample in samples if not sample.remove_sample]
    completed = [sample.reward for sample in samples if getattr(sample.status, "value", sample.status) == "completed"]
    trainable = [sample for sample in samples if sample.has_trainable_tokens]
    prompt_lengths = [len(sample.prompt_ids) for sample in samples if not sample.remove_sample]
    response_lengths = [len(sample.response_ids) for sample in samples if not sample.remove_sample]
    trainable_tokens = [sum(int(value) for value in sample.response_mask) for sample in samples if not sample.remove_sample]
    out: dict[str, float] = {
        "polar/sample_count": float(len(samples)),
        "polar/trainable_sample_count": float(len(trainable)),
        "polar/placeholder_count": float(sum(1 for sample in samples if sample.remove_sample)),
    }
    if prompt_lengths:
        out["polar/rollout_prompt_tokens_mean"] = float(mean(prompt_lengths))
        out["polar/rollout_prompt_tokens_max"] = float(max(prompt_lengths))
        out["polar/rollout_prompt_tokens_min"] = float(min(prompt_lengths))
    if response_lengths:
        out["polar/rollout_response_tokens_mean"] = float(mean(response_lengths))
        out["polar/rollout_response_tokens_max"] = float(max(response_lengths))
        out["polar/rollout_response_tokens_min"] = float(min(response_lengths))
    if trainable_tokens:
        out["polar/rollout_trainable_tokens_mean"] = float(mean(trainable_tokens))
        out["polar/rollout_trainable_tokens_sum"] = float(sum(trainable_tokens))
    if rewards:
        out["polar/reward_mean"] = float(mean(rewards))
        out["polar/reward_std"] = float(pstdev(rewards)) if len(rewards) > 1 else 0.0
    if completed:
        out["polar/reward_mean_completed"] = float(mean(completed))
    out.update(polar_extra_metrics(samples, rewards))
    return out


def polar_extra_metrics(samples: list[VerlPolarSample], rewards: list[float] | None = None) -> dict[str, float]:
    """Compact Prompt-grounded-compatible Polar metrics for logging."""
    rewards = [sample.reward for sample in samples if not sample.remove_sample] if rewards is None else rewards
    out: dict[str, float] = {}
    seen: set[str] = set()
    register_to_init_queue_ms: list[float] = []
    init_ms: list[float] = []
    run_ms: list[float] = []
    postrun_ms: list[float] = []
    session_is_placeholder: dict[str, bool] = {}
    completed_session_rewards: list[float] = []
    policy_staleness: list[float] = []
    resolved_values: list[float] = []
    sent_max_tokens: list[float] = []
    native_prompt_tokens: list[float] = []
    driver_timing: dict[str, list[float]] = {}
    stitched_trace_count = 0
    split_trace_count = 0
    builder_prefix_merged_sessions = 0
    builder_prefix_split_sessions = 0
    builder_prefix_prompt_grounded_single_sessions = 0
    builder_per_request_sessions = 0
    prompt_alignment_mismatch_count_sum = 0.0
    prompt_alignment_mismatch_sample_count = 0
    prompt_alignment_prompt_grounded_drift_tokens_sum = 0.0
    adapter_prompt_alignment_mismatch_count_sum = 0.0
    adapter_prompt_alignment_mismatch_sample_count = 0
    prompt_grounded_single_reset_count_sum = 0.0
    prompt_grounded_single_truncate_count_sum = 0.0
    prompt_grounded_single_truncated_tokens_sum = 0.0
    prompt_grounded_single_partial_masked_tokens_sum = 0.0
    prompt_grounded_single_context_tail_tokens_sum = 0.0

    for sample in samples:
        polar_meta = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
        if not isinstance(polar_meta, dict):
            continue
        trace_meta = polar_meta.get("trace_metadata") or {}
        if isinstance(trace_meta, dict) and trace_meta.get("adapter_stitched_trace_count"):
            stitched_trace_count += 1
        else:
            split_trace_count += 1
        if isinstance(trace_meta, dict):
            prompt_mismatch = _optional_float(trace_meta.get("prompt_alignment_mismatch_count"))
            if prompt_mismatch is not None:
                prompt_alignment_mismatch_count_sum += prompt_mismatch
                if prompt_mismatch > 0:
                    prompt_alignment_mismatch_sample_count += 1
            prompt_drift = _optional_float(trace_meta.get("prompt_alignment_prompt_grounded_drift_tokens_sum"))
            if prompt_drift is not None:
                prompt_alignment_prompt_grounded_drift_tokens_sum += prompt_drift
            adapter_mismatch = _optional_float(trace_meta.get("adapter_prompt_alignment_mismatch_count"))
            if adapter_mismatch is not None:
                adapter_prompt_alignment_mismatch_count_sum += adapter_mismatch
                if adapter_mismatch > 0:
                    adapter_prompt_alignment_mismatch_sample_count += 1
            prompt_grounded_single_reset_count_sum += _optional_float(trace_meta.get("prompt_grounded_single_reset_count")) or 0.0
            prompt_grounded_single_truncate_count_sum += _optional_float(trace_meta.get("prompt_grounded_single_truncate_count")) or 0.0
            prompt_grounded_single_truncated_tokens_sum += _optional_float(trace_meta.get("prompt_grounded_single_truncated_tokens")) or 0.0
            prompt_grounded_single_partial_masked_tokens_sum += _optional_float(trace_meta.get("prompt_grounded_single_partial_masked_tokens")) or 0.0
            prompt_grounded_single_context_tail_tokens_sum += _optional_float(trace_meta.get("prompt_grounded_single_context_tail_tokens")) or 0.0
        for key in ("polar_policy_staleness", "policy_staleness"):
            if key in polar_meta:
                policy_staleness.append(float(polar_meta[key]))
                break
        session_id = polar_meta.get("session_id")
        placeholder = bool(polar_meta.get("placeholder") or sample.remove_sample)
        if not session_id or session_id in seen:
            continue
        seen.add(str(session_id))
        trajectory_meta = polar_meta.get("trajectory_metadata") or {}
        if isinstance(trajectory_meta, dict):
            builder = trajectory_meta.get("builder")
            if builder == "prefix_merging":
                stats = trajectory_meta.get("reconstruction_stats") or {}
                trace_count = int(trajectory_meta.get("trace_count") or 0)
                completions_total = int(stats.get("completions_total") or 0) if isinstance(stats, dict) else 0
                if completions_total > trace_count:
                    builder_prefix_merged_sessions += 1
                if trace_count > 1:
                    builder_prefix_split_sessions += 1
            elif builder == "prefix_merging_prompt_grounded_single":
                builder_prefix_prompt_grounded_single_sessions += 1
                stats = trajectory_meta.get("reconstruction_stats") or {}
                trace_count = int(trajectory_meta.get("trace_count") or 0)
                completions_total = int(stats.get("completions_total") or 0) if isinstance(stats, dict) else 0
                if completions_total > trace_count:
                    builder_prefix_merged_sessions += 1
                if trace_count > 1:
                    builder_prefix_split_sessions += 1
            elif builder == "per_request":
                builder_per_request_sessions += 1
        timing = polar_meta.get("timing") or {}
        if isinstance(timing, dict) and timing:
            register_to_init_queue_ms.append(float(timing.get("register_to_init_queue_ms", 0.0)))
            init_ms.append(float(timing.get("init_ms", 0.0)))
            run_ms.append(float(timing.get("run_ms", 0.0)))
            postrun_ms.append(float(timing.get("postrun_ms", 0.0)))
        session_is_placeholder[str(session_id)] = placeholder
        if _session_status(sample) == "COMPLETED" and not placeholder:
            completed_session_rewards.append(float(sample.reward))
        trace_meta = polar_meta.get("trace_metadata") or {}
        if isinstance(trace_meta, dict):
            for item in trace_meta.get("completion_metadata") or []:
                if not isinstance(item, dict):
                    continue
                timing = item.get("driver_timing")
                if not isinstance(timing, dict):
                    continue
                for key, value in timing.items():
                    try:
                        driver_timing.setdefault(str(key), []).append(float(value))
                    except (TypeError, ValueError):
                        pass
        evaluation = (polar_meta.get("trajectory_metadata") or {}).get("evaluation") or {}
        if isinstance(evaluation, dict):
            timing = evaluation.get("driver_timing")
            if isinstance(timing, dict):
                for key, value in timing.items():
                    try:
                        driver_timing.setdefault(str(key), []).append(float(value))
                    except (TypeError, ValueError):
                        pass
            for key in (
                "driver_turns",
                "driver_tool_turns",
                "driver_tool_blocks",
                "driver_num_turns",
                "driver_num_turns_tool_messages",
                "driver_response_budget_used",
                "driver_cumulative_completion_tokens",
            ):
                if key in evaluation:
                    try:
                        driver_timing.setdefault(key, []).append(float(evaluation[key]))
                    except (TypeError, ValueError):
                        pass
            for value in evaluation.get("artifact_sent_max_tokens") or []:
                try:
                    sent_max_tokens.append(float(value))
                except (TypeError, ValueError):
                    pass
            for value in evaluation.get("artifact_native_prompt_tokens") or []:
                try:
                    native_prompt_tokens.append(float(value))
                except (TypeError, ValueError):
                    pass
        report = (evaluation.get("report") or {}) if isinstance(evaluation, dict) else {}
        if isinstance(report, dict) and "resolved" in report:
            resolved_values.append(1.0 if report.get("resolved") else 0.0)

    if init_ms:
        out["polar/session_ms/register_to_init_queue_mean"] = mean(register_to_init_queue_ms)
        out["polar/session_ms/init_mean"] = mean(init_ms)
        out["polar/session_ms/run_mean"] = mean(run_ms)
        out["polar/session_ms/postrun_mean"] = mean(postrun_ms)
    if rewards:
        out["polar/reward_mean"] = mean(rewards)
        out["polar/reward_std"] = pstdev(rewards) if len(rewards) > 1 else 0.0
    if completed_session_rewards:
        out["polar/reward_mean_completed"] = mean(completed_session_rewards)
    if policy_staleness:
        out["polar/staleness/mean"] = mean(policy_staleness)
    if seen:
        empty_sessions = sum(1 for value in session_is_placeholder.values() if value)
        out["polar/rollout_success_rate"] = (len(seen) - empty_sessions) / len(seen)
    out["polar/trace/adapter_stitched_sample_count"] = float(stitched_trace_count)
    out["polar/trace/split_sample_count"] = float(split_trace_count)
    out["polar/trace/builder_prefix_merged_sessions"] = float(builder_prefix_merged_sessions)
    out["polar/trace/builder_prefix_split_sessions"] = float(builder_prefix_split_sessions)
    out["polar/trace/builder_prefix_prompt_grounded_single_sessions"] = float(builder_prefix_prompt_grounded_single_sessions)
    out["polar/trace/builder_per_request_sessions"] = float(builder_per_request_sessions)
    out["polar/trace/prompt_alignment_mismatch_count_sum"] = float(prompt_alignment_mismatch_count_sum)
    out["polar/trace/prompt_alignment_mismatch_sample_count"] = float(prompt_alignment_mismatch_sample_count)
    out["polar/trace/prompt_alignment_prompt_grounded_drift_tokens_sum"] = float(prompt_alignment_prompt_grounded_drift_tokens_sum)
    out["polar/trace/adapter_prompt_alignment_mismatch_count_sum"] = float(adapter_prompt_alignment_mismatch_count_sum)
    out["polar/trace/adapter_prompt_alignment_mismatch_sample_count"] = float(adapter_prompt_alignment_mismatch_sample_count)
    out["polar/trace/prompt_grounded_single_reset_count_sum"] = float(prompt_grounded_single_reset_count_sum)
    out["polar/trace/prompt_grounded_single_truncate_count_sum"] = float(prompt_grounded_single_truncate_count_sum)
    out["polar/trace/prompt_grounded_single_truncated_tokens_sum"] = float(prompt_grounded_single_truncated_tokens_sum)
    out["polar/trace/prompt_grounded_single_partial_masked_tokens_sum"] = float(prompt_grounded_single_partial_masked_tokens_sum)
    out["polar/trace/prompt_grounded_single_context_tail_tokens_sum"] = float(prompt_grounded_single_context_tail_tokens_sum)
    if resolved_values:
        out["polar/eval/resolved_rate"] = mean(resolved_values)
    if sent_max_tokens:
        out["polar/search/sent_max_tokens_mean"] = mean(sent_max_tokens)
        out["polar/search/sent_max_tokens_max"] = max(sent_max_tokens)
        out["polar/search/sent_max_tokens_min"] = min(sent_max_tokens)
    for key, values in driver_timing.items():
        if values:
            out[f"polar/driver/{key}_mean"] = mean(values)
            out[f"polar/driver/{key}_max"] = max(values)
            out[f"polar/driver/{key}_sum"] = sum(values)
    if native_prompt_tokens:
        out["polar/search/native_prompt_tokens_mean"] = mean(native_prompt_tokens)
        out["polar/search/native_prompt_tokens_max"] = max(native_prompt_tokens)
        out["polar/search/native_prompt_tokens_min"] = min(native_prompt_tokens)
    return {key: float(value) for key, value in out.items()}


def _session_status(sample: VerlPolarSample) -> str | None:
    polar_meta = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    status = polar_meta.get("session_status") if isinstance(polar_meta, dict) else None
    return str(getattr(status, "value", status)) if status is not None else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_metrics_prefix(metrics: dict[str, float], prefix: str = "polar") -> dict[str, float]:
    """Rewrite canonical ``polar/`` metric keys to a configured prefix.

    Internal helpers emit Prompt-grounded-compatible ``polar/...`` keys.  Experiments can
    override the public prefix without changing every helper.
    """
    prefix = (prefix or "polar").strip().strip("/")
    if prefix == "polar":
        return {key: float(value) for key, value in metrics.items()}
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("polar/"):
            out[f"{prefix}/{key[len('polar/'):]}"] = float(value)
        else:
            out[key] = float(value)
    return out
