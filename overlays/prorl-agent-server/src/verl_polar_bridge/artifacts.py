"""Artifact helpers for inspecting Polar trajectories."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from verl_polar_bridge.adapter import VerlPolarSample


def dump_longest_trace(samples: list[VerlPolarSample], path: str | Path) -> Path | None:
    """Dump the longest response trace as JSON for debugging."""
    if not samples:
        return None
    sample = max(samples, key=lambda item: len(item.response_ids))
    payload: dict[str, Any] = {
        "uid": sample.uid,
        "group_index": sample.group_index,
        "trajectory_index": sample.trajectory_index,
        "trace_index": sample.trace_index,
        "response_length": len(sample.response_ids),
        "reward": sample.reward,
        "status": getattr(sample.status, "value", str(sample.status)),
        "prompt": _to_jsonable(sample.prompt),
        "response": sample.response,
        "metadata": _to_jsonable(sample.metadata),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out


def dump_aborted_samples(samples: list[VerlPolarSample], path: str | Path) -> Path | None:
    """Dump non-completed/placeholder samples as JSONL.

    This is intentionally sample-based: it captures trajectories that made it
    through conversion but were aborted/failed/truncated/placeholder.  Scheduler
    drop reasons are handled by :func:`dump_dropped_events`.
    """
    rows = [
        _sample_summary(sample)
        for sample in samples
        if sample.remove_sample or str(getattr(sample.status, "value", sample.status)) != "completed"
    ]
    if not rows:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out, rows)
    return out


def dump_dropped_events(events: list[dict[str, Any]], path: str | Path) -> Path | None:
    """Dump scheduler drop/rejection events as JSONL."""
    if not events:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out, events)
    return out


def dump_validation_fanout(payload: dict[str, Any], path: str | Path) -> Path | None:
    """Dump validation fan-out diagnostics as JSON."""
    if not payload:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True))
    return out


def dump_full_trajectory_samples(
    samples: list[VerlPolarSample],
    path: str | Path,
    *,
    global_steps: int,
    limit: int = 8,
    require_subagent: bool = False,
    validate: bool = False,
) -> Path | None:
    """Dump full, human-readable Polar samples grouped by parent rollout.

    Unlike :func:`dump_longest_trace`, this artifact is intentionally
    parent-level: subagent and final/main segment samples that came from one
    original rollout are written together so a reader can inspect the full
    prompt-grounded fanout/stitch view.
    """

    rows = full_trajectory_payloads(
        samples,
        global_steps=global_steps,
        limit=limit,
        require_subagent=require_subagent,
        validate=validate,
    )
    if not rows:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out, rows)
    return out


def full_trajectory_payloads(
    samples: list[VerlPolarSample],
    *,
    global_steps: int,
    limit: int = 8,
    require_subagent: bool = False,
    validate: bool = False,
) -> list[dict[str, Any]]:
    """Build JSONL rows for full trajectory artifacts."""

    grouped: dict[tuple[Any, ...], list[VerlPolarSample]] = {}
    for sample in samples:
        grouped.setdefault(_parent_trajectory_key(sample), []).append(sample)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        group_samples = grouped[key]
        ordered = sorted(
            group_samples,
            key=lambda sample: (
                _segment_sort_key(sample),
                int(getattr(sample, "trace_index", 0)),
            ),
        )
        has_subagent = any(_segment_kind(sample) == "subagent" for sample in ordered)
        if require_subagent and not has_subagent:
            continue
        rows.append(
            {
                "global_steps": int(global_steps),
                "validate": bool(validate),
                "uid": ordered[0].uid if ordered else None,
                "group_index": int(getattr(ordered[0], "group_index", 0)) if ordered else None,
                "trajectory_index": int(getattr(ordered[0], "trajectory_index", 0)) if ordered else None,
                "parent_key": _to_jsonable(key),
                "sample_count": len(ordered),
                "has_subagent": has_subagent,
                "reward_sum": float(sum(float(sample.reward) for sample in ordered)),
                "original_reward": _first_original_reward(ordered),
                "response_length_sum": int(sum(len(sample.response_ids) for sample in ordered)),
                "trainable_tokens_sum": int(sum(sum(int(v) for v in sample.response_mask) for sample in ordered)),
                "segments": [_full_segment_payload(sample) for sample in ordered],
                "stitched_readable": [_readable_segment_payload(sample) for sample in ordered],
            }
        )
        if len(rows) >= max(0, int(limit)):
            break
    return rows


def validation_fanout_payload(
    samples: list[VerlPolarSample],
    *,
    input_rows: int,
    global_steps: int,
    selected_samples: list[VerlPolarSample] | None = None,
) -> dict[str, Any] | None:
    """Build diagnostics for validation output fan-out.

    Returns ``None`` when the output is already one-to-one and no group has more
    than one sample.
    """
    grouped: dict[int, list[VerlPolarSample]] = {}
    for sample in samples:
        grouped.setdefault(int(sample.group_index), []).append(sample)
    fanout_groups = {group: group_samples for group, group_samples in grouped.items() if len(group_samples) != 1}
    if len(samples) == int(input_rows) and not fanout_groups:
        return None
    selected_by_group: dict[int, VerlPolarSample] = {}
    for sample in selected_samples or []:
        selected_by_group[int(sample.group_index)] = sample
    return {
        "global_steps": int(global_steps),
        "input_rows": int(input_rows),
        "output_samples_before_select": len(samples),
        "output_samples_after_select": len(selected_samples) if selected_samples is not None else None,
        "fanout_group_count": len(fanout_groups),
        "selected_samples": [_sample_summary(sample) for sample in selected_samples or []],
        "groups": [
            {
                "group_index": group_index,
                "sample_count": len(group_samples),
                "selected_trace_index": getattr(selected_by_group.get(group_index), "trace_index", None),
                "samples": [_sample_summary(sample) for sample in group_samples],
            }
            for group_index, group_samples in sorted(grouped.items())
            if len(group_samples) != 1
        ],
    }


def resolve_artifacts_dir(default_name: str = "polar_artifacts") -> Path | None:
    """Resolve the directory for training-time debug artifacts.

    Prefer an explicit ``POLAR_ARTIFACTS_DIR``.  When running through the smoke
    scripts, ``LOG_DIR`` is already scoped to the current job, so place artifacts
    under ``$LOG_DIR/artifacts``.  Return ``None`` when neither is set so library
    callers do not unexpectedly write into the cwd.
    """
    explicit = os.getenv("POLAR_ARTIFACTS_DIR")
    if explicit:
        return Path(explicit)
    log_dir = os.getenv("LOG_DIR")
    if log_dir:
        return Path(log_dir) / "artifacts"
    default = os.getenv("POLAR_ARTIFACTS_DEFAULT_DIR")
    if default:
        return Path(default)
    return None


def dump_eval_artifacts(result: Any, directory: str | Path, *, longest_trace_name: str = "longest_trace.json") -> dict[str, Path]:
    """Dump eval outputs/metrics and the longest trace for debugging."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = out_dir / "outputs.json"
    metrics_path = out_dir / "metrics.json"
    serializable_outputs = [_jsonable_output(output) for output in getattr(result, "outputs", [])]
    outputs_path.write_text(json.dumps(serializable_outputs, ensure_ascii=False, indent=2))
    metrics_path.write_text(json.dumps(getattr(result, "metrics", {}), ensure_ascii=False, indent=2, sort_keys=True))
    paths = {"outputs": outputs_path, "metrics": metrics_path}

    samples = list(getattr(result, "samples", None) or [])
    if not samples:
        for output in getattr(result, "outputs", []) or []:
            if isinstance(output, dict):
                samples.extend(output.get("samples") or [])
    trace_path = dump_longest_trace(samples, out_dir / longest_trace_name) if samples else None
    if trace_path is not None:
        paths["longest_trace"] = trace_path
    return paths


def _jsonable_output(output: Any) -> Any:
    if not isinstance(output, dict):
        return _to_jsonable(output)
    out = dict(output)
    if "samples" in out:
        out["samples"] = [_sample_summary(sample) for sample in out.get("samples") or []]
    return _to_jsonable(out)


def _full_segment_payload(sample: VerlPolarSample) -> dict[str, Any]:
    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    polar = polar if isinstance(polar, dict) else {}
    trace_debug = polar.get("trace_debug") if isinstance(polar, dict) else None
    trace_debug = trace_debug if isinstance(trace_debug, dict) else {}
    return {
        **_sample_summary(sample),
        "prompt_length": len(sample.prompt_ids),
        "response_length": len(sample.response_ids),
        "trainable_tokens": int(sum(int(v) for v in sample.response_mask)),
        "rollout_logprob_count": len(sample.rollout_log_probs),
        "prompt": _to_jsonable(sample.prompt),
        "response": sample.response,
        "segment_kind": polar.get("segment_kind"),
        "merge_group_id": polar.get("merge_group_id"),
        "segment_group_id": polar.get("segment_group_id"),
        "parent_merge_group_id": polar.get("parent_merge_group_id"),
        "segment_idx": polar.get("segment_idx"),
        "num_segments": polar.get("num_segments"),
        "segment_weight": polar.get("segment_weight"),
        "original_reward": polar.get("original_reward"),
        "segment_reward": polar.get("segment_reward"),
        "trace_response_messages": _to_jsonable(trace_debug.get("response_messages")),
    }


def _readable_segment_payload(sample: VerlPolarSample) -> dict[str, Any]:
    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    polar = polar if isinstance(polar, dict) else {}
    return {
        "segment_kind": polar.get("segment_kind") or "unknown",
        "merge_group_id": polar.get("merge_group_id"),
        "parent_merge_group_id": polar.get("parent_merge_group_id"),
        "response_length": len(sample.response_ids),
        "trainable_tokens": int(sum(int(v) for v in sample.response_mask)),
        "reward": float(sample.reward),
        "response": sample.response,
    }


def _sample_summary(sample: VerlPolarSample) -> dict[str, Any]:
    return {
        "uid": sample.uid,
        "group_index": sample.group_index,
        "trajectory_index": sample.trajectory_index,
        "trace_index": sample.trace_index,
        "response_length": len(sample.response_ids),
        "reward": sample.reward,
        "status": getattr(sample.status, "value", str(sample.status)),
        "metadata": _to_jsonable(sample.metadata),
    }


def _parent_trajectory_key(sample: VerlPolarSample) -> tuple[Any, ...]:
    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    polar = polar if isinstance(polar, dict) else {}
    merge_group_id = polar.get("merge_group_id")
    parent_merge_group_id = polar.get("parent_merge_group_id")
    segment_kind = polar.get("segment_kind")
    if parent_merge_group_id is not None:
        parent_id = str(parent_merge_group_id)
    elif segment_kind in {"final", "main"} and merge_group_id is not None:
        parent_id = str(merge_group_id)
    else:
        # Fall back to original row/sample identity. This keeps validation rows
        # and non-subagent trajectories grouped sensibly even when no explicit
        # merge-group metadata is present.
        parent_id = str(polar.get("sample_uid") or sample.uid)
    return (
        int(sample.group_index),
        int(sample.trajectory_index),
        parent_id,
    )


def _segment_kind(sample: VerlPolarSample) -> str:
    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    if isinstance(polar, dict) and polar.get("segment_kind") is not None:
        return str(polar.get("segment_kind"))
    return ""


def _segment_sort_key(sample: VerlPolarSample) -> tuple[int, int, int]:
    polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
    polar = polar if isinstance(polar, dict) else {}
    kind = str(polar.get("segment_kind") or "")
    if kind == "subagent":
        major = 0
    elif kind in {"final", "main"}:
        major = 1
    else:
        major = 2
    dispatch = _safe_int(polar.get("dispatch_index"), default=_safe_int(polar.get("merge_group_index"), default=0))
    segment_idx = _safe_int(polar.get("segment_idx"), default=_safe_int(polar.get("segment_index"), default=0))
    return (major, dispatch, segment_idx)


def _first_original_reward(samples: list[VerlPolarSample]) -> float | None:
    for sample in samples:
        polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
        if isinstance(polar, dict) and polar.get("original_reward") is not None:
            return float(polar.get("original_reward"))
    return None


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(_to_jsonable(key)): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _to_jsonable(getattr(value, "value"))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_jsonable(model_dump(mode="python"))
        except TypeError:
            return _to_jsonable(model_dump())
    if hasattr(value, "__dict__"):
        return _to_jsonable({key: item for key, item in vars(value).items() if not key.startswith("_")})
    return str(value)
