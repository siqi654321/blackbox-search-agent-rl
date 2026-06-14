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
