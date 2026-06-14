"""Built-in trajectory builder that emits one trace per request/completion."""

from __future__ import annotations

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.builder.record_utils import build_trace_from_completion, is_internal_completion_record
from polar.trajectory.models import CompletionSession, Trajectory


class PerRequestBuilder(BaseTrajectoryBuilder):
    """Convert every stored completion record into one trajectory trace."""

    async def build(self, session: CompletionSession) -> Trajectory:
        raw_completion_count = len(session.completions)
        completions = [
            completion
            for completion in session.completions
            if not is_internal_completion_record(completion)
        ]
        skipped_internal_count = raw_completion_count - len(completions)

        if not completions:
            return Trajectory(
                status="ERROR",
                metadata={
                    "builder": "per_request",
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

        return Trajectory(
            status="COMPLETED",
            metadata={
                "builder": "per_request",
                "session_id": session.session_id,
                "task_id": session.task_id,
                "api_type": session.api_type,
                "model_requested": session.model_requested,
                "model_used": session.model_used,
                "record_count": len(completions),
                "record_count_raw": raw_completion_count,
                "record_count_skipped_internal": skipped_internal_count,
                "task_metadata": dict(session.metadata),
                "trace_count": len(completions),
                **_top_level_scheduler_metadata(session.metadata),
            },
            traces=[
                _with_builder_metadata(build_trace_from_completion(completion), index)
                for index, completion in enumerate(completions)
            ],
        )


def _top_level_scheduler_metadata(metadata: dict) -> dict:
    keys = {"group_id", "policy_version", "rollout_step"}
    return {key: metadata[key] for key in keys if key in metadata}


def _with_builder_metadata(trace, index: int):
    metadata = dict(getattr(trace, "metadata", {}) or {})
    metadata.setdefault("builder", "per_request")
    metadata.setdefault("builder_trace_index", int(index))
    metadata.setdefault("split_reason", metadata.get("split_reason") or "per_request")
    try:
        return trace.model_copy(update={"metadata": metadata})
    except Exception:
        trace.metadata = metadata
        return trace
