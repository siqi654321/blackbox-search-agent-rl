"""VERL AgentLoopManager entry point for Polar-backed rollouts."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from polar.http_utils import polar_async_client, polar_http_timeout
from verl_polar_bridge.artifacts import (
    dump_aborted_samples,
    dump_dropped_events,
    dump_full_trajectory_samples,
    dump_longest_trace,
    dump_validation_fanout,
    resolve_artifacts_dir,
    validation_fanout_payload,
)
from verl_polar_bridge.client import PolarGatewayClient
from verl_polar_bridge.config import resolve_polar_verl_config
from verl_polar_bridge.debug_utils import env_flag, env_int, stable_hash, token_preview
from verl_polar_bridge.dataproto import prompt_rows_to_samples, samples_to_dataproto
from verl_polar_bridge.adapter import VerlPolarSample, VerlPolarStatus
from verl_polar_bridge.metrics import apply_metrics_prefix, summarize_samples
from verl_polar_bridge.scheduler import AsyncPolarScheduler, PolarCompletedGroup
from verl_polar_bridge.variable_pack import (
    build_packed_variable_payload,
    compact_samples_for_fixed_output,
    resolve_packed_variable_config,
)

logger = logging.getLogger(__name__)


class PolarAgentLoopManager:
    """Drop-in manager class configured via VERL ``agent_loop_manager_class``.

    This first implementation establishes the lifecycle/delegation boundary and
    validates Polar configuration.  Full DataProto generation is implemented in
    later plan phases; until then callers get a clear error instead of silently
    falling back to native VERL rollout.
    """

    def __init__(
        self,
        config: Any,
        worker_group: Any = None,
        rollout_resource_pool: Any = None,
        reward_loop_worker_handles: list[Any] | None = None,
        *,
        native_manager: Any = None,
    ) -> None:
        self.config = config
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.reward_loop_worker_handles = reward_loop_worker_handles
        self.polar_config = resolve_polar_verl_config(config)
        _log_resolved_polar_config(self.polar_config, config)
        self.async_scheduler_factory = AsyncPolarScheduler
        self.native_manager = native_manager
        self.rollout_replicas = getattr(native_manager, "rollout_replicas", []) if native_manager is not None else []
        self.server_handles = getattr(native_manager, "server_handles", []) if native_manager is not None else []
        self.server_addresses = getattr(native_manager, "server_addresses", []) if native_manager is not None else []
        self.native_openai_bridge: Any | None = None
        self._policy_version = 0
        self._admission_paused = False
        try:
            from verl_polar_bridge.hooks import register_manager

            register_manager(self)
        except Exception:
            pass

    @classmethod
    def create(
        cls,
        config: Any,
        worker_group: Any = None,
        rollout_resource_pool: Any = None,
        reward_loop_worker_handles: list[Any] | None = None,
    ) -> "PolarAgentLoopManager":
        return _run_coro_sync(
            cls.create_async(
                config=config,
                worker_group=worker_group,
                rollout_resource_pool=rollout_resource_pool,
                reward_loop_worker_handles=reward_loop_worker_handles,
            )
        )

    @classmethod
    async def create_async(
        cls,
        config: Any,
        worker_group: Any = None,
        rollout_resource_pool: Any = None,
        reward_loop_worker_handles: list[Any] | None = None,
    ) -> "PolarAgentLoopManager":
        native_manager = await _create_native_agent_loop_manager(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        instance = cls(
            config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
            native_manager=native_manager,
        )
        instance.configure_polar_gateway_upstreams()
        return instance

    def generate_sequences(self, prompts: Any) -> Any:
        """Generate VERL rollout batch through Polar.

        VERL's PPO trainer calls this method synchronously, while Polar task
        submission is callback/poll based.  The public contract therefore stays
        sync and this method drives an internal async batch scheduler that
        admits groups up to Polar backpressure limits, waits for completions,
        drops stale/invalid groups, and then packs accepted traces into a VERL
        ``DataProto``.
        """
        return _run_coro_sync(self.generate_sequences_async(prompts))

    async def generate_sequences_async(self, prompts: Any) -> Any:
        """Async implementation used by ``generate_sequences`` and tests."""
        rows = prompt_rows_to_samples(prompts)
        rollout_config = _get_path(self.config, "actor_rollout_ref.rollout", default={}) or {}
        prompt_length = int(_get_value(rollout_config, "prompt_length", 0) or 0)
        response_length = int(_get_value(rollout_config, "response_length", 0) or 0)
        if prompt_length <= 0 or response_length <= 0:
            raise ValueError(
                "actor_rollout_ref.rollout.prompt_length and response_length must be set "
                "to build VERL DataProto from Polar traces"
            )
        pad_token_id = int(_get_path(self.config, "actor_rollout_ref.model.pad_token_id", default=0) or 0)
        # VERL's PPO trainer has already expanded ``gen_batch`` with
        # ``batch.repeat(rollout.n, interleave=True)`` before calling an
        # AgentLoopManager.  Native VERL agent-loop workers therefore generate
        # one trajectory per input row.  Keep the Polar bridge row-based too:
        # submitting ``num_samples=rollout.n`` here would silently create
        # n^2 rollouts and break GRPO/PPO alignment for n > 1.
        num_rollouts_per_task = 1
        global_steps = int(getattr(prompts, "meta_info", {}).get("global_steps", 0) or 0)
        max_tokens = prompt_length + response_length

        scheduler = self._build_async_scheduler()
        scheduler.set_rollout_context(global_steps)
        scheduler.update_policy_version(max(self._policy_version, global_steps))
        completed_groups: list[PolarCompletedGroup] = []
        timeout = None if self.polar_config.request_timeout is None else polar_http_timeout(self.polar_config.request_timeout)
        callback_context = scheduler if self._scheduler_needs_callback_listener(scheduler) else _null_async_context()
        async with callback_context:
            async with polar_async_client(timeout=timeout) as client:
                for group_index, row in enumerate(rows):
                    scheduler.submit_group(
                        row,
                        group_index=group_index,
                        rollout_id=group_index,
                        num_rollouts=num_rollouts_per_task,
                        global_steps=global_steps,
                        uid=getattr(row, "uid", None),
                        max_tokens=max_tokens,
                    )
                while _scheduler_has_work(scheduler):
                    await scheduler.run_until_capacity(client)
                    completed_groups.extend(
                        await scheduler.drain_completed(max_groups=len(rows), rollout_id=global_steps)
                    )
                    if not scheduler.active:
                        continue
                    completed = await scheduler.wait_for_next(timeout=0.5)
                    if completed is not None:
                        scheduler.completed_buffer.append(completed)
                    completed_groups.extend(
                        await scheduler.drain_completed(max_groups=len(rows), rollout_id=global_steps)
                    )

        samples = [sample for group in completed_groups for sample in group.samples]
        polar_metrics = scheduler.snapshot_metrics()
        validation_fanout = None
        if bool(getattr(prompts, "meta_info", {}).get("validate", False)):
            selected_samples = self._select_one_validation_sample_per_group(samples, rows=rows, prompts=prompts)
            validation_fanout = validation_fanout_payload(
                samples,
                input_rows=len(rows),
                global_steps=global_steps,
                selected_samples=selected_samples,
            )
            if validation_fanout is not None:
                logger.warning(
                    "POLAR_VALIDATION_FANOUT_DEBUG summary=%s",
                    {
                        "global_steps": validation_fanout.get("global_steps"),
                        "input_rows": validation_fanout.get("input_rows"),
                        "output_samples_before_select": validation_fanout.get("output_samples_before_select"),
                        "output_samples_after_select": validation_fanout.get("output_samples_after_select"),
                        "fanout_group_count": validation_fanout.get("fanout_group_count"),
                    },
                )
                if _env_flag("POLAR_VALIDATION_FANOUT_VERBOSE"):
                    logger.warning("POLAR_VALIDATION_FANOUT_DEBUG_FULL %s", validation_fanout)
            samples = selected_samples
        is_validate = bool(getattr(prompts, "meta_info", {}).get("validate", False))
        if not samples:
            raise RuntimeError(
                "Polar rollout produced no accepted trainable samples; "
                f"validate={is_validate} "
                f"input_rows={len(rows)} completed_groups={len(completed_groups)} "
                f"metrics={polar_metrics}"
            )
        packed_variable_payload = None
        packed_variable_metrics: dict[str, float] = {}
        packed_variable_config = resolve_packed_variable_config()
        if packed_variable_config.enabled and not is_validate:
            packed_variable_payload, packed_variable_metrics = build_packed_variable_payload(
                samples,
                prompt_length=prompt_length,
                response_length=response_length,
                max_pack_tokens=packed_variable_config.max_pack_tokens,
            )
        fixed_output_samples = samples
        if (
            packed_variable_payload is not None
            and _env_flag("POLAR_PACKED_VARIABLE_ACTOR_UPDATE")
            and _env_flag("POLAR_PACKED_VARIABLE_COMPACT_FIXED_OUTPUT")
        ):
            fixed_output_samples = compact_samples_for_fixed_output(samples)
            packed_variable_metrics["polar/packed_variable/fixed_output_compacted"] = 1.0
            packed_variable_metrics["polar/packed_variable/fixed_output_sample_count"] = float(len(fixed_output_samples))
        output = samples_to_dataproto(
            fixed_output_samples,
            prompt_length=prompt_length,
            response_length=response_length,
            pad_token_id=pad_token_id,
        )
        if packed_variable_payload is not None:
            output.meta_info["polar_packed_variable_train_payload"] = packed_variable_payload
        if not self.polar_config.dynamic_history_enable:
            output.meta_info.pop("polar_dynamic_history", None)
        if is_validate:
            self._make_validation_union_safe(output, prompts)
        polar_metrics.update(summarize_samples(samples))
        polar_metrics.update(packed_variable_metrics)
        full_artifact_paths = self._maybe_dump_full_trajectory_artifacts(
            samples,
            global_steps=global_steps,
            validate=is_validate,
        )
        if full_artifact_paths.get("validation") is not None:
            polar_metrics["polar/artifacts/validation_trajectories_dumped"] = 1.0
        if full_artifact_paths.get("subagent") is not None:
            polar_metrics["polar/artifacts/subagent_trajectories_dumped"] = 1.0
        if full_artifact_paths.get("full") is not None:
            polar_metrics["polar/artifacts/full_trajectories_dumped"] = 1.0
        alignment_path = self._maybe_dump_alignment_debug_artifact(samples, output, global_steps=global_steps)
        if alignment_path is not None:
            polar_metrics["polar/artifacts/alignment_debug_dumped"] = 1.0
        artifact_path = self._maybe_dump_longest_trace(samples, global_steps=global_steps)
        if artifact_path is not None:
            polar_metrics["polar/artifacts/longest_trace_dumped"] = 1.0
        artifact_paths = self._maybe_dump_aborted_artifacts(scheduler, samples, global_steps=global_steps)
        if artifact_paths.get("aborted_samples") is not None:
            polar_metrics["polar/artifacts/aborted_samples_dumped"] = 1.0
        if artifact_paths.get("dropped_events") is not None:
            polar_metrics["polar/artifacts/dropped_events_dumped"] = 1.0
        fanout_path = self._maybe_dump_validation_fanout(validation_fanout, global_steps=global_steps)
        if fanout_path is not None:
            polar_metrics["polar/artifacts/validation_fanout_dumped"] = 1.0
        polar_metrics = apply_metrics_prefix(polar_metrics, self.polar_config.metrics_prefix)
        output.meta_info["timing"] = {}
        # VERL's native AgentLoopManager emits a list of per-sample metric dicts;
        # DataProto.concat knows how to merge this shape. Keep one aggregate dict
        # here so trainer integrations can reduce/log it when available.
        output.meta_info["metrics"] = [polar_metrics]
        output.meta_info["polar_metrics"] = polar_metrics
        output.meta_info["polar_scheduler_stats"] = polar_metrics
        if _env_flag("POLAR_MANAGER_METRICS_DEBUG"):
            logger.warning(
                "POLAR_MANAGER_METRICS_DEBUG sample_count=%s metric_count=%s metric_keys=%s",
                len(samples),
                len(polar_metrics),
                sorted(polar_metrics.keys())[:80],
            )
        return output

    def _maybe_dump_alignment_debug_artifact(self, samples: list[Any], output: Any, *, global_steps: int) -> Path | None:
        if not env_flag("POLAR_ALIGNMENT_DEBUG"):
            return None
        artifacts_dir = resolve_artifacts_dir()
        if artifacts_dir is None:
            return None
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        limit = env_int("POLAR_ALIGNMENT_DEBUG_ARTIFACT_LIMIT", 16)
        path = artifacts_dir / f"alignment_debug_step_{int(global_steps):06d}.json"
        rows = []
        tensors = getattr(output, "batch", None)
        if tensors is None:
            tensors = {}
        for idx, sample in enumerate(samples[: max(0, limit)]):
            polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
            trace_meta = polar.get("trace_metadata", {}) if isinstance(polar, dict) else {}
            row = {
                "row": idx,
                "uid": sample.uid,
                "source_uid": polar.get("source_uid") if isinstance(polar, dict) else None,
                "group_index": sample.group_index,
                "trace_index": sample.trace_index,
                "status": str(getattr(sample.status, "value", sample.status)),
                "reward": float(sample.reward),
                "prompt_ids": token_preview(sample.prompt_ids, limit=32),
                "response_ids": token_preview(sample.response_ids, limit=64),
                "response_hash": stable_hash(sample.response_ids),
                "response_mask": token_preview(sample.response_mask, limit=64),
                "rollout_log_probs_head": [float(v) for v in sample.rollout_log_probs[:64]],
                "trainable_positions": [i for i, v in enumerate(sample.response_mask) if int(v)][:128],
                "trace_metadata": trace_meta,
                "polar_metadata_keys": sorted(str(k) for k in polar.keys()) if isinstance(polar, dict) else [],
            }
            try:
                if "responses" in tensors:
                    row["tensor_response"] = token_preview(tensors["responses"][idx].detach().cpu().tolist(), limit=64)
                if "response_mask" in tensors:
                    row["tensor_response_mask"] = token_preview(tensors["response_mask"][idx].detach().cpu().tolist(), limit=64)
                if "rollout_log_probs" in tensors:
                    row["tensor_rollout_log_probs_head"] = [float(v) for v in tensors["rollout_log_probs"][idx].detach().cpu().tolist()[:64]]
            except Exception as exc:
                row["tensor_debug_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        payload = {
            "global_steps": int(global_steps),
            "sample_count": len(samples),
            "output_batch_len": len(output),
            "tensor_keys": sorted(str(k) for k in tensors.keys()),
            "non_tensor_keys": sorted(str(k) for k in (getattr(output, "non_tensor_batch", None) if getattr(output, "non_tensor_batch", None) is not None else {}).keys()),
            "rows": rows,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        logger.warning("POLAR_ALIGNMENT_ARTIFACT_DEBUG path=%s sample_count=%s", path, len(samples))
        return path


    def _make_validation_union_safe(self, output: Any, prompts: Any) -> None:
        """Avoid VERL validation ``test_batch.union(test_output_gen_batch)`` key collisions.

        The training path has a Polar-specific dynamic-history alignment branch
        before union.  VERL's validation path does not; it directly unions the
        original validation batch with rollout output.  Therefore duplicate
        non-tensor keys such as ``uid``/``source_uid`` must either be identical
        arrays or absent on the rollout output.  Keep rollout provenance in
        ``polar_metadata`` and drop the colliding helper keys for validation.
        """
        non_tensors = getattr(output, "non_tensor_batch", None)
        if not isinstance(non_tensors, dict):
            return
        for key in ("uid", "source_uid", "raw_prompt", "multi_modal_inputs"):
            if key in getattr(prompts, "non_tensor_batch", {}):
                non_tensors.pop(key, None)
        # Validation does not run the trainer-side dynamic-history alignment
        # branch, so leaving the flag set only confuses debugging.
        getattr(output, "meta_info", {}).pop("polar_dynamic_history", None)

    def _select_one_validation_sample_per_group(self, samples: list[Any], *, rows: list[Any], prompts: Any) -> list[Any]:
        """Keep validation rollout output one-to-one with VERL validation inputs.

        VERL validation directly unions ``test_batch`` with rollout output, so
        the output batch size must remain exactly the padded validation input
        size.  If Polar emits multiple traces for a session, choose the same kind
        of preferred row as the training prune fallback: non-placeholder,
        trainable, then longest sequence.  If a validation group was dropped by
        acceptance filtering, insert an all-zero placeholder for that exact
        group index instead of shortening the output; otherwise VERL's later
        unpad+union path can see fewer rows than the original validation batch.
        """
        grouped: dict[int, list[Any]] = {}
        for sample in samples:
            grouped.setdefault(int(getattr(sample, "group_index", 0)), []).append(sample)
        selected: list[Any] = []
        for group_index, row in enumerate(rows):
            if group_index in grouped:
                selected.append(max(grouped[group_index], key=_validation_sample_preference))
            else:
                selected.append(_validation_placeholder_sample(row=row, group_index=group_index, prompts=prompts))
        return selected

    def _maybe_dump_longest_trace(self, samples: list[Any], *, global_steps: int) -> Path | None:
        if not self.polar_config.metrics_log_longest_trace_artifact:
            return None
        interval = max(1, int(self.polar_config.metrics_longest_trace_interval))
        if int(global_steps) % interval != 0:
            return None
        artifacts_dir = resolve_artifacts_dir()
        if artifacts_dir is None:
            return None
        path = artifacts_dir / f"longest_trace_step_{int(global_steps):06d}.json"
        try:
            return dump_longest_trace(samples, path)
        except Exception:
            logger.exception("Failed to dump Polar longest trace artifact to %s", path)
            return None

    def _maybe_dump_aborted_artifacts(
        self,
        scheduler: AsyncPolarScheduler,
        samples: list[Any],
        *,
        global_steps: int,
    ) -> dict[str, Path | None]:
        if not self.polar_config.metrics_log_longest_trace_artifact:
            return {}
        interval = max(1, int(self.polar_config.metrics_longest_trace_interval))
        if int(global_steps) % interval != 0:
            return {}
        artifacts_dir = resolve_artifacts_dir()
        if artifacts_dir is None:
            return {}
        paths: dict[str, Path | None] = {"aborted_samples": None, "dropped_events": None}
        aborted_path = artifacts_dir / f"aborted_samples_step_{int(global_steps):06d}.jsonl"
        dropped_path = artifacts_dir / f"dropped_events_step_{int(global_steps):06d}.jsonl"
        try:
            paths["aborted_samples"] = dump_aborted_samples(samples, aborted_path)
        except Exception:
            logger.exception("Failed to dump Polar aborted sample artifact to %s", aborted_path)
        try:
            paths["dropped_events"] = dump_dropped_events(list(getattr(scheduler.stats, "dropped_events", []) or []), dropped_path)
        except Exception:
            logger.exception("Failed to dump Polar dropped event artifact to %s", dropped_path)
        return paths

    def _maybe_dump_validation_fanout(self, payload: dict[str, Any] | None, *, global_steps: int) -> Path | None:
        if not payload:
            return None
        artifacts_dir = resolve_artifacts_dir()
        if artifacts_dir is None:
            return None
        path = artifacts_dir / f"validation_fanout_step_{int(global_steps):06d}.json"
        try:
            return dump_validation_fanout(payload, path)
        except Exception:
            logger.exception("Failed to dump Polar validation fanout artifact to %s", path)
            return None

    def _maybe_dump_full_trajectory_artifacts(
        self,
        samples: list[Any],
        *,
        global_steps: int,
        validate: bool,
    ) -> dict[str, Path | None]:
        paths: dict[str, Path | None] = {"validation": None, "subagent": None, "full": None}
        artifacts_dir = resolve_artifacts_dir()
        if artifacts_dir is None:
            return paths

        if validate and env_flag("POLAR_VALIDATION_TRAJECTORY_ARTIFACT"):
            limit = env_int("POLAR_VALIDATION_TRAJECTORY_LIMIT", 8)
            path = artifacts_dir / f"validation_trajectories_step_{int(global_steps):06d}.jsonl"
            try:
                paths["validation"] = dump_full_trajectory_samples(
                    samples,
                    path,
                    global_steps=global_steps,
                    limit=limit,
                    require_subagent=False,
                    validate=True,
                )
            except Exception:
                logger.exception("Failed to dump Polar validation trajectory artifact to %s", path)

        if (not validate) and env_flag("POLAR_SUBAGENT_TRAJECTORY_ARTIFACT"):
            limit = env_int("POLAR_SUBAGENT_TRAJECTORY_LIMIT", 8)
            path = artifacts_dir / f"subagent_trajectories_step_{int(global_steps):06d}.jsonl"
            try:
                paths["subagent"] = dump_full_trajectory_samples(
                    samples,
                    path,
                    global_steps=global_steps,
                    limit=limit,
                    require_subagent=True,
                    validate=False,
                )
            except Exception:
                logger.exception("Failed to dump Polar subagent trajectory artifact to %s", path)

        if env_flag("POLAR_FULL_TRAJECTORY_ARTIFACT"):
            limit = env_int("POLAR_FULL_TRAJECTORY_LIMIT", 8)
            suffix = "validation" if validate else "train"
            path = artifacts_dir / f"full_trajectories_{suffix}_step_{int(global_steps):06d}.jsonl"
            try:
                paths["full"] = dump_full_trajectory_samples(
                    samples,
                    path,
                    global_steps=global_steps,
                    limit=limit,
                    require_subagent=env_flag("POLAR_FULL_TRAJECTORY_REQUIRE_SUBAGENT"),
                    validate=validate,
                )
            except Exception:
                logger.exception("Failed to dump Polar full trajectory artifact to %s", path)
        return paths

    def update_policy_version(self, policy_version: int) -> None:
        """Hook for trainer/rollout code after serving weights are updated."""
        self._policy_version = max(self._policy_version, int(policy_version))

    def prepare_policy_update(self, policy_version: int) -> None:
        """Pause Polar admission/generation before an overlapping weight sync."""
        del policy_version
        self._admission_paused = True
        if not self.polar_config.allow_weight_update_overlap:
            return
        try:
            self._pause_gateway_generation()
        except Exception:
            try:
                self._resume_gateway_generation()
            finally:
                self._admission_paused = False
            raise

    def finish_policy_update(self, policy_version: int) -> None:
        """Resume Polar gateway/admission after an overlapping weight sync."""
        self.update_policy_version(policy_version)
        try:
            if self.polar_config.allow_weight_update_overlap:
                self._resume_gateway_generation()
        finally:
            self._admission_paused = False

    def abort_policy_update(self, policy_version: int) -> None:
        """Resume gateway/admission after a failed weight sync without advancing version."""
        del policy_version
        try:
            if self.polar_config.allow_weight_update_overlap:
                self._resume_gateway_generation()
        finally:
            self._admission_paused = False

    def _build_async_scheduler(self) -> AsyncPolarScheduler:
        scheduler = self.async_scheduler_factory(trainer_config=self.config, polar_config=self.polar_config)
        if self._admission_paused and hasattr(scheduler, "pause_admission"):
            scheduler.pause_admission()
        return scheduler

    def _scheduler_needs_callback_listener(self, scheduler: AsyncPolarScheduler) -> bool:
        return isinstance(scheduler, AsyncPolarScheduler)

    def configure_polar_gateway_upstreams(self) -> None:
        """Point Polar gateway at VERL/Ray-managed rollout serving.

        VERL starts the SGLang/vLLM rollout replicas through the native
        AgentLoopManager. Polar must use those policy servers, not an external
        retrieval-summary SGLang.  VERL SGLang servers expose ``/generate``
        rather than OpenAI chat completions, so we expose one local
        OpenAI-compatible bridge that load-balances over all VERL-managed
        endpoints.
        """
        if not self.polar_config.gateway_url:
            return
        if self.server_addresses:
            base_generate_urls = [_server_address_to_openai_base_url(address) for address in self.server_addresses]
            if self.native_openai_bridge is None:
                from verl_polar_bridge.native_openai_server import start_native_openai_bridge

                self.native_openai_bridge = start_native_openai_bridge(
                    sglang_base_urls=base_generate_urls,
                    tokenizer_name_or_path=self.polar_config.tokenizer_name_or_path or _get_path(self.config, "actor_rollout_ref.model.path"),
                    model_name=_get_path(self.config, "actor_rollout_ref.model.path"),
                )
            base_url = self.native_openai_bridge.base_url
            logger.warning(
                "POLAR_GATEWAY_UPSTREAMS server_count=%s bridge=%s upstreams=%s",
                len(base_generate_urls),
                base_url,
                base_generate_urls,
            )
        elif self.server_handles:
            raise RuntimeError(
                "Polar Prompt-grounded bridge requires VERL rollout server_addresses for SGLang /generate; "
                "server_handles-only Ray actor path is disabled to avoid patching VERL internals."
            )
        else:
            return
        client = PolarGatewayClient(
            self.polar_config.gateway_url,
            timeout=max(self.polar_config.gateway_control_timeout, 5.0),
        )
        client.update_upstream(base_url, timeout_seconds=self.polar_config.gateway_control_timeout)

    def _pause_gateway_generation(self) -> None:
        if not self.polar_config.gateway_url:
            raise RuntimeError(
                "polar.gateway_url or polar.topology_path is required when "
                "polar.allow_weight_update_overlap is enabled"
            )
        client = PolarGatewayClient(
            self.polar_config.gateway_url,
            timeout=max(self.polar_config.weight_update_pause_timeout + 5.0, 10.0),
        )
        client.pause_generation(timeout_seconds=self.polar_config.weight_update_pause_timeout)

    def _resume_gateway_generation(self) -> None:
        if not self.polar_config.gateway_url:
            return
        client = PolarGatewayClient(
            self.polar_config.gateway_url,
            timeout=max(self.polar_config.gateway_control_timeout, 5.0),
        )
        client.resume_generation()

    async def clear_kv_cache(self) -> Any:
        if self.native_manager is not None and hasattr(self.native_manager, "clear_kv_cache"):
            return await self.native_manager.clear_kv_cache()
        return None

    async def start_profile(self) -> Any:
        if self.native_manager is not None and hasattr(self.native_manager, "start_profile"):
            return await self.native_manager.start_profile()
        return None

    async def stop_profile(self) -> Any:
        if self.native_manager is not None and hasattr(self.native_manager, "stop_profile"):
            return await self.native_manager.stop_profile()
        return None


def _get_path(obj: Any, path: str, *, default: Any = None) -> Any:
    current = obj
    for part in path.split("."):
        current = _get_value(current, part, default=None)
        if current is None:
            return default
    return current


def _scheduler_has_work(scheduler: AsyncPolarScheduler) -> bool:
    return bool(
        scheduler.deferred_queue
        or scheduler.active
        or scheduler.output_queue.qsize()
        or scheduler.completed_buffer
    )


def _log_resolved_polar_config(polar_config: Any, raw_config: Any) -> None:
    task_template = getattr(polar_config, "task_template", {}) or {}
    agent = task_template.get("agent") if isinstance(task_template, dict) else None
    builder = task_template.get("builder") if isinstance(task_template, dict) else None
    evaluator = task_template.get("evaluator") if isinstance(task_template, dict) else None
    runtime = task_template.get("runtime") if isinstance(task_template, dict) else None
    agent = agent if isinstance(agent, dict) else {}
    builder = builder if isinstance(builder, dict) else {}
    evaluator = evaluator if isinstance(evaluator, dict) else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    logger.warning(
        "POLAR_RESOLVED_TASK_TEMPLATE_DEBUG "
        "env_search=%s "
        "agent_import_path=%s agent_harness=%s builder_strategy=%s evaluator_strategy=%s "
        "runtime_import_path=%s agent_settings=%s agent_env_sampling=%s task_template_keys=%s",
        os.environ.get("POLAR_SEARCH_HARNESS"),
        agent.get("import_path"),
        agent.get("harness"),
        builder.get("strategy"),
        evaluator.get("strategy"),
        runtime.get("import_path"),
        agent.get("settings"),
        {key: (agent.get("env") or {}).get(key) for key in ("SEARCH_TEMPERATURE", "SEARCH_TOP_P", "SEARCH_DO_SAMPLE")} if isinstance(agent.get("env"), dict) else None,
        sorted(task_template.keys()) if isinstance(task_template, dict) else type(task_template).__name__,
    )


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Some VERL/Ray workers already own an event loop.  Keep the sync public
    # API usable by running the coroutine in a short-lived helper thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


@asynccontextmanager
async def _null_async_context():
    yield None


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    get_method = getattr(obj, "get", None)
    if callable(get_method):
        try:
            return get_method(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _validation_sample_preference(sample: Any) -> tuple[int, int, int, int]:
    remove_sample = bool(getattr(sample, "remove_sample", False))
    response_mask = getattr(sample, "response_mask", []) or []
    response_ids = getattr(sample, "response_ids", []) or []
    prompt_ids = getattr(sample, "prompt_ids", []) or []
    trainable_tokens = sum(int(value) for value in response_mask)
    seqlen = len(prompt_ids) + len(response_ids)
    trace_index = int(getattr(sample, "trace_index", 0) or 0)
    return (0 if remove_sample else 1, int(trainable_tokens), int(seqlen), -trace_index)


def _validation_placeholder_sample(*, row: Any, group_index: int, prompts: Any) -> VerlPolarSample:
    """Return one validation-safe dummy rollout row for a dropped group.

    The row is all-zero on the response side and marked ``remove_sample`` so
    Polar metrics/artifacts can identify it.  It exists solely to preserve the
    one-to-one batch-size contract required by VERL validation union.
    """

    prompt_ids = _prompt_ids_for_row(prompts, group_index)
    uid = str(getattr(row, "uid", group_index))
    metadata = {
        "placeholder": True,
        "validation_placeholder": True,
        "group_index": int(group_index),
        "sample_uid": uid,
        "source_uid": uid,
        "session_status": "VALIDATION_DROPPED",
    }
    return VerlPolarSample(
        uid=uid,
        group_index=int(group_index),
        trajectory_index=0,
        trace_index=-1,
        prompt_ids=prompt_ids,
        response_ids=[0],
        response_mask=[0],
        rollout_log_probs=[0.0],
        reward=0.0,
        status=VerlPolarStatus.ABORTED,
        prompt=getattr(row, "prompt", ""),
        response="",
        metadata={"polar": metadata},
        remove_sample=True,
    )


def _prompt_ids_for_row(prompts: Any, group_index: int) -> list[int]:
    batch = getattr(prompts, "batch", None)
    if batch is None or "prompts" not in batch:
        return [0]
    try:
        values = batch["prompts"][int(group_index)]
        if hasattr(values, "detach"):
            values = values.detach().cpu().tolist()
        elif hasattr(values, "tolist"):
            values = values.tolist()
        return [int(v) for v in values]
    except Exception:
        return [0]


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


async def _create_native_agent_loop_manager(
    *,
    config: Any,
    worker_group: Any = None,
    rollout_resource_pool: Any = None,
    reward_loop_worker_handles: list[Any] | None = None,
) -> Any:
    from verl.experimental.agent_loop import AgentLoopManager as NativeAgentLoopManager

    return await NativeAgentLoopManager.create(
        config=config,
        worker_group=worker_group,
        rollout_resource_pool=rollout_resource_pool,
        reward_loop_worker_handles=reward_loop_worker_handles,
    )


def _server_address_to_openai_base_url(address: Any) -> str:
    text = str(address).strip()
    if not text:
        raise ValueError("empty VERL rollout server address")
    if text.startswith("http://") or text.startswith("https://"):
        base = text.rstrip("/")
    else:
        base = f"http://{text}".rstrip("/")
    # Polar's SGLangClient posts to absolute OpenAI-compatible paths such as
    # /v1/chat/completions.  Store only the upstream origin here; otherwise a
    # configured base ending in /v1 can become /v1/v1/chat/completions.
    return base[:-3] if base.endswith("/v1") else base
