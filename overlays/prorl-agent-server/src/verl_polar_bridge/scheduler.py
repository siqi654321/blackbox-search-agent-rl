"""Polar task schedulers for VERL bridge development."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any

import httpx
from fastapi import FastAPI, Request

from verl_polar_bridge.adapter import VerlPolarSample, task_result_to_verl_samples
from verl_polar_bridge.client import PolarRolloutClient
from verl_polar_bridge.config import PolarVerlConfig, render_instruction, render_task_payload
from verl_polar_bridge.debug_utils import debug_print, env_flag, env_int, messages_summary, stable_hash, text_preview
from verl_polar_bridge._messages import prompt_to_instruction_text

logger = logging.getLogger(__name__)

_CALLBACK_FALLBACK_POLL_SECONDS = 60.0


@dataclass
class SchedulerStats:
    submitted_tasks: int = 0
    accepted_tasks: int = 0
    dropped_tasks: int = 0
    accepted_samples: int = 0
    accepted_trainable_samples: int = 0
    accepted_trainable_tokens: int = 0
    dropped_reasons: dict[str, int] = field(default_factory=dict)
    logprob_errors: int = 0
    truncated_samples: int = 0
    truncated_tokens: int = 0
    stale_groups: int = 0
    output_queue_full_waits: int = 0
    admission_pauses: int = 0
    deferred_queue_dequeues: int = 0
    dropped_events: list[dict[str, Any]] = field(default_factory=list)

    def drop(self, reason: str, **metadata: Any) -> None:
        self.dropped_tasks += 1
        self.dropped_reasons[reason] = self.dropped_reasons.get(reason, 0) + 1
        event = {"reason": reason}
        event.update({key: value for key, value in metadata.items() if value is not None})
        self.dropped_events.append(event)


@dataclass
class PolarPendingGroup:
    """Prompt group admitted into the async Polar scheduler."""

    group_id: int
    sample: Any
    group_index: int
    rollout_id: int
    num_rollouts: int
    global_steps: int
    uid: str | None
    policy_version: int
    submitted_rollout_id: int
    session_cost: int
    max_tokens: int | None = None


@dataclass
class PolarCompletedGroup:
    """Completed Polar task retained until the trainer drains it."""

    group_id: int
    samples: list[VerlPolarSample]
    task_id: str
    submitted_rollout_id: int
    policy_version: int
    session_count: int


class PolarScheduler:
    """Submit one VERL prompt group to Polar and convert completed traces."""

    def __init__(
        self,
        *,
        trainer_config: Any,
        polar_config: PolarVerlConfig,
        client: PolarRolloutClient | None = None,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        self.trainer_config = trainer_config
        self.polar_config = polar_config
        self.client = client or PolarRolloutClient(
            polar_config.rollout_server_url,
            timeout=polar_config.request_timeout,
        )
        self.poll_interval_seconds = poll_interval_seconds
        self.stats = SchedulerStats()

    def rollout_group(
        self,
        sample: Any,
        *,
        group_index: int,
        rollout_id: int,
        num_rollouts: int,
        global_steps: int,
        uid: str | None = None,
        max_tokens: int | None = None,
    ) -> list[VerlPolarSample]:
        raw_prompt = _sample_get(sample, "prompt", sample)
        if _env_flag("POLAR_SEARCH_RAW_CHAT_INSTRUCTION", default=True) and _is_search_harness(self.polar_config):
            prompt_text = _prompt_to_raw_chat_instruction(raw_prompt)
        else:
            prompt_text = prompt_to_instruction_text(raw_prompt)
        instruction = render_instruction(
            trainer_config=self.trainer_config,
            config=self.polar_config,
            sample=sample,
            prompt_text=prompt_text,
            rollout_id=rollout_id,
            task_position=group_index,
            num_rollouts=num_rollouts,
            global_steps=global_steps,
            uid=uid,
        )
        payload = render_task_payload(
            trainer_config=self.trainer_config,
            config=self.polar_config,
            sample=sample,
            instruction=instruction,
            rollout_id=rollout_id,
            task_position=group_index,
            num_rollouts=num_rollouts,
            global_steps=global_steps,
            uid=uid,
        )
        agent = payload.get("agent") if isinstance(payload, dict) else None
        agent = agent if isinstance(agent, dict) else {}
        _debug_rollout_payload(
            "sync_build_payload",
            {
                "group_index": group_index,
                "rollout_id": rollout_id,
                "num_rollouts": num_rollouts,
                "global_steps": global_steps,
                "uid": uid,
                "is_search_harness": _is_search_harness(self.polar_config),
                "agent_import_path": agent.get("import_path"),
                "agent_settings": agent.get("settings"),
                "agent_env_sampling": {key: (agent.get("env") or {}).get(key) for key in ("SEARCH_TEMPERATURE", "SEARCH_TOP_P", "SEARCH_DO_SAMPLE")} if isinstance(agent.get("env"), dict) else None,
                "sample": _sample_debug_summary(sample, raw_prompt, prompt_text, instruction),
            },
        )
        self.stats.submitted_tasks += 1
        submitted = self.client.submit_task(payload)
        task_result = self._ensure_terminal_result(payload["task_id"], submitted)
        samples = _convert_and_record(
            task_result=task_result,
            group_index=group_index,
            uid=uid,
            reward_key=self.polar_config.reward_key,
            max_tokens=max_tokens,
            min_complete_accept_fraction=self.polar_config.min_complete_accept_fraction,
            reject_logprob_error=self.polar_config.acceptance_reject_logprob_error,
            overflow_policy=self.polar_config.overflow_policy,
            stitch_traces=self.polar_config.stitch_traces,
            stats=self.stats,
        )
        _debug_rollout_payload(
            "sync_task_result_converted",
            {
                "group_index": group_index,
                "uid": uid,
                "task_result": _task_result_debug_summary(task_result),
                "sample_count": len(samples),
                "sample_prompt_lens": [len(sample.prompt_ids) for sample in samples],
                "sample_response_lens": [len(sample.response_ids) for sample in samples],
                "sample_loss_tokens": [sum(int(v) for v in sample.response_mask) for sample in samples],
                "sample_rewards": [float(sample.reward) for sample in samples],
            },
        )
        return samples

    def _ensure_terminal_result(self, task_id: str, submitted: Any) -> Any:
        if _is_task_result(submitted):
            return submitted
        if _is_task_status(submitted):
            status = submitted if submitted.status in {"completed", "failed"} else self.client.wait_task(
                task_id,
                poll_interval_seconds=self.poll_interval_seconds,
                timeout_seconds=self.polar_config.request_timeout,
            )
            return _make_task_result(status)
        status = self.client.wait_task(
            task_id,
            poll_interval_seconds=self.poll_interval_seconds,
            timeout_seconds=self.polar_config.request_timeout,
        )
        return _make_task_result(status)

    def _accept_task(self, task_result: Any) -> str | None:
        return _task_rejection_reason(task_result, self.polar_config.min_complete_accept_fraction)


class PolarCallbackWaiter:
    """Async callback-first waiter with polling fallback for Polar tasks.

    This mirrors the critical VERL+Polar behavior: register the task event
    before submit, inject ``callback_url`` into the payload, accept trainer-side
    ``TaskResult`` callbacks at ``/callback/task_result``, and defensively poll
    ``/rollout/task/{task_id}`` when callbacks are delayed or lost.
    """

    def __init__(
        self,
        *,
        polar_config: PolarVerlConfig,
        fallback_poll_seconds: float = _CALLBACK_FALLBACK_POLL_SECONDS,
    ) -> None:
        self.polar_config = polar_config
        self.fallback_poll_seconds = fallback_poll_seconds
        self.callback_url: str | None = None
        self._task_events: dict[str, asyncio.Event] = {}
        self._task_results: dict[str, Any] = {}
        self._server: Any = None
        self._server_task: asyncio.Task[Any] | None = None

    async def __aenter__(self) -> "PolarCallbackWaiter":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()

    async def start(self) -> None:
        if self._server is not None:
            return
        import uvicorn

        app = FastAPI()

        @app.post("/callback/task_result")
        async def on_task_result(request: Request) -> dict[str, Any]:
            payload = await request.json()
            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            if not task_id:
                return {"ok": False, "reason": "missing task_id"}
            try:
                result = _parse_task_result_payload(payload)
            except Exception:
                logger.exception("Invalid callback payload for task %s", task_id)
                return {"ok": False, "reason": "invalid payload"}
            self._task_results[task_id] = result
            event = self._task_events.get(task_id)
            if event is not None:
                event.set()
            return {"ok": True}

        config = uvicorn.Config(
            app=app,
            host=self.polar_config.callback_host,
            port=self.polar_config.callback_port,
            log_level="warning",
            lifespan="on",
        )
        server = uvicorn.Server(config)
        self._server = server
        self._server_task = asyncio.create_task(server.serve(), name="verl-polar-callback-listener")
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        self.callback_url = f"http://{self.polar_config.callback_host}:{port}/callback/task_result"

    async def stop(self) -> None:
        server = self._server
        task = self._server_task
        self._server = None
        self._server_task = None
        self.callback_url = None
        self._task_events.clear()
        self._task_results.clear()
        if server is None:
            return
        server.should_exit = True
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()

    async def submit_with_callback(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> Any:
        if self.callback_url is None:
            await self.start()
        task_id = str(payload["task_id"])
        event = asyncio.Event()
        self._task_events[task_id] = event
        payload["callback_url"] = self.callback_url
        try:
            response = await client.post(
                f"{self.polar_config.rollout_server_url}/rollout/task/submit",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return await self.await_task_result(client, task_id, event)
        finally:
            self._task_events.pop(task_id, None)
            self._task_results.pop(task_id, None)

    async def await_task_result(self, client: httpx.AsyncClient, task_id: str, event: asyncio.Event) -> Any:
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=self.fallback_poll_seconds)
            except asyncio.TimeoutError:
                status = await _poll_task_status(client, self.polar_config.rollout_server_url, task_id)
                if str(status.status) in {"completed", "failed"}:
                    return _make_task_result(status)
                continue
            result = self._task_results.get(task_id)
            if result is not None:
                return result
            status = await _poll_task_status(client, self.polar_config.rollout_server_url, task_id)
            return _make_task_result(status)


class AsyncPolarScheduler:
    """Callback-capable async scheduler with prompt-grounded queue semantics.

    The class can still be used directly via ``rollout_group`` for tests and
    simple scripts, but it also exposes queue-oriented APIs for the VERL manager:
    admit prompt groups with ``submit_group``/``run_until_capacity``, collect
    finished groups with ``drain_completed``, and drop stale groups according to
    ``max_off_policy_steps``.
    """

    def __init__(
        self,
        *,
        trainer_config: Any,
        polar_config: PolarVerlConfig,
        fallback_poll_seconds: float = _CALLBACK_FALLBACK_POLL_SECONDS,
    ) -> None:
        self.trainer_config = trainer_config
        self.polar_config = polar_config
        self.callback_waiter = PolarCallbackWaiter(
            polar_config=polar_config,
            fallback_poll_seconds=fallback_poll_seconds,
        )
        self.stats = SchedulerStats()
        self.output_queue: asyncio.Queue[PolarCompletedGroup] = asyncio.Queue(
            maxsize=max(32, polar_config.max_concurrency * 2)
        )
        self.deferred_queue: deque[PolarPendingGroup] = deque()
        self.completed_buffer: deque[PolarCompletedGroup] = deque()
        self.active: dict[asyncio.Task[PolarCompletedGroup | None], PolarPendingGroup] = {}
        self._group_counter = 0
        self._current_rollout_id = 0
        self._policy_version = 0
        self._admission_paused = False
        self._active_sessions = 0

    async def __aenter__(self) -> "AsyncPolarScheduler":
        await self.callback_waiter.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.callback_waiter.stop()

    async def rollout_group(
        self,
        sample: Any,
        *,
        group_index: int,
        rollout_id: int,
        num_rollouts: int,
        global_steps: int,
        uid: str | None = None,
        max_tokens: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> list[VerlPolarSample]:
        payload = _build_rollout_payload(
            trainer_config=self.trainer_config,
            polar_config=self.polar_config,
            sample=sample,
            group_index=group_index,
            rollout_id=rollout_id,
            num_rollouts=num_rollouts,
            global_steps=global_steps,
            uid=uid,
        )
        self.stats.submitted_tasks += 1
        owns_client = client is None
        timeout = self.polar_config.request_timeout
        if client is None:
            client = httpx.AsyncClient(timeout=timeout)
        try:
            task_result = await self.callback_waiter.submit_with_callback(client, payload)
            samples = _convert_and_record(
                task_result=task_result,
                group_index=group_index,
                uid=uid,
                reward_key=self.polar_config.reward_key,
                max_tokens=max_tokens,
                min_complete_accept_fraction=self.polar_config.min_complete_accept_fraction,
                reject_logprob_error=self.polar_config.acceptance_reject_logprob_error,
                overflow_policy=self.polar_config.overflow_policy,
                stitch_traces=self.polar_config.stitch_traces,
                stats=self.stats,
            )
            return samples
        finally:
            if owns_client:
                await client.aclose()

    def set_rollout_context(self, rollout_id: int) -> None:
        self._current_rollout_id = max(self._current_rollout_id, int(rollout_id))

    def update_policy_version(self, policy_version: int) -> None:
        self._policy_version = max(self._policy_version, int(policy_version))

    def pause_admission(self) -> None:
        if not self._admission_paused:
            self.stats.admission_pauses += 1
        self._admission_paused = True

    def resume_admission(self) -> None:
        self._admission_paused = False

    def submit_group(
        self,
        sample: Any,
        *,
        group_index: int,
        rollout_id: int,
        num_rollouts: int,
        global_steps: int,
        uid: str | None = None,
        max_tokens: int | None = None,
        session_cost: int | None = None,
    ) -> PolarPendingGroup:
        pending = PolarPendingGroup(
            group_id=self._group_counter,
            sample=sample,
            group_index=group_index,
            rollout_id=rollout_id,
            num_rollouts=num_rollouts,
            global_steps=global_steps,
            uid=uid,
            policy_version=self._policy_version,
            submitted_rollout_id=self._current_rollout_id,
            session_cost=int(session_cost or num_rollouts or 1),
            max_tokens=max_tokens,
        )
        self._group_counter += 1
        self.deferred_queue.append(pending)
        return pending

    async def run_until_capacity(self, client: httpx.AsyncClient) -> None:
        """Admit deferred groups until backpressure limits are reached."""
        self._collect_finished_tasks()
        while self._can_admit_next():
            pending = self.deferred_queue.popleft()
            self.stats.deferred_queue_dequeues += 1
            if pending.session_cost > self.polar_config.max_session_concurrency:
                self.stats.drop(
                    "session_cost_exceeds_limit",
                    group_id=pending.group_id,
                    group_index=pending.group_index,
                    uid=pending.uid,
                    session_cost=pending.session_cost,
                    max_session_concurrency=self.polar_config.max_session_concurrency,
                )
                continue
            if self._active_sessions + pending.session_cost > self.polar_config.max_session_concurrency:
                self.deferred_queue.appendleft(pending)
                self.stats.deferred_queue_dequeues -= 1
                break
            task = asyncio.create_task(
                self._submit_pending(client, pending),
                name=f"verl-polar-rollout-{pending.group_id}",
            )
            self.active[task] = pending
            self._active_sessions += pending.session_cost

    async def wait_for_next(self, timeout: float | None = None) -> PolarCompletedGroup | None:
        if self.output_queue.qsize() == 0 and self.active:
            done, _ = await asyncio.wait(self.active.keys(), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pending = self.active.pop(task)
                self._active_sessions -= pending.session_cost
                try:
                    completed = task.result()
                except Exception as exc:
                    logger.warning(
                        "Polar active task failed for group %s while waiting for next result: %s",
                        pending.group_id,
                        exc,
                    )
                    self.stats.drop(
                        "active_task_exception",
                        group_id=pending.group_id,
                        uid=pending.uid,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    continue
                if completed is not None:
                    await self._emit_completed(completed)
        try:
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def drain_completed(self, *, max_groups: int, rollout_id: int) -> list[PolarCompletedGroup]:
        while True:
            try:
                self.completed_buffer.append(self.output_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        accepted: list[PolarCompletedGroup] = []
        while self.completed_buffer and len(accepted) < max_groups:
            completed = self.completed_buffer.popleft()
            staleness = max(0, int(rollout_id) - completed.policy_version)
            if staleness > self.polar_config.max_off_policy_steps:
                self.stats.stale_groups += 1
                self.stats.drop(
                    "stale_group",
                    group_id=completed.group_id,
                    task_id=completed.task_id,
                    submitted_rollout_id=completed.submitted_rollout_id,
                    policy_version=completed.policy_version,
                    accepted_rollout_id=rollout_id,
                    staleness=staleness,
                )
                continue
            annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=rollout_id,
                staleness=staleness,
                policy_version=completed.policy_version,
                scheduler_group_id=completed.group_id,
            )
            accepted.append(completed)
        return accepted

    def snapshot_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "polar/submitted_tasks": float(self.stats.submitted_tasks),
            "polar/accepted_tasks": float(self.stats.accepted_tasks),
            "polar/dropped_tasks": float(self.stats.dropped_tasks),
            "polar/accepted_samples": float(self.stats.accepted_samples),
            "polar/accepted_trainable_samples": float(self.stats.accepted_trainable_samples),
            "polar/accepted_trainable_tokens": float(self.stats.accepted_trainable_tokens),
            "polar/logprob_errors": float(self.stats.logprob_errors),
            "polar/truncated_samples": float(self.stats.truncated_samples),
            "polar/truncated_tokens": float(self.stats.truncated_tokens),
            "polar/stale_groups": float(self.stats.stale_groups),
            "polar/output_queue_full_waits": float(self.stats.output_queue_full_waits),
            "polar/deferred_queue_dequeues": float(self.stats.deferred_queue_dequeues),
            "polar/completed_groups": float(self.stats.accepted_tasks),
            "polar/dropped_groups": float(self.stats.dropped_tasks),
            "polar/scheduler/active_groups": float(len(self.active)),
            "polar/scheduler/active_sessions": float(self._active_sessions),
            "polar/scheduler/completed_buffer": float(len(self.completed_buffer)),
            "polar/scheduler/output_queue": float(self.output_queue.qsize()),
            "polar/scheduler/deferred_queue": float(len(self.deferred_queue)),
            "polar/scheduler/policy_version": float(self._policy_version),
            "polar/scheduler/admission_paused": float(self._admission_paused),
            "polar/scheduler/admission_pauses": float(self.stats.admission_pauses),
        }
        for reason, count in self.stats.dropped_reasons.items():
            metrics[f"polar/dropped/{reason}"] = float(count)
            metrics[f"polar/dropped_{reason}_groups"] = float(count)
        return metrics

    async def _submit_pending(self, client: httpx.AsyncClient, pending: PolarPendingGroup) -> PolarCompletedGroup | None:
        payload = _build_rollout_payload(
            trainer_config=self.trainer_config,
            polar_config=self.polar_config,
            sample=pending.sample,
            group_index=pending.group_index,
            rollout_id=pending.rollout_id,
            num_rollouts=pending.num_rollouts,
            global_steps=pending.global_steps,
            uid=pending.uid,
        )
        _attach_scheduler_metadata(
            payload,
            group_id=pending.group_id,
            policy_version=pending.policy_version,
            rollout_step=pending.submitted_rollout_id,
        )
        self.stats.submitted_tasks += 1
        try:
            task_result = await self.callback_waiter.submit_with_callback(client, payload)
        except Exception as exc:
            logger.warning(
                "Polar task submit/wait failed for group %s task %s: %s",
                pending.group_id,
                payload.get("task_id"),
                exc,
            )
            self.stats.drop(
                "submit_exception",
                task_id=str(payload.get("task_id")),
                group_id=pending.group_id,
                uid=pending.uid,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None
        samples = _convert_and_record(
            task_result=task_result,
            group_index=pending.group_index,
            uid=pending.uid,
            reward_key=self.polar_config.reward_key,
            max_tokens=pending.max_tokens,
            min_complete_accept_fraction=self.polar_config.min_complete_accept_fraction,
            reject_logprob_error=self.polar_config.acceptance_reject_logprob_error,
            overflow_policy=self.polar_config.overflow_policy,
            stitch_traces=self.polar_config.stitch_traces,
            stats=self.stats,
        )
        _debug_rollout_payload(
            "task_result_converted",
            {
                "group_id": pending.group_id,
                "group_index": pending.group_index,
                "uid": pending.uid,
                "task_result": _task_result_debug_summary(task_result),
                "sample_count": len(samples),
                "sample_prompt_lens": [len(sample.prompt_ids) for sample in samples],
                "sample_response_lens": [len(sample.response_ids) for sample in samples],
                "sample_loss_tokens": [sum(int(v) for v in sample.response_mask) for sample in samples],
                "sample_rewards": [float(sample.reward) for sample in samples],
                "sample_status": [str(sample.status) for sample in samples],
            },
        )
        if not samples:
            return None
        return PolarCompletedGroup(
            group_id=pending.group_id,
            samples=samples,
            task_id=str(getattr(task_result, "task_id", payload["task_id"])),
            submitted_rollout_id=pending.submitted_rollout_id,
            policy_version=pending.policy_version,
            session_count=len(getattr(task_result, "results", []) or []),
        )

    async def _emit_completed(self, completed: PolarCompletedGroup) -> None:
        while True:
            try:
                self.output_queue.put_nowait(completed)
                return
            except asyncio.QueueFull:
                self.stats.output_queue_full_waits += 1
                await asyncio.sleep(0.1)

    def _collect_finished_tasks(self) -> None:
        for task in [task for task in self.active if task.done()]:
            pending = self.active.pop(task)
            self._active_sessions -= pending.session_cost
            try:
                completed = task.result()
            except Exception as exc:
                logger.warning(
                    "Polar active task failed for group %s: %s",
                    pending.group_id,
                    exc,
                )
                self.stats.drop(
                    "active_task_exception",
                    group_id=pending.group_id,
                    uid=pending.uid,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue
            if completed is not None:
                self.completed_buffer.append(completed)

    def _can_admit_next(self) -> bool:
        if self._admission_paused or not self.deferred_queue:
            return False
        if len(self.active) >= self.polar_config.max_concurrency:
            return False
        if self._active_sessions >= self.polar_config.max_session_concurrency:
            return False
        owned_groups = len(self.active) + self.output_queue.qsize() + len(self.completed_buffer)
        batch_size = max(1, self.polar_config.max_concurrency // max(1, self.polar_config.max_async_level))
        return owned_groups < batch_size * self.polar_config.max_async_level


def _search_rollout_debug_enabled() -> bool:
    return env_flag("POLAR_SEARCH_ROLLOUT_DEBUG", default=False)


def _search_rollout_debug_limit() -> int:
    return env_int("POLAR_SEARCH_ROLLOUT_DEBUG_LIMIT", 16)


def _debug_rollout_payload(event: str, payload: dict[str, Any]) -> None:
    if _search_rollout_debug_enabled():
        debug_print("POLAR_SEARCH_ROLLOUT_DEBUG", {"event": event, **payload}, stream="stderr")


def _sample_debug_summary(sample: Any, raw_prompt: Any, prompt_text: str, instruction: str) -> dict[str, Any]:
    return {
        "sample_type": type(sample).__name__,
        "uid": str(_sample_get(sample, "uid", "")),
        "index": _sample_get(sample, "index", None),
        "raw_prompt_type": type(raw_prompt).__name__,
        "raw_prompt_summary": messages_summary(raw_prompt) if isinstance(raw_prompt, list) else text_preview(raw_prompt),
        "prompt_text": text_preview(prompt_text),
        "instruction": text_preview(instruction),
        "prompt_text_eq_instruction": prompt_text == instruction,
        "prompt_text_hash": stable_hash(prompt_text),
        "instruction_hash": stable_hash(instruction),
    }


def _task_result_debug_summary(task_result: Any) -> dict[str, Any]:
    results = list(getattr(task_result, "results", []) or [])
    out: dict[str, Any] = {
        "task_id": getattr(task_result, "task_id", None),
        "status": str(getattr(task_result, "status", "")),
        "result_count": len(results),
    }
    session_summaries = []
    for result in results[:_search_rollout_debug_limit()]:
        traj = getattr(result, "trajectory", None)
        traces = list(getattr(traj, "traces", []) or []) if traj is not None else []
        session_summaries.append({
            "session_id": getattr(result, "session_id", None),
            "session_status": str(getattr(result, "status", "")),
            "trajectory_status": getattr(traj, "status", None),
            "trajectory_error": getattr(traj, "error", None),
            "trace_count": len(traces),
            "trace_prompt_lens": [len(getattr(t, "prompt_ids", []) or []) for t in traces],
            "trace_response_lens": [len(getattr(t, "response_ids", []) or []) for t in traces],
            "trace_loss_tokens": [sum(int(v) for v in (getattr(t, "loss_mask", []) or [])) for t in traces],
            "trajectory_metadata": getattr(traj, "metadata", {}) if traj is not None else {},
        })
    out["sessions"] = session_summaries
    return out


def _sample_get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _is_task_result(value: Any) -> bool:
    return all(hasattr(value, key) for key in ("task_id", "status", "results", "result_paths")) and not hasattr(
        value, "total_sessions"
    )


def _is_task_status(value: Any) -> bool:
    return all(hasattr(value, key) for key in ("task_id", "status", "results", "total_sessions"))


def _make_task_result(status: Any) -> Any:
    try:
        from polar.rollout.models import TaskResult

        result = TaskResult(
            task_id=status.task_id,
            status=status.status,
            results=status.results,
            result_paths=getattr(status, "result_paths", []),
        )
        total_sessions = getattr(status, "total_sessions", None)
        if total_sessions is not None:
            try:
                # Pydantic models allow extra attributes when the backing
                # object does; tests/fallback SimpleNamespace rely on the same
                # field so result-count acceptance can validate TaskStatus
                # responses too.
                setattr(result, "total_sessions", int(total_sessions))
            except Exception:
                pass
        return result
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(
            task_id=status.task_id,
            status=status.status,
            results=status.results,
            result_paths=getattr(status, "result_paths", []),
            total_sessions=getattr(status, "total_sessions", None),
        )


def _build_rollout_payload(
    *,
    trainer_config: Any,
    polar_config: PolarVerlConfig,
    sample: Any,
    group_index: int,
    rollout_id: int,
    num_rollouts: int,
    global_steps: int,
    uid: str | None,
) -> dict[str, Any]:
    raw_prompt = _sample_get(sample, "prompt", sample)
    if _env_flag("POLAR_SEARCH_RAW_CHAT_INSTRUCTION", default=True) and _is_search_harness(polar_config):
        prompt_text = _prompt_to_raw_chat_instruction(raw_prompt)
    else:
        prompt_text = prompt_to_instruction_text(raw_prompt)
    instruction = render_instruction(
        trainer_config=trainer_config,
        config=polar_config,
        sample=sample,
        prompt_text=prompt_text,
        rollout_id=rollout_id,
        task_position=group_index,
        num_rollouts=num_rollouts,
        global_steps=global_steps,
        uid=uid,
    )
    payload = render_task_payload(
        trainer_config=trainer_config,
        config=polar_config,
        sample=sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=group_index,
        num_rollouts=num_rollouts,
        global_steps=global_steps,
        uid=uid,
    )
    agent = payload.get("agent") if isinstance(payload, dict) else None
    agent = agent if isinstance(agent, dict) else {}
    _debug_rollout_payload(
        "build_payload",
        {
            "group_index": group_index,
            "rollout_id": rollout_id,
            "num_rollouts": num_rollouts,
            "global_steps": global_steps,
            "uid": uid,
            "is_search_harness": _is_search_harness(polar_config),
            "raw_chat_instruction_enabled": _env_flag("POLAR_SEARCH_RAW_CHAT_INSTRUCTION", default=True),
            "agent_import_path": agent.get("import_path"),
            "agent_settings": agent.get("settings"),
            "agent_env_sampling": {key: (agent.get("env") or {}).get(key) for key in ("SEARCH_TEMPERATURE", "SEARCH_TOP_P", "SEARCH_DO_SAMPLE")} if isinstance(agent.get("env"), dict) else None,
            "sample": _sample_debug_summary(sample, raw_prompt, prompt_text, instruction),
        },
    )
    return payload


def _attach_scheduler_metadata(
    payload: dict[str, Any], *, group_id: int, policy_version: int, rollout_step: int) -> None:
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        payload["metadata"] = metadata = {}
    metadata.update(
        {
            "polar_scheduler_group_id": int(group_id),
            "polar_policy_version": int(policy_version),
            "rollout_step": int(rollout_step),
        }
    )
    session_metadata = metadata.setdefault("session_metadata", {})
    if isinstance(session_metadata, dict):
        session_metadata.update(
            {
                "polar_scheduler_group_id": int(group_id),
                "polar_policy_version": int(policy_version),
                "rollout_step": int(rollout_step),
            }
        )


def annotate_accepted_samples(
    samples: list[VerlPolarSample],
    *,
    accepted_rollout_id: int,
    staleness: int,
    policy_version: int,
    scheduler_group_id: int,
) -> None:
    for sample in samples:
        polar = sample.metadata.setdefault("polar", {})
        if not isinstance(polar, dict):
            sample.metadata["polar"] = polar = {}
        polar.update(
            {
                "accepted_rollout_id": int(accepted_rollout_id),
                "polar_policy_staleness": int(staleness),
                "polar_policy_version": int(policy_version),
                "polar_scheduler_group_id": int(scheduler_group_id),
            }
        )


def _convert_and_record(
    *,
    task_result: Any,
    group_index: int,
    uid: str | None,
    reward_key: str,
    max_tokens: int | None,
    min_complete_accept_fraction: float,
    reject_logprob_error: bool,
    overflow_policy: str,
    stitch_traces: bool,
    stats: SchedulerStats,
) -> list[VerlPolarSample]:
    task_id = str(getattr(task_result, "task_id", ""))
    rejection = _task_rejection_reason(task_result, min_complete_accept_fraction)
    if rejection is not None:
        stats.drop(
            rejection,
            task_id=task_id,
            group_index=group_index,
            uid=uid,
            status=getattr(task_result, "status", None),
            result_count=len(getattr(task_result, "results", []) or []),
            expected_num_samples=_expected_num_samples(task_result),
            min_complete_accept_fraction=min_complete_accept_fraction,
        )
        return []
    try:
        samples = task_result_to_verl_samples(
            task_result,
            group_index=group_index,
            uid=uid,
            reward_key=reward_key,
            max_tokens=max_tokens,
            overflow_policy=overflow_policy,
            stitch_traces=stitch_traces,
        )
    except Exception as exc:
        if exc.__class__.__name__ == "RolloutLogprobError":
            stats.logprob_errors += 1
            if not reject_logprob_error:
                logger.warning(
                    "Polar rollout logprob alignment failed for task %s; dropping group anyway to avoid "
                    "unsafe token/logprob reconstruction. Set polar.acceptance.reject_logprob_error=true "
                    "for fail-closed accounting.",
                    getattr(task_result, "task_id", "<unknown>"),
            )
            stats.drop("logprob_error", task_id=task_id, group_index=group_index, uid=uid, error=str(exc))
            return []
        raise
    if not samples:
        stats.drop("no_samples", task_id=task_id, group_index=group_index, uid=uid)
        return []
    trainable_tokens = count_trainable_tokens(samples)
    if trainable_tokens <= 0:
        stats.drop(
            "no_trainable_tokens",
            task_id=task_id,
            group_index=group_index,
            uid=uid,
            sample_count=len(samples),
            sample_summaries=_dropped_sample_summaries(samples),
        )
        return []
    trainable_fraction_rejection = _trainable_complete_fraction_rejection_reason(
        task_result,
        samples,
        min_complete_accept_fraction,
    )
    if trainable_fraction_rejection is not None:
        stats.drop(
            trainable_fraction_rejection,
            task_id=task_id,
            group_index=group_index,
            uid=uid,
            status=getattr(task_result, "status", None),
            result_count=len(getattr(task_result, "results", []) or []),
            expected_num_samples=_expected_num_samples(task_result),
            min_complete_accept_fraction=min_complete_accept_fraction,
        )
        return []
    _record_overflow_stats(samples, max_tokens=max_tokens, overflow_policy=overflow_policy, stats=stats)
    stats.accepted_tasks += 1
    stats.accepted_samples += len(samples)
    stats.accepted_trainable_samples += sum(1 for sample in samples if sample.has_trainable_tokens)
    stats.accepted_trainable_tokens += trainable_tokens
    return samples


def _record_overflow_stats(
    samples: list[VerlPolarSample],
    *,
    max_tokens: int | None,
    overflow_policy: str,
    stats: SchedulerStats,
) -> None:
    if str(overflow_policy or "").strip().lower() != "verl_truncate" or max_tokens is None:
        return
    for sample in samples:
        polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
        if not isinstance(polar, dict):
            continue
        raw_total_len = int(polar.get("raw_total_len", len(sample.prompt_ids) + len(sample.response_ids)) or 0)
        if raw_total_len <= int(max_tokens):
            continue
        stats.truncated_samples += 1
        stats.truncated_tokens += raw_total_len - int(max_tokens)


def _task_rejection_reason(task_result: Any, min_complete_accept_fraction: float) -> str | None:
    if task_result.status != "completed":
        return "task_not_completed"
    if not task_result.results:
        return "empty_results"
    expected = _expected_num_samples(task_result)
    if expected is not None and len(task_result.results) != expected:
        return "result_count_mismatch"
    completed = sum(1 for result in task_result.results if str(result.status) == "COMPLETED")
    fraction = completed / max(len(task_result.results), 1)
    if fraction < min_complete_accept_fraction:
        return "complete_fraction_below_threshold"
    return None


def _trainable_complete_fraction_rejection_reason(
    task_result: Any,
    samples: list[VerlPolarSample],
    min_complete_accept_fraction: float,
) -> str | None:
    """Reject partial batches where completed sessions yielded no trainable row.

    Task-level status can be ``completed`` even when some individual sessions
    are TIMEOUT/ERROR or when a nominally completed session only produced
    placeholder/all-zero-mask samples.  VERL/GRPO should not silently train a
    group whose effective trainable completed-session fraction is below the
    configured acceptance threshold.
    """
    results = list(getattr(task_result, "results", []) or [])
    if not results:
        return None
    if min_complete_accept_fraction <= 0:
        return None

    completed_session_ids = {
        str(getattr(result, "session_id", idx))
        for idx, result in enumerate(results)
        if str(getattr(result, "status", "")) == "COMPLETED"
    }
    if not completed_session_ids:
        return "complete_trainable_fraction_below_threshold"

    trainable_completed_session_ids: set[str] = set()
    for sample in samples:
        if not sample.has_trainable_tokens:
            continue
        polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
        if not isinstance(polar, dict):
            continue
        session_id = polar.get("session_id")
        session_status = polar.get("session_status")
        if session_id is None:
            continue
        if str(session_status) == "COMPLETED" and str(session_id) in completed_session_ids:
            trainable_completed_session_ids.add(str(session_id))

    fraction = len(trainable_completed_session_ids) / max(len(results), 1)
    if fraction < min_complete_accept_fraction:
        return "complete_trainable_fraction_below_threshold"
    return None


def _parse_task_result_payload(payload: dict[str, Any]) -> Any:
    try:
        from polar.rollout.models import TaskResult

        return TaskResult.model_validate(payload)
    except (ImportError, SyntaxError, AttributeError, TypeError):
        from types import SimpleNamespace

        if "task_id" not in payload or "status" not in payload or "results" not in payload:
            raise
        return SimpleNamespace(
            task_id=payload["task_id"],
            status=payload["status"],
            results=payload.get("results") or [],
            result_paths=payload.get("result_paths") or [],
        )


async def _poll_task_status(client: httpx.AsyncClient, base_url: str, task_id: str) -> Any:
    response = await client.get(f"{base_url.rstrip('/')}/rollout/task/{task_id}")
    response.raise_for_status()
    payload = response.json()
    try:
        from polar.rollout.models import TaskStatus

        return TaskStatus.model_validate(payload)
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(
            task_id=payload["task_id"],
            status=payload["status"],
            results=payload.get("results") or [],
            result_paths=payload.get("result_paths") or [],
            total_sessions=payload.get("total_sessions", len(payload.get("results") or [])),
        )



def _dropped_sample_summaries(samples: list[VerlPolarSample], *, limit: int = 4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples[: max(0, int(limit))]:
        polar = sample.metadata.get("polar", {}) if isinstance(sample.metadata, dict) else {}
        trace_meta = polar.get("trace_metadata", {}) if isinstance(polar, dict) else {}
        rows.append(
            {
                "uid": sample.uid,
                "group_index": int(sample.group_index),
                "trace_index": int(sample.trace_index),
                "status": str(getattr(sample.status, "value", sample.status)),
                "remove_sample": bool(sample.remove_sample),
                "reward": float(sample.reward),
                "prompt_len": len(sample.prompt_ids),
                "response_len": len(sample.response_ids),
                "trainable_tokens": int(sum(int(value) for value in sample.response_mask)),
                "finish_reason": (polar.get("trace_debug", {}) or {}).get("finish_reason") if isinstance(polar, dict) else None,
                "session_status": polar.get("session_status") if isinstance(polar, dict) else None,
                "trajectory_status": polar.get("trajectory_status") if isinstance(polar, dict) else None,
                "trace_builder": trace_meta.get("builder") if isinstance(trace_meta, dict) else None,
                "completion_count": trace_meta.get("completion_count") if isinstance(trace_meta, dict) else None,
            }
        )
    return rows

def count_trainable_tokens(samples: list[VerlPolarSample]) -> int:
    return sum(sum(int(value) for value in sample.response_mask) for sample in samples if not sample.remove_sample)


def _expected_num_samples(task_result: Any) -> int | None:
    metadata = getattr(task_result, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("num_samples") is not None:
        return int(metadata["num_samples"])
    status_total = getattr(task_result, "total_sessions", None)
    if status_total is not None:
        return int(status_total)
    for result in getattr(task_result, "results", []) or []:
        result_metadata = getattr(result, "metadata", None)
        if isinstance(result_metadata, dict):
            for key in ("task_num_samples", "num_samples", "expected_num_samples"):
                if result_metadata.get(key) is not None:
                    return int(result_metadata[key])
    return None


def _is_search_harness(config: PolarVerlConfig) -> bool:
    task_template = getattr(config, "task_template", {}) or {}
    agent = task_template.get("agent") if isinstance(task_template, dict) else None
    import_path = agent.get("import_path") if isinstance(agent, dict) else ""
    return "SearchR1Harness" in str(import_path)


def _prompt_to_raw_chat_instruction(prompt: Any) -> str:
    if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
        try:
            return json.dumps(prompt, ensure_ascii=False)
        except Exception:
            pass
    return prompt_to_instruction_text(prompt)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
