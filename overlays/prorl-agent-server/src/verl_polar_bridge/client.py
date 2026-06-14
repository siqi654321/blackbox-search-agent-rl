"""Small HTTP client for Polar rollout server and gateway APIs."""

from __future__ import annotations

import time
from typing import Any

import httpx

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polar.rollout.models import TaskRequest, TaskResult, TaskStatus


class PolarClientError(RuntimeError):
    """Raised when Polar rollout server APIs fail."""


class PolarRolloutClient:
    """Synchronous client used by the VERL bridge scheduler.

    Polar's rollout server uses the async task API:
    ``POST /rollout/task/submit`` followed by polling
    ``GET /rollout/task/{task_id}`` until a terminal status is reached.
    Keep the endpoint names centralized here so scheduler/manager code does
    not depend on REST details.
    """

    def __init__(self, base_url: str, *, timeout: float | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def submit_task(self, payload: dict[str, Any] | "TaskRequest") -> "TaskResult | TaskStatus | dict[str, Any]":
        data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
        response = httpx.post(
            f"{self.base_url}/rollout/task/submit",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return _parse_task_response(body)

    def get_task_status(self, task_id: str) -> "TaskStatus":
        response = httpx.get(f"{self.base_url}/rollout/task/{task_id}", timeout=self.timeout)
        response.raise_for_status()
        return _parse_task_status(response.json())

    def wait_task(
        self,
        task_id: str,
        *,
        poll_interval_seconds: float = 5.0,
        timeout_seconds: float | None = None,
    ) -> "TaskStatus":
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            status = self.get_task_status(task_id)
            if str(status.status) in {"completed", "failed"}:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Polar task {task_id}")
            time.sleep(poll_interval_seconds)


class PolarGatewayClient:
    """Small client for Polar gateway SGLang admission/generation control."""

    def __init__(self, gateway_url: str, *, timeout: float | None = None) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout

    def pause_generation(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        params = {} if timeout_seconds is None else {"timeout_seconds": timeout_seconds}
        request_timeout = self.timeout
        if timeout_seconds is not None:
            request_timeout = max(float(timeout_seconds) + 5.0, float(self.timeout or 0), 10.0)
        response = httpx.post(
            f"{self.gateway_url}/admin/sglang/pause",
            params=params,
            timeout=request_timeout,
        )
        response.raise_for_status()
        return _safe_json(response)

    def resume_generation(self) -> dict[str, Any]:
        response = httpx.post(f"{self.gateway_url}/admin/sglang/resume", timeout=self.timeout)
        response.raise_for_status()
        return _safe_json(response)

    def update_upstream(self, base_url: str, *, timeout_seconds: float = 300.0) -> dict[str, Any]:
        response = httpx.post(
            f"{self.gateway_url}/admin/sglang/upstream",
            json={"base_url": base_url, "timeout_seconds": timeout_seconds},
            timeout=max(float(timeout_seconds) + 5.0, float(self.timeout or 0), 10.0),
        )
        response.raise_for_status()
        return _safe_json(response)


def _parse_task_response(body: dict[str, Any]) -> "TaskResult | TaskStatus | dict[str, Any]":
    if "results" in body and "total_sessions" in body:
        return _parse_task_status(body)
    if "results" in body:
        return _parse_task_result(body)
    return body


def _parse_task_status(body: dict[str, Any]) -> "TaskStatus":
    from polar.rollout.models import TaskStatus

    return TaskStatus.model_validate(body)


def _parse_task_result(body: dict[str, Any]) -> "TaskResult":
    from polar.rollout.models import TaskResult

    return TaskResult.model_validate(body)


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {"value": body}
