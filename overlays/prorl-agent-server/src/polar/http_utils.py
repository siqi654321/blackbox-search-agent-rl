"""HTTPX tuning helpers for high-concurrency Polar rollout paths."""

from __future__ import annotations

import os
from typing import Any

import httpx


def polar_http_limits() -> httpx.Limits:
    """Return process-wide HTTP pool limits for Polar service clients.

    The SearchR1 long-run path can issue hundreds of concurrent local HTTP
    requests across trainer -> rollout -> gateway -> bridge.  httpx defaults are
    too small for that workload and can raise PoolTimeout under transient tail
    latency.  Keep this env-configurable so smoke tests and production runs can
    tune independently without code changes.
    """

    return httpx.Limits(
        max_connections=_env_int("POLAR_HTTP_MAX_CONNECTIONS", 1024),
        max_keepalive_connections=_env_int("POLAR_HTTP_MAX_KEEPALIVE_CONNECTIONS", 512),
        keepalive_expiry=_env_float("POLAR_HTTP_KEEPALIVE_EXPIRY", 30.0),
    )


def polar_http_timeout(
    timeout: float | httpx.Timeout | None,
    *,
    connect: float | None = None,
    pool: float | None = None,
) -> httpx.Timeout | None:
    """Return an httpx Timeout with an enlarged configurable pool timeout."""

    if timeout is None:
        return None
    pool_timeout = _env_float("POLAR_HTTP_POOL_TIMEOUT", 300.0) if pool is None else float(pool)
    if isinstance(timeout, httpx.Timeout):
        return httpx.Timeout(
            connect=timeout.connect if connect is None else float(connect),
            read=timeout.read,
            write=timeout.write,
            pool=pool_timeout,
        )
    value = float(timeout)
    return httpx.Timeout(
        value,
        connect=value if connect is None else float(connect),
        pool=pool_timeout,
    )


def polar_async_client(*, timeout: float | httpx.Timeout | None = None, **kwargs: Any) -> httpx.AsyncClient:
    """Construct an AsyncClient with Polar high-concurrency defaults."""

    kwargs.setdefault("limits", polar_http_limits())
    kwargs["timeout"] = polar_http_timeout(timeout)
    return httpx.AsyncClient(**kwargs)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return float(default)
    try:
        parsed = float(value)
    except ValueError:
        return float(default)
    return parsed if parsed > 0 else float(default)
