"""Module-level trainer hooks for VERL + Polar policy updates."""

from __future__ import annotations

from typing import Any

_global_manager: Any | None = None


def register_manager(manager: Any | None) -> None:
    """Register the active ``PolarAgentLoopManager`` for module-level hooks."""
    global _global_manager
    _global_manager = manager


def update_policy_version(config: Any, policy_version: int) -> None:
    """Optional hook called after serving weights are updated."""
    del config
    if _global_manager is not None:
        _global_manager.update_policy_version(policy_version)


def prepare_policy_update(config: Any, policy_version: int) -> None:
    """Optional hook called before overlapping serving-weight sync."""
    del config
    if _global_manager is not None:
        _global_manager.prepare_policy_update(policy_version)


def finish_policy_update(config: Any, policy_version: int) -> None:
    """Optional hook called after overlapping serving-weight sync."""
    del config
    if _global_manager is not None:
        _global_manager.finish_policy_update(policy_version)


def abort_policy_update(config: Any, policy_version: int) -> None:
    """Optional hook called when serving-weight sync fails after prepare."""
    del config
    if _global_manager is not None:
        abort = getattr(_global_manager, "abort_policy_update", None)
        if callable(abort):
            abort(policy_version)
        else:
            _global_manager.finish_policy_update(policy_version)
