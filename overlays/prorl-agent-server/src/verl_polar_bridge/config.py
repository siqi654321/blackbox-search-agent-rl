"""Configuration and template helpers for VERL-driven Polar rollouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

_PLACEHOLDER_RE = re.compile(r"{([^{}]+)}")
_MISSING = object()


@dataclass(frozen=True)
class PolarVerlConfig:
    """Resolved Polar settings consumed by the VERL agent-loop bridge."""

    rollout_server_url: str
    gateway_url: str | None
    topology_path: str | None
    task_template: dict[str, Any]
    task_id_template: str
    instruction_template: str | None
    reward_key: str
    max_concurrency: int
    max_session_concurrency: int
    max_async_level: int
    max_off_policy_steps: int
    request_timeout: float | None
    gateway_control_timeout: float
    weight_update_pause_timeout: float
    allow_weight_update_overlap: bool
    callback_host: str
    callback_port: int
    scoring_mode: str
    min_complete_accept_fraction: float
    tokenizer_name_or_path: str | None
    add_generation_prompt: bool
    eval_dataset_name: str
    dynamic_history_enable: bool
    dynamic_history_mode: str
    overflow_policy: str
    acceptance_reject_logprob_error: bool
    metrics_prefix: str
    metrics_log_longest_trace_artifact: bool
    metrics_longest_trace_interval: int
    stitch_traces: bool


def resolve_polar_verl_config(config: Any) -> PolarVerlConfig:
    """Resolve Polar bridge settings from a VERL/OmegaConf-style config tree.

    Preferred layout is ``config.polar``. For compatibility with existing
    experiments, the helper also accepts flat ``polar_*`` keys on ``config``.
    """

    polar = _get(config, "polar", default=None) or config
    topology_path = _get_any(polar, ("topology_path", "polar_topology_path"), default=None)
    rollout_server_url = _get_any(
        polar,
        ("rollout_url", "rollout_server_url", "polar_rollout_url"),
        default=None,
    )
    topology = None
    if rollout_server_url is None and topology_path:
        from polar.config import TopologyConfig

        topology = TopologyConfig.load(topology_path)
        rollout_server_url = topology.rollout.public_url
    if rollout_server_url is None:
        raise ValueError(
            "Polar rollout URL is not configured. Set polar.rollout_url or "
            "polar.topology_path in the VERL config."
        )

    task_template = deepcopy(_get_any(polar, ("task_template", "polar_task_template"), default={}) or {})
    task_template = _from_namespace(_to_plain(task_template))
    if not isinstance(task_template, dict):
        # Some Hydra CLI override patterns can materialize nested additions as
        # a scalar placeholder. Fall back to the SearchR1 template when the
        # user selected the built-in search harness via sibling overrides/env.
        fallback_template = _maybe_build_search_task_template(polar)
        if fallback_template is not None:
            task_template = fallback_template
        else:
            raise ValueError(f"polar.task_template must be a mapping, got {type(task_template).__name__}: {task_template!r}")
    if "agent" not in task_template:
        fallback_template = _maybe_build_search_task_template(polar)
        if fallback_template is not None:
            task_template = fallback_template
        else:
            raise ValueError(f"polar.task_template must include an agent spec; got keys={list(task_template.keys())}")
    task_template = _apply_search_task_settings_overlay(task_template, polar)

    max_async_level = int(_get_any(polar, ("max_async_level", "polar_max_async_level"), default=2) or 2)
    if max_async_level <= 0:
        raise ValueError("polar.max_async_level must be greater than 0")

    rollout_batch_size = int(_get_path(config, "data.train_batch_size", default=1) or 1)
    if rollout_batch_size <= 0:
        raise ValueError("data.train_batch_size must be greater than 0")

    group_size = int(_get_path(config, "actor_rollout_ref.rollout.n", default=1) or 1)
    if group_size <= 0:
        raise ValueError("actor_rollout_ref.rollout.n must be greater than 0")

    update_weights_interval = int(_get_path(config, "trainer.update_weights_interval", default=1) or 1)
    if update_weights_interval <= 0:
        raise ValueError("trainer.update_weights_interval must be greater than 0")

    request_timeout = _get_any(polar, ("request_timeout", "polar_request_timeout"), default=None)
    if request_timeout is not None:
        request_timeout = float(request_timeout)
        if request_timeout <= 0:
            raise ValueError("polar.request_timeout must be greater than 0")

    gateway_control_timeout = float(
        _get_any(
            polar,
            ("weight_update.gateway_control_timeout", "gateway_control_timeout", "polar_gateway_control_timeout"),
            default=30.0,
        )
        or 30.0
    )
    if gateway_control_timeout <= 0:
        raise ValueError("polar.gateway_control_timeout must be greater than 0")

    weight_update_pause_timeout = float(
        _get_any(
            polar,
            ("weight_update.pause_timeout", "weight_update_pause_timeout", "polar_weight_update_pause_timeout"),
            default=300.0,
        )
        or 300.0
    )
    if weight_update_pause_timeout <= 0:
        raise ValueError("polar.weight_update_pause_timeout must be greater than 0")

    callback_host = str(_get_any(polar, ("callback_host", "polar_callback_host"), default="127.0.0.1")).strip()
    if not callback_host:
        raise ValueError("polar.callback_host must be a non-empty host or IP")
    if callback_host in {"0.0.0.0", "::"}:
        raise ValueError("polar.callback_host must be reachable by the rollout server, not a wildcard bind address")

    callback_port = int(_get_any(polar, ("callback_port", "polar_callback_port"), default=0) or 0)
    if callback_port < 0:
        raise ValueError("polar.callback_port must be greater than or equal to 0")

    scoring_mode = str(_get_any(polar, ("scoring_mode", "polar_scoring_mode"), default="group")).strip().lower()
    if scoring_mode not in {"group", "individual"}:
        raise ValueError("polar.scoring_mode must be 'group' or 'individual'")

    min_complete_accept_fraction = float(
        _get_any(
            polar,
            ("acceptance.min_complete_accept_fraction", "min_complete_accept_fraction", "polar_min_complete_accept_fraction"),
            default=0.0,
        )
        or 0.0
    )
    if not 0.0 <= min_complete_accept_fraction <= 1.0:
        raise ValueError("polar.min_complete_accept_fraction must be between 0 and 1")

    explicit_max_concurrency = _get_any(polar, ("max_concurrency", "polar_max_concurrency"), default=None)
    if explicit_max_concurrency is None:
        max_concurrency = rollout_batch_size * max_async_level
    else:
        max_concurrency = int(explicit_max_concurrency)
        if max_concurrency <= 0:
            raise ValueError("polar.max_concurrency must be greater than 0")

    explicit_max_session_concurrency = _get_any(
        polar,
        ("max_session_concurrency", "polar_max_session_concurrency"),
        default=None,
    )
    if explicit_max_session_concurrency is None:
        max_session_concurrency = max_concurrency * group_size
    else:
        max_session_concurrency = int(explicit_max_session_concurrency)
        if max_session_concurrency <= 0:
            raise ValueError("polar.max_session_concurrency must be greater than 0")

    tokenizer_name_or_path = _get_any(
        polar,
        ("tokenizer_name_or_path", "polar_tokenizer_name_or_path"),
        default=_get_path(config, "actor_rollout_ref.model.path", default=None),
    )

    gateway_url = _get_any(polar, ("gateway_url", "polar_gateway_url"), default=None)
    if gateway_url is None and topology_path:
        if topology is None:
            from polar.config import TopologyConfig

            topology = TopologyConfig.load(topology_path)
        if getattr(topology.gateway, "nodes", None):
            gateway_url = topology.gateway.nodes[0].public_url

    allow_weight_update_overlap = bool(
        _get_any(
            polar,
            ("weight_update.allow_overlap", "allow_weight_update_overlap", "polar_allow_weight_update_overlap"),
            default=False,
        )
    )
    dynamic_history_enable = bool(
        _get_any(polar, ("dynamic_history.enable", "dynamic_history_enable"), default=False)
    )
    dynamic_history_mode = str(
        _get_any(polar, ("dynamic_history.mode", "dynamic_history_mode"), default="append") or "append"
    ).strip()
    if dynamic_history_mode not in {"append", "trace", "session"}:
        raise ValueError("polar.dynamic_history.mode must be one of: append, trace, session")

    overflow_policy = str(
        _get_any(polar, ("overflow_policy", "trajectory_overflow_policy"), default="drop") or "drop"
    ).strip().lower()
    if overflow_policy not in {"drop", "verl_truncate", "none"}:
        raise ValueError("polar.overflow_policy must be one of: drop, verl_truncate, none")

    acceptance_reject_logprob_error = bool(
        _get_any(polar, ("acceptance.reject_logprob_error", "reject_logprob_error"), default=True)
    )
    metrics_prefix = str(_get_any(polar, ("metrics.prefix", "metrics_prefix"), default="polar") or "polar").strip()
    if not metrics_prefix:
        raise ValueError("polar.metrics.prefix must be non-empty")
    metrics_log_longest_trace_artifact = bool(
        _get_any(polar, ("metrics.log_longest_trace_artifact", "log_longest_trace_artifact"), default=False)
    )
    metrics_longest_trace_interval = int(
        _get_any(polar, ("metrics.longest_trace_interval", "longest_trace_interval"), default=1) or 1
    )
    if metrics_longest_trace_interval <= 0:
        raise ValueError("polar.metrics.longest_trace_interval must be greater than 0")

    # Default remains baseline-compatible stitching: append-only Search traces are
    # merged into one VERL row when safe. Complex/prompt-grounded all-trace runs can
    # disable this so every builder trace becomes its own trainable segment.
    stitch_traces = bool(_get_any(polar, ("training.stitch_traces", "stitch_traces"), default=True))

    return PolarVerlConfig(
        rollout_server_url=str(rollout_server_url).rstrip("/"),
        gateway_url=str(gateway_url).rstrip("/") if gateway_url else None,
        topology_path=str(topology_path) if topology_path else None,
        task_template=task_template,
        task_id_template=str(
            _get_any(
                polar,
                ("task_id_template", "polar_task_id_template"),
                default="polar-verl-{global_steps}-{uid}-{task_position}",
            )
        ),
        instruction_template=_get_any(polar, ("instruction_template", "polar_instruction_template"), default=None),
        reward_key=str(_get_any(polar, ("reward_key", "polar_reward_key"), default="score") or "score"),
        max_concurrency=max_concurrency,
        max_session_concurrency=max_session_concurrency,
        max_async_level=max_async_level,
        max_off_policy_steps=max_async_level + update_weights_interval,
        request_timeout=request_timeout,
        gateway_control_timeout=gateway_control_timeout,
        weight_update_pause_timeout=weight_update_pause_timeout,
        allow_weight_update_overlap=allow_weight_update_overlap,
        callback_host=callback_host,
        callback_port=callback_port,
        scoring_mode=scoring_mode,
        min_complete_accept_fraction=min_complete_accept_fraction,
        tokenizer_name_or_path=str(tokenizer_name_or_path) if tokenizer_name_or_path else None,
        add_generation_prompt=bool(_get_any(polar, ("add_generation_prompt", "polar_add_generation_prompt"), default=True)),
        eval_dataset_name=str(_get_any(polar, ("eval_dataset_name", "polar_eval_dataset_name"), default="polar_eval")),
        dynamic_history_enable=dynamic_history_enable,
        dynamic_history_mode=dynamic_history_mode,
        overflow_policy=overflow_policy,
        acceptance_reject_logprob_error=acceptance_reject_logprob_error,
        metrics_prefix=metrics_prefix,
        metrics_log_longest_trace_artifact=metrics_log_longest_trace_artifact,
        metrics_longest_trace_interval=metrics_longest_trace_interval,
        stitch_traces=stitch_traces,
    )


def render_task_payload(
    *,
    trainer_config: Any,
    config: PolarVerlConfig,
    sample: Any,
    instruction: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
    global_steps: int = 0,
    uid: str | None = None,
) -> dict[str, Any]:
    context = _build_context(
        trainer_config=trainer_config,
        sample=sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=num_rollouts,
        global_steps=global_steps,
        uid=uid,
    )
    payload = _render_template_value(deepcopy(config.task_template), context)
    if not isinstance(payload, dict):
        raise ValueError("polar.task_template must render to a mapping")
    payload["task_id"] = str(_render_template_value(config.task_id_template, context))
    payload["instruction"] = instruction
    payload["num_samples"] = num_rollouts
    if config.request_timeout is not None:
        payload.setdefault("timeout_seconds", float(config.request_timeout))
    payload.setdefault("metadata", {})
    if isinstance(payload["metadata"], dict):
        payload["metadata"].setdefault("source_uid", context["uid"])
        payload["metadata"].setdefault("rollout_id", rollout_id)
        payload["metadata"].setdefault("global_steps", global_steps)
    return payload


def render_instruction(
    *,
    trainer_config: Any,
    config: PolarVerlConfig,
    sample: Any,
    prompt_text: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
    global_steps: int = 0,
    uid: str | None = None,
) -> str:
    if not config.instruction_template:
        return prompt_text
    context = _build_context(
        trainer_config=trainer_config,
        sample=sample,
        instruction=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=num_rollouts,
        global_steps=global_steps,
        uid=uid,
    )
    rendered = _render_template_value(config.instruction_template, context)
    if not isinstance(rendered, str):
        raise ValueError("polar.instruction_template must render to a string")
    return rendered


def render_topology_template(topology_path: str | Path, trainer_config: Any) -> dict[str, Any]:
    """Load a Polar topology template and point every gateway node at VERL's router."""
    from polar.config import TopologyConfig

    router_url = resolve_verl_router_base_url(trainer_config)
    if router_url is None:
        raise ValueError("VERL rollout router URL is not configured")

    topology = TopologyConfig.load(topology_path)

    return {
        "rollout": topology.rollout.model_dump(mode="python"),
        "gateway": {
            "heartbeat_interval_seconds": topology.gateway.heartbeat_interval_seconds,
            "rollout_server_url": topology.gateway.rollout_server_url,
            "nodes": [
                {
                    **node.model_dump(mode="python", exclude={"sglang"}),
                    "sglang": {"base_url": router_url},
                }
                for node in topology.gateway.nodes
            ],
        },
    }


def resolve_verl_router_base_url(config: Any) -> str | None:
    for path in (
        "polar.router_base_url",
        "actor_rollout_ref.rollout.router_base_url",
        "actor_rollout_ref.rollout.sglang_router_base_url",
    ):
        value = _get_path(config, path, default=None)
        if value not in (None, ""):
            return str(value).rstrip("/")
    ip = _get_path(config, "actor_rollout_ref.rollout.sglang_router_ip", default=None)
    port = _get_path(config, "actor_rollout_ref.rollout.sglang_router_port", default=None)
    if ip in (None, "") or port in (None, ""):
        return None
    return f"http://{ip}:{port}"


def _build_context(
    *,
    trainer_config: Any,
    sample: Any,
    instruction: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
    global_steps: int,
    uid: str | None,
) -> dict[str, Any]:
    metadata = deepcopy(_sample_get(sample, "metadata", {}) or {})
    extra_info = deepcopy(_sample_get(sample, "extra_info", {}) or {})
    reward_model = deepcopy(_sample_get(sample, "reward_model", {}) or {})
    resolved_uid = uid or str(_sample_get(sample, "uid", None) or extra_info.get("uid") or metadata.get("uid") or rollout_id)
    return {
        "args": _to_namespace(_to_plain(trainer_config)),
        "config": _to_namespace(_to_plain(trainer_config)),
        "instruction": instruction,
        "num_rollouts": num_rollouts,
        "num_samples": num_rollouts,
        "rollout_id": rollout_id,
        "global_steps": global_steps,
        "uid": resolved_uid,
        "sample": SimpleNamespace(
            prompt=deepcopy(_sample_get(sample, "prompt", "")),
            response=deepcopy(_sample_get(sample, "response", "")),
            label=_sample_get(sample, "label", None),
            metadata=_to_namespace(metadata),
            extra_info=_to_namespace(extra_info),
            reward_model=_to_namespace(reward_model),
            index=_sample_get(sample, "index", None),
            group_index=_sample_get(sample, "group_index", None),
            status=_sample_get(sample, "status", None),
            uid=resolved_uid,
        ),
        "extra_info": _to_namespace(extra_info),
        "reward_model": _to_namespace(reward_model),
        "task_position": task_position,
    }


def _render_template_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if match := re.fullmatch(r"{([^{}]+)}", value):
            resolved = deepcopy(_resolve_path(context, match.group(1)))
            return _from_namespace(resolved)

        def replace(match: re.Match[str]) -> str:
            resolved = _resolve_path(context, match.group(1))
            return "" if resolved is None else str(resolved)

        return _PLACEHOLDER_RE.sub(replace, value)
    if isinstance(value, list):
        return [_render_template_value(item, context) for item in value]
    if isinstance(value, dict):
        return {str(key): _render_template_value(item, context) for key, item in value.items()}
    return value


def _resolve_path(context: dict[str, Any], path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise ValueError(f"Unknown template variable: {path}")
            current = current[part]
            continue
        if hasattr(current, part):
            current = getattr(current, part)
            continue
        raise ValueError(f"Unknown template variable: {path}")
    return current


def _env_flag(name: str, *, default: bool = False) -> bool:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _get_any(obj: Any, keys: tuple[str, ...], *, default: Any = None) -> Any:
    for key in keys:
        value = _get_path(obj, key, default=_MISSING) if "." in key else _get(obj, key, default=_MISSING)
        if value is not _MISSING:
            return value
    return default


def _get_path(obj: Any, path: str, *, default: Any = None) -> Any:
    current = obj
    for part in path.split("."):
        current = _get(current, part, default=_MISSING)
        if current is _MISSING:
            return default
    return current


def _get(obj: Any, key: str, *, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    get_method = getattr(obj, "get", None)
    if callable(get_method):
        try:
            return get_method(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _sample_get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{str(key): _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _from_namespace(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {k: _from_namespace(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _from_namespace(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_namespace(item) for item in value]
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    # OmegaConf is optional in this package; avoid importing it unless present.
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if hasattr(value, "__dict__"):
        return {str(k): _to_plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _apply_search_task_settings_overlay(task_template: dict[str, Any], polar: Any) -> dict[str, Any]:
    """Apply rollout sampling knobs to the final SearchR1 task template.

    This runs after fallback/explicit template selection so it covers both the
    built-in SearchR1 template and YAML/CLI materialized SearchR1 templates.
    It deliberately avoids black-box harnesses.
    """
    agent = task_template.get("agent") if isinstance(task_template, dict) else None
    if not isinstance(agent, dict):
        return task_template
    import_path = str(agent.get("import_path") or "")
    harness = str(agent.get("harness") or "")
    if "SearchR1Harness" not in import_path and "SearchR1Harness" not in harness:
        return task_template

    settings = agent.setdefault("settings", {})
    if not isinstance(settings, dict):
        settings = {}
        agent["settings"] = settings

    sampling = _search_sampling_settings(polar)
    settings.update(sampling)
    settings.setdefault(
        "max_model_len",
        int(
            _get_path(
                polar,
                "search.max_model_len",
                default=os.environ.get("SEARCH_MAX_MODEL_LEN", os.environ.get("POLAR_ROLLOUT_MAX_MODEL_LEN", "40000")),
            )
        ),
    )

    # Also pass these through agent.env so runtimes that do not use settings in
    # process construction still expose the same values to the driver.
    env = agent.setdefault("env", {})
    if isinstance(env, dict):
        env.update(
            {
                "SEARCH_TEMPERATURE": str(sampling["temperature"]),
                "SEARCH_TOP_P": str(sampling["top_p"]),
                "SEARCH_TOP_K": str(sampling["top_k"]),
                "SEARCH_REPETITION_PENALTY": str(sampling["repetition_penalty"]),
                "SEARCH_DO_SAMPLE": "true" if sampling["do_sample"] else "false",
                "SEARCH_MAX_MODEL_LEN": str(settings["max_model_len"]),
            }
        )
    return task_template


def _search_topk_settings(polar: Any) -> dict[str, int]:
    import os

    value = _get_path(polar, "search.topk", default=os.environ.get("SEARCH_TOPK"))
    if value in (None, ""):
        return {}
    return {"topk": int(value)}


def _search_sampling_settings(polar: Any) -> dict[str, Any]:
    import os

    return {
        "temperature": float(
            _get_path(
                polar,
                "search.temperature",
                default=os.environ.get("SEARCH_TEMPERATURE", os.environ.get("SMOKE_ROLLOUT_TEMPERATURE", "1.0")),
            )
        ),
        "top_p": float(
            _get_path(
                polar,
                "search.top_p",
                default=os.environ.get("SEARCH_TOP_P", os.environ.get("SMOKE_ROLLOUT_TOP_P", "1.0")),
            )
        ),
        "top_k": int(
            _get_path(
                polar,
                "search.top_k",
                default=os.environ.get(
                    "SEARCH_TOP_K",
                    os.environ.get("POLAR_ROLLOUT_TOP_K", os.environ.get("SMOKE_ROLLOUT_TOP_K", "-1")),
                ),
            )
        ),
        "repetition_penalty": float(
            _get_path(
                polar,
                "search.repetition_penalty",
                default=os.environ.get(
                    "SEARCH_REPETITION_PENALTY",
                    os.environ.get("POLAR_ROLLOUT_REPETITION_PENALTY", "1.0"),
                ),
            )
        ),
        "do_sample": _as_bool(
            _get_path(
                polar,
                "search.do_sample",
                default=os.environ.get(
                    "SEARCH_DO_SAMPLE",
                    os.environ.get("SMOKE_ROLLOUT_DO_SAMPLE", "true"),
                ),
            ),
            default=True,
        ),
    }


def _maybe_build_search_task_template(polar: Any) -> dict[str, Any] | None:
    import os

    text = str(_get_any(polar, ("task_template", "polar_task_template"), default=""))
    marker_values = [
        text,
        str(_get_path(polar, "task_template.agent.import_path", default="")),
        os.environ.get("POLAR_SEARCH_HARNESS", ""),
    ]
    if not any("SearchR1Harness" in value or value == "1" for value in marker_values):
        return None
    retrieval_url = (
        _get_path(polar, "task_template.agent.settings.retrieval_url", default=None)
        or os.environ.get("SEARCH_RETRIEVAL_URL")
        or "http://127.0.0.1:1249"
    )
    model_name = (
        _get_path(polar, "task_template.agent.model_name", default=None)
        or os.environ.get("POLAR_SEARCH_MODEL_NAME")
        or "qwen3-search-policy"
    )
    repo_root = os.environ.get("PRO_RL_REPO_ROOT") or str(Path(__file__).resolve().parents[2])
    pythonpath = os.environ.get("PYTHONPATH", "")
    runtime_pythonpath = f"{repo_root}/src:{pythonpath}" if pythonpath else f"{repo_root}/src"
    return {
        "runtime": {
            "backend": "docker",
            "image": "local",
            "import_path": "verl_polar_bridge.search_agent.local_runtime:LocalRuntime",
            "network": "host",
            "workdir": repo_root,
            "env": {
                "PYTHONPATH": runtime_pythonpath,
                "PRO_RL_REPO_ROOT": repo_root,
            },
        },
        "agent": {
            "import_path": "verl_polar_bridge.search_agent.harness:SearchR1Harness",
            "model_name": model_name,
            "settings": {
                "retrieval_url": retrieval_url,
                "max_assistant_turns": int(
                    _get_path(polar, "search.max_turns", default=os.environ.get("SEARCH_MAX_TURNS", "100"))
                ),
                "max_tokens": int(
                    _get_path(polar, "search.max_tokens", default=os.environ.get("SEARCH_MAX_TOKENS", "2048"))
                ),
                **_search_sampling_settings(polar),
                "max_model_len": int(
                    _get_path(
                        polar,
                        "search.max_model_len",
                        default=os.environ.get(
                            "SEARCH_MAX_MODEL_LEN",
                            os.environ.get("POLAR_ROLLOUT_MAX_MODEL_LEN", "40000"),
                        ),
                    )
                ),
                **_search_topk_settings(polar),
                "tool_config_path": os.environ.get("POLAR_SEARCH_TOOL_CONFIG_PATH") or os.environ.get("STANDALONE_TOOL_CONFIG_PATH", ""),
                "max_tool_response_length": int(os.environ.get("SEARCH_MAX_TOOL_RESPONSE_LENGTH", "2048")),
                "tool_response_truncate_side": os.environ.get("SEARCH_TOOL_RESPONSE_TRUNCATE_SIDE", "middle"),
                "format": os.environ.get("SEARCH_TOOL_FORMAT", "hermes"),
            },
            "env": {"PYTHONPATH": runtime_pythonpath},
        },
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "verl_polar_bridge.search_agent.evaluator:SearchR1Evaluator",
            "config": {"reward_key": "score"},
        },
        "metadata": {
            "uid": "{uid}",
            "source_uid": "{sample.metadata.source_uid}",
            "rollout_uid": "{sample.uid}",
            "reward_model": "{reward_model}",
            "extra_info": "{extra_info}",
        },
    }
