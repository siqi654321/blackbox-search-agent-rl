"""OpenAI-compatible bridge backed by VERL-managed SGLang /generate HTTP.

This mirrors The reference implementation's rollout path: tokenize/apply chat template locally, call the
SGLang native /generate endpoint with input_ids, then convert SGLang's native
output_token_logprobs into an OpenAI-like chat.completion response that carries
input_token_ids/token_ids/logprobs for Polar trajectory building.
"""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import uuid4

import httpx
import torch
from polar.http_utils import polar_async_client, polar_http_timeout
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

from verl_polar_bridge.debug_utils import messages_summary, stable_hash, token_preview

logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    return os.environ.get("POLAR_NATIVE_DEBUG", "1").lower() not in {"0", "false", "no", "off"}


def _deep_debug_enabled() -> bool:
    return os.environ.get("POLAR_NATIVE_DEEP_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _deep_debug_verbose() -> bool:
    return os.environ.get("POLAR_NATIVE_DEEP_DEBUG_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}


def _alignment_debug_enabled() -> bool:
    return os.environ.get("POLAR_ALIGNMENT_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _alignment_debug_limit() -> int:
    try:
        return max(0, int(os.environ.get("POLAR_ALIGNMENT_DEBUG_LIMIT", "8")))
    except Exception:
        return 8


_ALIGNMENT_DEBUG_EMITTED = 0


def _alignment_debug_allowed() -> bool:
    global _ALIGNMENT_DEBUG_EMITTED
    if not _alignment_debug_enabled():
        return False
    limit = _alignment_debug_limit()
    if limit and _ALIGNMENT_DEBUG_EMITTED >= limit:
        return False
    _ALIGNMENT_DEBUG_EMITTED += 1
    return True


def _logprob_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"len": 0}
    vals = [float(v) for v in values]
    return {
        "len": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / max(len(vals), 1),
        "zero_count": sum(1 for v in vals if abs(v) < 1e-12),
        "head": vals[:8],
        "tail": vals[-8:] if len(vals) > 8 else vals,
    }


def _shape_preview(value: Any, *, limit: int = 8) -> dict[str, Any]:
    info: dict[str, Any] = {"type": type(value).__name__}
    try:
        if value is None:
            info["is_none"] = True
            return info
        if isinstance(value, torch.Tensor):
            info.update({"shape": list(value.shape), "dtype": str(value.dtype), "preview": value.detach().cpu().flatten()[:limit].tolist()})
            return info
        if isinstance(value, (list, tuple)):
            info["len"] = len(value)
            info["preview"] = list(value[:limit])
            if value:
                info["first_type"] = type(value[0]).__name__
                if isinstance(value[0], (list, tuple)):
                    info["first_len"] = len(value[0])
                    info["first_preview"] = list(value[0][:limit])
            return info
    except Exception as exc:
        info["error"] = repr(exc)
    return info


@dataclass
class NativeOpenAIBridgeHandle:
    base_url: str
    port: int
    thread: threading.Thread
    server: uvicorn.Server
    upstream_urls: tuple[str, ...] = ()

    def stop(self) -> None:
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)


def start_native_openai_bridge(
    *,
    sglang_base_url: str | None = None,
    sglang_base_urls: Sequence[str] | None = None,
    tokenizer_name_or_path: str,
    model_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> NativeOpenAIBridgeHandle:
    upstream_urls = _normalize_upstream_urls(sglang_base_urls or ([sglang_base_url] if sglang_base_url else []))
    if not upstream_urls:
        raise ValueError("sglang_base_url or sglang_base_urls is required for native OpenAI bridge")
    if not tokenizer_name_or_path:
        raise ValueError("tokenizer_name_or_path is required for native OpenAI bridge")
    if port == 0:
        port = _free_port(host)

    for upstream_url in upstream_urls:
        _probe_sglang_generate(upstream_url)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    served_model = model_name or tokenizer_name_or_path
    app = _build_app(sglang_base_urls=upstream_urls, tokenizer=tokenizer, model_name=served_model)
    config = uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False)
    server = uvicorn.Server(config)

    def _run_server() -> None:
        try:
            server.run()
        except Exception:
            logger.exception("VERL SGLang /generate OpenAI bridge crashed")
            raise

    thread = threading.Thread(target=_run_server, name=f"verl-polar-sglang-generate-openai-{port}", daemon=True)
    thread.start()
    _wait_ready(host, port)
    return NativeOpenAIBridgeHandle(
        base_url=f"http://{host}:{port}",
        port=port,
        thread=thread,
        server=server,
        upstream_urls=tuple(upstream_urls),
    )


def _normalize_upstream_urls(values: Sequence[str | None]) -> tuple[str, ...]:
    urls: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        url = str(value).strip().rstrip("/")
        if not url or url in seen:
            continue
        urls.append(url)
        seen.add(url)
    return tuple(urls)


def _build_app(*, sglang_base_urls: Sequence[str], tokenizer: Any, model_name: str) -> FastAPI:
    app = FastAPI()
    upstream_urls = tuple(sglang_base_urls)
    upstream_lock = threading.Lock()
    upstream_inflight = {url: 0 for url in upstream_urls}
    upstream_request_count = {url: 0 for url in upstream_urls}
    upstream_generated_tokens = {url: 0 for url in upstream_urls}
    upstream_generate_s = {url: 0.0 for url in upstream_urls}
    session_to_upstream: dict[str, str] = {}
    next_upstream_index = 0

    def _choose_upstream(session_key: str) -> str:
        nonlocal next_upstream_index
        if not upstream_urls:
            raise RuntimeError("no SGLang upstreams configured")
        if len(upstream_urls) == 1:
            return upstream_urls[0]
        key = session_key or uuid4().hex
        with upstream_lock:
            cached = session_to_upstream.get(key)
            if cached is not None:
                return cached
            # Sticky round-robin: preserve multi-turn prefix-cache locality for
            # a session while spreading new sessions across all VERL replicas.
            url = upstream_urls[next_upstream_index % len(upstream_urls)]
            next_upstream_index += 1
            session_to_upstream[key] = url
            return url

    def _mark_upstream_start(url: str) -> None:
        with upstream_lock:
            upstream_inflight[url] = upstream_inflight.get(url, 0) + 1

    def _mark_upstream_done(url: str, *, elapsed_s: float = 0.0, completion_tokens: int = 0) -> None:
        with upstream_lock:
            upstream_inflight[url] = max(0, upstream_inflight.get(url, 0) - 1)
            upstream_request_count[url] = upstream_request_count.get(url, 0) + 1
            upstream_generated_tokens[url] = upstream_generated_tokens.get(url, 0) + int(completion_tokens or 0)
            upstream_generate_s[url] = upstream_generate_s.get(url, 0.0) + float(elapsed_s or 0.0)


    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "verl-sglang-generate",
            "model": model_name,
            "sglang_base_url": upstream_urls[0] if upstream_urls else "",
            "sglang_base_urls": list(upstream_urls),
            "upstream_inflight": dict(upstream_inflight),
            "upstream_request_count": dict(upstream_request_count),
            "upstream_generated_tokens": dict(upstream_generated_tokens),
            "upstream_generate_s": dict(upstream_generate_s),
            "session_upstreams": len(session_to_upstream),
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        try:
            return await _chat_completions_impl(request)
        except Exception as exc:
            logger.exception("VERL SGLang /generate bridge completion failed")
            return JSONResponse({"error": {"message": f"VERL SGLang /generate bridge failed: {type(exc).__name__}: {exc}"}}, status_code=500)

    async def _chat_completions_impl(request: Request) -> JSONResponse:
        body = await request.json()
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            return JSONResponse({"error": {"message": "messages must be a list"}}, status_code=400)
        session_key = _session_key(request, body)
        request_id = uuid4().hex
        add_generation_prompt = bool(body.get("add_generation_prompt", True))
        chat_kwargs = dict(body.get("chat_template_kwargs") or {})
        tools = body.get("tools")
        if tools:
            # Match VERL ToolAgentLoop/RLHFDataset semantics: tool schemas are
            # passed as the explicit apply_chat_template(..., tools=...)
            # argument, not as a chat_template_kwargs entry.  Some tokenizers
            # render the same tokens either way, but others treat the explicit
            # parameter specially; using the same call shape removes a subtle
            # prompt-length/rollout drift between standalone VERL and Polar.
            chat_kwargs.pop("tools", None)
        request_t0 = time.perf_counter()
        prompt_render_t0 = time.perf_counter()
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **chat_kwargs,
        )
        prompt_ids_list = _as_1d_int_list(prompt_ids)
        prompt_render_s = time.perf_counter() - prompt_render_t0
        if _is_internal_prompt_probe(body):
            now = int(time.time())
            return JSONResponse(
                {
                    "id": f"chatcmpl-{uuid4().hex}",
                    "object": "chat.completion",
                    "created": now,
                    "model": str(body.get("model") or model_name),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                            "input_token_ids": prompt_ids_list,
                            "token_ids": [],
                            "logprobs": {"content": []},
                            "matched_stop": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt_ids_list),
                        "completion_tokens": 0,
                        "total_tokens": len(prompt_ids_list),
                    },
                    "metadata": {
                        "backend": "verl-sglang-generate",
                        "internal": True,
                        "bridge_timing": {
                            "bridge_total_s": time.perf_counter() - request_t0,
                            "prompt_render_s": prompt_render_s,
                            "prompt_tokens": len(prompt_ids_list),
                            "completion_tokens": 0,
                        },
                    },
                }
            )
        sampling_params = _sampling_params(body, prompt_tokens=len(prompt_ids_list))
        payload = {
            "input_ids": [prompt_ids_list],
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        if body.get("return_routed_experts") is not None:
            payload["return_routed_experts"] = body.get("return_routed_experts")

        if _debug_enabled():
            logger.warning(
                "POLAR_SGLANG_GENERATE_BRIDGE_DEBUG phase=before_post request_id=%s session_key=%s upstream=%s "
                "messages=%s add_generation_prompt=%s tools=%s tool_names=%s chat_kwargs=%s "
                "prompt_ids=%s sampling_params=%s body_keys=%s",
                request_id,
                session_key,
                _choose_upstream(session_key),
                len(messages),
                add_generation_prompt,
                bool(tools),
                _tool_names(tools),
                _debug_chat_kwargs(chat_kwargs),
                _shape_preview(prompt_ids_list),
                sampling_params.copy(),
                sorted(body.keys()),
            )
            if _deep_debug_enabled():
                logger.warning(
                    "POLAR_SGLANG_GENERATE_BRIDGE_DEEP_DEBUG phase=before_post request_id=%s payload=%s",
                    request_id,
                    {
                        "session_key": session_key,
                        "upstream": _choose_upstream(session_key),
                        "messages": messages_summary(messages),
                        "tools_hash": stable_hash(tools),
                        "tools": tools if _deep_debug_verbose() else {"count": len(tools) if isinstance(tools, list) else 0, "names": _tool_names(tools)},
                        "prompt_ids": token_preview(prompt_ids_list),
                        "sampling_params": sampling_params.copy(),
                        "extra_body": body.get("extra_body") if isinstance(body.get("extra_body"), dict) else None,
                    },
                )

        headers = {"X-SMG-Routing-Key": session_key} if session_key else None
        timeout = polar_http_timeout(float(os.environ.get("POLAR_SGLANG_GENERATE_TIMEOUT", "300")))
        upstream_url = _choose_upstream(session_key)
        _mark_upstream_start(upstream_url)
        upstream_t0 = time.perf_counter()
        try:
            async with polar_async_client(timeout=timeout) as client:
                response = await client.post(f"{upstream_url}/generate", json=payload, headers=headers)
        except Exception:
            upstream_elapsed_s = time.perf_counter() - upstream_t0
            _mark_upstream_done(upstream_url, elapsed_s=upstream_elapsed_s, completion_tokens=0)
            raise
        upstream_elapsed_s = time.perf_counter() - upstream_t0
        if response.status_code >= 400:
            _mark_upstream_done(upstream_url, elapsed_s=upstream_elapsed_s, completion_tokens=0)
            raise RuntimeError(f"SGLang /generate returned {response.status_code}: {response.text[:2000]}")
        response_json_t0 = time.perf_counter()
        raw_output = response.json()
        response_json_s = time.perf_counter() - response_json_t0
        extract_t0 = time.perf_counter()
        output = _unwrap_singleton(raw_output)
        token_ids, log_probs = _extract_output_tokens_and_logprobs(output)
        extract_logprobs_s = time.perf_counter() - extract_t0
        _mark_upstream_done(upstream_url, elapsed_s=upstream_elapsed_s, completion_tokens=len(token_ids))
        decode_text_s = 0.0
        text = output.get("text") if isinstance(output, dict) else None
        if text is None:
            decode_t0 = time.perf_counter()
            text = tokenizer.decode(token_ids, skip_special_tokens=False)
            decode_text_s = time.perf_counter() - decode_t0
        finish_reason = _extract_finish_reason(output)

        if _debug_enabled():
            logger.warning(
                "POLAR_SGLANG_GENERATE_BRIDGE_DEBUG phase=after_post request_id=%s raw_type=%s output_keys=%s "
                "prompt_len=%s token_len=%s logprob_len=%s finish_reason=%s text_preview=%r",
                request_id,
                type(raw_output).__name__,
                sorted(output.keys()) if isinstance(output, dict) else None,
                len(prompt_ids_list),
                len(token_ids),
                len(log_probs),
                finish_reason,
                str(text)[:120],
            )
            if _alignment_debug_allowed():
                meta = output.get("meta_info") if isinstance(output, dict) else {}
                if not isinstance(meta, dict):
                    meta = {}
                raw_items = meta.get("output_token_logprobs") or []
                logger.warning(
                    "POLAR_NATIVE_ALIGNMENT_DEBUG %s",
                    {
                        "request_id": request_id,
                        "session_key": session_key,
                        "upstream": _choose_upstream(session_key),
                        "prompt": token_preview(prompt_ids_list),
                        "token_ids": token_preview(token_ids),
                        "log_probs": _logprob_stats(log_probs),
                        "finish_reason": finish_reason,
                        "raw_output_keys": sorted(output.keys()) if isinstance(output, dict) else None,
                        "meta_keys": sorted(meta.keys()),
                        "raw_output_token_logprobs_shape": _shape_preview(raw_items),
                        "raw_output_ids": _shape_preview(output.get("output_ids") if isinstance(output, dict) else None),
                        "meta_output_ids": _shape_preview(meta.get("output_ids") or meta.get("output_token_ids")),
                    },
                )
            if _deep_debug_enabled():
                logger.warning(
                    "POLAR_SGLANG_GENERATE_BRIDGE_DEEP_DEBUG phase=after_post request_id=%s payload=%s",
                    request_id,
                    {
                        "prompt_ids": token_preview(prompt_ids_list),
                        "token_ids": token_preview(token_ids),
                        "logprob_len": len(log_probs),
                        "finish_reason": finish_reason,
                        "text_hash": stable_hash(text or ""),
                        "text_head": str(text or "")[:240],
                        "text_tail": str(text or "")[-240:],
                    },
                )

        now = int(time.time())
        logprob_content_t0 = time.perf_counter()
        logprob_content = _logprob_content(tokenizer, token_ids, log_probs)
        logprob_content_s = time.perf_counter() - logprob_content_t0
        meta = output.get("meta_info") if isinstance(output, dict) else {}
        raw_logprobs = meta.get("output_token_logprobs") if isinstance(meta, dict) else None
        scheduled_max_new_tokens = sampling_params.get("max_new_tokens")
        response_body = {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": str(body.get("model") or model_name),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason or "stop",
                    "input_token_ids": prompt_ids_list,
                    "token_ids": token_ids,
                    "logprobs": {"content": logprob_content},
                    "matched_stop": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids_list),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids_list) + len(token_ids),
            },
            "metadata": {
                "backend": "verl-sglang-generate",
                **(_safe_meta(output) or {}),
                "bridge_timing": {
                    "bridge_total_s": time.perf_counter() - request_t0,
                    "upstream_generate_s": upstream_elapsed_s,
                    "prompt_render_s": prompt_render_s,
                    "response_json_s": response_json_s,
                    "extract_logprobs_s": extract_logprobs_s,
                    "decode_text_s": decode_text_s,
                    "logprob_content_s": logprob_content_s,
                    "prompt_tokens": len(prompt_ids_list),
                    "completion_tokens": len(token_ids),
                    "meta_output_token_logprobs_len": len(raw_logprobs) if isinstance(raw_logprobs, list) else 0,
                    "scheduled_max_new_tokens": int(scheduled_max_new_tokens)
                    if isinstance(scheduled_max_new_tokens, int)
                    else 0,
                    "bridge_schedule_enabled": 1 if _bridge_schedule_config(body) is not None else 0,
                    "upstream_url": upstream_url,
                },
            },
        }
        return JSONResponse(response_body)

    return app


def _is_internal_prompt_probe(body: dict[str, Any]) -> bool:
    marker = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
    if marker.get("polar_skip_trajectory") is True and marker.get("polar_internal") is True:
        if marker.get("purpose") in {"prompt_token_count", "tool_response_token_count"}:
            return True
    return body.get("max_tokens") == 0 and bool(marker.get("polar_internal"))


def _probe_sglang_generate(sglang_base_url: str) -> None:
    payload = {
        "input_ids": [[151644, 872, 198]],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True,
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            response = client.post(f"{sglang_base_url}/generate", json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"status={response.status_code} body={response.text[:2000]}")
        output = _unwrap_singleton(response.json())
        token_ids, log_probs = _extract_output_tokens_and_logprobs(output)
        if _debug_enabled():
            logger.warning(
                "POLAR_SGLANG_GENERATE_BRIDGE_DEBUG phase=probe_ok upstream=%s output_type=%s token_len=%s logprob_len=%s",
                sglang_base_url,
                type(output).__name__,
                len(token_ids),
                len(log_probs),
            )
    except Exception as exc:
        raise RuntimeError(
            f"VERL-managed SGLang endpoint {sglang_base_url}/generate is not usable for Polar bridge: {type(exc).__name__}: {exc}"
        ) from exc


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") is not None:
            names.append(str(function.get("name")))
        elif tool.get("name") is not None:
            names.append(str(tool.get("name")))
    return names


def _debug_chat_kwargs(chat_kwargs: dict[str, Any]) -> dict[str, Any]:
    preview = dict(chat_kwargs)
    tools = preview.get("tools")
    if tools is not None:
        preview["tools"] = {"count": len(tools) if isinstance(tools, list) else None, "names": _tool_names(tools)}
    return preview


def _sampling_params(body: dict[str, Any], *, prompt_tokens: int | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "stop_token_ids",
        "skip_special_tokens",
        "no_stop_trim",
        "spaces_between_special_tokens",
    ):
        if key in body and body[key] is not None:
            params[key] = body[key]
    if "do_sample" in body and body["do_sample"] is not None:
        do_sample = _as_bool(body["do_sample"], default=True)
        # SGLang /generate does not use a HuggingFace-style `do_sample` knob.
        # Standalone VERL's SGLang agent-loop path likewise sends only
        # temperature/top_p/top_k.  For deterministic compare, emulate
        # do_sample=false by forcing greedy-compatible params instead of
        # forwarding an unknown sampling key to SGLang.
        if not do_sample:
            params["temperature"] = 0
            params["top_p"] = 1.0
            params["top_k"] = 1
    if "max_tokens" in body and body["max_tokens"] is not None:
        params["max_new_tokens"] = int(body["max_tokens"])
    elif "max_new_tokens" in body and body["max_new_tokens"] is not None:
        params["max_new_tokens"] = int(body["max_new_tokens"])
    schedule = _bridge_schedule_config(body)
    if schedule is not None and prompt_tokens is not None:
        params["max_new_tokens"] = _native_turn_max_tokens(
            prompt_tokens=prompt_tokens,
            prompt_length=schedule["prompt_length"],
            response_length=schedule["response_length"],
            max_model_len=schedule["max_model_len"],
        )
    return params


def _bridge_schedule_config(body: dict[str, Any]) -> dict[str, int] | None:
    extra = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
    enabled = extra.get("polar_bridge_schedule_max_tokens")
    if enabled is not True:
        return None
    prompt_length = _int_or_none(extra.get("polar_prompt_length"))
    response_length = _int_or_none(extra.get("polar_response_length"))
    max_model_len = _int_or_none(extra.get("polar_max_model_len"))
    if prompt_length is None or response_length is None or max_model_len is None:
        return None
    return {
        "prompt_length": prompt_length,
        "response_length": response_length,
        "max_model_len": max_model_len,
    }


def _native_turn_max_tokens(
    *,
    prompt_tokens: int,
    prompt_length: int,
    response_length: int,
    max_model_len: int,
) -> int:
    """Match VERL async SGLang server's default per-turn max_new_tokens."""

    prompt_tokens = max(0, int(prompt_tokens))
    train_window = max(1, int(prompt_length) + int(response_length) - prompt_tokens)
    model_window = max(1, int(max_model_len) - prompt_tokens)
    return max(1, min(train_window, model_window))


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _session_key(request: Request, body: dict[str, Any]) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1]
    return str(body.get("session_id") or body.get("user") or uuid4().hex)


def _unwrap_singleton(output: Any) -> dict[str, Any]:
    if isinstance(output, list):
        if len(output) != 1:
            raise RuntimeError(f"expected singleton SGLang /generate output, got list length {len(output)}")
        output = output[0]
    if not isinstance(output, dict):
        raise RuntimeError(f"expected SGLang /generate output dict, got {type(output).__name__}")
    return output


def _extract_output_tokens_and_logprobs(output: dict[str, Any]) -> tuple[list[int], list[float]]:
    meta = output.get("meta_info") or {}
    items = meta.get("output_token_logprobs") or []
    token_ids: list[int] = []
    log_probs: list[float] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                log_probs.append(float(item[0]))
                token_ids.append(int(item[1]))
    if not token_ids:
        for key in ("output_ids", "token_ids"):
            ids = output.get(key)
            if isinstance(ids, list):
                token_ids = [int(x) for x in ids]
                break
        if not token_ids:
            for key in ("output_ids", "token_ids", "completion_token_ids"):
                ids = meta.get(key)
                if isinstance(ids, list):
                    token_ids = [int(x) for x in ids]
                    break
    if len(log_probs) < len(token_ids):
        log_probs.extend([0.0] * (len(token_ids) - len(log_probs)))
    elif len(log_probs) > len(token_ids):
        log_probs = log_probs[: len(token_ids)]
    return token_ids, log_probs


def _extract_finish_reason(output: dict[str, Any]) -> str | None:
    meta = output.get("meta_info") or {}
    finish = meta.get("finish_reason") or output.get("finish_reason")
    if isinstance(finish, dict):
        return finish.get("type") or finish.get("reason")
    if finish is not None:
        return str(finish)
    return None


def _safe_meta(output: dict[str, Any]) -> dict[str, Any]:
    meta = output.get("meta_info") or {}
    return meta if isinstance(meta, dict) else {}


def _as_1d_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    return [int(x) for x in value]


def _logprob_content(tokenizer: Any, token_ids: list[int], log_probs: list[Any]) -> list[dict[str, Any]]:
    content = []
    for idx, token_id in enumerate(token_ids):
        lp = log_probs[idx] if idx < len(log_probs) else 0.0
        if isinstance(lp, torch.Tensor):
            lp = lp.detach().cpu().item()
        content.append(
            {
                "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "token_id": int(token_id),
                "logprob": float(lp),
                "top_logprobs": [],
            }
        )
    return content


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_ready(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"native OpenAI bridge did not start on {host}:{port}: {last_error}")
