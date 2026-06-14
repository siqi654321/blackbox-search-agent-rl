"""Small opt-in debug helpers for VERL+Polar Search alignment."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def debug_print(marker: str, payload: dict[str, Any], *, stream: str = "stderr") -> None:
    line = marker + " " + json.dumps(payload, ensure_ascii=False, default=str)
    try:
        if stream == "logger":
            logger.warning(line)
        elif stream == "stdout":
            print(line, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)
    except Exception:
        logger.warning("%s %s", marker, payload)


def stable_hash(value: Any) -> str:
    try:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        data = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha1(data).hexdigest()[:16]


def token_preview(ids: Any, *, limit: int = 12) -> dict[str, Any]:
    values = list(ids or [])
    out: dict[str, Any] = {"len": len(values), "hash": stable_hash(values)}
    if len(values) <= limit * 2:
        out["ids"] = values
    else:
        out["head"] = values[:limit]
        out["tail"] = values[-limit:]
    return out


def text_preview(text: Any, *, limit: int = 240) -> dict[str, Any]:
    s = "" if text is None else str(text)
    return {"len": len(s), "hash": stable_hash(s), "head": s[:limit], "tail": s[-limit:] if len(s) > limit else s}


def messages_summary(messages: Any, *, text_limit: int = 120) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {"type": type(messages).__name__, "hash": stable_hash(messages), "preview": text_preview(messages, limit=text_limit)}
    roles: list[str] = []
    content_lens: list[int] = []
    items: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            items.append({"idx": idx, "type": type(msg).__name__, "preview": text_preview(msg, limit=text_limit)})
            continue
        role = str(msg.get("role", ""))
        content = msg.get("content")
        content_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        roles.append(role)
        content_lens.append(len(content_text or ""))
        entry = {
            "idx": idx,
            "role": role,
            "content_len": len(content_text or ""),
            "content_hash": stable_hash(content_text or ""),
            "has_tool_calls": bool(msg.get("tool_calls")),
            "content_head": (content_text or "")[:text_limit],
        }
        if idx < 4 or idx >= max(4, len(messages) - 2):
            items.append(entry)
    return {
        "type": "list",
        "count": len(messages),
        "roles": roles,
        "content_lens": content_lens,
        "hash": stable_hash(messages),
        "items": items,
    }
