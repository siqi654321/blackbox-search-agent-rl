"""Base transformer interface with SGLang request enhancement."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any


class BaseTransformer(ABC):
    """Abstract base class for API transformers.

    Transforms requests from source API format to OpenAI format (for SGLang),
    and transforms responses back to source API format.
    """

    @abstractmethod
    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Transform request body to OpenAI/SGLang format."""
        pass

    @abstractmethod
    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        """Transform response back to source API format."""
        pass

    @abstractmethod
    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Transform a streaming chunk to source API format."""
        pass

    def is_streaming_request(self, body: dict[str, Any]) -> bool:
        """Check if request is for streaming response."""
        return body.get("stream", False)

    def create_stream_state(self, original_request: dict[str, Any]) -> Any | None:
        """Create per-request stream state when chunk transforms need memory."""
        return None

    @staticmethod
    def _is_qwen_model(model_name: str | None) -> bool:
        if not model_name:
            return False
        return "qwen" in model_name.lower()

    @staticmethod
    def _merge_developer_role(request: dict[str, Any]) -> dict[str, Any]:
        """Rename 'developer' role to 'system' and merge all system messages into one."""
        messages = request.get("messages")
        if not isinstance(messages, list):
            return request

        # Rename developer -> system
        normalized = [
            {**msg, "role": "system"} if isinstance(msg, dict) and msg.get("role") == "developer" else msg
            for msg in messages
        ]

        # Merge multiple system messages into one at the top
        system_parts: list[str] = []
        non_system: list[Any] = []
        for msg in normalized:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content", "")
                text = content if isinstance(content, str) else str(content) if content else ""
                if text:
                    system_parts.append(text)
            else:
                non_system.append(msg)

        if len(system_parts) > 1:
            request["messages"] = [{"role": "system", "content": "\n\n".join(system_parts)}, *non_system]
        else:
            request["messages"] = normalized
        return request

    def _enhance_for_training(
        self,
        request: dict[str, Any],
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Apply model compatibility fixes and request fields needed for training."""
        request.pop("_polar_model_served", None)

        if self._is_qwen_model(model_name):
            # Qwen chat templates do not support the developer role.  Do not
            # force enable_thinking here: standalone VERL does not pass this
            # kwarg by default, and Qwen3 templates render different generation
            # prompts when enable_thinking=False.  Preserve caller-provided
            # chat_template_kwargs, and keep the old behavior only behind an
            # explicit compatibility switch.
            request = self._merge_developer_role(request)
            if _env_flag("POLAR_QWEN_DISABLE_THINKING", default=False):
                chat_template_kwargs = dict(request.get("chat_template_kwargs") or {})
                chat_template_kwargs.setdefault("enable_thinking", False)
                request["chat_template_kwargs"] = chat_template_kwargs

        # OpenAI-compatible SGLang accepts `logprobs`, but VERL/SGLang
        # training traces also need token ids and prompt ids.  The SGLang
        # native knobs below make the response include meta_info fields such as
        # input_token_ids, output_token_ids and output_token_logprobs.
        request["logprobs"] = True
        request.setdefault("top_logprobs", 1)
        request["return_logprob"] = True
        request.setdefault("logprob_start_len", 0)
        request.setdefault("top_logprobs_num", 1)
        # Some OpenAI-compatible entrypoints drop unknown top-level fields when
        # constructing sampling params, but preserve `extra_body` and/or
        # `extra_body.sampling_params`.  Keep the native SGLang knobs in all
        # three places so at least one survives to the server implementation.
        extra_body = dict(request.get("extra_body") or {})
        extra_body.setdefault("return_logprob", True)
        extra_body.setdefault("logprob_start_len", 0)
        extra_body.setdefault("top_logprobs_num", 1)
        sampling_params = dict(extra_body.get("sampling_params") or {})
        sampling_params.setdefault("return_logprob", True)
        sampling_params.setdefault("logprob_start_len", 0)
        sampling_params.setdefault("top_logprobs_num", 1)
        extra_body["sampling_params"] = sampling_params
        request["extra_body"] = extra_body
        return request


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
