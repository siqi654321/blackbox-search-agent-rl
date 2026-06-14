"""SearchR1-like Polar harness."""

from __future__ import annotations

import os
import shlex

from polar.agent.base import BaseHarness
from polar.runtime.models import ExecInput

from verl_polar_bridge.debug_utils import debug_print, env_flag


class SearchR1Harness(BaseHarness):
    """Run a lightweight SearchR1/Qwen-style tool loop inside the Polar runtime.

    The harness writes the task instruction to a file and invokes
    ``verl_polar_bridge.search_agent.driver``. The driver calls the Polar gateway
    via ``OPENAI_BASE_URL`` for policy completions and the configured retrieval
    service for search results.
    """

    def run_steps(self, instruction: str) -> list[ExecInput]:
        retrieval_url = str(
            self.settings.get("retrieval_url")
            or _env_value(self.env, "SEARCH_RETRIEVAL_URL", "")
            or ""
        )
        if not retrieval_url:
            raise ValueError("SearchR1Harness requires agent.settings.retrieval_url")
        model = self.model_name or str(self.settings.get("model") or "qwen3-search-policy")
        max_turns = int(self.settings.get("max_assistant_turns", self.settings.get("max_turns", 8)))
        max_tokens = int(self.settings.get("max_tokens", self.settings.get("max_response_tokens", 2048)))
        prompt_length = int(
            self.settings.get("prompt_length", _env_value(self.env, "SEARCH_PROMPT_LENGTH", 4096))
        )
        max_model_len = int(
            self.settings.get("max_model_len", _env_value(self.env, "SEARCH_MAX_MODEL_LEN", 40000))
        )
        bridge_max_tokens = _bool_text(
            self.settings.get(
                "bridge_max_tokens",
                _env_value(self.env, "POLAR_SEARCH_BRIDGE_MAX_TOKENS", "true"),
            )
        )
        temperature = float(
            self.settings.get("temperature", _env_value(self.env, "SEARCH_TEMPERATURE", 1.0))
        )
        top_p = float(self.settings.get("top_p", _env_value(self.env, "SEARCH_TOP_P", 1.0)))
        top_k = int(self.settings.get("top_k", _env_value(self.env, "SEARCH_TOP_K", -1)))
        repetition_penalty = float(
            self.settings.get("repetition_penalty", _env_value(self.env, "SEARCH_REPETITION_PENALTY", 1.0))
        )
        do_sample = _bool_text(
            self.settings.get(
                "do_sample",
                _env_value(
                    self.env,
                    "SEARCH_DO_SAMPLE",
                    _env_value(self.env, "SMOKE_ROLLOUT_DO_SAMPLE", "true"),
                ),
            )
        )
        tool_config_path = str(
            self.settings.get(
                "tool_config_path", _env_value(self.env, "POLAR_SEARCH_TOOL_CONFIG_PATH", "")
            )
            or ""
        )
        topk_value = self.settings.get("topk", _env_value(self.env, "SEARCH_TOPK", ""))
        topk_arg = f"--topk {int(topk_value)} " if str(topk_value).strip() else ""
        max_tool_response_length = int(self.settings.get("max_tool_response_length", 2048))
        tool_response_truncate_side = str(self.settings.get("tool_response_truncate_side", "middle"))
        # Do not JSON-encode here.  For SearchR1 alignment the scheduler may
        # pass the native VERL raw_chat as a JSON *array string*; double
        # encoding it turns `[{"role": ...}]` into the literal user message
        # `"[{\"role\": ...}]"`, adding a fixed prompt offset and collapsing
        # standalone's system/user messages into one user message.
        instruction_literal = repr(str(instruction))
        command = "\n".join(
            [
                "set -euo pipefail",
                "mkdir -p \"${ARTIFACTS_DIR:-.}\"",
                "export POLAR_SEARCH_INSTRUCTION_FILE=\"${ARTIFACTS_DIR:-.}/search_instruction.txt\"",
                f"python - <<'PY'\nimport os, pathlib\npathlib.Path(os.environ['POLAR_SEARCH_INSTRUCTION_FILE']).write_text({instruction_literal}, encoding='utf-8')\nPY",
                "python -m verl_polar_bridge.search_agent.driver "
                "--instruction-file \"$POLAR_SEARCH_INSTRUCTION_FILE\" "
                "--output-file \"${ARTIFACTS_DIR:-.}/search_agent_output.json\" "
                f"--retrieval-url {shlex.quote(retrieval_url)} "
                f"--model {shlex.quote(model)} "
                f"--max-turns {max_turns} "
                f"--max-tokens {max_tokens} "
                f"--prompt-length {prompt_length} "
                f"--max-model-len {max_model_len} "
                f"--temperature {temperature} "
                f"--top-p {top_p} "
                f"--top-k {top_k} "
                f"--repetition-penalty {repetition_penalty} "
                f"--do-sample {do_sample} "
                f"{topk_arg}"
                f"--max-tool-response-length {max_tool_response_length} "
                f"--tool-response-truncate-side {shlex.quote(tool_response_truncate_side)}",
            ]
        )
        step_env = {
            "SEARCH_RETRIEVAL_URL": retrieval_url,
            "SEARCH_TEMPERATURE": str(temperature),
            "SEARCH_TOP_P": str(top_p),
            "SEARCH_TOP_K": str(top_k),
            "SEARCH_REPETITION_PENALTY": str(repetition_penalty),
            "SEARCH_DO_SAMPLE": do_sample,
            "POLAR_SEARCH_TOOL_CONFIG_PATH": tool_config_path,
            "STANDALONE_TOOL_CONFIG_PATH": tool_config_path,
            "SEARCH_MAX_MODEL_LEN": str(max_model_len),
            "POLAR_SEARCH_BRIDGE_MAX_TOKENS": bridge_max_tokens,
        }
        if str(topk_value).strip():
            step_env["SEARCH_TOPK"] = str(int(topk_value))
        if env_flag("POLAR_SEARCH_HARNESS_DEBUG", default=env_flag("POLAR_SEARCH_ROLLOUT_DEBUG", default=False)):
            debug_print(
                "POLAR_SEARCH_HARNESS_DEBUG",
                {
                    "event": "run_steps",
                    "settings": dict(self.settings),
                    "agent_env": dict(self.env),
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": do_sample,
                    "command_sampling": {
                        "temperature_arg": f"--temperature {temperature}",
                        "top_p_arg": f"--top-p {top_p}",
                        "top_k_arg": f"--top-k {top_k}",
                        "repetition_penalty_arg": f"--repetition-penalty {repetition_penalty}",
                        "do_sample_arg": f"--do-sample {do_sample}",
                    },
                    "step_env": step_env,
                },
                stream="stderr",
            )
        return [
            ExecInput(
                command=command,
                env=step_env,
            )
        ]


def _env_value(mapping: dict[str, str], name: str, default: object) -> object:
    if name in mapping:
        return mapping[name]
    return os.environ.get(name, default)


def _bool_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"0", "false", "no", "off"}:
        return "false"
    if text in {"1", "true", "yes", "on"}:
        return "true"
    return "true"
