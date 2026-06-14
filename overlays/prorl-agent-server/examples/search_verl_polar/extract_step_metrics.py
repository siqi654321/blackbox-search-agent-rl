#!/usr/bin/env python3
"""Extract final VERL step metrics from console logs.

The VERL trainer prints one long line per step in the form:

    step:1 - key:value - key:value ...

This helper parses the last step line and emits either JSON or TSV so shell
scripts can compare standalone VERL against VERL+Polar without depending on
WandB or TensorBoard.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"step:(?P<step>\d+)\s+-\s+(?P<body>.*)")
PAIR_RE = re.compile(r"(?P<key>[A-Za-z0-9_./-]+):(?P<value>[-+0-9.eE]+)")


DEFAULT_KEYS = [
    "polar/submitted_tasks",
    "polar/accepted_samples",
    "polar/reward_mean",
    "polar/reward_std",
    "polar/staleness/mean",
    "polar/driver/driver_total_s_mean",
    "polar/driver/completion_s_mean",
    "polar/driver/completion_overhead_s_mean",
    "polar/driver/prompt_probe_s_mean",
    "polar/driver/prompt_probe_bridge_total_s_mean",
    "polar/driver/prompt_probe_prompt_render_s_mean",
    "polar/driver/tool_token_probe_s_mean",
    "polar/driver/tool_token_probe_bridge_total_s_mean",
    "polar/driver/tool_token_probe_prompt_render_s_mean",
    "polar/driver/retrieval_s_mean",
    "polar/driver/bridge_total_s_mean",
    "polar/driver/upstream_generate_s_mean",
    "polar/driver/bridge_overhead_s_mean",
    "polar/driver/prompt_render_s_mean",
    "polar/driver/response_json_s_mean",
    "polar/driver/extract_logprobs_s_mean",
    "polar/driver/decode_text_s_mean",
    "polar/driver/logprob_content_s_mean",
    "polar/driver/prompt_tokens_mean",
    "polar/driver/completion_tokens_mean",
    "polar/driver/meta_output_token_logprobs_len_mean",
    "polar/driver/scheduled_max_new_tokens_mean",
    "polar/driver/bridge_schedule_enabled_mean",
    "polar/driver/non_completion_overhead_s_mean",
    "polar/driver/completion_count_mean",
    "polar/driver/prompt_probe_count_mean",
    "polar/driver/tool_token_probe_count_mean",
    "polar/driver/retrieval_count_mean",
    "polar/driver/driver_turns_mean",
    "polar/driver/driver_tool_turns_mean",
    "critic/score/mean",
    "critic/score/max",
    "critic/score/min",
    "critic/rewards/mean",
    "response_length/mean",
    "response_length/max",
    "response_length/min",
    "prompt_length/mean",
    "training/rollout_actor_probs_pearson_corr",
    "training/rollout_probs_diff_mean",
    "training/rollout_probs_diff_std",
    "training/rollout_probs_diff_max",
    "rollout_corr/log_ppl_abs_diff",
    "rollout_corr/kl",
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/lr",
    "perf/throughput",
    "perf/time_per_step",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "training/global_step",
]


def parse_last_step(log_path: Path) -> dict[str, Any]:
    last_match: re.Match[str] | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        match = STEP_RE.search(line)
        if match:
            last_match = match
    if last_match is None:
        raise SystemExit(f"No 'step:<n> - ...' metrics line found in {log_path}")

    metrics: dict[str, Any] = {"step": int(last_match.group("step"))}
    body = last_match.group("body")
    for match in PAIR_RE.finditer(body):
        value = float(match.group("value"))
        if math.isfinite(value) and value.is_integer():
            # Keep integers compact for readability; downstream JSON consumers
            # still see a number.
            value = int(value)
        metrics[match.group("key")] = value
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="VERL console log containing step metrics")
    parser.add_argument("--format", choices=["json", "tsv"], default="json")
    parser.add_argument(
        "--keys",
        default=",".join(DEFAULT_KEYS),
        help="Comma-separated keys for TSV output. JSON output always includes all parsed keys.",
    )
    parser.add_argument("--label", default="", help="Optional label column for TSV output")
    parser.add_argument("--header", action="store_true", help="Emit TSV header")
    args = parser.parse_args()

    metrics = parse_last_step(args.log)

    if args.format == "json":
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2))
        return

    keys = [key for key in args.keys.split(",") if key]
    columns = (["label"] if args.label else []) + ["step"] + keys
    if args.header:
        print("\t".join(columns))
    row: list[str] = []
    if args.label:
        row.append(args.label)
    row.append(str(metrics.get("step", "")))
    for key in keys:
        row.append(str(metrics.get(key, "")))
    print("\t".join(row))


if __name__ == "__main__":
    main()
