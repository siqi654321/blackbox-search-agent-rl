#!/usr/bin/env python3
"""Normalize SearchR1-like VERL data for VERL + Polar.

This script is intentionally schema-tolerant. Use --prompt-key/--answer-key
when your parquet uses project-specific column names.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path)
        return
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _same_path(input_path: Path, output_path: Path) -> bool:
    """Return True when input and output name the same filesystem target.

    Compare by samefile when both paths exist, and fall back to normalized
    absolute paths so we still catch the common dangerous case where the output
    is an existing training parquet specified through a different relative form.
    """
    try:
        return input_path.exists() and output_path.exists() and input_path.samefile(output_path)
    except OSError:
        pass
    return os.path.abspath(os.fspath(input_path)) == os.path.abspath(os.fspath(output_path))


def _first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if _is_present(value):
            return _jsonable(value)
    return default


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    return True


def _jsonable(value: Any) -> Any:
    # pandas may return numpy arrays/scalars for parquet object columns. Convert
    # them before writing json/parquet so downstream template rendering sees
    # normal Python objects.
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return [_jsonable(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt-key", default=None)
    ap.add_argument("--answer-key", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="Optional row limit for tiny smoke/validation datasets")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if _same_path(in_path, out_path):
        raise SystemExit(
            "Refusing to overwrite input data: --input and --output resolve to "
            f"the same path ({out_path}). Use a separate prepared-data output, "
            "for example $LOG_DIR/train_polar_compare.parquet."
        )
    rows = _load_rows(in_path)
    out = []
    prompt_keys = [k for k in [args.prompt_key, "raw_prompt", "prompt", "question", "input"] if k]
    answer_keys = [k for k in [args.answer_key, "answer", "ground_truth", "target", "label"] if k]
    if args.max_rows is not None:
        if args.max_rows < 0:
            raise SystemExit("--max-rows must be non-negative")
        rows = rows[: args.max_rows]
    for i, row in enumerate(rows):
        uid = str(_first_present(row, ["uid", "source_uid", "id", "qid"], None) or uuid.uuid5(uuid.NAMESPACE_URL, f"search-{i}"))
        prompt = _first_present(row, prompt_keys, "")
        raw_reward_model = _jsonable(row.get("reward_model"))
        reward_model = raw_reward_model if isinstance(raw_reward_model, dict) else {}
        answer = _first_present(row, answer_keys, None)
        if answer is not None and "ground_truth" not in reward_model:
            reward_model = {**reward_model, "ground_truth": answer, "answer": answer}
        raw_extra_info = _jsonable(row.get("extra_info"))
        extra_info = raw_extra_info if isinstance(raw_extra_info, dict) else {}
        data_source = str(_first_present(row, ["data_source"], "searchR1_nq") or "searchR1_nq")
        if not data_source.startswith("searchR1_"):
            data_source = "searchR1_" + data_source
        question = _first_present(row, ["question", "input"], None)
        if question is None and isinstance(prompt, list):
            for message in reversed(prompt):
                if isinstance(message, dict) and message.get("role") == "user":
                    question = message.get("content")
                    break
        if question is None and isinstance(prompt, str):
            question = prompt
        tools_kwargs = extra_info.get("tools_kwargs")
        if not isinstance(tools_kwargs, dict):
            tools_kwargs = {
                "search": {
                    "create_kwargs": {
                        "ground_truth": reward_model.get("ground_truth"),
                        "question": question,
                        "data_source": data_source,
                    }
                }
            }
            extra_info["tools_kwargs"] = tools_kwargs
        extra_info.setdefault("need_tools_kwargs", True)
        extra_info.setdefault("question", question)
        extra_info.setdefault("index", i)
        for key, value in row.items():
            if key not in {"prompt", "raw_prompt", "reward_model", "extra_info"} and key not in reward_model:
                extra_info.setdefault(key, _jsonable(value))
        out.append({
            "uid": uid,
            "source_uid": uid,
            "data_source": data_source,
            "prompt": prompt,
            "raw_prompt": prompt,
            "reward_model": reward_model,
            "extra_info": extra_info,
        })
    _write_rows(out_path, out)
    print(f"wrote {len(out)} rows to {out_path}")
    if out:
        print(json.dumps(out[0], ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
