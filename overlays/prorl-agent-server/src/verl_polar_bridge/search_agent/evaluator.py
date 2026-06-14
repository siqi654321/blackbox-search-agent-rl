"""SearchR1 evaluator for Polar trajectories.

Keep the scoring logic aligned with the SearchR1 reward function used by the
standalone VERL baseline.  The Polar-specific part of this file is limited to
finding the trajectory text/metadata and wrapping the score into EvalResult.
"""

from __future__ import annotations

import json
import os
import random
import sys
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory
from verl_polar_bridge.debug_utils import messages_summary, token_preview

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class SearchR1Evaluator(BaseTrajectoryEvaluator):
    """Score SearchR1 answers using standalone-compatible logic."""

    def __init__(self, reward_key: str = "score", **kwargs: Any) -> None:
        self.reward_key = reward_key
        self.kwargs = kwargs

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        metadata = _task_metadata(trajectory)
        ground_truth = _ground_truth(metadata)
        prediction_text, prediction_source = _prediction_text_with_source(trajectory, runtime)
        artifact_metadata = _artifact_metadata(runtime)
        prediction = extract_solution(prediction_text)
        score = compute_score(prediction_text, ground_truth) if ground_truth is not None else 0.0
        if _env_flag("POLAR_SEARCH_REWARD_DEBUG", default=False):
            _debug_reward({
                "event": "reward",
                "score": float(score),
                "uid": metadata.get("uid"),
                "source_uid": metadata.get("source_uid") or metadata.get("uid"),
                "task_id": metadata.get("task_id"),
                "session_id": metadata.get("session_id"),
                "rollout_id": metadata.get("rollout_id"),
                "rollout_step": metadata.get("rollout_step"),
                "global_steps": metadata.get("global_steps"),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "prediction_source": prediction_source,
                "has_answer": prediction is not None,
                "open_answer_tags": (prediction_text or "").count("<answer>"),
                "close_answer_tags": (prediction_text or "").count("</answer>"),
                "prediction_text_len_chars": len(prediction_text or ""),
                "prediction_text_tail": (prediction_text or "")[-600:],
                "trace_count": len(trajectory.traces or []),
                "trace_debug": [
                    {
                        "idx": idx,
                        "prompt_ids": token_preview(getattr(trace, "prompt_ids", []) or []),
                        "response_ids": token_preview(getattr(trace, "response_ids", []) or []),
                        "loss_tokens": sum(int(v) for v in (getattr(trace, "loss_mask", []) or [])),
                        "logprob_len": len(getattr(trace, "response_logprobs", None) or []),
                        "finish_reason": getattr(trace, "finish_reason", None),
                        "prompt_messages": messages_summary(getattr(trace, "prompt_messages", []) or []),
                        "response_messages": messages_summary(getattr(trace, "response_messages", []) or []),
                        "metadata_keys": sorted(str(k) for k in ((getattr(trace, "metadata", {}) or {}).keys())),
                    }
                    for idx, trace in enumerate((trajectory.traces or [])[:4])
                ],
                "task_metadata_keys": sorted(str(k) for k in metadata.keys()),
            })
        return EvalResult(
            outcome_reward=float(score),
            trace_rewards=[float(score) for _ in trajectory.traces] if trajectory.traces else None,
            metadata={
                self.reward_key: float(score),
                "ground_truth": ground_truth,
                "prediction": prediction,
                **artifact_metadata,
        },
    )


def _artifact_metadata(runtime: dict[str, Any]) -> dict[str, Any]:
    artifacts_dir = runtime.get("artifacts_dir")
    if not artifacts_dir:
        return {}
    path = Path(str(artifacts_dir)) / "search_agent_output.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    timing = data.get("timing")
    if isinstance(timing, dict):
        out["driver_timing"] = {
            str(key): float(value)
            for key, value in timing.items()
            if isinstance(value, (int, float))
        }
    transcript = data.get("transcript")
    if isinstance(transcript, list):
        assistant_turns = [item for item in transcript if isinstance(item, dict) and "assistant" in item]
        tool_turns = [item for item in transcript if isinstance(item, dict) and "tool" in item]
        out["driver_turns"] = len(assistant_turns)
        out["driver_tool_turns"] = len(tool_turns)
    if "response_budget_used" in data:
        try:
            out["driver_response_budget_used"] = float(data["response_budget_used"])
        except (TypeError, ValueError):
            pass
    if "cumulative_completion_tokens" in data:
        try:
            out["driver_cumulative_completion_tokens"] = float(data["cumulative_completion_tokens"])
        except (TypeError, ValueError):
            pass
    return out


def _task_metadata(trajectory: Trajectory) -> dict[str, Any]:
    meta = dict(trajectory.metadata or {})
    task_meta = meta.get("task_metadata")
    if isinstance(task_meta, dict):
        return task_meta
    return meta


def _ground_truth(metadata: dict[str, Any]) -> Any:
    rm = metadata.get("reward_model")
    if isinstance(rm, str):
        try:
            rm = json.loads(rm)
        except Exception:
            rm = None
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if gt is not None:
            return gt
    extra = metadata.get("extra_info")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except Exception:
            extra = None
    if isinstance(extra, dict):
        return extra.get("ground_truth")
    return metadata.get("ground_truth")


def _prediction_text_with_source(trajectory: Trajectory, runtime: dict[str, Any]) -> tuple[str, str]:
    # Baseline VERL reward runs after AgentLoopOutput has been truncated to
    # ``response_length`` and decodes those token ids with
    # ``skip_special_tokens=True``.  If Polar scores the full harness artifact,
    # an answer emitted after the 2048-token training window can incorrectly
    # receive reward even though standalone would score the clipped prefix.
    # Decode Polar's native response_ids prefix for reward alignment.  This is
    # decode-only; token ids/logprobs still come from native SGLang output and
    # are never reconstructed from text.
    if _env_flag("POLAR_SEARCH_REWARD_SCORE_TRUNCATED", default=True):
        decoded = _decode_truncated_response_text(trajectory)
        if decoded:
            return decoded, "trajectory_response_ids_truncated"

    artifacts_dir = runtime.get("artifacts_dir")
    if artifacts_dir:
        path = Path(str(artifacts_dir)) / "search_agent_output.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("final"):
                    return str(data["final"]), "artifact_final"
            except Exception:
                pass
    texts: list[str] = []
    for trace in trajectory.traces:
        for msg in trace.response_messages or []:
            content = msg.get("content") if isinstance(msg, dict) else None
            if content:
                texts.append(str(content))
    return "\n".join(texts), "trajectory_response_messages"


def _prediction_text(trajectory: Trajectory, runtime: dict[str, Any]) -> str:
    return _prediction_text_with_source(trajectory, runtime)[0]


_TOKENIZER_CACHE: Any = None
_TOKENIZER_CACHE_PATH: str | None = None


def _decode_truncated_response_text(trajectory: Trajectory) -> str:
    tokenizer = _reward_tokenizer()
    if tokenizer is None:
        return ""
    max_tokens = _env_int("POLAR_SEARCH_REWARD_MAX_RESPONSE_TOKENS", _env_int("SEARCH_MAX_TOKENS", 2048))
    texts: list[str] = []
    for trace in trajectory.traces or []:
        ids = list(getattr(trace, "response_ids", []) or [])
        if not ids:
            continue
        try:
            texts.append(tokenizer.decode(ids[:max_tokens], skip_special_tokens=True))
        except Exception:
            continue
    return "\n".join(text for text in texts if text)


def _reward_tokenizer() -> Any:
    global _TOKENIZER_CACHE, _TOKENIZER_CACHE_PATH
    path = (
        os.environ.get("POLAR_SEARCH_REWARD_TOKENIZER_PATH")
        or os.environ.get("SEARCH_REWARD_TOKENIZER_PATH")
        or os.environ.get("MODEL_PATH")
        or os.environ.get("TOKENIZER_PATH")
    )
    if not path:
        return None
    if _TOKENIZER_CACHE is not None and _TOKENIZER_CACHE_PATH == path:
        return _TOKENIZER_CACHE
    try:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _TOKENIZER_CACHE_PATH = path
        return _TOKENIZER_CACHE
    except Exception as exc:
        if _env_flag("POLAR_SEARCH_REWARD_DEBUG", default=False):
            _debug_reward({"event": "tokenizer_load_failed", "path": path, "error": repr(exc)})
        return None


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _debug_reward(payload: dict[str, Any]) -> None:
    print("POLAR_SEARCH_REWARD_DEBUG " + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


# The functions below mirror the SearchR1 QA exact-match scorer,
# which compare_standalone_vs_polar.sh copies into VERL for the standalone run.

def normalize_answer(s: Any) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _normalize_for_long_answer(text: str) -> str:
    """Normalization for long answers (less aggressive than EM normalization)."""
    text = _safe_str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _simple_tokenize(text: str, max_tokens: int = 2048) -> list[str]:
    """A lightweight tokenizer that works for both EN and CJK."""
    text = _normalize_for_long_answer(text)
    if not text:
        return []

    if " " in text:
        toks = [t for t in text.split(" ") if t]
    else:
        toks = [c for c in text if not c.isspace()]
    return toks[:max_tokens]


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Compute LCS length with O(min(n,m)) memory."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(pred: str, ref: str, max_tokens: int = 2048) -> float:
    """ROUGE-L F1 on token sequences."""
    p = _simple_tokenize(pred, max_tokens=max_tokens)
    r = _simple_tokenize(ref, max_tokens=max_tokens)
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    prec = lcs / max(len(p), 1)
    rec = lcs / max(len(r), 1)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def _token_f1(pred: str, ref: str, max_tokens: int = 4096) -> float:
    """Token-level F1 on multisets (bag-of-tokens)."""
    p = _simple_tokenize(pred, max_tokens=max_tokens)
    r = _simple_tokenize(ref, max_tokens=max_tokens)
    if not p or not r:
        return 0.0

    pc = Counter(p)
    rc = Counter(r)
    common = pc & rc
    tp = sum(common.values())
    if tp == 0:
        return 0.0
    prec = tp / max(len(p), 1)
    rec = tp / max(len(r), 1)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def _ngram_repetition_ratio(tokens: list[str], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    uniq = len(set(ngrams))
    return 1.0 - (uniq / len(ngrams))


def _apply_anti_gibberish_penalty(score: float, pred: str) -> float:
    """Penalize obvious degeneration for long answers."""
    pred = _normalize_for_long_answer(pred)
    toks = _simple_tokenize(pred, max_tokens=8192)
    if not toks:
        return 0.0

    rep4 = _ngram_repetition_ratio(toks, n=4)
    rep_penalty = max(0.0, rep4 - 0.2)
    score = score * (1.0 - min(0.7, rep_penalty))

    if len(toks) > 3000:
        score = score * 0.85
    if len(toks) > 6000:
        score = score * 0.7

    return float(max(0.0, min(1.0, score)))


def em_check(prediction: Any, golden_answers: Any) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction: Any, golden_answers: Any) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str: str) -> str | None:
    """Extract the answer from the solution string."""
    match = re.finditer(_ANSWER_RE, solution_str or "")
    matches = list(match)
    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()


def count_answer_tags(text: str) -> tuple[int, int]:
    opening_tags = (text or "").count("<answer>")
    closing_tags = (text or "").count("</answer>")
    return opening_tags, closing_tags


def compute_score(
    solution_str: str,
    ground_truth: Any,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Search-R1-like scoring function.

    This intentionally follows the baseline reward implementation used by
    compare_standalone_vs_polar.sh.  It defaults to EM on <answer>...</answer>
    and supports the long-answer metrics used by that baseline file.
    """
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target")
        metric = ground_truth.get("metric")
    else:
        targets = ground_truth
        metric = None

    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str or "")
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Metric: {metric}")
        print(f"Golden answers: {targets}")
        print(f"Solution str: {repr(solution_str)}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")

    if answer is None:
        return 0.0

    if open_count > 10 or close_count > 10:
        score = score / 4

    if metric in {"rouge_l", "rouge-l", "rouge_l_f1", "rouge"}:
        refs: Iterable[str]
        if isinstance(targets, str):
            refs = [targets]
        elif isinstance(targets, list):
            refs = [str(x) for x in targets]
        else:
            refs = [str(targets)]

        best = 0.0
        for ref in refs:
            best = max(best, _rouge_l_f1(answer, ref))
        best = _apply_anti_gibberish_penalty(best, answer)
        return float(best * score)

    if metric in {"token_f1", "f1", "bag_f1"}:
        if isinstance(targets, str):
            refs = [targets]
        elif isinstance(targets, list):
            refs = [str(x) for x in targets]
        else:
            refs = [str(targets)]
        best = 0.0
        for ref in refs:
            best = max(best, _token_f1(answer, ref))
        best = _apply_anti_gibberish_penalty(best, answer)
        return float(best * score)

    if isinstance(targets, dict) and "target" in targets:
        targets = targets["target"]

    if em_check(answer, targets):
        return float(score)
    return float(format_score)


def compute_score_subem(
    solution_str: str,
    ground_truth: Any,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Substring exact-match score, matching the baseline helper."""
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        if isinstance(ground_truth, dict):
            print(f"Golden answers: {ground_truth.get('target')}")
        else:
            print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0.0
    targets = ground_truth.get("target") if isinstance(ground_truth, dict) else ground_truth
    if subem_check(answer, targets):
        return float(score)
    return float(format_score)
