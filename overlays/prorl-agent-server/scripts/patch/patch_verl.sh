#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${VERL_ROOT:-../verl}}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_FILE="${ROOT}/verl/trainer/ppo/ray_trainer.py"

if [[ ! -d "${ROOT}" ]]; then
  echo "VERL checkout not found: ${ROOT}" >&2
  exit 1
fi
if [[ ! -f "${TRAINER_FILE}" ]]; then
  echo "VERL trainer file not found: ${TRAINER_FILE}" >&2
  exit 1
fi

required_markers=(
  "_verl_polar_update_weights_with_hooks"
  "_verl_polar_reduce_rollout_metrics"
  "metrics.update(polar_rollout_metrics)"
  "_verl_polar_expand_batch_by_source_uid"
  "_verl_polar_prepare_fanout_training_batch"
)

count_markers() {
  local file="$1"
  shift
  local count=0
  local marker
  for marker in "$@"; do
    if grep -q "${marker}" "${file}"; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

apply_minimal_polar_patch() {
  echo "Applying minimal VERL Polar dynamic-history/fanout patch to ${TRAINER_FILE}"
  TRAINER_FILE="${TRAINER_FILE}" python3 - <<'PY_PATCH'
import os
from pathlib import Path

path = Path(os.environ["TRAINER_FILE"])
s = path.read_text()

def ensure_import(s: str, line: str) -> str:
    if line in s:
        return s
    lines = s.splitlines()
    insert_at = 0
    for i, current in enumerate(lines):
        if current.startswith("import ") or current.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, line)
    return "\n".join(lines) + ("\n" if s.endswith("\n") else "")

s = ensure_import(s, "import os")
s = ensure_import(s, "import numpy as np")
s = ensure_import(s, "from collections import defaultdict")
s = s.replace("from typing import Optional", "from typing import Any, Optional")
if "from typing import Any, Optional" not in s and "from typing import" in s:
    # Best-effort for newer typing import layouts.
    s = s.replace("from typing import ", "from typing import Any, ", 1)

# Weight sync hook: replace all trainer-to-rollout update calls with the Polar hook.
s = s.replace(
    "self.checkpoint_manager.update_weights(self.global_steps)",
    "_verl_polar_update_weights_with_hooks(self, self.global_steps)",
)

# Preserve stable source_uid before VERL overwrites uid with PPO grouping ids.
old_uid = '''                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
'''
new_uid = '''                # add uid to batch
                # Preserve the dataset uid as source_uid before assigning the
                # per-rollout UUID used by PPO advantage grouping.  Polar keeps
                # source_uid for dynamic-history alignment/provenance.
                if "source_uid" not in batch.non_tensor_batch:
                    existing_uid = batch.non_tensor_batch.get("uid")
                    if existing_uid is not None:
                        batch.non_tensor_batch["source_uid"] = np.array(
                            [str(x) for x in existing_uid], dtype=object
                        )
                    else:
                        batch.non_tensor_batch["source_uid"] = np.array(
                            [str(i) for i in range(len(batch.batch))], dtype=object
                        )
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
'''
if "Preserve the dataset uid as source_uid" not in s and old_uid in s:
    s = s.replace(old_uid, new_uid, 1)

# Rollout metrics + dynamic-history alignment + fixed-DataProto fanout.
metrics_anchor = '''                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
'''
metrics_insert = metrics_anchor + '''                        polar_rollout_metrics = _verl_polar_reduce_rollout_metrics(gen_batch_output)
                        _verl_polar_debug_rollout_metrics(gen_batch_output, polar_rollout_metrics)
                        metrics.update(polar_rollout_metrics)

                        if gen_batch_output.meta_info.get("polar_already_aligned_batch", False):
                            batch = gen_batch_output
                            gen_batch_output = None
                        elif gen_batch_output.meta_info.get("polar_dynamic_history", False):
                            batch = _verl_polar_expand_batch_by_source_uid(batch, gen_batch_output)
                            try:
                                polar_dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
                            except Exception:
                                polar_dp_size = 1
                            if _verl_polar_env_flag("POLAR_FANOUT_TRAINING", "1"):
                                batch, fanout_metrics = _verl_polar_prepare_fanout_training_batch(
                                    batch,
                                    dp_size=polar_dp_size,
                                    ppo_mini_batch_size=_verl_polar_actor_ppo_mini_batch_size(self),
                                )
                                metrics.update(fanout_metrics)
                            gen_batch_output = None
'''
if "polar_rollout_metrics = _verl_polar_reduce_rollout_metrics(gen_batch_output)" not in s:
    if metrics_anchor not in s:
        raise SystemExit("Cannot find generate_sequences timing anchor in ray_trainer.py")
    s = s.replace(metrics_anchor, metrics_insert, 1)
else:
    # If an older/full patch is present, collapse its dynamic-history branch to
    # the minimal fixed-DataProto prompt_grounded_single/fanout branch.
    branch_start = '                        elif gen_batch_output.meta_info.get("polar_dynamic_history", False):'
    branch_end_marker = '\n\n                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:'
    branch_start_idx = s.find(branch_start)
    if branch_start_idx >= 0:
        branch_end_idx = s.find(branch_end_marker, branch_start_idx)
        if branch_end_idx >= 0:
            branch = '''                        elif gen_batch_output.meta_info.get("polar_dynamic_history", False):
                            batch = _verl_polar_expand_batch_by_source_uid(batch, gen_batch_output)
                            try:
                                polar_dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
                            except Exception:
                                polar_dp_size = 1
                            if _verl_polar_env_flag("POLAR_FANOUT_TRAINING", "1"):
                                batch, fanout_metrics = _verl_polar_prepare_fanout_training_batch(
                                    batch,
                                    dp_size=polar_dp_size,
                                    ppo_mini_batch_size=_verl_polar_actor_ppo_mini_batch_size(self),
                                )
                                metrics.update(fanout_metrics)
                            gen_batch_output = None
'''
            s = s[:branch_start_idx] + branch + s[branch_end_idx:]

old_union = '''                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
'''
new_union = '''                    # repeat to align with repeated responses in rollout
                    if gen_batch_output is not None:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
'''
if old_union in s:
    s = s.replace(old_union, new_union, 1)

helpers = r'''


def _verl_polar_reduce_rollout_metrics(gen_batch_output):
    """Extract optional Polar rollout metrics from AgentLoopManager output."""
    meta_info = getattr(gen_batch_output, "meta_info", {}) or {}
    metrics = meta_info.get("metrics")
    if not metrics:
        metrics = meta_info.get("polar_metrics") or meta_info.get("polar_scheduler_stats")
    if not metrics:
        return {}
    if isinstance(metrics, list):
        merged = defaultdict(list)
        for item in metrics:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                merged[key].append(value)
        return reduce_metrics(merged) if merged else {}
    if isinstance(metrics, dict):
        if any(isinstance(value, (list, tuple, np.ndarray)) for value in metrics.values()):
            return reduce_metrics(dict(metrics))
        return dict(metrics)
    return {}


def _verl_polar_debug_rollout_metrics(gen_batch_output, reduced_metrics):
    """Optional one-line diagnostics for Polar rollout metric plumbing."""
    if str(os.environ.get("POLAR_TRAINER_METRICS_DEBUG", "0")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    meta_info = getattr(gen_batch_output, "meta_info", {}) or {}
    polar_keys = sorted(key for key in reduced_metrics if str(key).startswith("polar/"))
    print(
        "POLAR_TRAINER_METRICS_DEBUG "
        f"meta_keys={sorted(meta_info.keys())} "
        f"raw_metrics_type={type(meta_info.get('metrics')).__name__} "
        f"polar_metrics_type={type(meta_info.get('polar_metrics')).__name__} "
        f"reduced_count={len(reduced_metrics)} "
        f"polar_key_count={len(polar_keys)} "
        f"polar_keys={polar_keys[:80]}",
        flush=True,
    )


def _verl_polar_update_weights_with_hooks(trainer, global_steps):
    manager = getattr(trainer, "async_rollout_manager", None)
    if hasattr(manager, "prepare_policy_update"):
        manager.prepare_policy_update(global_steps)
    try:
        result = trainer.checkpoint_manager.update_weights(global_steps)
    except BaseException:
        if hasattr(manager, "abort_policy_update"):
            manager.abort_policy_update(global_steps)
        elif hasattr(manager, "finish_policy_update"):
            manager.finish_policy_update(global_steps)
        raise
    if hasattr(manager, "update_policy_version"):
        manager.update_policy_version(global_steps)
    if hasattr(manager, "finish_policy_update"):
        manager.finish_policy_update(global_steps)
    return result


def _verl_polar_env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _verl_polar_lcm(a: int, b: int) -> int:
    import math

    a = max(1, int(a or 1))
    b = max(1, int(b or 1))
    return abs(a * b) // math.gcd(a, b)


def _verl_polar_actor_ppo_mini_batch_size(trainer) -> int:
    """Return the fixed-DataProto PPO mini-batch divisor."""
    try:
        actor_cfg = trainer.config.actor_rollout_ref.actor
        return max(1, int(actor_cfg.ppo_mini_batch_size))
    except Exception:
        return 1


def _verl_polar_prepare_fanout_training_batch(batch, *, dp_size: int, ppo_mini_batch_size: int):
    """Keep Polar dynamic-history/fanout rows for actual PPO update."""
    prefix = "polar/fanout_training"
    input_samples = len(batch)
    divisor = _verl_polar_lcm(dp_size, ppo_mini_batch_size)
    out = batch
    pad_size = 0
    if input_samples > 0 and input_samples % divisor != 0:
        out, pad_size = pad_dataproto_to_divisor(batch, divisor)
    out.meta_info["polar_fanout_training_enabled"] = True
    out.meta_info["polar_fanout_training_input_samples"] = int(input_samples)
    out.meta_info["polar_fanout_training_output_samples"] = int(len(out))
    out.meta_info["polar_fanout_training_pad_size"] = int(pad_size)
    return out, {
        f"{prefix}/enabled": 1.0,
        f"{prefix}/input_samples": float(input_samples),
        f"{prefix}/output_samples": float(len(out)),
        f"{prefix}/dp_size": float(max(1, int(dp_size or 1))),
        f"{prefix}/ppo_mini_batch_size": float(max(1, int(ppo_mini_batch_size or 1))),
        f"{prefix}/divisor": float(divisor),
        f"{prefix}/pad_size": float(pad_size),
        f"{prefix}/pad_fraction": float(pad_size / max(len(out), 1)),
        f"{prefix}/no_prune": 1.0,
    }


def _verl_polar_expand_batch_by_source_uid(batch, gen_batch_output):
    import numpy as np
    import torch
    from verl.protocol import DataProto

    source_uids = gen_batch_output.non_tensor_batch.get("source_uid")
    if source_uids is None:
        source_uids = gen_batch_output.non_tensor_batch.get("uid")
    if source_uids is None:
        raise ValueError("Polar dynamic-history rollout output requires non_tensor_batch['source_uid'] or ['uid']")

    uid_to_index = {}
    base_source_uids = batch.non_tensor_batch.get("source_uid")
    if base_source_uids is not None:
        uid_to_index.update({str(uid): i for i, uid in enumerate(base_source_uids)})
    base_uids = batch.non_tensor_batch.get("uid")
    if base_uids is not None:
        uid_to_index.update({str(uid): i for i, uid in enumerate(base_uids)})
    if not uid_to_index:
        raise ValueError("Polar dynamic-history alignment requires original batch non_tensor_batch['source_uid'] or ['uid']")

    indices = []
    for uid in source_uids:
        key = str(uid)
        if key not in uid_to_index:
            raise ValueError(f"Polar dynamic-history source_uid {key!r} not found in original batch uid/source_uid")
        indices.append(uid_to_index[key])

    device = batch.batch.device if batch.batch is not None else None
    torch_indices = torch.tensor(indices, dtype=torch.long, device=device)
    np_indices = np.asarray(indices, dtype=np.int64)
    expanded_batch = batch.batch[torch_indices] if batch.batch is not None else None
    expanded_non_tensors = {key: value[np_indices] for key, value in batch.non_tensor_batch.items()}
    if "source_uid" in expanded_non_tensors and "source_uid" in gen_batch_output.non_tensor_batch:
        expanded_non_tensors.pop("source_uid", None)
    expanded_meta_info = batch.meta_info.copy()
    for key in ("metrics", "polar_metrics", "polar_scheduler_stats"):
        if key in gen_batch_output.meta_info and key not in expanded_meta_info:
            expanded_meta_info[key] = gen_batch_output.meta_info[key]
    expanded = DataProto(batch=expanded_batch, non_tensor_batch=expanded_non_tensors, meta_info=expanded_meta_info)
    return expanded.union(gen_batch_output)
'''

if "def _verl_polar_expand_batch_by_source_uid" not in s:
    s = s.rstrip() + helpers + "\n"
else:
    # Ensure fanout helpers exist on older minimal patches.
    anchor = "\ndef _verl_polar_expand_batch_by_source_uid"
    needed = []
    if "def _verl_polar_env_flag" not in s:
        needed.append(helpers[helpers.index("\ndef _verl_polar_env_flag"):helpers.index("\ndef _verl_polar_expand_batch_by_source_uid")])
    elif "def _verl_polar_prepare_fanout_training_batch" not in s:
        needed.append(helpers[helpers.index("\ndef _verl_polar_lcm"):helpers.index("\ndef _verl_polar_expand_batch_by_source_uid")])
    if needed:
        pos = s.find(anchor)
        s = s[:pos] + "".join(needed) + s[pos:]

# Upgrade old expand helper variants: source_uid-safe union and metrics-only meta propagation.
old_uid_map = '''    base_uids = batch.non_tensor_batch.get("uid")
    if base_uids is None:
        raise ValueError("Polar dynamic-history alignment requires original batch non_tensor_batch['uid']")
    uid_to_index = {str(uid): i for i, uid in enumerate(base_uids)}
'''
new_uid_map = '''    uid_to_index = {}
    base_source_uids = batch.non_tensor_batch.get("source_uid")
    if base_source_uids is not None:
        uid_to_index.update({str(uid): i for i, uid in enumerate(base_source_uids)})
    base_uids = batch.non_tensor_batch.get("uid")
    if base_uids is not None:
        uid_to_index.update({str(uid): i for i, uid in enumerate(base_uids)})
    if not uid_to_index:
        raise ValueError("Polar dynamic-history alignment requires original batch non_tensor_batch['source_uid'] or ['uid']")
'''
s = s.replace(old_uid_map, new_uid_map)
old_expand_union = '''    expanded_batch = batch.batch[torch_indices] if batch.batch is not None else None
    expanded_non_tensors = {key: value[np_indices] for key, value in batch.non_tensor_batch.items()}
    expanded = DataProto(batch=expanded_batch, non_tensor_batch=expanded_non_tensors, meta_info=batch.meta_info.copy())
    return expanded.union(gen_batch_output)
'''
new_expand_union = '''    expanded_batch = batch.batch[torch_indices] if batch.batch is not None else None
    expanded_non_tensors = {key: value[np_indices] for key, value in batch.non_tensor_batch.items()}
    if "source_uid" in expanded_non_tensors and "source_uid" in gen_batch_output.non_tensor_batch:
        expanded_non_tensors.pop("source_uid", None)
    expanded_meta_info = batch.meta_info.copy()
    for key in ("metrics", "polar_metrics", "polar_scheduler_stats"):
        if key in gen_batch_output.meta_info and key not in expanded_meta_info:
            expanded_meta_info[key] = gen_batch_output.meta_info[key]
    expanded = DataProto(batch=expanded_batch, non_tensor_batch=expanded_non_tensors, meta_info=expanded_meta_info)
    return expanded.union(gen_batch_output)
'''
s = s.replace(old_expand_union, new_expand_union)
# Remove packed payload key from older full-patch meta propagation if present.
s = s.replace('        "polar_packed_variable_train_payload",\n', '')

path.write_text(s)
PY_PATCH

  PYTHONPYCACHEPREFIX=/tmp/pro_rl_pycache python3 -m py_compile "${TRAINER_FILE}"
}

marker_count="$(count_markers "${TRAINER_FILE}" "${required_markers[@]}")"
apply_minimal_polar_patch >/dev/null
if [[ "${marker_count}" -eq "${#required_markers[@]}" ]]; then
  echo "VERL Polar patch already appears to be fully applied to ${ROOT}"
else
  echo "Applied VERL Polar patch to ${ROOT}"
fi
