#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${VERL_ROOT:-../verl}}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_FILE="${ROOT}/verl/trainer/ppo/ray_trainer.py"
WORKER_LOSSES_FILE="${ROOT}/verl/workers/utils/losses.py"

if [[ ! -d "${ROOT}" ]]; then
  echo "VERL checkout not found: ${ROOT}" >&2
  exit 1
fi
if [[ ! -f "${TRAINER_FILE}" ]]; then
  echo "VERL trainer file not found: ${TRAINER_FILE}" >&2
  exit 1
fi
if [[ ! -f "${WORKER_LOSSES_FILE}" ]]; then
  echo "VERL worker losses file not found: ${WORKER_LOSSES_FILE}" >&2
  exit 1
fi

required_markers=(
  "_verl_polar_update_weights_with_hooks"
  "_verl_polar_reduce_rollout_metrics"
  "metrics.update(polar_rollout_metrics)"
  "_verl_polar_expand_batch_by_source_uid"
  "_verl_polar_prepare_fanout_training_batch"
)
worker_required_markers=(
  "polar_packed_variable"
  "_polar_shift_nested_rows"
  "_polar_flatten_if_nested"
  "POLAR_PACKED_VARIABLE_PPO_LOSS_ERROR"
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


# Packed-variable actor update hook.  Keep the heavy implementation in
# verl_polar_bridge.verl_packed_patch so launch-time patching only has to insert
# a tiny stable call site.  Use a regex because local/remote VERL patches may
# differ in comments around the temperature assignment.
if (
    "return packed_variable_actor_update(self, packed_variable_payload)" not in s
    and "return _verl_polar_packed_variable_actor_update(self, packed_variable_payload)" not in s
):
    import re

    update_actor_pattern = re.compile(
        r'(        batch\.meta_info\["temperature"\] = [^\n]+\n)'
        r'(        # update actor\n)'
    )
    update_actor_replacement = (
        r'\1'
        '        packed_variable_payload = batch.meta_info.get("polar_packed_variable_train_payload")\n'
        '        if (\n'
        '            self.use_legacy_worker_impl == "disable"\n'
        '            and _verl_polar_env_flag("POLAR_PACKED_VARIABLE_ACTOR_UPDATE")\n'
        '            and packed_variable_payload is not None\n'
        '        ):\n'
        '            from verl_polar_bridge.verl_packed_patch import packed_variable_actor_update\n'
        '\n'
        '            return packed_variable_actor_update(self, packed_variable_payload)\n'
        r'\2'
    )
    s, update_actor_replacements = update_actor_pattern.subn(update_actor_replacement, s, count=1)
    if update_actor_replacements == 0:
        print("WARN: could not find _update_actor temperature anchor for packed-variable actor update hook", flush=True)

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
                        if _verl_polar_env_flag("POLAR_PACKED_VARIABLE_ACTOR_DRY_RUN"):
                            packed_variable_payload = gen_batch_output.meta_info.get("polar_packed_variable_train_payload")
                            if packed_variable_payload is not None:
                                from verl_polar_bridge.verl_packed_patch import compute_packed_variable_actor_dry_run

                                metrics.update(compute_packed_variable_actor_dry_run(self, packed_variable_payload))
                            else:
                                metrics["polar/packed_variable_actor/enabled"] = 1.0
                                metrics["polar/packed_variable_actor/valid"] = 0.0
                                metrics["polar/packed_variable_actor/missing_payload"] = 1.0

                        if gen_batch_output.meta_info.get("polar_already_aligned_batch", False):
                            batch = gen_batch_output
                            gen_batch_output = None
                        elif gen_batch_output.meta_info.get("polar_dynamic_history", False):
                            if "polar_packed_variable_train_payload" in gen_batch_output.meta_info:
                                batch.meta_info["polar_packed_variable_train_payload"] = gen_batch_output.meta_info["polar_packed_variable_train_payload"]
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
    # Patch/update existing rollout metrics block with packed-variable dry-run.
    dry_run_snippet = '''                        if _verl_polar_env_flag("POLAR_PACKED_VARIABLE_ACTOR_DRY_RUN"):
                            packed_variable_payload = gen_batch_output.meta_info.get("polar_packed_variable_train_payload")
                            if packed_variable_payload is not None:
                                from verl_polar_bridge.verl_packed_patch import compute_packed_variable_actor_dry_run

                                metrics.update(compute_packed_variable_actor_dry_run(self, packed_variable_payload))
                            else:
                                metrics["polar/packed_variable_actor/enabled"] = 1.0
                                metrics["polar/packed_variable_actor/valid"] = 0.0
                                metrics["polar/packed_variable_actor/missing_payload"] = 1.0

'''
    if "POLAR_PACKED_VARIABLE_ACTOR_DRY_RUN" not in s:
        anchor = '                        metrics.update(polar_rollout_metrics)\n'
        if anchor in s:
            s = s.replace(anchor, anchor + dry_run_snippet, 1)

    branch_start = '                        elif gen_batch_output.meta_info.get("polar_dynamic_history", False):'
    branch_end_marker = '\n\n                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:'
    branch_start_idx = s.find(branch_start)
    if branch_start_idx >= 0 and "polar_packed_variable_train_payload" not in s[branch_start_idx:branch_start_idx + 500]:
        branch_end_idx = s.find(branch_end_marker, branch_start_idx)
        if branch_end_idx >= 0:
            branch = '''                        elif gen_batch_output.meta_info.get("polar_dynamic_history", False):
                            if "polar_packed_variable_train_payload" in gen_batch_output.meta_info:
                                batch.meta_info["polar_packed_variable_train_payload"] = gen_batch_output.meta_info["polar_packed_variable_train_payload"]
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

# Packed-variable actor update runs old/ref/rollout-correction inside the
# packed update hook.  Map its internal timing breakdown back to VERL's
# standard timing_raw buckets before compute_timing_metrics() logs wandb keys.
actor_metrics_anchor = '''                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
'''
actor_metrics_replacement = actor_metrics_anchor + '''                        _verl_polar_align_packed_update_timing(timing_raw, actor_output_metrics)
'''
if "_verl_polar_align_packed_update_timing(timing_raw, actor_output_metrics)" not in s:
    if actor_metrics_anchor in s:
        s = s.replace(actor_metrics_anchor, actor_metrics_replacement, 1)
    else:
        print("WARN: could not find actor_output_metrics anchor for packed timing alignment", flush=True)

# Packed-variable actor update consumes its own padding-free payload in
# _update_actor().  The fixed padded DataProto path should not recompute
# old/ref log-probs, rollout-correction, fixed advantages, critic values, or
# fixed-data metrics when POLAR_PACKED_VARIABLE_ACTOR_UPDATE=1.  Besides saving
# time, this avoids requiring fixed batch tensors that are irrelevant to the
# packed update.
if "packed_variable_update_enabled = (" not in s:
    old_mode_select = '''                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
'''
    new_mode_select = '''                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    packed_variable_update_enabled = (
                        _verl_polar_env_flag("POLAR_PACKED_VARIABLE_ACTOR_UPDATE")
                        and batch.meta_info.get("polar_packed_variable_train_payload") is not None
                    )
                    if packed_variable_update_enabled:
                        metrics["polar/packed_variable_update/skip_fixed_old_log_prob"] = 1.0
                        metrics["polar/packed_variable_update/skip_fixed_ref_log_prob"] = 1.0
                    elif bypass_recomputing_logprobs:  # Use `rollout_log_probs`
'''
    if old_mode_select in s:
        s = s.replace(old_mode_select, new_mode_select, 1)
    else:
        print("WARN: could not find rollout-correction mode-selection anchor for packed update skip", flush=True)

    old_assert = '''                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
'''
    new_assert = '''                    if not packed_variable_update_enabled:
                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
'''
    if old_assert in s:
        s = s.replace(old_assert, new_assert, 1)

    old_ref = '''                    if self.use_reference_policy:
                        # compute reference log_prob
'''
    new_ref = '''                    if self.use_reference_policy and not packed_variable_update_enabled:
                        # compute reference log_prob
'''
    if old_ref in s:
        s = s.replace(old_ref, new_ref, 1)

    old_values = '''                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
'''
    new_values = '''                    if self.use_critic and not packed_variable_update_enabled:
                        with marked_timer("values", timing_raw, color="cyan"):
'''
    if old_values in s:
        s = s.replace(old_values, new_values, 1)

    old_kl_reward = '''                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
'''
    new_kl_reward = '''                        if packed_variable_update_enabled and self.config.algorithm.use_kl_in_reward:
                            metrics["polar/packed_variable_update/skip_kl_in_reward"] = 1.0
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        elif self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
'''
    if old_kl_reward in s:
        s = s.replace(old_kl_reward, new_kl_reward, 1)

    old_rollout_corr_guard = '''                            and not bypass_recomputing_logprobs
                        ):
'''
    new_rollout_corr_guard = '''                            and not bypass_recomputing_logprobs
                            and not packed_variable_update_enabled
                        ):
'''
    if old_rollout_corr_guard in s:
        s = s.replace(old_rollout_corr_guard, new_rollout_corr_guard, 1)
    else:
        old_rollout_corr_guard_with_comment = '''                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
'''
        new_rollout_corr_guard_with_comment = '''                            and not bypass_recomputing_logprobs
                            and not packed_variable_update_enabled  # Only in decoupled fixed DataProto mode
                        ):
'''
        if old_rollout_corr_guard_with_comment in s:
            s = s.replace(old_rollout_corr_guard_with_comment, new_rollout_corr_guard_with_comment, 1)

    old_adv = '''                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
'''
    new_adv = '''                        if packed_variable_update_enabled:
                            metrics["polar/packed_variable_update/skip_fixed_advantage"] = 1.0
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )
'''
    if old_adv in s:
        s = s.replace(old_adv, new_adv, 1)

    old_update_critic = '''                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
'''
    new_update_critic = '''                    # update critic
                    if self.use_critic and not packed_variable_update_enabled:
                        with marked_timer("update_critic", timing_raw, color="pink"):
'''
    if old_update_critic in s:
        s = s.replace(old_update_critic, new_update_critic, 1)

    old_data_metrics = '''                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
'''
    new_data_metrics = '''                packed_variable_metrics_only = (
                    _verl_polar_env_flag("POLAR_PACKED_VARIABLE_ACTOR_UPDATE")
                    and batch.meta_info.get("polar_packed_variable_train_payload") is not None
                    and (batch.batch is None or "advantages" not in batch.batch)
                )
                metrics_batch = _verl_polar_metrics_batch_without_fanout_padding(batch, metrics)
                if packed_variable_metrics_only:
                    # Packed update emits packed-native score/reward/length metrics
                    # from verl_polar_bridge.packed_actor.  Do not compute fixed
                    # padded partial metrics here; keep this block to avoid native
                    # compute_data_metrics requiring fixed advantages/returns.
                    metrics["polar/packed_variable_update/skip_fixed_data_metrics"] = 1.0
                    metrics["polar/packed_variable_update/packed_native_data_metrics_expected"] = 1.0
                    metrics["polar/packed_variable_update/skip_fixed_variance_metrics"] = 1.0
                else:
                    metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=metrics_batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=metrics_batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                if not packed_variable_metrics_only:
                    metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))
'''
    if old_data_metrics in s:
        s = s.replace(old_data_metrics, new_data_metrics, 1)

    # Remove the older fixed-batch partial metrics variant if it has already
    # been applied.  Packed update now emits score/reward/length metrics from
    # the packed-native sample path instead.
    old_partial_block = '''                if packed_variable_metrics_only:
                    # Keep fixed-batch scalar/length metrics comparable with the
                    # original pad true-long path even though packed update skips
                    # fixed advantages/returns.  This emits critic/rewards/*,
                    # critic/score/*, response_length/*, prompt_length/*, etc.
                    metrics.update(
                        _verl_polar_compute_fixed_data_metrics_without_advantages(
                            batch=metrics_batch,
                            use_critic=self.use_critic,
                        )
                    )
                    metrics["polar/packed_variable_update/skip_fixed_data_metrics"] = 0.0
                    metrics["polar/packed_variable_update/fixed_data_metrics_partial"] = 1.0
                    metrics["polar/packed_variable_update/skip_fixed_variance_metrics"] = 1.0
                else:
                    metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
'''
    new_native_expected_block = '''                if packed_variable_metrics_only:
                    # Packed update emits packed-native score/reward/length metrics
                    # from verl_polar_bridge.packed_actor.  Do not compute fixed
                    # padded partial metrics here; keep this block to avoid native
                    # compute_data_metrics requiring fixed advantages/returns.
                    metrics["polar/packed_variable_update/skip_fixed_data_metrics"] = 1.0
                    metrics["polar/packed_variable_update/packed_native_data_metrics_expected"] = 1.0
                    metrics["polar/packed_variable_update/skip_fixed_variance_metrics"] = 1.0
                else:
                    metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
'''
    if old_partial_block in s:
        s = s.replace(old_partial_block, new_native_expected_block, 1)
    partial_helper = "\ndef _verl_polar_compute_fixed_data_metrics_without_advantages("
    partial_start = s.find(partial_helper)
    if partial_start >= 0:
        partial_end = s.find("\ndef _verl_polar_metrics_batch_without_fanout_padding(", partial_start)
        if partial_end >= 0:
            s = s[:partial_start] + s[partial_end:]

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


def _verl_polar_align_packed_update_timing(timing_raw, actor_output_metrics):
    """Map packed-update timers back to standard VERL timing buckets."""
    prefix = "polar/packed_variable_update/timing_s/"
    if f"{prefix}update_worker" not in actor_output_metrics:
        return

    def _metric(name: str, default: float = 0.0) -> float:
        try:
            return float(actor_output_metrics.get(f"{prefix}{name}", default) or 0.0)
        except (TypeError, ValueError):
            return float(default)

    # Match the fixed DataProto pad path timing semantics:
    # - old_log_prob/ref are standalone pre-update forward passes;
    # - rollout correction is part of the driver-side adv stage;
    # - update_actor is only the worker PPO update, not the whole packed hook.
    timing_raw["old_log_prob"] = _metric("old_log_prob")
    timing_raw["ref"] = _metric("ref_log_prob")
    timing_raw["adv"] = _metric("adv") + _metric("rollout_correction")
    timing_raw["update_actor"] = _metric("update_worker", float(timing_raw.get("update_actor", 0.0) or 0.0))


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
    neutralize_pad_rows = _verl_polar_env_flag("POLAR_FANOUT_NEUTRALIZE_PAD_ROWS", "0")
    if input_samples > 0 and input_samples % divisor != 0:
        out, pad_size = pad_dataproto_to_divisor(batch, divisor)
        if neutralize_pad_rows:
            _verl_polar_neutralize_fanout_padding_rows(out, input_samples=input_samples, pad_size=pad_size)
    out.meta_info["polar_fanout_training_enabled"] = True
    out.meta_info["polar_fanout_training_input_samples"] = int(input_samples)
    out.meta_info["polar_fanout_training_output_samples"] = int(len(out))
    out.meta_info["polar_fanout_training_pad_size"] = int(pad_size)
    out.meta_info["polar_fanout_training_pad_rows_neutralized"] = bool(pad_size > 0 and neutralize_pad_rows)
    return out, {
        f"{prefix}/enabled": 1.0,
        f"{prefix}/input_samples": float(input_samples),
        f"{prefix}/output_samples": float(len(out)),
        f"{prefix}/dp_size": float(max(1, int(dp_size or 1))),
        f"{prefix}/ppo_mini_batch_size": float(max(1, int(ppo_mini_batch_size or 1))),
        f"{prefix}/divisor": float(divisor),
        f"{prefix}/pad_size": float(pad_size),
        f"{prefix}/pad_fraction": float(pad_size / max(len(out), 1)),
        # Default keeps historical pad-path semantics: pad rows are duplicated
        # real rows and can affect loss/metrics.  Set POLAR_FANOUT_NEUTRALIZE_PAD_ROWS=1
        # only for the future fixed behavior.
        f"{prefix}/pad_rows_neutralized": float(bool(pad_size > 0 and neutralize_pad_rows)),
        f"{prefix}/pad_rows_affect_loss": float(bool(pad_size > 0 and not neutralize_pad_rows)),
        f"{prefix}/no_prune": 1.0,
    }


def _verl_polar_neutralize_fanout_padding_rows(batch, *, input_samples: int, pad_size: int):
    """Make duplicated fanout pad rows zero-loss and group-isolated.

    ``pad_dataproto_to_divisor`` appends duplicates from the front.  If left as
    normal samples, those rows can change GRPO group mean/std and contribute
    duplicate PPO loss.  Keep the rows only as divisibility placeholders:
    - unique uid/source_uid so they do not join a real GRPO group;
    - zero response_mask/rm_scores/rollout_log_probs so rewards, advantages,
      rollout correction, and PPO loss are exactly zero on pad rows.
    """
    if int(pad_size or 0) <= 0:
        return batch
    import numpy as _np

    start = int(input_samples)
    end = start + int(pad_size)
    if getattr(batch, "batch", None) is not None:
        for key in ("response_mask", "rm_scores", "rollout_log_probs", "token_level_scores", "token_level_rewards", "advantages", "returns", "rollout_is_weights"):
            if key in batch.batch:
                batch.batch[key][start:end].zero_()
    for key in ("uid", "source_uid"):
        values = batch.non_tensor_batch.get(key)
        if values is None:
            continue
        values = values.copy()
        for i in range(start, min(end, len(values))):
            values[i] = f"__polar_fanout_pad__{i}"
        batch.non_tensor_batch[key] = values.astype(object) if isinstance(values, _np.ndarray) else values
    return batch


def _verl_polar_metrics_batch_without_fanout_padding(batch, metrics):
    """Optionally return an unpadded view for logging-only metrics.

    Default intentionally keeps historical pad-path semantics so packed-update
    comparison is against the original pad implementation.  The pad-row issue is
    tracked by metrics and can be fixed later with an explicit env switch.
    """
    try:
        pad_size = int((getattr(batch, "meta_info", {}) or {}).get("polar_fanout_training_pad_size", 0) or 0)
    except Exception:
        pad_size = 0
    exclude = _verl_polar_env_flag("POLAR_FANOUT_EXCLUDE_PAD_ROWS_FROM_METRICS", "0")
    if exclude and pad_size > 0 and len(batch) > pad_size:
        metrics["polar/fanout_training/pad_rows_excluded_from_metrics"] = 1.0
        return unpad_dataproto(batch, pad_size)
    metrics["polar/fanout_training/pad_rows_excluded_from_metrics"] = 0.0
    return batch

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
    for key in ("polar_packed_variable_train_payload", "metrics", "polar_metrics", "polar_scheduler_stats"):
        if key in gen_batch_output.meta_info and key not in expanded_meta_info:
            expanded_meta_info[key] = gen_batch_output.meta_info[key]
    expanded = DataProto(batch=expanded_batch, non_tensor_batch=expanded_non_tensors, meta_info=expanded_meta_info)
    return expanded.union(gen_batch_output)
'''

if "def _verl_polar_expand_batch_by_source_uid" not in s:
    s = s.rstrip() + helpers + "\n"
elif "POLAR_PACKED_VARIABLE_ACTOR_UPDATE" not in s:
    # Ensure fanout helpers exist on older minimal patches.  Full packed-variable
    # trainer patches carry their own helpers and must not be downgraded here.
    anchor = "\ndef _verl_polar_expand_batch_by_source_uid"
    needed = []
    if "def _verl_polar_env_flag" not in s:
        needed.append(helpers[helpers.index("\ndef _verl_polar_env_flag"):helpers.index("\ndef _verl_polar_expand_batch_by_source_uid")])
    elif "def _verl_polar_prepare_fanout_training_batch" not in s:
        needed.append(helpers[helpers.index("\ndef _verl_polar_lcm"):helpers.index("\ndef _verl_polar_expand_batch_by_source_uid")])
    if needed:
        pos = s.find(anchor)
        s = s[:pos] + "".join(needed) + s[pos:]

# Upgrade existing trainer helper variants with packed timing alignment.
timing_align_helper = r'''

def _verl_polar_align_packed_update_timing(timing_raw, actor_output_metrics):
    """Map packed-update timers back to standard VERL timing buckets."""
    prefix = "polar/packed_variable_update/timing_s/"
    if f"{prefix}update_worker" not in actor_output_metrics:
        return

    def _metric(name: str, default: float = 0.0) -> float:
        try:
            return float(actor_output_metrics.get(f"{prefix}{name}", default) or 0.0)
        except (TypeError, ValueError):
            return float(default)

    # Match the fixed DataProto pad path timing semantics:
    # - old_log_prob/ref are standalone pre-update forward passes;
    # - rollout correction is part of the driver-side adv stage;
    # - update_actor is only the worker PPO update, not the whole packed hook.
    timing_raw["old_log_prob"] = _metric("old_log_prob")
    timing_raw["ref"] = _metric("ref_log_prob")
    timing_raw["adv"] = _metric("adv") + _metric("rollout_correction")
    timing_raw["update_actor"] = _metric("update_worker", float(timing_raw.get("update_actor", 0.0) or 0.0))
'''
if "def _verl_polar_align_packed_update_timing" not in s:
    anchor = "\ndef _verl_polar_update_weights_with_hooks"
    pos = s.find(anchor)
    if pos >= 0:
        s = s[:pos] + timing_align_helper + s[pos:]

# Upgrade existing fanout helper variants with optional pad-row neutralization.
# Default remains the historical pad path: pad_dataproto_to_divisor duplicates
# real rows, so true-long comparisons still match the original implementation.
# Set POLAR_FANOUT_NEUTRALIZE_PAD_ROWS=1 later to fix the recorded pad-row issue.
neutralize_helper = r'''

def _verl_polar_neutralize_fanout_padding_rows(batch, *, input_samples: int, pad_size: int):
    """Make duplicated fanout pad rows zero-loss and group-isolated.

    ``pad_dataproto_to_divisor`` appends duplicates from the front.  If left as
    normal samples, those rows can change GRPO group mean/std and contribute
    duplicate PPO loss.  Keep the rows only as divisibility placeholders:
    - unique uid/source_uid so they do not join a real GRPO group;
    - zero response_mask/rm_scores/rollout_log_probs so rewards, advantages,
      rollout correction, and PPO loss are exactly zero on pad rows.
    """
    if int(pad_size or 0) <= 0:
        return batch
    import numpy as _np

    start = int(input_samples)
    end = start + int(pad_size)
    if getattr(batch, "batch", None) is not None:
        for key in ("response_mask", "rm_scores", "rollout_log_probs", "token_level_scores", "token_level_rewards", "advantages", "returns", "rollout_is_weights"):
            if key in batch.batch:
                batch.batch[key][start:end].zero_()
    for key in ("uid", "source_uid"):
        values = batch.non_tensor_batch.get(key)
        if values is None:
            continue
        values = values.copy()
        for i in range(start, min(end, len(values))):
            values[i] = f"__polar_fanout_pad__{i}"
        batch.non_tensor_batch[key] = values.astype(object) if isinstance(values, _np.ndarray) else values
    return batch
'''
metrics_helper = r'''

def _verl_polar_metrics_batch_without_fanout_padding(batch, metrics):
    """Optionally return an unpadded view for logging-only metrics.

    Default intentionally keeps historical pad-path semantics so packed-update
    comparison is against the original pad implementation.  The pad-row issue is
    tracked by metrics and can be fixed later with an explicit env switch.
    """
    try:
        pad_size = int((getattr(batch, "meta_info", {}) or {}).get("polar_fanout_training_pad_size", 0) or 0)
    except Exception:
        pad_size = 0
    exclude = _verl_polar_env_flag("POLAR_FANOUT_EXCLUDE_PAD_ROWS_FROM_METRICS", "0")
    if exclude and pad_size > 0 and len(batch) > pad_size:
        metrics["polar/fanout_training/pad_rows_excluded_from_metrics"] = 1.0
        return unpad_dataproto(batch, pad_size)
    metrics["polar/fanout_training/pad_rows_excluded_from_metrics"] = 0.0
    return batch
'''
if "def _verl_polar_prepare_fanout_training_batch" in s:
    old_pad_call = '''    if input_samples > 0 and input_samples % divisor != 0:
        out, pad_size = pad_dataproto_to_divisor(batch, divisor)
'''
    new_pad_call = '''    if input_samples > 0 and input_samples % divisor != 0:
        out, pad_size = pad_dataproto_to_divisor(batch, divisor)
        if _verl_polar_env_flag("POLAR_FANOUT_NEUTRALIZE_PAD_ROWS", "0"):
            _verl_polar_neutralize_fanout_padding_rows(out, input_samples=input_samples, pad_size=pad_size)
'''
    if "_verl_polar_neutralize_fanout_padding_rows(out, input_samples=input_samples, pad_size=pad_size)" not in s:
        s = s.replace(old_pad_call, new_pad_call, 1)
    if 'out.meta_info["polar_fanout_training_pad_rows_neutralized"]' not in s:
        s = s.replace(
            '    out.meta_info["polar_fanout_training_pad_size"] = int(pad_size)\n',
            '    out.meta_info["polar_fanout_training_pad_size"] = int(pad_size)\n'
            '    out.meta_info["polar_fanout_training_pad_rows_neutralized"] = bool(pad_size > 0)\n',
            1,
        )
    if 'pad_rows_neutralized' not in s:
        s = s.replace(
            '        f"{prefix}/pad_fraction": float(pad_size / max(len(out), 1)),\n',
            '        f"{prefix}/pad_fraction": float(pad_size / max(len(out), 1)),\n'
            '        f"{prefix}/pad_rows_neutralized": float(bool(pad_size > 0)),\n'
            '        f"{prefix}/pad_rows_affect_loss": 0.0,\n',
            1,
        )
    if "def _verl_polar_neutralize_fanout_padding_rows" not in s:
        anchor = "\ndef _verl_polar_expand_batch_by_source_uid"
        pos = s.find(anchor)
        if pos >= 0:
            s = s[:pos] + neutralize_helper + s[pos:]
    if "def _verl_polar_metrics_batch_without_fanout_padding" not in s:
        anchor = "\ndef _verl_polar_expand_batch_by_source_uid"
        pos = s.find(anchor)
        if pos >= 0:
            s = s[:pos] + metrics_helper + s[pos:]

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
if "POLAR_PACKED_VARIABLE_ACTOR_UPDATE" not in s:
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
    for key in ("polar_packed_variable_train_payload", "metrics", "polar_metrics", "polar_scheduler_stats"):
        if key in gen_batch_output.meta_info and key not in expanded_meta_info:
            expanded_meta_info[key] = gen_batch_output.meta_info[key]
    expanded = DataProto(batch=expanded_batch, non_tensor_batch=expanded_non_tensors, meta_info=expanded_meta_info)
    return expanded.union(gen_batch_output)
'''
if "POLAR_PACKED_VARIABLE_ACTOR_UPDATE" not in s:
    s = s.replace(old_expand_union, new_expand_union)
# Keep packed payload propagation when full packed-variable hooks are present.

path.write_text(s)
PY_PATCH

  WORKER_LOSSES_FILE="${WORKER_LOSSES_FILE}" python3 - <<'PY_LOSSES'
import os
import re
from pathlib import Path

path = Path(os.environ["WORKER_LOSSES_FILE"])
s = path.read_text()

def ensure_import(src: str, line: str) -> str:
    if line in src:
        return src
    lines = src.splitlines()
    insert_at = 0
    for i, current in enumerate(lines):
        if current.startswith("import ") or current.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, line)
    return "\n".join(lines) + ("\n" if src.endswith("\n") else "")

if "POLAR_PACKED_VARIABLE_PPO_LOSS_ERROR" not in s:
    s = ensure_import(s, "import os")
    s = ensure_import(s, "import traceback")
    s = ensure_import(s, "from verl.utils.torch_functional import masked_mean, masked_sum")

    helpers = r'''

def _polar_debug_enabled() -> bool:
    return os.getenv("POLAR_PACKED_VARIABLE_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _polar_tensor_debug(name: str, value) -> dict:
    try:
        if isinstance(value, torch.Tensor):
            out = {"name": name, "is_nested": bool(value.is_nested), "dtype": str(value.dtype), "device": str(value.device)}
            if value.is_nested:
                offsets = value.offsets().detach().cpu().tolist()
                out.update({
                    "batch": int(value.size(0)),
                    "offsets_head": offsets[:8],
                    "offsets_tail": offsets[-8:],
                    "values_shape": list(value.values().shape),
                })
            else:
                out["shape"] = list(value.shape)
            return out
        return {"name": name, "type": type(value).__name__}
    except Exception as exc:
        return {"name": name, "debug_error": f"{type(exc).__name__}: {exc}"}


def _polar_debug_print(label: str, payload: dict) -> None:
    if not _polar_debug_enabled():
        return
    try:
        print(f"POLAR_PACKED_VARIABLE_DEBUG {label} {payload}", flush=True)
    except Exception:
        pass


def _polar_shift_nested_rows(values: torch.Tensor, *, drop: str) -> torch.Tensor:
    """Shift full-sequence nested rows for packed-variable token prediction."""

    if values.is_nested:
        if drop == "first":
            rows = [row[1:] for row in values.unbind()]
        elif drop == "last":
            rows = [row[:-1] for row in values.unbind()]
        else:
            raise ValueError(f"unexpected drop={drop!r}")
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged).contiguous()
    if drop == "first":
        return values[..., 1:]
    if drop == "last":
        return values[..., :-1]
    raise ValueError(f"unexpected drop={drop!r}")


def _polar_flatten_if_nested(values):
    """Flatten packed-variable nested/jagged tensors to token vectors."""

    if values is None:
        return None
    if isinstance(values, torch.Tensor) and values.is_nested:
        return values.values().contiguous()
    return values
'''
    if "_polar_shift_nested_rows" not in s:
        marker = "\ndef ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):"
        if marker not in s:
            raise SystemExit("Cannot find ppo_loss marker in losses.py")
        s = s.replace(marker, helpers + marker, 1)

    if 'polar_packed_variable = bool(tu.get(data, "polar_packed_variable", False))' not in s:
        start_pattern = re.compile(
            r'(def ppo_loss\(config: ActorConfig, model_output, data: TensorDict, dp_group=None\):\n'
            r'    """Computes ppo loss from model output \(log_prob, entropy, values, etc\. \) and old_log_probs from data\."""\n)'
            r'    log_prob = no_padding_2_padding\(model_output\["log_probs"\], data\)\n'
            r'    entropy = model_output.get\("entropy", None\)\n'
            r'    if entropy is not None:\n'
            r'        entropy = no_padding_2_padding\(entropy, data\)\n'
        )
        start_replacement = r'''\1    polar_packed_variable = bool(tu.get(data, "polar_packed_variable", False))
    if polar_packed_variable:
        # Packed-variable tensors are full-sequence nested/jagged rows.  VERL's
        # model output at position p predicts token p+1, so shift loss-side
        # tensors before applying the native completion loss mask.
        log_prob = model_output["log_probs"]
        entropy = model_output.get("entropy", None)
        response_mask = _polar_shift_nested_rows(data["loss_mask"], drop="first").to(torch.bool)
        old_log_prob = _polar_shift_nested_rows(data["old_log_probs"], drop="first").to(torch.float32)
        advantages = _polar_shift_nested_rows(data["advantages"], drop="first").to(torch.float32)
        log_prob = _polar_shift_nested_rows(log_prob, drop="last").to(torch.float32)
        if entropy is not None:
            entropy = _polar_shift_nested_rows(entropy, drop="last").to(torch.float32)
        _polar_debug_print(
            "ppo_loss_packed_inputs",
            {
                "input_ids": _polar_tensor_debug("input_ids", data.get("input_ids")),
                "loss_mask": _polar_tensor_debug("loss_mask", data.get("loss_mask")),
                "old_log_probs": _polar_tensor_debug("old_log_probs", data.get("old_log_probs")),
                "advantages": _polar_tensor_debug("advantages", data.get("advantages")),
                "model_log_probs": _polar_tensor_debug("model_log_probs", model_output.get("log_probs")),
                "shifted_response_mask_sum": int((response_mask.values() if response_mask.is_nested else response_mask).sum().detach().item()),
            },
        )
    else:
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        entropy = model_output.get("entropy", None)
        if entropy is not None:
            entropy = no_padding_2_padding(entropy, data)
        response_mask = data["response_mask"].to(bool)
        old_log_prob = data["old_log_probs"]
        advantages = data["advantages"]
'''
        s, n = start_pattern.subn(start_replacement, s, count=1)
        if n != 1:
            raise SystemExit("Cannot patch ppo_loss start block in losses.py")

        response_block = '''    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)
'''
        response_replacement = '''    if not polar_packed_variable:
        response_mask = data["response_mask"].to(bool)
        old_log_prob = data["old_log_probs"]
        advantages = data["advantages"]
    # compute policy loss
    rollout_is_weights = data.get("rollout_is_weights", None)
'''
        if response_block in s:
            s = s.replace(response_block, response_replacement, 1)

    if "POLAR_PACKED_VARIABLE_PPO_LOSS_ERROR" not in s:
        policy_pattern = re.compile(
            r'    # compute policy loss\n'
            r'    rollout_is_weights = data.get\("rollout_is_weights", None\)\n\n'
            r'    loss_agg_mode = config.loss_agg_mode\n\n'
            r'    loss_mode = config.policy_loss.get\("loss_mode", "vanilla"\)\n\n'
            r'    policy_loss_fn = get_policy_loss_fn\(loss_mode\)\n'
            r'    pg_loss, pg_metrics = policy_loss_fn\(\n'
            r'        old_log_prob=old_log_prob,\n'
            r'        log_prob=log_prob,\n'
            r'        advantages=advantages,\n'
            r'        response_mask=response_mask,\n'
            r'        loss_agg_mode=loss_agg_mode,\n'
            r'        config=config,\n'
            r'        rollout_is_weights=rollout_is_weights,\n'
            r'    \)\n'
        )
        policy_replacement = '''    # compute policy loss
    rollout_is_weights = data.get("rollout_is_weights", None)
    if polar_packed_variable and rollout_is_weights is not None:
        rollout_is_weights = _polar_shift_nested_rows(rollout_is_weights, drop="first").to(torch.float32)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    packed_loss_scale = None
    packed_legacy_loss_scale = False
    if polar_packed_variable:
        if loss_agg_mode != "token-mean":
            raise NotImplementedError("packed-variable PPO currently supports loss_agg_mode=token-mean only")
        packed_legacy_loss_scale = bool(tu.get(data, "polar_packed_legacy_loss_scale", False))
        packed_loss_scale = data.get("polar_packed_loss_scale", None)
        if packed_loss_scale is not None:
            packed_loss_scale = _polar_shift_nested_rows(packed_loss_scale, drop="first").to(torch.float32)
        log_prob = _polar_flatten_if_nested(log_prob)
        old_log_prob = _polar_flatten_if_nested(old_log_prob)
        advantages = _polar_flatten_if_nested(advantages)
        response_mask = _polar_flatten_if_nested(response_mask)
        if entropy is not None:
            entropy = _polar_flatten_if_nested(entropy)
        if rollout_is_weights is not None:
            rollout_is_weights = _polar_flatten_if_nested(rollout_is_weights)
        packed_loss_scale = _polar_flatten_if_nested(packed_loss_scale)
        if packed_loss_scale is not None:
            packed_loss_scale = packed_loss_scale.to(torch.float32)
        if packed_legacy_loss_scale:
            if packed_loss_scale is None:
                raise ValueError("packed legacy loss scale enabled but polar_packed_loss_scale is missing")
            batch_num_tokens = data["batch_num_tokens"]
            if not isinstance(batch_num_tokens, torch.Tensor):
                batch_num_tokens = torch.tensor(float(batch_num_tokens), device=response_mask.device, dtype=torch.float32)
            else:
                batch_num_tokens = batch_num_tokens.to(device=response_mask.device, dtype=torch.float32)
            global_mini_batch_size = tu.get(data, "polar_packed_global_mini_batch_size", tu.get(data, "global_batch_size", 1))
            if isinstance(global_mini_batch_size, torch.Tensor):
                global_mini_batch_size_float = float(global_mini_batch_size.detach().item())
            else:
                global_mini_batch_size_float = float(global_mini_batch_size)
            global_mini_batch_size_float = max(1.0, global_mini_batch_size_float)
            legacy_loss_weight = packed_loss_scale.detach() * (batch_num_tokens / global_mini_batch_size_float)
            if rollout_is_weights is None:
                rollout_is_weights = legacy_loss_weight
            else:
                rollout_is_weights = rollout_is_weights * legacy_loss_weight
            scale_mask = response_mask.to(torch.float32) * legacy_loss_weight
            pg_losses_unclipped_debug = -advantages
            metrics["pg_loss_token_mean_debug"] = Metric(
                value=masked_mean(pg_losses_unclipped_debug, response_mask), aggregation=AggregationType.MEAN
            )
            metrics["pg_loss_scaled_token_mean_debug"] = Metric(
                value=masked_mean(pg_losses_unclipped_debug, scale_mask), aggregation=AggregationType.MEAN
            )
            metrics["pg_loss_scale_mean_debug"] = Metric(
                value=masked_mean(legacy_loss_weight, response_mask), aggregation=AggregationType.MEAN
            )
            metrics["pg_loss_scale_sum_debug"] = Metric(
                value=masked_sum(legacy_loss_weight, response_mask), aggregation=AggregationType.SUM
            )
            metrics["adv_token_mean_debug"] = Metric(
                value=masked_mean(advantages, response_mask), aggregation=AggregationType.MEAN
            )
            metrics["adv_scaled_token_mean_debug"] = Metric(
                value=masked_mean(advantages, scale_mask), aggregation=AggregationType.MEAN
            )
            metrics["loss_scale_mask_tokens_debug"] = Metric(
                value=scale_mask.sum().to(torch.float32), aggregation=AggregationType.SUM
            )
            metrics["loss_scale_enabled_debug"] = Metric(
                value=torch.tensor(1.0, device=response_mask.device), aggregation=AggregationType.MEAN
            )
        _polar_debug_print(
            "ppo_loss_packed_flattened",
            {
                "log_prob": _polar_tensor_debug("flat_log_prob", log_prob),
                "old_log_prob": _polar_tensor_debug("flat_old_log_prob", old_log_prob),
                "advantages": _polar_tensor_debug("flat_advantages", advantages),
                "response_mask": _polar_tensor_debug("flat_response_mask", response_mask),
                "packed_loss_scale": _polar_tensor_debug("flat_packed_loss_scale", packed_loss_scale),
                "rollout_is_weights": _polar_tensor_debug("flat_rollout_is_weights", rollout_is_weights),
                "response_mask_sum": int(response_mask.sum().detach().item()),
            },
        )

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    try:
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )
    except Exception as exc:
        if polar_packed_variable:
            print(
                "POLAR_PACKED_VARIABLE_PPO_LOSS_ERROR "
                f"{type(exc).__name__}: {exc}\\\\n{traceback.format_exc()}",
                flush=True,
            )
        raise
'''
        s, n = policy_pattern.subn(policy_replacement, s, count=1)
        if n != 1:
            raise SystemExit("Cannot patch ppo_loss policy-loss block in losses.py")

    # Ensure older already-patched losses.py gets the KL debug imports too.
    if "from verl.utils.torch_functional import masked_mean, masked_sum" not in s:
        s = ensure_import(s, "from verl.utils.torch_functional import masked_mean, masked_sum")

    kl_pattern = '''    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
'''
    kl_replacement = '''    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        if polar_packed_variable:
            ref_log_prob = _polar_flatten_if_nested(_polar_shift_nested_rows(ref_log_prob, drop="first").to(torch.float32))
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        if polar_packed_variable:
            metrics["kl_loss_token_mean_debug"] = Metric(
                value=masked_mean(kld, response_mask), aggregation=AggregationType.MEAN
            )
            metrics["kl_loss_token_sum_debug"] = Metric(
                value=masked_sum(kld, response_mask), aggregation=AggregationType.SUM
            )
            metrics["kl_loss_mask_tokens_debug"] = Metric(
                value=response_mask.sum().to(torch.float32), aggregation=AggregationType.SUM
            )
            batch_num_tokens_debug = data["batch_num_tokens"]
            if not isinstance(batch_num_tokens_debug, torch.Tensor):
                batch_num_tokens_debug = torch.tensor(
                    float(batch_num_tokens_debug), device=kld.device, dtype=torch.float32
                )
            metrics["kl_loss_batch_num_tokens_debug"] = Metric(
                value=batch_num_tokens_debug.to(torch.float32), aggregation=AggregationType.MEAN
            )
            dp_size_debug = data["dp_size"]
            if not isinstance(dp_size_debug, torch.Tensor):
                dp_size_debug = torch.tensor(float(dp_size_debug), device=kld.device, dtype=torch.float32)
            metrics["kl_loss_dp_size_debug"] = Metric(
                value=dp_size_debug.to(torch.float32), aggregation=AggregationType.MEAN
            )
'''
    if kl_pattern in s:
        s = s.replace(kl_pattern, kl_replacement, 1)



# Upgrade the packed policy-loss block even when losses.py was already patched
# by an older patch_verl.sh run.  Older patched files contain the
# POLAR_PACKED_VARIABLE_PPO_LOSS_ERROR marker, so the main block above is skipped
# and the legacy per-sample loss scale would never be applied.
policy_replacement_v2 = '''    # compute policy loss
    rollout_is_weights = data.get("rollout_is_weights", None)
    if polar_packed_variable and rollout_is_weights is not None:
        rollout_is_weights = _polar_shift_nested_rows(rollout_is_weights, drop="first").to(torch.float32)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    packed_loss_scale = None
    packed_legacy_loss_scale = False
    if polar_packed_variable:
        if loss_agg_mode != "token-mean":
            raise NotImplementedError("packed-variable PPO currently supports loss_agg_mode=token-mean only")
        packed_legacy_loss_scale = bool(tu.get(data, "polar_packed_legacy_loss_scale", False))
        packed_loss_scale = data.get("polar_packed_loss_scale", None)
        if packed_loss_scale is not None:
            packed_loss_scale = _polar_shift_nested_rows(packed_loss_scale, drop="first").to(torch.float32)
        log_prob = _polar_flatten_if_nested(log_prob)
        old_log_prob = _polar_flatten_if_nested(old_log_prob)
        advantages = _polar_flatten_if_nested(advantages)
        response_mask = _polar_flatten_if_nested(response_mask)
        if entropy is not None:
            entropy = _polar_flatten_if_nested(entropy)
        if rollout_is_weights is not None:
            rollout_is_weights = _polar_flatten_if_nested(rollout_is_weights)
        packed_loss_scale = _polar_flatten_if_nested(packed_loss_scale)
        if packed_loss_scale is not None:
            packed_loss_scale = packed_loss_scale.to(torch.float32)
        if packed_legacy_loss_scale:
            if packed_loss_scale is None:
                raise ValueError("packed legacy loss scale enabled but polar_packed_loss_scale is missing")
            batch_num_tokens = data["batch_num_tokens"]
            if not isinstance(batch_num_tokens, torch.Tensor):
                batch_num_tokens = torch.tensor(float(batch_num_tokens), device=response_mask.device, dtype=torch.float32)
            else:
                batch_num_tokens = batch_num_tokens.to(device=response_mask.device, dtype=torch.float32)
            global_mini_batch_size = tu.get(data, "polar_packed_global_mini_batch_size", tu.get(data, "global_batch_size", 1))
            if isinstance(global_mini_batch_size, torch.Tensor):
                global_mini_batch_size_float = float(global_mini_batch_size.detach().item())
            else:
                global_mini_batch_size_float = float(global_mini_batch_size)
            global_mini_batch_size_float = max(1.0, global_mini_batch_size_float)
            legacy_loss_weight = packed_loss_scale.detach() * (batch_num_tokens / global_mini_batch_size_float)
            if rollout_is_weights is None:
                rollout_is_weights = legacy_loss_weight
            else:
                rollout_is_weights = rollout_is_weights * legacy_loss_weight
            scale_mask = response_mask.to(torch.float32) * legacy_loss_weight
            pg_losses_unclipped_debug = -advantages
            metrics["pg_loss_token_mean_debug"] = Metric(
                value=masked_mean(pg_losses_unclipped_debug, response_mask), aggregation=AggregationType.MEAN
            )
            metrics["pg_loss_scaled_token_mean_debug"] = Metric(
                value=masked_mean(pg_losses_unclipped_debug, scale_mask), aggregation=AggregationType.MEAN
            )
            metrics["pg_loss_scale_mean_debug"] = Metric(
                value=masked_mean(legacy_loss_weight, response_mask), aggregation=AggregationType.MEAN
            )
            metrics["pg_loss_scale_sum_debug"] = Metric(
                value=masked_sum(legacy_loss_weight, response_mask), aggregation=AggregationType.SUM
            )
            metrics["adv_token_mean_debug"] = Metric(
                value=masked_mean(advantages, response_mask), aggregation=AggregationType.MEAN
            )
            metrics["adv_scaled_token_mean_debug"] = Metric(
                value=masked_mean(advantages, scale_mask), aggregation=AggregationType.MEAN
            )
            metrics["loss_scale_mask_tokens_debug"] = Metric(
                value=scale_mask.sum().to(torch.float32), aggregation=AggregationType.SUM
            )
            metrics["loss_scale_enabled_debug"] = Metric(
                value=torch.tensor(1.0, device=response_mask.device), aggregation=AggregationType.MEAN
            )
        _polar_debug_print(
            "ppo_loss_packed_flattened",
            {
                "log_prob": _polar_tensor_debug("flat_log_prob", log_prob),
                "old_log_prob": _polar_tensor_debug("flat_old_log_prob", old_log_prob),
                "advantages": _polar_tensor_debug("flat_advantages", advantages),
                "response_mask": _polar_tensor_debug("flat_response_mask", response_mask),
                "packed_loss_scale": _polar_tensor_debug("flat_packed_loss_scale", packed_loss_scale),
                "rollout_is_weights": _polar_tensor_debug("flat_rollout_is_weights", rollout_is_weights),
                "response_mask_sum": int(response_mask.sum().detach().item()),
            },
        )

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    try:
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )
    except Exception as exc:
        if polar_packed_variable:
            print(
                "POLAR_PACKED_VARIABLE_PPO_LOSS_ERROR "
                f"{type(exc).__name__}: {exc}\\n{traceback.format_exc()}",
                flush=True,
            )
        raise
'''
if (
    "pg_loss_scaled_token_mean_debug" not in s
    or 'bool(data.get("polar_packed_legacy_loss_scale", False))' in s
    or 'data.get("polar_packed_global_mini_batch_size", data.get("global_batch_size", 1))' in s
):
    policy_block_pattern = re.compile(
        r'    # compute policy loss\n'
        r'    rollout_is_weights = data\.get\("rollout_is_weights", None\)\n'
        r'.*?'
        r'(?=\n    # AggregationType\.MEAN for pg metrics:)',
        re.S,
    )
    s, n = policy_block_pattern.subn(policy_replacement_v2, s, count=1)
    if n != 1:
        raise SystemExit("Cannot upgrade ppo_loss policy-loss block in losses.py")

# Ensure KL debug imports/metrics are present even when losses.py was patched by
# an older patch_verl.sh run before these diagnostics existed.
if "from verl.utils.torch_functional import masked_mean, masked_sum" not in s:
    s = ensure_import(s, "from verl.utils.torch_functional import masked_mean, masked_sum")
if (
    "kl_loss_token_mean_debug" not in s
    and 'kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)\n        kl_loss = agg_loss(' in s
):
    s = s.replace(
        'kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)\n        kl_loss = agg_loss(',
        r'''kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        if polar_packed_variable:
            metrics["kl_loss_token_mean_debug"] = Metric(
                value=masked_mean(kld, response_mask), aggregation=AggregationType.MEAN
            )
            metrics["kl_loss_token_sum_debug"] = Metric(
                value=masked_sum(kld, response_mask), aggregation=AggregationType.SUM
            )
            metrics["kl_loss_mask_tokens_debug"] = Metric(
                value=response_mask.sum().to(torch.float32), aggregation=AggregationType.SUM
            )
            batch_num_tokens_debug = data["batch_num_tokens"]
            if not isinstance(batch_num_tokens_debug, torch.Tensor):
                batch_num_tokens_debug = torch.tensor(
                    float(batch_num_tokens_debug), device=kld.device, dtype=torch.float32
                )
            metrics["kl_loss_batch_num_tokens_debug"] = Metric(
                value=batch_num_tokens_debug.to(torch.float32), aggregation=AggregationType.MEAN
            )
            dp_size_debug = data["dp_size"]
            if not isinstance(dp_size_debug, torch.Tensor):
                dp_size_debug = torch.tensor(float(dp_size_debug), device=kld.device, dtype=torch.float32)
            metrics["kl_loss_dp_size_debug"] = Metric(
                value=dp_size_debug.to(torch.float32), aggregation=AggregationType.MEAN
            )
        kl_loss = agg_loss(''',
        1,
    )

path.write_text(s)
PY_LOSSES

  PYTHONPYCACHEPREFIX=/tmp/pro_rl_pycache python3 -m py_compile "${TRAINER_FILE}" "${WORKER_LOSSES_FILE}"
}

marker_count="$(count_markers "${TRAINER_FILE}" "${required_markers[@]}")"
worker_marker_count="$(count_markers "${WORKER_LOSSES_FILE}" "${worker_required_markers[@]}")"
apply_minimal_polar_patch >/dev/null
if [[ "${marker_count}" -eq "${#required_markers[@]}" && "${worker_marker_count}" -eq "${#worker_required_markers[@]}" ]]; then
  echo "VERL Polar patch already appears to be fully applied to ${ROOT}"
else
  echo "Applied VERL Polar patch to ${ROOT}"
fi
