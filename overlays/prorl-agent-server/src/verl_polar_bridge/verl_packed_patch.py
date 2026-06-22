"""Small trainer-side packed-variable hooks used by ``patch_verl.sh``.

Keeping the heavier logic in this importable module lets the tracked VERL patch
stay small and makes remote launch-time patching deterministic.
"""

from __future__ import annotations

import time
from typing import Any


def compute_packed_variable_actor_dry_run(trainer: Any, payload: dict[str, Any]) -> dict[str, float]:
    """Compute packed-variable actor logprob parity without updating weights."""

    prefix = "polar/packed_variable_actor"
    metrics: dict[str, float] = {f"{prefix}/enabled": 1.0}
    try:
        if getattr(trainer, "use_legacy_worker_impl", None) != "disable":
            metrics[f"{prefix}/valid"] = 0.0
            metrics[f"{prefix}/unsupported_legacy_worker"] = 1.0
            return metrics
        from verl_polar_bridge.packed_actor import (
            actor_samples_to_nested_tensordict,
            packed_actor_logprob_parity_metrics,
            packed_variable_payload_to_actor_samples,
            partition_actor_samples_by_tokens,
        )

        samples = packed_variable_payload_to_actor_samples(payload)
        dp_rank_mapping, collect_mask, dp_size = _actor_dispatch_info(trainer)
        partitions = partition_actor_samples_by_tokens(samples, dp_size=dp_size)
        shard_samples = [[samples[idx] for idx in part] for part in partitions]
        ordered_samples = [sample for shard in shard_samples for sample in shard]
        temperature = _logprob_temperature(trainer.config.actor_rollout_ref.rollout.temperature)
        dp_shards = [
            actor_samples_to_nested_tensordict(
                shard,
                temperature=temperature,
                calculate_entropy=False,
            )
            for shard in shard_samples
        ]
        worker_group = trainer.actor_rollout_wg
        raw_method_name = _raw_worker_method(
            worker_group,
            candidates=(
                "compute_log_prob",
                "actor_rollout_compute_log_prob",
                "actor_rollout_ref_compute_log_prob",
            ),
        )
        output = _run_raw_worker_method(
            worker_group,
            raw_method_name=raw_method_name,
            dp_shards=dp_shards,
            collect_mask=collect_mask,
            dp_rank_mapping=dp_rank_mapping,
        )
        metrics[f"{prefix}/dp_size"] = float(dp_size)
        metrics[f"{prefix}/worker_count"] = float(len(dp_rank_mapping))
        metrics[f"{prefix}/num_packs"] = float(len(payload.get("packs") or []))
        metrics[f"{prefix}/num_samples"] = float(len(samples))
        metrics[f"{prefix}/padding_free"] = 1.0
        metrics.update(packed_actor_logprob_parity_metrics(ordered_samples, output["log_probs"], prefix=prefix))
    except Exception as exc:
        metrics[f"{prefix}/valid"] = 0.0
        metrics[f"{prefix}/error"] = 1.0
        print(f"POLAR_PACKED_VARIABLE_ACTOR_DRY_RUN_ERROR {type(exc).__name__}: {exc}", flush=True)
    return metrics


def packed_variable_actor_update(trainer: Any, payload: dict[str, Any]) -> Any:
    """Run a real actor PPO update from padding-free packed-variable payload."""

    prefix = "polar/packed_variable_update"
    metrics: dict[str, float] = {f"{prefix}/enabled": 1.0}
    try:
        if getattr(trainer, "use_legacy_worker_impl", None) != "disable":
            metrics[f"{prefix}/valid"] = 0.0
            metrics[f"{prefix}/unsupported_legacy_worker"] = 1.0
            return _empty_dataproto_with_metrics(metrics)

        actor_cfg = trainer.config.actor_rollout_ref.actor
        use_kl_loss = bool(getattr(actor_cfg, "use_kl_loss", False))
        metrics[f"{prefix}/use_kl_loss"] = float(use_kl_loss)

        from verl_polar_bridge.packed_actor import (
            actor_samples_to_training_tensordict,
            apply_smoke_reward_diversity,
            assign_old_log_probs_from_nested_output,
            assign_ref_log_probs_from_nested_output,
            compute_group_grpo_advantages,
            concat_training_tensordicts,
            flatten_parent_sample_groups,
            group_actor_samples_by_parent,
            pad_actor_samples_to_divisor,
            pad_parent_minibatches_to_equal_row_count,
            pad_parent_sample_groups_to_divisor,
            parent_sample_loss_denominators,
            packed_actor_data_metrics,
            packed_training_payload_metrics,
            packed_variable_payload_to_actor_samples,
            partition_actor_samples_equal_count_by_tokens,
            partition_actor_samples_equal_count_row_order,
            partition_parent_sample_groups_row_order,
        )

        samples = packed_variable_payload_to_actor_samples(payload)
        input_sample_count = len(samples)
        norm_adv_by_std = bool(trainer.config.algorithm.get("norm_adv_by_std_in_grpo", True))
        smoke_reward_diversity_enabled = _env_flag("POLAR_PACKED_VARIABLE_SMOKE_REWARD_DIVERSITY")
        metrics[f"{prefix}/smoke_reward_diversity_requested"] = float(bool(smoke_reward_diversity_enabled))
        parent_advantage_level = str(__import__("os").environ.get("POLAR_PACKED_ADVANTAGE_LEVEL", "segment")).strip().lower() in {"parent", "parent_sample", "rollout"}
        parent_sample_loss = _env_flag("POLAR_PACKED_PARENT_SAMPLE_LOSS", default=False)
        segment_weight_loss_enabled = _env_flag("POLAR_PACKED_SEGMENT_WEIGHT_LOSS", default=True) and not parent_sample_loss
        metrics[f"{prefix}/segment_weight_loss_enabled"] = float(bool(segment_weight_loss_enabled))
        metrics[f"{prefix}/parent_advantage_level"] = float(bool(parent_advantage_level))
        metrics[f"{prefix}/parent_sample_loss_enabled"] = float(bool(parent_sample_loss))
        if smoke_reward_diversity_enabled:
            original_advantages_by_sample = compute_group_grpo_advantages(
                samples, norm_by_std=norm_adv_by_std, parent_level=parent_advantage_level
            )
            original_metrics = packed_training_payload_metrics(
                samples,
                advantages_by_sample=original_advantages_by_sample,
                prefix=prefix,
            )
            metrics[f"{prefix}/smoke_reward_diversity_original_advantage_abs_mean"] = float(
                original_metrics.get(f"{prefix}/advantage_abs_mean", 0.0)
            )
            samples, diversity_metrics = apply_smoke_reward_diversity(samples, prefix=prefix)
            metrics.update(diversity_metrics)
            input_sample_count = len(samples)

        dp_rank_mapping, collect_mask, dp_size = _actor_dispatch_info(trainer)
        rollout_config = trainer.config.actor_rollout_ref.rollout
        temperature = _logprob_temperature(rollout_config.temperature)
        configured_prompt_length = int(getattr(rollout_config, "prompt_length", 0) or 0)
        configured_response_length = int(getattr(rollout_config, "response_length", 0) or 0)
        configured_mini_batch_size = int(getattr(actor_cfg, "ppo_mini_batch_size", len(samples)) or len(samples))
        configured_mini_batch_size = max(1, configured_mini_batch_size)
        rollout_n = int(getattr(rollout_config, "n", 1) or 1)
        fixed_global_mini_batch_size = configured_mini_batch_size * max(1, rollout_n)
        worker_dp_size = max(1, int(dp_size or 1))
        parent_aware_minibatch = bool(parent_sample_loss and parent_advantage_level)
        metrics[f"{prefix}/parent_aware_minibatch"] = float(bool(parent_aware_minibatch))
        if fixed_global_mini_batch_size % worker_dp_size != 0:
            raise RuntimeError(
                "packed update would desync collectives: global mini-batch size not divisible by actor DP size "
                f"global_mini_batch_size={fixed_global_mini_batch_size}, dp_size={worker_dp_size}"
            )
        local_mini_batch_size = max(1, fixed_global_mini_batch_size // worker_dp_size)

        # Match the fixed DataProto true-long update path.  In parent-sample
        # mode, pad and partition logical parent rollout slots, then flatten
        # segments only after each local parent mini-batch is selected.  This
        # prevents one subagent fanout from shifting all later PPO mini-batch
        # boundaries by one segment row.
        pad_divisor = _lcm(max(1, worker_dp_size), max(1, fixed_global_mini_batch_size))
        partition_mode = str(
            getattr(actor_cfg, "packed_variable_partition_mode", None)
            or __import__("os").environ.get("POLAR_PACKED_VARIABLE_PARTITION_MODE", "row_order")
        ).strip().lower().replace("-", "_")

        if parent_aware_minibatch:
            if partition_mode not in {"row", "row_order", "contiguous", "fixed", "pad"}:
                raise ValueError(
                    "parent-aware packed update currently requires "
                    "POLAR_PACKED_VARIABLE_PARTITION_MODE=row_order"
                )
            parent_groups_before_pad = group_actor_samples_by_parent(samples)
            parent_input_count = len(parent_groups_before_pad)
            segment_input_count = len(samples)
            parent_groups, pad_size = pad_parent_sample_groups_to_divisor(parent_groups_before_pad, divisor=pad_divisor)
            parent_shards = partition_parent_sample_groups_row_order(parent_groups, dp_size=worker_dp_size)
            shard_parent_lengths = [len(shard) for shard in parent_shards]
            if any(length % local_mini_batch_size != 0 for length in shard_parent_lengths):
                raise RuntimeError(
                    "packed parent-aware update would desync collectives: non-divisible parent shard lengths "
                    f"shard_parent_lengths={shard_parent_lengths}, local_mini_batch_size={local_mini_batch_size}, "
                    f"parent_input_count={parent_input_count}, parent_output_count={len(parent_groups)}, "
                    f"pad_size={pad_size}, divisor={pad_divisor}"
                )
            minibatch_counts = [length // local_mini_batch_size for length in shard_parent_lengths]
            if len(set(minibatch_counts)) != 1:
                raise RuntimeError(
                    "packed parent-aware update would desync collectives: unequal parent minibatch counts "
                    f"{minibatch_counts}; shard_parent_lengths={shard_parent_lengths}, "
                    f"local_mini_batch_size={local_mini_batch_size}"
                )
            shard_parent_minibatches = [
                [shard[i : i + local_mini_batch_size] for i in range(0, len(shard), local_mini_batch_size)]
                for shard in parent_shards
            ]
            shard_samples = [flatten_parent_sample_groups(shard) for shard in parent_shards]
            samples = flatten_parent_sample_groups(parent_groups)
            shard_lengths = [len(shard) for shard in shard_samples]
            normalized_partition_mode = "parent_row_order"
            metrics[f"{prefix}/parent_rows_before_pad"] = float(parent_input_count)
            metrics[f"{prefix}/parent_rows_after_pad"] = float(len(parent_groups))
            metrics[f"{prefix}/segment_rows_before_parent_pad"] = float(segment_input_count)
            metrics[f"{prefix}/segment_rows_after_parent_pad"] = float(len(samples))
            metrics[f"{prefix}/parent_pad_size"] = float(pad_size)
            metrics[f"{prefix}/max_shard_parent_samples"] = float(max(shard_parent_lengths, default=0))
            metrics[f"{prefix}/min_shard_parent_samples"] = float(min(shard_parent_lengths, default=0))
        else:
            # Segment-row fallback used by pure prompt_grounded_single and legacy packed runs.
            samples, pad_size = pad_actor_samples_to_divisor(samples, divisor=pad_divisor)
            if partition_mode in {"token", "token_balance", "token_balanced", "by_tokens"}:
                partitions = partition_actor_samples_equal_count_by_tokens(samples, dp_size=dp_size)
                normalized_partition_mode = "token_balance"
            elif partition_mode in {"row", "row_order", "contiguous", "fixed", "pad"}:
                partitions = partition_actor_samples_equal_count_row_order(samples, dp_size=dp_size)
                normalized_partition_mode = "row_order"
            else:
                raise ValueError(
                    "unsupported POLAR_PACKED_VARIABLE_PARTITION_MODE "
                    f"{partition_mode!r}; expected row_order or token_balance"
                )
            shard_samples = [[samples[idx] for idx in part] for part in partitions]
            shard_lengths = [len(shard) for shard in shard_samples]
            if len(set(shard_lengths)) != 1:
                raise RuntimeError(
                    "packed update would desync collectives: unequal shard sample counts "
                    f"{shard_lengths}; input_samples={input_sample_count}, output_samples={len(samples)}, "
                    f"pad_size={pad_size}, dp_size={dp_size}, divisor={pad_divisor}"
                )
            if any(length % local_mini_batch_size != 0 for length in shard_lengths):
                raise RuntimeError(
                    "packed update would desync collectives: non-divisible shard lengths "
                    f"shard_lengths={shard_lengths}, local_mini_batch_size={local_mini_batch_size}, "
                    f"input_samples={input_sample_count}, output_samples={len(samples)}, pad_size={pad_size}, "
                    f"divisor={pad_divisor}"
                )
            minibatch_counts = [length // local_mini_batch_size for length in shard_lengths]
            if len(set(minibatch_counts)) != 1:
                raise RuntimeError(
                    "packed update would desync collectives: unequal minibatch counts "
                    f"{minibatch_counts}; shard_lengths={shard_lengths}, local_mini_batch_size={local_mini_batch_size}"
                )
            shard_parent_minibatches = None

        ppo_mini_batch_size = int(local_mini_batch_size * worker_dp_size)
        adv_t0 = time.perf_counter()
        advantages_by_sample = compute_group_grpo_advantages(
            samples, norm_by_std=norm_adv_by_std, parent_level=parent_advantage_level
        )
        metrics[f"{prefix}/timing_s/adv"] = float(time.perf_counter() - adv_t0)
        metrics.update(packed_training_payload_metrics(samples, advantages_by_sample=advantages_by_sample, prefix=prefix))
        metrics.update(
            packed_actor_data_metrics(
                samples,
                prefix=prefix,
                max_prompt_length=configured_prompt_length,
                max_response_length=configured_response_length,
                advantages_by_sample=advantages_by_sample,
            )
        )
        metrics[f"{prefix}/input_samples"] = float(input_sample_count)
        metrics[f"{prefix}/output_samples"] = float(len(samples))
        metrics[f"{prefix}/dp_size"] = float(dp_size)
        metrics[f"{prefix}/worker_count"] = float(len(dp_rank_mapping))
        metrics[f"{prefix}/num_packs"] = float(len(payload.get("packs") or []))
        metrics[f"{prefix}/padding_free"] = 1.0
        metrics[f"{prefix}/pad_size"] = float(pad_size)
        metrics[f"{prefix}/pad_divisor"] = float(pad_divisor)
        metrics[f"{prefix}/pad_compat_original_pad_path"] = 1.0
        metrics[f"{prefix}/pad_rows_affect_loss"] = float(bool(pad_size > 0))
        metrics[f"{prefix}/partition_mode_row_order"] = float(normalized_partition_mode == "row_order")
        metrics[f"{prefix}/partition_mode_token_balance"] = float(normalized_partition_mode == "token_balance")
        metrics[f"{prefix}/equal_count_partition"] = 1.0
        metrics[f"{prefix}/max_shard_samples"] = float(max(shard_lengths, default=0))
        metrics[f"{prefix}/min_shard_samples"] = float(min(shard_lengths, default=0))
        metrics[f"{prefix}/fixed_global_mini_batch_size"] = float(fixed_global_mini_batch_size)
        metrics[f"{prefix}/ppo_mini_batch_size"] = float(ppo_mini_batch_size)
        metrics[f"{prefix}/local_mini_batch_size"] = float(local_mini_batch_size)
        metrics[f"{prefix}/minibatch_count_per_rank"] = float(minibatch_counts[0] if minibatch_counts else 0)

        legacy_loss_scale = _env_flag("POLAR_PACKED_VARIABLE_LEGACY_LOSS_SCALE", default=True)
        metrics[f"{prefix}/legacy_loss_scale_enabled"] = float(bool(legacy_loss_scale))
        if legacy_loss_scale:
            if parent_sample_loss:
                parent_denoms = parent_sample_loss_denominators(samples)
                trainable_token_counts = [max(1, int(parent_denoms.get(int(sample.sample_index), sample.trainable_tokens or 0))) for sample in samples]
            else:
                trainable_token_counts = [max(1, int(sample.trainable_tokens or 0)) for sample in samples]
            inv_token_counts = [1.0 / float(count) for count in trainable_token_counts]
            metrics[f"{prefix}/legacy_loss_scale_inv_tokens_mean"] = float(
                sum(inv_token_counts) / max(1, len(inv_token_counts))
            )
            metrics[f"{prefix}/legacy_loss_scale_inv_tokens_min"] = float(min(inv_token_counts, default=0.0))
            metrics[f"{prefix}/legacy_loss_scale_inv_tokens_max"] = float(max(inv_token_counts, default=0.0))
        if parent_aware_minibatch and shard_parent_minibatches is not None:
            update_shard_minibatch_samples, parent_minibatch_segment_rows, parent_minibatch_row_pad = (
                pad_parent_minibatches_to_equal_row_count(shard_parent_minibatches)
            )
            metrics[f"{prefix}/parent_minibatch_segment_rows"] = float(parent_minibatch_segment_rows)
            metrics[f"{prefix}/parent_minibatch_row_pad"] = float(parent_minibatch_row_pad)
            update_shard_samples = [
                [sample for minibatch in shard_minibatches for sample in minibatch]
                for shard_minibatches in update_shard_minibatch_samples
            ]
            metrics[f"{prefix}/update_row_pad_rows"] = float(
                sum(max(0, len(update_shard) - len(metric_shard)) for update_shard, metric_shard in zip(update_shard_samples, shard_samples, strict=True))
            )
            dp_shards = [
                concat_training_tensordicts(
                    [
                        actor_samples_to_training_tensordict(
                            minibatch_samples,
                            temperature=temperature,
                            advantages_by_sample=advantages_by_sample,
                            calculate_entropy=bool(float(getattr(actor_cfg, "entropy_coeff", 0.0)) != 0.0),
                            compute_loss=True,
                            legacy_loss_scale=legacy_loss_scale,
                            global_mini_batch_size=ppo_mini_batch_size,
                            parent_sample_loss=parent_sample_loss,
                        )
                        for minibatch_samples in shard_minibatches
                    ]
                )
                for shard_minibatches in update_shard_minibatch_samples
            ]
            # After equal row padding, every parent-aware mini-batch has the
            # same segment-row count, so VERL's row-based DataLoader can split
            # exactly at parent mini-batch boundaries.
            actor_update_mini_batch_sizes = [int(parent_minibatch_segment_rows * worker_dp_size) for _ in update_shard_samples]
        else:
            dp_shards = [
                actor_samples_to_training_tensordict(
                    shard,
                    temperature=temperature,
                    advantages_by_sample=advantages_by_sample,
                    calculate_entropy=bool(float(getattr(actor_cfg, "entropy_coeff", 0.0)) != 0.0),
                    compute_loss=True,
                    legacy_loss_scale=legacy_loss_scale,
                    global_mini_batch_size=ppo_mini_batch_size,
                    parent_sample_loss=parent_sample_loss,
                )
                for shard in shard_samples
            ]
            update_shard_samples = shard_samples
            actor_update_mini_batch_sizes = [int(ppo_mini_batch_size) for _ in shard_samples]
        metrics[f"{prefix}/actor_update_mini_batch_size"] = float(max(actor_update_mini_batch_sizes, default=0))
        metrics[f"{prefix}/actor_update_mini_batch_size_min"] = float(min(actor_update_mini_batch_sizes, default=0))
        metrics[f"{prefix}/actor_update_mini_batch_size_max"] = float(max(actor_update_mini_batch_sizes, default=0))

        # Match fixed padded DataProto decoupled mode: old_log_probs are
        # recomputed once from the current actor before PPO mini-batch updates,
        # rather than reusing rollout engine log-probs.
        old_log_prob_t0 = time.perf_counter()
        old_log_prob_outputs = _compute_actor_log_probs_for_shards(
            trainer,
            shard_samples=update_shard_samples,
            temperature=temperature,
            calculate_entropy=False,
        )
        old_logprob_tokens = 0
        for shard_td, old_output in zip(dp_shards, old_log_prob_outputs, strict=True):
            old_logprob_tokens += assign_old_log_probs_from_nested_output(shard_td, old_output["log_probs"])
        metrics[f"{prefix}/old_log_prob_recomputed"] = 1.0
        metrics[f"{prefix}/old_log_prob_tokens"] = float(old_logprob_tokens)
        metrics[f"{prefix}/timing_s/old_log_prob"] = float(time.perf_counter() - old_log_prob_t0)

        if use_kl_loss:
            ref_log_prob_t0 = time.perf_counter()
            ref_log_prob_outputs = _compute_ref_log_probs_for_shards(
                trainer,
                shard_samples=update_shard_samples,
                temperature=temperature,
            )
            ref_tokens = 0
            for shard_td, ref_output in zip(dp_shards, ref_log_prob_outputs, strict=True):
                ref_tokens += assign_ref_log_probs_from_nested_output(shard_td, ref_output["log_probs"])
            metrics[f"{prefix}/ref_log_prob_enabled"] = 1.0
            metrics[f"{prefix}/ref_log_prob_valid"] = 1.0
            metrics[f"{prefix}/ref_log_prob_tokens"] = float(ref_tokens)
            metrics[f"{prefix}/timing_s/ref_log_prob"] = float(time.perf_counter() - ref_log_prob_t0)
        else:
            metrics[f"{prefix}/ref_log_prob_enabled"] = 0.0
            metrics[f"{prefix}/timing_s/ref_log_prob"] = 0.0

        rollout_corr_config = trainer.config.algorithm.get("rollout_correction", None)
        if rollout_corr_config is not None and not bool(rollout_corr_config.get("bypass_mode", False)):
            # Match the fixed decoupled pad path: compute rollout IS weights and
            # optional rejection masks from recomputed actor old_log_probs vs
            # native rollout log-probs before actor update.
            rollout_corr_t0 = time.perf_counter()
            rollout_corr_metrics = []
            for shard_td in dp_shards:
                rollout_corr_metrics.append(_apply_rollout_correction_to_packed_shard(shard_td, rollout_corr_config))
            metrics.update(_reduce_metric_dicts(rollout_corr_metrics))
            metrics[f"{prefix}/rollout_correction_applied"] = 1.0
            metrics[f"{prefix}/rollout_correction_metrics_native"] = 1.0
            metrics[f"{prefix}/timing_s/rollout_correction"] = float(time.perf_counter() - rollout_corr_t0)
        else:
            metrics[f"{prefix}/rollout_correction_applied"] = 0.0
            metrics[f"{prefix}/timing_s/rollout_correction"] = 0.0

        for shard_idx, shard in enumerate(dp_shards):
            from verl.utils import tensordict_utils as tu

            tu.assign_non_tensor(
                shard,
                temperature=float(temperature),
                compute_loss=True,
                calculate_entropy=bool(float(getattr(actor_cfg, "entropy_coeff", 0.0)) != 0.0),
                polar_packed_legacy_loss_scale=bool(legacy_loss_scale),
                polar_packed_global_mini_batch_size=int(ppo_mini_batch_size),
                polar_packed_parent_sample_loss=bool(parent_sample_loss),
                polar_packed_variable=True,
                global_batch_size=float(len(samples)),
                mini_batch_size=actor_update_mini_batch_sizes[shard_idx],
                epochs=int(getattr(actor_cfg, "ppo_epochs", 1) or 1),
                seed=int(getattr(actor_cfg, "data_loader_seed", 1) or 1),
                dataloader_kwargs={"shuffle": bool(getattr(actor_cfg, "shuffle", False))},
            )

        _debug_print(
            "actor_update_shards",
            {
                "dp_size": dp_size,
                "dp_rank_mapping": dp_rank_mapping,
                "collect_mask": collect_mask,
                "shard_sample_counts": shard_lengths,
                "update_shard_sample_counts": [len(shard) for shard in update_shard_samples],
                "shard_token_counts": [sum(sample.token_length for sample in shard) for shard in shard_samples],
                "update_shard_token_counts": [sum(sample.token_length for sample in shard) for shard in update_shard_samples],
                "shard_trainable_tokens": [sum(sample.trainable_tokens for sample in shard) for shard in shard_samples],
                "update_shard_trainable_tokens": [sum(sample.trainable_tokens for sample in shard) for shard in update_shard_samples],
                "input_sample_count": input_sample_count,
                "output_sample_count": len(samples),
                "pad_size": pad_size,
                "pad_divisor": pad_divisor,
                "partition_mode": normalized_partition_mode,
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "local_mini_batch_size": local_mini_batch_size,
                "minibatch_counts": minibatch_counts,
                "actor_update_mini_batch_sizes": actor_update_mini_batch_sizes,
                "parent_aware_minibatch": parent_aware_minibatch,
                "use_kl_loss": use_kl_loss,
                "legacy_loss_scale": legacy_loss_scale,
            },
        )

        worker_group = trainer.actor_rollout_wg
        raw_method_name = _raw_worker_method(
            worker_group,
            candidates=("update_actor", "actor_rollout_update_actor", "actor_rollout_ref_update_actor"),
        )
        outputs = []
        update_worker_t0 = time.perf_counter()
        for worker, dp_rank in zip(worker_group.workers, dp_rank_mapping, strict=True):
            outputs.append(worker_group._execute_remote_single_worker(worker, raw_method_name, dp_shards[int(dp_rank)]))
        import ray

        raw_outputs = ray.get(outputs)
        metrics[f"{prefix}/timing_s/update_worker"] = float(time.perf_counter() - update_worker_t0)
        worker_metrics = _collect_worker_metrics_outputs(raw_outputs, collect_mask, dp_rank_mapping)
        _debug_print("actor_update_worker_metrics_raw", {"worker_metrics": worker_metrics})
        if worker_metrics:
            from verl.utils.metric import reduce_metrics
            from verl.utils.py_functional import rename_dict

            renamed = reduce_metrics(rename_dict(worker_metrics, "actor/"))
            renamed.update(_legacy_sum_actor_loss_metrics(worker_metrics, dp_size=dp_size))
            if "actor/mfu" in renamed:
                renamed["perf/mfu/actor"] = renamed.pop("actor/mfu")
            for key, value in renamed.items():
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    pass
        metrics[f"{prefix}/valid"] = 1.0
        metrics[f"{prefix}/updated"] = 1.0
    except Exception as exc:
        import traceback

        metrics[f"{prefix}/valid"] = 0.0
        metrics[f"{prefix}/error"] = 1.0
        print(
            "POLAR_PACKED_VARIABLE_ACTOR_UPDATE_ERROR "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            flush=True,
        )
    return _empty_dataproto_with_metrics(metrics)


def _compute_actor_log_probs_for_shards(
    trainer: Any,
    *,
    shard_samples: list[list[Any]],
    temperature: float,
    calculate_entropy: bool,
) -> list[Any]:
    from verl_polar_bridge.packed_actor import actor_samples_to_nested_tensordict

    dp_rank_mapping, collect_mask, _dp_size = _actor_dispatch_info(trainer)
    dp_shards = [
        actor_samples_to_nested_tensordict(
            shard,
            temperature=temperature,
            calculate_entropy=calculate_entropy,
        )
        for shard in shard_samples
    ]
    worker_group = trainer.actor_rollout_wg
    raw_method_name = _raw_worker_method(
        worker_group,
        candidates=(
            "compute_log_prob",
            "actor_rollout_compute_log_prob",
            "actor_rollout_ref_compute_log_prob",
        ),
    )
    return _run_raw_worker_method(
        worker_group,
        raw_method_name=raw_method_name,
        dp_shards=dp_shards,
        collect_mask=collect_mask,
        dp_rank_mapping=dp_rank_mapping,
    )


def _compute_ref_log_probs_for_shards(
    trainer: Any,
    *,
    shard_samples: list[list[Any]],
    temperature: float,
) -> list[Any]:
    from verl.utils import tensordict_utils as tu
    from verl_polar_bridge.packed_actor import actor_samples_to_nested_tensordict

    ref_in_actor = bool(getattr(trainer, "ref_in_actor", False))
    worker_group = trainer.actor_rollout_wg if ref_in_actor else trainer.ref_policy_wg
    role = "actor" if ref_in_actor else "ref"
    dp_rank_mapping, collect_mask, ref_dp_size = _dispatch_info_for_role(trainer, worker_group, role=role)
    if int(ref_dp_size) != len(shard_samples):
        raise RuntimeError(
            "packed-variable KL currently requires ref DP size to match actor DP size; "
            f"ref_dp_size={ref_dp_size}, actor_dp_size={len(shard_samples)}"
        )
    dp_shards = [
        actor_samples_to_nested_tensordict(
            shard,
            temperature=temperature,
            calculate_entropy=False,
        )
        for shard in shard_samples
    ]
    for shard in dp_shards:
        tu.assign_non_tensor(shard, compute_loss=False, calculate_entropy=False)
        if ref_in_actor:
            tu.assign_non_tensor(shard, no_lora_adapter=True)

    raw_method_name = _raw_worker_method(
        worker_group,
        candidates=(
            ("compute_log_prob", "actor_rollout_compute_log_prob", "actor_rollout_ref_compute_log_prob")
            if ref_in_actor
            else ("compute_ref_log_prob", "ref_policy_compute_ref_log_prob", "actor_rollout_ref_compute_ref_log_prob")
        ),
    )
    return _run_raw_worker_method(
        worker_group,
        raw_method_name=raw_method_name,
        dp_shards=dp_shards,
        collect_mask=collect_mask,
        dp_rank_mapping=dp_rank_mapping,
    )


def _apply_rollout_correction_to_packed_shard(shard_td: Any, rollout_corr_config: Any) -> dict[str, float]:
    import torch
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    # Keep a sample/sequence dimension for rollout-corr diagnostics.  The
    # fixed padded DataProto path calls compute_offpolicy_metrics on a 2D
    # (batch, response_len) tensor, so log_ppl_diff_min/max are per-sequence
    # extrema.  Flattening all packed trainable tokens into one vector made
    # those extrema collapse to one shard-level mean and biased min/max toward
    # the global mean (for example packed min stayed positive while pad min was
    # negative).  Pad only inside this diagnostic/correction helper; training
    # tensors stay jagged/padding-free.
    old_log_prob, rollout_log_prob, response_mask = _trainable_tokens_to_padded_rows(
        shard_td["old_log_probs"],
        shard_td["rollout_log_probs"],
        shard_td["loss_mask"],
    )
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = bool(rollout_corr_config.get("rollout_is_batch_normalize", False))
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_is_weights_proto, modified_response_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
    )
    if rollout_is_weights_proto is not None:
        weights = rollout_is_weights_proto.batch["rollout_is_weights"].to(torch.float32)[response_mask.to(torch.bool)]
        _assign_flat_trainable_values(shard_td, "rollout_is_weights", weights)
    if not bool(modified_response_mask.all().item()):
        updated_loss_mask = _flat_trainable_mask_to_nested_loss_mask(
            shard_td["loss_mask"],
            modified_response_mask.to(torch.bool)[response_mask.to(torch.bool)],
        )
        shard_td["loss_mask"] = updated_loss_mask.contiguous()
    return {str(key): float(value) for key, value in metrics.items()}


def _trainable_tokens_to_padded_rows(old_values: Any, rollout_values: Any, loss_mask: Any) -> tuple[Any, Any, Any]:
    import torch

    old_rows = list(old_values.unbind()) if isinstance(old_values, torch.Tensor) and old_values.is_nested else list(old_values)
    rollout_rows = (
        list(rollout_values.unbind())
        if isinstance(rollout_values, torch.Tensor) and rollout_values.is_nested
        else list(rollout_values)
    )
    mask_rows = list(loss_mask.unbind()) if isinstance(loss_mask, torch.Tensor) and loss_mask.is_nested else list(loss_mask)
    old_out = []
    rollout_out = []
    mask_out = []
    max_len = 0
    for old_row, rollout_row, mask_row in zip(old_rows, rollout_rows, mask_rows, strict=True):
        trainable = mask_row[1:].to(torch.bool)
        if int(trainable.sum().item()) <= 0:
            continue
        old_trainable = old_row[1:].to(torch.float32)[trainable]
        rollout_trainable = rollout_row[1:].to(torch.float32)[trainable]
        old_out.append(old_trainable)
        rollout_out.append(rollout_trainable)
        mask_out.append(torch.ones_like(old_trainable, dtype=torch.bool))
        max_len = max(max_len, int(old_trainable.numel()))
    if not old_out:
        raise ValueError("packed rollout correction has no trainable tokens")
    device = old_out[0].device
    batch = len(old_out)
    old_padded = torch.zeros((batch, max_len), dtype=torch.float32, device=device)
    rollout_padded = torch.zeros((batch, max_len), dtype=torch.float32, device=device)
    mask_padded = torch.zeros((batch, max_len), dtype=torch.bool, device=device)
    for idx, (old_row, rollout_row, mask_row) in enumerate(zip(old_out, rollout_out, mask_out, strict=True)):
        length = int(old_row.numel())
        old_padded[idx, :length] = old_row
        rollout_padded[idx, :length] = rollout_row
        mask_padded[idx, :length] = mask_row
    return old_padded, rollout_padded, mask_padded


def _flatten_trainable_tokens(values: Any, loss_mask: Any, *, logprob_drop: str) -> Any:
    import torch

    value_rows = list(values.unbind()) if isinstance(values, torch.Tensor) and values.is_nested else list(values)
    mask_rows = list(loss_mask.unbind()) if isinstance(loss_mask, torch.Tensor) and loss_mask.is_nested else list(loss_mask)
    out = []
    for value_row, mask_row in zip(value_rows, mask_rows, strict=True):
        if logprob_drop == "last":
            shifted_values = value_row[:-1]
            shifted_mask = mask_row[1:].to(torch.bool)
        elif logprob_drop == "first":
            shifted_values = value_row[1:]
            shifted_mask = mask_row[1:].to(torch.bool)
        elif logprob_drop == "none":
            shifted_values = value_row[1:]
            shifted_mask = mask_row[1:].to(torch.bool)
        else:
            raise ValueError(f"unexpected logprob_drop={logprob_drop!r}")
        if int(shifted_mask.sum().item()) > 0:
            out.append(shifted_values[shifted_mask])
    if not out:
        raise ValueError("packed rollout correction has no trainable tokens")
    return torch.cat(out).to(torch.float32)


def _assign_flat_trainable_values(shard_td: Any, key: str, flat_values: Any) -> None:
    import torch

    loss_rows = list(shard_td["loss_mask"].unbind())
    rows = []
    cursor = 0
    for loss_row in loss_rows:
        count = int(loss_row[1:].sum().item())
        row = torch.ones_like(loss_row, dtype=torch.float32)
        if count > 0:
            row[1:][loss_row[1:].to(torch.bool)] = flat_values[cursor : cursor + count].to(row.device)
            cursor += count
        rows.append(row)
    if cursor != int(flat_values.numel()):
        raise ValueError(f"packed {key} flat assignment consumed {cursor} != {int(flat_values.numel())}")
    shard_td[key] = torch.nested.as_nested_tensor(rows, layout=torch.jagged).contiguous()


def _flat_trainable_mask_to_nested_loss_mask(loss_mask: Any, flat_mask: Any) -> Any:
    import torch

    rows = []
    cursor = 0
    for loss_row in loss_mask.unbind():
        updated = loss_row.clone()
        trainable = loss_row[1:].to(torch.bool)
        count = int(trainable.sum().item())
        if count > 0:
            keep = flat_mask[cursor : cursor + count].to(dtype=updated.dtype, device=updated.device)
            response_part = updated[1:]
            response_part[trainable] = keep
            updated[1:] = response_part
            cursor += count
        rows.append(updated)
    if cursor != int(flat_mask.numel()):
        raise ValueError(f"packed loss_mask assignment consumed {cursor} != {int(flat_mask.numel())}")
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _collect_worker_metrics_outputs(outputs: list[Any], collect_mask: list[bool], dp_rank_mapping: list[int]) -> dict[str, Any]:
    from verl.utils import tensordict_utils as tu

    selected = [
        (int(dp_rank), output)
        for output, collect, dp_rank in zip(outputs, collect_mask, dp_rank_mapping, strict=True)
        if collect
    ]
    if not selected:
        raise RuntimeError("packed-variable actor update collected no worker outputs")
    selected.sort(key=lambda item: item[0])
    merged: dict[str, list[Any]] = {}
    for _dp_rank, output in selected:
        worker_metrics = tu.get(output, "metrics", {})
        if not isinstance(worker_metrics, dict):
            continue
        for key, value in worker_metrics.items():
            merged.setdefault(key, [])
            if isinstance(value, list):
                merged[key].extend(value)
            else:
                merged[key].append(value)
    return merged


def _legacy_sum_actor_loss_metrics(worker_metrics: dict[str, Any], *, dp_size: int) -> dict[str, float]:
    """Match legacy VERL actor loss metric scale for packed update.

    The new engine returns one metric value per PPO mini-batch per collected DP
    rank.  ``reduce_metrics`` would average those values, while the legacy
    pad/DataProto actor accumulates loss metrics across PPO mini-batches after
    DP averaging.  Keep that step-sum under the canonical ``actor/kl_loss`` key
    so packed/pad dashboards use the same convention, and expose the previous
    mini-batch mean as a debug metric.
    """

    out: dict[str, float] = {}
    dp_reduce_factor = max(1, int(dp_size or 1))
    for metric_name in ("pg_loss", "kl_loss", "loss"):
        values = _numeric_metric_values(worker_metrics.get(metric_name))
        if not values:
            values = _numeric_metric_values(worker_metrics.get(f"actor/{metric_name}"))
        if not values:
            continue
        out[f"actor/{metric_name}"] = float(sum(values) / dp_reduce_factor)
        out[f"actor/{metric_name}_minibatch_mean_debug"] = float(sum(values) / len(values))
        out[f"actor/{metric_name}_minibatch_count_debug"] = float(len(values) / dp_reduce_factor)
        out[f"actor/{metric_name}_raw_metric_count_debug"] = float(len(values))
        out[f"actor/{metric_name}_dp_reduce_factor_debug"] = float(dp_reduce_factor)
    return out


def _numeric_metric_values(value: Any) -> list[float]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    out: list[float] = []
    for item in values:
        try:
            if hasattr(item, "aggregate"):
                item = item.aggregate()
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _reduce_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, list[float]] = {}
    for item in items:
        for key, value in (item or {}).items():
            try:
                merged.setdefault(str(key), []).append(float(value))
            except (TypeError, ValueError):
                pass
    out: dict[str, float] = {}
    for key, values in merged.items():
        if not values:
            continue
        if "max" in key:
            out[key] = max(values)
        elif "min" in key:
            out[key] = min(values)
        else:
            out[key] = sum(values) / len(values)
    return out


def _empty_dataproto_with_metrics(metrics: dict[str, float]) -> Any:
    from verl import DataProto

    return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})


def _env_flag(name: str, default: object = "0") -> bool:
    import os

    raw = os.environ.get(name)
    if raw is None:
        raw = "1" if default is True else "0" if default is False else default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _lcm(a: int, b: int) -> int:
    import math

    a = max(1, int(a or 1))
    b = max(1, int(b or 1))
    return abs(a * b) // math.gcd(a, b)


def _debug_print(label: str, payload: dict[str, Any]) -> None:
    if not _env_flag("POLAR_PACKED_VARIABLE_DEBUG"):
        return
    try:
        print(f"POLAR_PACKED_VARIABLE_DEBUG {label} {payload}", flush=True)
    except Exception:
        pass


def _actor_dispatch_info(trainer: Any) -> tuple[list[int], list[bool], int]:
    return _dispatch_info_for_role(trainer, trainer.actor_rollout_wg, role="actor")


def _dispatch_info_for_role(trainer: Any, worker_group: Any, *, role: str) -> tuple[list[int], list[bool], int]:
    if role not in worker_group._dispatch_info:
        worker_group._dispatch_info[role] = worker_group._query_dispatch_info(role)
    if role not in worker_group._collect_info:
        worker_group._collect_info[role] = worker_group._query_collect_info(role)
    dp_rank_mapping = list(worker_group._dispatch_info[role])
    collect_mask = list(worker_group._collect_info[role])
    dp_size = max(dp_rank_mapping) + 1 if dp_rank_mapping else 1
    return dp_rank_mapping, collect_mask, dp_size


def _raw_worker_method(worker_group: Any, *, candidates: tuple[str, ...]) -> str:
    if getattr(worker_group, "fused_worker_used", False):
        return candidates[0]
    for candidate in candidates:
        if all(hasattr(worker, candidate) for worker in worker_group.workers):
            return candidate
    raise RuntimeError(f"cannot find packed-variable worker method; tried={candidates}")


def _run_raw_worker_method(
    worker_group: Any,
    *,
    raw_method_name: str,
    dp_shards: list[Any],
    collect_mask: list[bool],
    dp_rank_mapping: list[int],
) -> list[Any]:
    import ray

    outputs = []
    for worker, dp_rank in zip(worker_group.workers, dp_rank_mapping, strict=True):
        outputs.append(worker_group._execute_remote_single_worker(worker, raw_method_name, dp_shards[int(dp_rank)]))
    raw_outputs = ray.get(outputs)
    selected = [
        (int(dp_rank), output)
        for output, collect, dp_rank in zip(raw_outputs, collect_mask, dp_rank_mapping, strict=True)
        if collect
    ]
    if not selected:
        raise RuntimeError(f"packed-variable {raw_method_name} collected no worker outputs")
    selected.sort(key=lambda item: item[0])
    return [output for _dp_rank, output in selected]


def _logprob_temperature(raw_temperature: Any) -> float:
    try:
        value = float(raw_temperature)
    except Exception:
        return 1.0
    return 1.0 if value <= 0.0 else value
