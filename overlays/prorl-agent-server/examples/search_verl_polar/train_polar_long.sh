#!/usr/bin/env bash
set -euo pipefail

# Long-run SearchR1 VERL+Polar trainer.  The PPO/training knobs intentionally
# mirror the pinned verl checkout; the only functional differences are:
#   1) rollout is routed through PolarAgentLoopManager;
#   2) Polar service/gateway URLs are provided via +polar.*;
#   3) Search tool execution is performed by the Polar SearchR1 harness.
# Start retrieval-summary SGLang, retrieval server, Polar rollout, and Polar
# gateway before running this script (or use launch_polar_long.sh if present).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${LOG_DIR:-}" && "$LOG_DIR" != /* ]]; then
  export LOG_DIR="$ROOT/$LOG_DIR"
fi

: "${CONFIG_PATH:=${SEARCH_CONFIG_DIR:-/path/to/search/config}}"
: "${CONFIG_NAME:=search_multiturn_grpo}"
: "${MODEL_PATH:=${POLICY_MODEL_PATH:-/path/to/policy/model}}"
: "${TRAIN_DATA:=/path/to/train.parquet}"
: "${VAL_DATA:=/path/to/val.parquet}"
: "${TOOL_CONFIG:=$CONFIG_PATH/tool_config/search_tool_config.yaml}"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export POLAR_ROLLOUT_URL="${POLAR_ROLLOUT_URL:-http://127.0.0.1:18080}"
export POLAR_GATEWAY_URL="${POLAR_GATEWAY_URL:-http://127.0.0.1:18100}"
export POLAR_TOPOLOGY_PATH="${POLAR_TOPOLOGY_PATH:-$ROOT/examples/search_verl_polar/topology.rendered.yaml}"
export POLAR_CALLBACK_HOST="${POLAR_CALLBACK_HOST:-127.0.0.1}"
export SEARCH_RETRIEVAL_URL="${SEARCH_RETRIEVAL_URL:-http://127.0.0.1:1249}"
export POLAR_SEARCH_HARNESS="${POLAR_SEARCH_HARNESS:-1}"
export POLAR_SEARCH_MODEL_NAME="${POLAR_SEARCH_MODEL_NAME:-$MODEL_PATH}"
export POLAR_SEARCH_TOOL_CONFIG_PATH="${POLAR_SEARCH_TOOL_CONFIG_PATH:-$TOOL_CONFIG}"
export STANDALONE_TOOL_CONFIG_PATH="${STANDALONE_TOOL_CONFIG_PATH:-$TOOL_CONFIG}"

VERL_ROOT="${VERL_ROOT:-$ROOT/../verl}"
if [[ "$VERL_ROOT" != /* ]]; then
  VERL_ROOT="$ROOT/$VERL_ROOT"
fi
export VERL_ROOT
export PYTHONPATH="$ROOT/src:$VERL_ROOT:${PYTHONPATH:-}"

# Align SearchR1 harness limits with standalone VERL train_verl.sh.
export SEARCH_MAX_TURNS="${SEARCH_MAX_TURNS:-${STANDALONE_MAX_ASSISTANT_TURNS:-100}}"
export SEARCH_MAX_TOKENS="${SEARCH_MAX_TOKENS:-${POLAR_MAX_RESPONSE_LENGTH:-35000}}"
export SEARCH_MAX_MODEL_LEN="${SEARCH_MAX_MODEL_LEN:-${POLAR_ROLLOUT_MAX_MODEL_LEN:-40000}}"
export SEARCH_MAX_TOOL_RESPONSE_LENGTH="${SEARCH_MAX_TOOL_RESPONSE_LENGTH:-${STANDALONE_MAX_TOOL_RESPONSE_LENGTH:-2048}}"
export SEARCH_TOOL_RESPONSE_TRUNCATE_SIDE="${SEARCH_TOOL_RESPONSE_TRUNCATE_SIDE:-middle}"
export SEARCH_TOP_K="${SEARCH_TOP_K:-${POLAR_ROLLOUT_TOP_K:--1}}"
export SEARCH_REPETITION_PENALTY="${SEARCH_REPETITION_PENALTY:-${POLAR_ROLLOUT_REPETITION_PENALTY:-1.0}}"
# Keep Polar's fixed DataProto training batch aligned with VERL's rollout.n fanout.
# Without this, accepted Polar traces can be pruned from train_batch_size*n back to
# train_batch_size, which makes critic/* metrics incomparable to standalone.
export POLAR_FANOUT_TRAINING="${POLAR_FANOUT_TRAINING:-1}"

POLAR_EFFECTIVE_TRAIN_BATCH_SIZE="${POLAR_TRAIN_BATCH_SIZE:-128}"
POLAR_EFFECTIVE_ROLLOUT_N="${POLAR_ROLLOUT_N:-8}"
POLAR_EFFECTIVE_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-8}"
POLAR_EFFECTIVE_MAX_CONCURRENCY="${POLAR_MAX_CONCURRENCY:-$(( POLAR_EFFECTIVE_TRAIN_BATCH_SIZE * POLAR_EFFECTIVE_MAX_ASYNC_LEVEL ))}"
POLAR_EFFECTIVE_MAX_SESSION_CONCURRENCY="${POLAR_MAX_SESSION_CONCURRENCY:-$(( POLAR_EFFECTIVE_MAX_CONCURRENCY * POLAR_EFFECTIVE_ROLLOUT_N ))}"

cd "$VERL_ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name="$CONFIG_NAME" \
  algorithm.adv_estimator=grpo \
  data.shuffle="${POLAR_DATA_SHUFFLE:-true}" \
  data.seed="${POLAR_DATA_SEED:-2026}" \
  data.train_batch_size="$POLAR_EFFECTIVE_TRAIN_BATCH_SIZE" \
  data.val_batch_size="${POLAR_VAL_BATCH_SIZE:-256}" \
  data.max_prompt_length="${POLAR_MAX_PROMPT_LENGTH:-4096}" \
  data.max_response_length="${POLAR_MAX_RESPONSE_LENGTH:-35000}" \
  data.filter_overlong_prompts="${POLAR_FILTER_OVERLONG_PROMPTS:-True}" \
  data.truncation="${POLAR_DATA_TRUNCATION:-error}" \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr="${POLAR_ACTOR_LR:-1e-6}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${POLAR_ACTOR_LR_WARMUP_STEPS_RATIO:-0.0}" \
  actor_rollout_ref.model.use_remove_padding="${POLAR_USE_REMOVE_PADDING:-True}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${POLAR_PPO_MINI_BATCH_SIZE:-32}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${POLAR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.actor.use_kl_loss="${POLAR_ACTOR_USE_KL_LOSS:-True}" \
  actor_rollout_ref.actor.kl_loss_coef="${POLAR_ACTOR_KL_LOSS_COEF:-0.0001}" \
  actor_rollout_ref.actor.clip_ratio_low="${POLAR_ACTOR_CLIP_RATIO_LOW:-0.2}" \
  actor_rollout_ref.actor.clip_ratio_high="${POLAR_ACTOR_CLIP_RATIO_HIGH:-0.28}" \
  actor_rollout_ref.actor.kl_loss_type="${POLAR_ACTOR_KL_LOSS_TYPE:-low_var_kl}" \
  actor_rollout_ref.rollout.temperature="${POLAR_ROLLOUT_TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${POLAR_ROLLOUT_TOP_P:-1.0}" \
  actor_rollout_ref.rollout.do_sample="${POLAR_ROLLOUT_DO_SAMPLE:-true}" \
  actor_rollout_ref.actor.entropy_coeff="${POLAR_ACTOR_ENTROPY_COEFF:-0}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${POLAR_ROLLOUT_TP_SIZE:-1}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${POLAR_ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}" \
  actor_rollout_ref.model.enable_gradient_checkpointing="${POLAR_ENABLE_GRADIENT_CHECKPOINTING:-True}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${POLAR_ACTOR_PARAM_OFFLOAD:-True}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${POLAR_ACTOR_OPTIMIZER_OFFLOAD:-True}" \
  actor_rollout_ref.rollout.max_model_len="${POLAR_ROLLOUT_MAX_MODEL_LEN:-40000}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${POLAR_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.gpu_memory_utilization="${POLAR_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}" \
  actor_rollout_ref.rollout.n="$POLAR_EFFECTIVE_ROLLOUT_N" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${POLAR_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${POLAR_REF_PARAM_OFFLOAD:-True}" \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_polar_bridge.manager.PolarAgentLoopManager \
  actor_rollout_ref.rollout.agent.num_workers="${POLAR_AGENT_NUM_WORKERS:-128}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${POLAR_MAX_TOOL_RESPONSE_LENGTH:-2048}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${POLAR_MAX_ASSISTANT_TURNS:-100}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${POLAR_MAX_USER_TURNS:-100}" \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.skip_tokenizer_init=false \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${POLAR_PROJECT_NAME:-search_r1_like_async_rl}" \
  trainer.experiment_name="${POLAR_EXPERIMENT_NAME:-qwen3-8b-asearcher-tis-datarand-flash-attn-polar}" \
  trainer.n_gpus_per_node="${POLAR_N_GPUS_PER_NODE:-4}" \
  actor_rollout_ref.rollout.n_gpus_per_node="${POLAR_N_GPUS_PER_NODE:-4}" \
  actor_rollout_ref.rollout.nnodes="${POLAR_ROLLOUT_NNODES:-0}" \
  actor_rollout_ref.rollout.data_parallel_size="${POLAR_ROLLOUT_DP_SIZE:-1}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${POLAR_VAL_TEMPERATURE:-0.7}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${POLAR_VAL_TOP_P:-0.8}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${POLAR_VAL_TOP_K:-20}" \
  actor_rollout_ref.rollout.val_kwargs.n="${POLAR_VAL_ROLLOUT_N:-1}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${POLAR_VAL_DO_SAMPLE:-false}" \
  trainer.nnodes="${POLAR_NNODES:-1}" \
  trainer.save_freq="${POLAR_SAVE_FREQ:-50}" \
  trainer.test_freq="${POLAR_TEST_FREQ:-50}" \
  trainer.default_local_dir="${POLAR_DEFAULT_LOCAL_DIR:-checkpoints/search_verl_polar/long}" \
  trainer.resume_mode="${POLAR_RESUME_MODE:-disable}" \
  trainer.resume_from_path="${POLAR_RESUME_FROM_PATH:-null}" \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  algorithm.rollout_correction.bypass_mode=false \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_is_batch_normalize=false \
  trainer.total_epochs="${POLAR_TOTAL_EPOCHS:-3}" \
  trainer.total_training_steps="${POLAR_TOTAL_TRAINING_STEPS:-null}" \
  +polar.enable=true \
  +polar.search.temperature="${SEARCH_TEMPERATURE:-${POLAR_ROLLOUT_TEMPERATURE:-1.0}}" \
  +polar.search.top_p="${SEARCH_TOP_P:-${POLAR_ROLLOUT_TOP_P:-1.0}}" \
  +polar.search.top_k="${SEARCH_TOP_K:-${POLAR_ROLLOUT_TOP_K:--1}}" \
  +polar.search.repetition_penalty="${SEARCH_REPETITION_PENALTY:-${POLAR_ROLLOUT_REPETITION_PENALTY:-1.0}}" \
  +polar.search.do_sample="${SEARCH_DO_SAMPLE:-${POLAR_ROLLOUT_DO_SAMPLE:-true}}" \
  +polar.search.max_model_len="$SEARCH_MAX_MODEL_LEN" \
  +polar.rollout_url="$POLAR_ROLLOUT_URL" \
  +polar.gateway_url="$POLAR_GATEWAY_URL" \
  +polar.topology_path="$POLAR_TOPOLOGY_PATH" \
  +polar.callback_host="$POLAR_CALLBACK_HOST" \
  +polar.max_concurrency="$POLAR_EFFECTIVE_MAX_CONCURRENCY" \
  +polar.max_session_concurrency="$POLAR_EFFECTIVE_MAX_SESSION_CONCURRENCY" \
  +polar.max_async_level="$POLAR_EFFECTIVE_MAX_ASYNC_LEVEL" \
  +polar.request_timeout="${POLAR_REQUEST_TIMEOUT:-2400}" \
  +polar.dynamic_history.enable="${POLAR_DYNAMIC_HISTORY_ENABLE:-true}" \
  +polar.dynamic_history.mode="${POLAR_DYNAMIC_HISTORY_MODE:-trace}" \
  +polar.overflow_policy="${POLAR_OVERFLOW_POLICY:-verl_truncate}" \
  +polar.weight_update.allow_overlap="${POLAR_ALLOW_WEIGHT_UPDATE_OVERLAP:-false}" \
  +polar.acceptance.reject_logprob_error="${POLAR_REJECT_LOGPROB_ERROR:-true}" \
  +polar.metrics.log_longest_trace_artifact="${POLAR_LOG_LONGEST_TRACE_ARTIFACT:-true}" \
  +polar.metrics.longest_trace_interval="${POLAR_LONGEST_TRACE_INTERVAL:-1}" \
  +polar.training.stitch_traces="${POLAR_STITCH_TRACES:-true}"

if [[ "${POLAR_SLEEP_INFINITY:-0}" == "1" ]]; then
  sleep infinity
fi
