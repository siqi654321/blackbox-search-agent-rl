# SearchR1 Polar prompt-grounded true long run

This overlay keeps the minimal SearchR1 + Polar + VERL path for a Qwen-family
policy model using `POLAR_PREFIX_MERGING_MODE=prompt_grounded_single` and fixed DataProto
fanout training. Packed-variable, variable-training, and alignment/outlier debug
patch paths are intentionally removed from the long-run command.

All cluster-specific paths below are placeholders. Replace them with your own
model, dataset, retrieval, and SearchR1 baseline file locations.

## Current long-run command

```bash
ray stop --force || true

lsof -ti tcp:30000 2>/dev/null | xargs -r kill -9 || true
lsof -ti tcp:1249 2>/dev/null | xargs -r kill -9 || true
lsof -ti tcp:18080 2>/dev/null | xargs -r kill -9 || true
lsof -ti tcp:18100 2>/dev/null | xargs -r kill -9 || true

pkill -f "main_ppo.py" || true
pkill -f "SGLangHttpServer" || true
pkill -f "sglang.srt" || true
pkill -f "sglang.launch_server" || true
pkill -f "retrieval_server_sglang_summarize.py" || true
pkill -f "polar.cli serve_rollout" || true
pkill -f "polar.cli serve_gateway" || true

sleep 10
nvidia-smi

LOG_DIR=logs/search_verl_polar_qwen3_4b_true_long_$(date +%Y%m%d_%H%M%S) \
VERL_ROOT=../verl \
CONFIG_PATH=/path/to/search/config \
CONFIG_NAME=search_multiturn_grpo \
MODEL_PATH=/path/to/policy/model \
POLAR_ROLLOUT_IS_TOKENIZER_PATH=/path/to/policy/model \
RAW_TRAIN_DATA=/path/to/train.parquet \
PREPARE_LONG_DATA=1 \
TOOL_CONFIG=/path/to/search/config/tool_config/search_tool_config.yaml \
POLAR_SEARCH_TOOL_CONFIG_PATH=/path/to/search/config/tool_config/search_tool_config.yaml \
STANDALONE_TOOL_CONFIG_PATH=/path/to/search/config/tool_config/search_tool_config.yaml \
SUMMARY_MODEL_PATH=/path/to/summary/model \
RETRIEVAL_INDEX_PATH=/path/to/retrieval/index.faiss \
RETRIEVAL_CORPUS_PATH=/path/to/retrieval/corpus.jsonl \
RETRIEVER_MODEL_PATH=/path/to/retriever/model \
RETRIEVAL_SERVER_SCRIPT=/path/to/retrieval_server_sglang_summarize.py \
BASELINE_TOOL_PARSER_SRC=/path/to/search_baseline/tool_parser.py \
BASELINE_SEARCH_TOOL_SRC=/path/to/search_baseline/search_tool.py \
BASELINE_SEARCH_UTILS_SRC=/path/to/search_baseline/search_r1_like_utils.py \
BASELINE_REWARD_SCORE_SRC=/path/to/search_baseline/search_r1_like_qa_em.py \
BASELINE_REWARD_INIT_SRC=/path/to/search_baseline/__init__.py \
SUMMARY_SGLANG_CUDA_VISIBLE_DEVICES=1 \
RETRIEVAL_CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_CUDA_VISIBLE_DEVICES=4,5,6,7 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
START_SUMMARY_SGLANG=1 \
START_RETRIEVAL_SERVER=1 \
START_POLAR_SERVICES=1 \
RESTART_POLAR_SERVICES=1 \
RESTART_POLAR_GATEWAY=1 \
RESTART_POLAR_ROLLOUT=1 \
INSTALL_DEPS=1 \
APPLY_VERL_PATCH=1 \
APPLY_SEARCH_BASELINE_PATCHES=1 \
POLAR_N_GPUS_PER_NODE=4 \
POLAR_ROLLOUT_TP_SIZE=1 \
POLAR_ROLLOUT_DP_SIZE=1 \
POLAR_TRAIN_BATCH_SIZE=128 \
POLAR_VAL_BATCH_SIZE=256 \
POLAR_PPO_MINI_BATCH_SIZE=32 \
POLAR_PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
POLAR_AGENT_NUM_WORKERS=128 \
POLAR_ROLLOUT_N=8 \
POLAR_TOTAL_EPOCHS=20 \
POLAR_TOTAL_TRAINING_STEPS=null \
POLAR_DATA_SHUFFLE=true \
POLAR_DATA_SEED=2026 \
POLAR_MAX_PROMPT_LENGTH=4096 \
POLAR_MAX_RESPONSE_LENGTH=35000 \
POLAR_FILTER_OVERLONG_PROMPTS=True \
POLAR_DATA_TRUNCATION=error \
POLAR_ROLLOUT_MAX_MODEL_LEN=40000 \
SEARCH_MAX_MODEL_LEN=40000 \
POLAR_ROLLOUT_GPU_MEMORY_UTILIZATION=0.5 \
POLAR_ROLLOUT_TEMPERATURE=1.0 \
POLAR_ROLLOUT_TOP_P=1.0 \
POLAR_ROLLOUT_TOP_K=-1 \
POLAR_ROLLOUT_REPETITION_PENALTY=1.0 \
POLAR_ROLLOUT_DO_SAMPLE=true \
SEARCH_TEMPERATURE=1.0 \
SEARCH_TOP_P=1.0 \
SEARCH_TOP_K=-1 \
SEARCH_REPETITION_PENALTY=1.0 \
SEARCH_DO_SAMPLE=true \
SEARCH_MAX_TURNS=100 \
SEARCH_MAX_TOKENS=35000 \
SEARCH_MAX_TOOL_RESPONSE_LENGTH=2048 \
SEARCH_TOOL_RESPONSE_TRUNCATE_SIDE=middle \
SEARCH_RETRIEVAL_TIMEOUT=6000 \
POLAR_FANOUT_TRAINING=1 \
POLAR_MAX_ASYNC_LEVEL=2 \
POLAR_MAX_CONCURRENCY=256 \
POLAR_MAX_SESSION_CONCURRENCY=2048 \
POLAR_REQUEST_TIMEOUT=3600 \
POLAR_SGLANG_GENERATE_TIMEOUT=3600 \
POLAR_OVERFLOW_POLICY=verl_truncate \
POLAR_DYNAMIC_HISTORY_ENABLE=true \
POLAR_DYNAMIC_HISTORY_MODE=trace \
POLAR_STITCH_TRACES=true \
POLAR_REJECT_LOGPROB_ERROR=true \
POLAR_SEARCH_BRIDGE_MAX_TOKENS=true \
POLAR_PREFIX_MERGING_MODE=prompt_grounded_single \
POLAR_SAVE_FREQ=-1 \
POLAR_TEST_FREQ=-1 \
POLAR_LOG_LONGEST_TRACE_ARTIFACT=true \
POLAR_LONGEST_TRACE_INTERVAL=10 \
POLAR_PROJECT_NAME=search_r1_like_async_rl \
POLAR_EXPERIMENT_NAME=qwen3-4b-true-long \
POLAR_DEFAULT_LOCAL_DIR=checkpoints/search_verl_polar/qwen3_4b_true_long \
POLAR_RESUME_MODE=disable \
POLAR_TRAINER_METRICS_DEBUG=1 \
POLAR_MANAGER_METRICS_DEBUG=1 \
RAY_DEDUP_LOGS=0 \
HYDRA_FULL_ERROR=1 \
PYTHONUNBUFFERED=1 \
nohup examples/search_verl_polar/launch_polar_long.sh >train.out &
```

## Notes

- `POLAR_PREFIX_MERGING_MODE=prompt_grounded_single` selects the prompt-grounded single-trajectory merge builder.
- `VERL_ROOT` points directly to the pinned verl checkout; no nested verl checkout is required.
- `POLAR_FANOUT_TRAINING=1` keeps Polar dynamic-history/fanout rows in the fixed DataProto PPO update path.
- `POLAR_SAVE_FREQ=-1` and `POLAR_TEST_FREQ=-1` disable periodic checkpoint/eval during this long run.
- Trainer row alignment uses the original VERL row uid in `gen_batch_output.non_tensor_batch["source_uid"]`; dataset/provenance source ids remain in `polar_metadata` only.
