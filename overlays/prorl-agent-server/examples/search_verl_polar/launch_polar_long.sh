#!/usr/bin/env bash
set -euo pipefail

# Full service launcher for long-run SearchR1 VERL+Polar training.
# Training arguments are kept in train_polar_long.sh and mirror the pinned verl checkout.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-$ROOT/logs/search_verl_polar_long/$(date +%Y%m%d_%H%M%S)}"
if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="$ROOT/$LOG_DIR"
fi
mkdir -p "$LOG_DIR"
export LOG_DIR

# Project paths: defaults mirror the pinned verl checkout.
: "${CONFIG_PATH:=${SEARCH_CONFIG_DIR:-/path/to/search/config}}"
: "${CONFIG_NAME:=search_multiturn_grpo}"
: "${MODEL_PATH:=${POLICY_MODEL_PATH:-/path/to/policy/model}}"
: "${SUMMARY_MODEL_PATH:=${SUMMARY_POLICY_MODEL_PATH:-/path/to/summary/model}}"
: "${RETRIEVAL_INDEX_PATH:=/path/to/retrieval/index.faiss}"
: "${RETRIEVAL_CORPUS_PATH:=/path/to/retrieval/corpus.jsonl}"
: "${RETRIEVER_MODEL_PATH:=/path/to/retriever/model}"
: "${TRAIN_DATA:=/path/to/train.parquet}"
: "${VAL_DATA:=/path/to/val.parquet}"
: "${RAW_TRAIN_DATA:=$TRAIN_DATA}"
: "${PREPARE_LONG_DATA:=0}"
: "${PREPARED_LONG_TRAIN_DATA:=$LOG_DIR/train_polar_long.parquet}"
: "${TOOL_CONFIG:=$CONFIG_PATH/tool_config/search_tool_config.yaml}"
export POLAR_SEARCH_TOOL_CONFIG_PATH="${POLAR_SEARCH_TOOL_CONFIG_PATH:-$TOOL_CONFIG}"
export STANDALONE_TOOL_CONFIG_PATH="${STANDALONE_TOOL_CONFIG_PATH:-$TOOL_CONFIG}"
export SEARCH_MAX_MODEL_LEN="${SEARCH_MAX_MODEL_LEN:-${POLAR_ROLLOUT_MAX_MODEL_LEN:-40000}}"
export SEARCH_TOP_K="${SEARCH_TOP_K:-${POLAR_ROLLOUT_TOP_K:--1}}"
export SEARCH_REPETITION_PENALTY="${SEARCH_REPETITION_PENALTY:-${POLAR_ROLLOUT_REPETITION_PENALTY:-1.0}}"
export SEARCH_RETRIEVAL_TIMEOUT="${SEARCH_RETRIEVAL_TIMEOUT:-6000}"

# Ports / URLs.
SUMMARY_SGLANG_HOST="${SUMMARY_SGLANG_HOST:-127.0.0.1}"
SUMMARY_SGLANG_PORT="${SUMMARY_SGLANG_PORT:-30000}"
RETRIEVAL_HOST="${RETRIEVAL_HOST:-127.0.0.1}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-1249}"
POLAR_ROLLOUT_HOST="${POLAR_ROLLOUT_HOST:-127.0.0.1}"
POLAR_ROLLOUT_PORT="${POLAR_ROLLOUT_PORT:-18080}"
POLAR_GATEWAY_HOST="${POLAR_GATEWAY_HOST:-127.0.0.1}"
POLAR_GATEWAY_PORT="${POLAR_GATEWAY_PORT:-18100}"
POLAR_NODE_ID="${POLAR_NODE_ID:-search-node-01}"

export SEARCH_RETRIEVAL_URL="${SEARCH_RETRIEVAL_URL:-http://$RETRIEVAL_HOST:$RETRIEVAL_PORT}"
export POLAR_ROLLOUT_URL="${POLAR_ROLLOUT_URL:-http://$POLAR_ROLLOUT_HOST:$POLAR_ROLLOUT_PORT}"
export POLAR_GATEWAY_URL="${POLAR_GATEWAY_URL:-http://$POLAR_GATEWAY_HOST:$POLAR_GATEWAY_PORT}"
export POLAR_TOPOLOGY_PATH="${POLAR_TOPOLOGY_PATH:-$ROOT/examples/search_verl_polar/topology.rendered.yaml}"
export POLAR_CALLBACK_HOST="${POLAR_CALLBACK_HOST:-127.0.0.1}"
POLICY_ROLLOUT_BASE_URL="${POLICY_ROLLOUT_BASE_URL:-http://127.0.0.1:12345/v1}"

VERL_ROOT="${VERL_ROOT:-$ROOT/../verl}"
if [[ "$VERL_ROOT" != /* ]]; then
  VERL_ROOT="$ROOT/$VERL_ROOT"
fi
export VERL_ROOT

# Latest true-long defaults from the ProRL SearchR1 runbook, renamed to the
# repository's prompt-grounded terminology and kept overrideable by env.
export POLAR_HTTP_MAX_CONNECTIONS="${POLAR_HTTP_MAX_CONNECTIONS:-2048}"
export POLAR_HTTP_MAX_KEEPALIVE_CONNECTIONS="${POLAR_HTTP_MAX_KEEPALIVE_CONNECTIONS:-1024}"
export POLAR_HTTP_POOL_TIMEOUT="${POLAR_HTTP_POOL_TIMEOUT:-600}"
export POLAR_HTTP_KEEPALIVE_EXPIRY="${POLAR_HTTP_KEEPALIVE_EXPIRY:-60}"

export POLAR_STITCH_BY_MERGE_GROUP="${POLAR_STITCH_BY_MERGE_GROUP:-1}"
export POLAR_PREFIX_MERGING_MODE="${POLAR_PREFIX_MERGING_MODE:-prompt_grounded_single}"

export POLAR_PACKED_VARIABLE_ENABLE="${POLAR_PACKED_VARIABLE_ENABLE:-1}"
export POLAR_PACKED_VARIABLE_ACTOR_UPDATE="${POLAR_PACKED_VARIABLE_ACTOR_UPDATE:-1}"
export POLAR_PACKED_VARIABLE_ACTOR_DRY_RUN="${POLAR_PACKED_VARIABLE_ACTOR_DRY_RUN:-0}"
export POLAR_PACKED_VARIABLE_DRY_RUN="${POLAR_PACKED_VARIABLE_DRY_RUN:-0}"
export POLAR_PACKED_VARIABLE_DEBUG="${POLAR_PACKED_VARIABLE_DEBUG:-0}"
export POLAR_PACKED_VARIABLE_COMPACT_FIXED_OUTPUT="${POLAR_PACKED_VARIABLE_COMPACT_FIXED_OUTPUT:-1}"
export POLAR_PACKED_VARIABLE_PARTITION_MODE="${POLAR_PACKED_VARIABLE_PARTITION_MODE:-row_order}"
export POLAR_PACKED_VARIABLE_LEGACY_LOSS_SCALE="${POLAR_PACKED_VARIABLE_LEGACY_LOSS_SCALE:-1}"
export POLAR_PACKED_VARIABLE_MINIBATCH_MODE="${POLAR_PACKED_VARIABLE_MINIBATCH_MODE:-row_pad}"
export POLAR_PACKED_VARIABLE_ROW_PAD="${POLAR_PACKED_VARIABLE_ROW_PAD:-1}"

export POLAR_SEGMENT_REWARD_MODE="${POLAR_SEGMENT_REWARD_MODE:-none}"
export POLAR_PACKED_ADVANTAGE_LEVEL="${POLAR_PACKED_ADVANTAGE_LEVEL:-parent}"
export POLAR_PACKED_PARENT_SAMPLE_LOSS="${POLAR_PACKED_PARENT_SAMPLE_LOSS:-1}"
export POLAR_PACKED_SEGMENT_WEIGHT_LOSS="${POLAR_PACKED_SEGMENT_WEIGHT_LOSS:-1}"

export POLAR_SEARCH_SUBAGENT_ENABLE="${POLAR_SEARCH_SUBAGENT_ENABLE:-1}"
export POLAR_SEARCH_MAX_SUBAGENTS="${POLAR_SEARCH_MAX_SUBAGENTS:-1}"
export POLAR_SEARCH_SUBAGENT_MAX_TURNS="${POLAR_SEARCH_SUBAGENT_MAX_TURNS:-2}"
export POLAR_SEARCH_SUBAGENT_MAX_TOKENS="${POLAR_SEARCH_SUBAGENT_MAX_TOKENS:-4096}"
export POLAR_SEARCH_SUBAGENT_REPORT_MAX_CHARS="${POLAR_SEARCH_SUBAGENT_REPORT_MAX_CHARS:-4096}"
export POLAR_SEARCH_SUBAGENT_REPORT_FORMAT="${POLAR_SEARCH_SUBAGENT_REPORT_FORMAT:-sections}"
export POLAR_SEARCH_SUBAGENT_CONTEXT_MAX_CHARS="${POLAR_SEARCH_SUBAGENT_CONTEXT_MAX_CHARS:-2048}"

export POLAR_SEARCH_WIPE_ENABLE="${POLAR_SEARCH_WIPE_ENABLE:-1}"
export POLAR_SEARCH_WIPE_MAX_TURNS="${POLAR_SEARCH_WIPE_MAX_TURNS:-4}"
export POLAR_SEARCH_WIPE_CONTEXT_RATIO="${POLAR_SEARCH_WIPE_CONTEXT_RATIO:-0.50}"
export POLAR_SEARCH_COMPACTION_ENABLE="${POLAR_SEARCH_COMPACTION_ENABLE:-$POLAR_SEARCH_WIPE_ENABLE}"
export POLAR_SEARCH_COMPACTION_MAX_TURNS="${POLAR_SEARCH_COMPACTION_MAX_TURNS:-$POLAR_SEARCH_WIPE_MAX_TURNS}"
export POLAR_SEARCH_COMPACTION_CONTEXT_RATIO="${POLAR_SEARCH_COMPACTION_CONTEXT_RATIO:-$POLAR_SEARCH_WIPE_CONTEXT_RATIO}"

export POLAR_SUBAGENT_WIPE_INTERACTION_ARTIFACT="${POLAR_SUBAGENT_WIPE_INTERACTION_ARTIFACT:-1}"
export POLAR_SUBAGENT_WIPE_INTERACTION_FORMAT="${POLAR_SUBAGENT_WIPE_INTERACTION_FORMAT:-html}"
export POLAR_SUBAGENT_WIPE_INTERACTION_TOOL_MAX_CHARS="${POLAR_SUBAGENT_WIPE_INTERACTION_TOOL_MAX_CHARS:-20000}"
export POLAR_SEARCH_DRIVER_DEBUG="${POLAR_SEARCH_DRIVER_DEBUG:-1}"
export POLAR_SEARCH_DRIVER_DEBUG_LIMIT="${POLAR_SEARCH_DRIVER_DEBUG_LIMIT:-32}"
export POLAR_FULL_TRAJECTORY_ARTIFACT="${POLAR_FULL_TRAJECTORY_ARTIFACT:-true}"
export POLAR_FULL_TRAJECTORY_REQUIRE_SUBAGENT="${POLAR_FULL_TRAJECTORY_REQUIRE_SUBAGENT:-false}"
export POLAR_FULL_TRAJECTORY_LIMIT="${POLAR_FULL_TRAJECTORY_LIMIT:-32}"
export POLAR_SUBAGENT_TRAJECTORY_ARTIFACT="${POLAR_SUBAGENT_TRAJECTORY_ARTIFACT:-true}"
export POLAR_SUBAGENT_TRAJECTORY_LIMIT="${POLAR_SUBAGENT_TRAJECTORY_LIMIT:-32}"

export POLAR_ACTOR_ENTROPY_COEFF="${POLAR_ACTOR_ENTROPY_COEFF:-0}"
export POLAR_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${POLAR_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export POLAR_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${POLAR_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export POLAR_ROLLOUT_GPU_MEMORY_UTILIZATION="${POLAR_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"

# GPU placement: same layout as the Search smoke/compare scripts.
SUMMARY_SGLANG_CUDA_VISIBLE_DEVICES="${SUMMARY_SGLANG_CUDA_VISIBLE_DEVICES:-1}"
RETRIEVAL_CUDA_VISIBLE_DEVICES="${RETRIEVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-4,5,6,7}"

# Lifecycle controls.
INSTALL_DEPS="${INSTALL_DEPS:-1}"
APPLY_VERL_PATCH="${APPLY_VERL_PATCH:-1}"
APPLY_SEARCH_BASELINE_PATCHES="${APPLY_SEARCH_BASELINE_PATCHES:-1}"
START_SUMMARY_SGLANG="${START_SUMMARY_SGLANG:-1}"
START_RETRIEVAL_SERVER="${START_RETRIEVAL_SERVER:-1}"
START_POLAR_SERVICES="${START_POLAR_SERVICES:-1}"
RESTART_POLAR_SERVICES="${RESTART_POLAR_SERVICES:-0}"
RESTART_POLAR_ROLLOUT="${RESTART_POLAR_ROLLOUT:-$RESTART_POLAR_SERVICES}"
RESTART_POLAR_GATEWAY="${RESTART_POLAR_GATEWAY:-$RESTART_POLAR_SERVICES}"
RUN_TRAIN="${RUN_TRAIN:-1}"

BASELINE_TOOL_PARSER_SRC="${BASELINE_TOOL_PARSER_SRC:-/path/to/search_baseline/tool_parser.py}"
BASELINE_SEARCH_TOOL_SRC="${BASELINE_SEARCH_TOOL_SRC:-/path/to/search_baseline/search_tool.py}"
BASELINE_SEARCH_UTILS_SRC="${BASELINE_SEARCH_UTILS_SRC:-/path/to/search_baseline/search_r1_like_utils.py}"
BASELINE_REWARD_SCORE_SRC="${BASELINE_REWARD_SCORE_SRC:-/path/to/search_baseline/search_r1_like_qa_em.py}"
BASELINE_REWARD_INIT_SRC="${BASELINE_REWARD_INIT_SRC:-/path/to/search_baseline/__init__.py}"
RETRIEVAL_SERVER_SCRIPT="${RETRIEVAL_SERVER_SCRIPT:-/path/to/retrieval_server_sglang_summarize.py}"

wait_port() {
  local host="$1" port="$2" name="$3" timeout="${4:-600}"
  local start now
  start=$(date +%s)
  echo "[wait] $name at $host:$port"
  while ! nc -z "$host" "$port" >/dev/null 2>&1; do
    sleep 1
    now=$(date +%s)
    if (( now - start > timeout )); then
      echo "Timed out waiting for $name at $host:$port" >&2
      return 1
    fi
  done
  echo "[wait] $name ready at $host:$port"
}

stop_port_processes() {
  local host="$1" port="$2" name="$3"
  if [[ "$host" != "127.0.0.1" && "$host" != "localhost" && "$host" != "0.0.0.0" ]]; then
    echo "[restart] refusing to stop non-local $name at $host:$port" >&2
    return 1
  fi
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
  fi
  if [[ -z "$pids" ]]; then
    echo "[restart] no existing $name listener on $host:$port"
    return 0
  fi
  echo "[restart] stopping $name listener(s) on $host:$port: $pids"
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! nc -z "$host" "$port" >/dev/null 2>&1; then
      echo "[restart] $name stopped"
      return 0
    fi
    sleep 1
  done
  echo "[restart] $name still alive; SIGKILL"
  kill -9 $pids 2>/dev/null || true
}

install_deps() {
  if [[ "$INSTALL_DEPS" != "1" ]]; then
    echo "[1/8] INSTALL_DEPS=$INSTALL_DEPS, skip"
    return
  fi
  echo "[1/8] installing repo and runtime deps"
  pip install -e .
  pip install faiss-gpu-cu12==1.8.0
  pip install fastapi
  pip install uvicorn
  pip install --upgrade numpy==1.26.4
}

maybe_patch_verl() {
  if [[ "$APPLY_VERL_PATCH" != "1" ]]; then
    echo "[2/8] APPLY_VERL_PATCH=$APPLY_VERL_PATCH, skip"
    return
  fi
  echo "[2/8] applying VERL patch"
  scripts/patch/patch_verl.sh "$VERL_ROOT"
  PYTHONPYCACHEPREFIX=/tmp/prorl_pycache python -m py_compile \
    "$VERL_ROOT/verl/trainer/ppo/ray_trainer.py"
}

copy_if_exists() {
  local src="$1" dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "[baseline-patch] missing source file: $src" >&2
    return 1
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "[baseline-patch] copied $src -> $dst"
}

apply_search_baseline_patches() {
  if [[ "$APPLY_SEARCH_BASELINE_PATCHES" != "1" ]]; then
    echo "[3/8] APPLY_SEARCH_BASELINE_PATCHES=$APPLY_SEARCH_BASELINE_PATCHES, skip"
    return
  fi
  echo "[3/8] applying SearchR1 baseline overrides"
  copy_if_exists "$BASELINE_TOOL_PARSER_SRC" "$VERL_ROOT/verl/experimental/agent_loop/tool_parser.py"
  copy_if_exists "$BASELINE_SEARCH_TOOL_SRC" "$VERL_ROOT/verl/tools/search_tool.py"
  copy_if_exists "$BASELINE_SEARCH_UTILS_SRC" "$VERL_ROOT/verl/tools/utils/search_r1_like_utils.py"
  copy_if_exists "$BASELINE_REWARD_SCORE_SRC" "$VERL_ROOT/verl/utils/reward_score/search_r1_like_qa_em.py"
  copy_if_exists "$BASELINE_REWARD_INIT_SRC" "$VERL_ROOT/verl/utils/reward_score/__init__.py"
  PYTHONPYCACHEPREFIX=/tmp/prorl_pycache python3 -m py_compile \
    "$VERL_ROOT/verl/experimental/agent_loop/tool_parser.py" \
    "$VERL_ROOT/verl/tools/search_tool.py" \
    "$VERL_ROOT/verl/tools/utils/search_r1_like_utils.py" \
    "$VERL_ROOT/verl/utils/reward_score/search_r1_like_qa_em.py" \
    "$VERL_ROOT/verl/utils/reward_score/__init__.py"
}

prepare_long_data() {
  if [[ "$PREPARE_LONG_DATA" != "1" ]]; then
    echo "[data] PREPARE_LONG_DATA=$PREPARE_LONG_DATA, using TRAIN_DATA=$TRAIN_DATA"
    return
  fi
  echo "[data] Preparing long Search data: raw=$RAW_TRAIN_DATA -> prepared=$PREPARED_LONG_TRAIN_DATA"
  python examples/search_verl_polar/prepare_data.py \
    --input "$RAW_TRAIN_DATA" \
    --output "$PREPARED_LONG_TRAIN_DATA"
  export TRAIN_DATA="$PREPARED_LONG_TRAIN_DATA"
  export VAL_DATA="${PREPARED_LONG_VAL_DATA:-$PREPARED_LONG_TRAIN_DATA}"
  echo "[data] Prepared long data selected: TRAIN_DATA=$TRAIN_DATA VAL_DATA=$VAL_DATA"
}

start_summary_sglang() {
  if [[ "$START_SUMMARY_SGLANG" != "1" ]]; then
    echo "[4/8] START_SUMMARY_SGLANG=$START_SUMMARY_SGLANG, skip"
    return
  fi
  if nc -z "$SUMMARY_SGLANG_HOST" "$SUMMARY_SGLANG_PORT" >/dev/null 2>&1; then
    echo "[4/8] summary SGLang already listening on $SUMMARY_SGLANG_HOST:$SUMMARY_SGLANG_PORT"
    return
  fi
  echo "[4/8] starting summary SGLang on $SUMMARY_SGLANG_HOST:$SUMMARY_SGLANG_PORT"
  CUDA_VISIBLE_DEVICES="$SUMMARY_SGLANG_CUDA_VISIBLE_DEVICES" nohup python -m sglang.launch_server \
    --model-path "$SUMMARY_MODEL_PATH" \
    --tensor-parallel-size="${SUMMARY_TP_SIZE:-1}" \
    --mem-fraction-static "${SUMMARY_MEM_FRACTION:-0.5}" \
    --port "$SUMMARY_SGLANG_PORT" \
    >"$LOG_DIR/summary_sglang.out" 2>&1 &
  wait_port "$SUMMARY_SGLANG_HOST" "$SUMMARY_SGLANG_PORT" "summary SGLang" 900
}

start_retrieval_server() {
  if [[ "$START_RETRIEVAL_SERVER" != "1" ]]; then
    echo "[5/8] START_RETRIEVAL_SERVER=$START_RETRIEVAL_SERVER, skip"
    return
  fi
  if nc -z "$RETRIEVAL_HOST" "$RETRIEVAL_PORT" >/dev/null 2>&1; then
    echo "[5/8] retrieval already listening on $RETRIEVAL_HOST:$RETRIEVAL_PORT"
    return
  fi
  echo "[5/8] starting retrieval server on $RETRIEVAL_HOST:$RETRIEVAL_PORT"
  CUDA_VISIBLE_DEVICES="$RETRIEVAL_CUDA_VISIBLE_DEVICES" nohup python3 \
    "$RETRIEVAL_SERVER_SCRIPT" \
    --index_path "$RETRIEVAL_INDEX_PATH" \
    --corpus_path "$RETRIEVAL_CORPUS_PATH" \
    --faiss_gpu \
    --retriever_name "${RETRIEVER_NAME:-e5}" \
    --retriever_model "$RETRIEVER_MODEL_PATH" \
    --sglang_base_url "http://$SUMMARY_SGLANG_HOST:$SUMMARY_SGLANG_PORT" \
    --sglang_model "${SUMMARY_SGLANG_MODEL_NAME:-qwen3-4b-instruct}" \
    --host 0.0.0.0 \
    --port "$RETRIEVAL_PORT" \
    >"$LOG_DIR/retrieval.out" 2>&1 &
  wait_port "$RETRIEVAL_HOST" "$RETRIEVAL_PORT" "retrieval server" 900
}

start_polar_services() {
  echo "[6/8] rendering Polar topology: $POLAR_TOPOLOGY_PATH"
  python examples/swegym_verl_grpo/render_topology.py \
    examples/search_verl_polar/topology.yaml \
    --router-base-url "$POLICY_ROLLOUT_BASE_URL" \
    --output "$POLAR_TOPOLOGY_PATH"

  if [[ "$START_POLAR_SERVICES" != "1" ]]; then
    echo "[6/8] START_POLAR_SERVICES=$START_POLAR_SERVICES, skip"
    return
  fi

  if [[ "$RESTART_POLAR_ROLLOUT" == "1" ]]; then
    stop_port_processes "$POLAR_ROLLOUT_HOST" "$POLAR_ROLLOUT_PORT" "Polar rollout"
  fi
  if ! nc -z "$POLAR_ROLLOUT_HOST" "$POLAR_ROLLOUT_PORT" >/dev/null 2>&1; then
    echo "[6/8] starting Polar rollout on $POLAR_ROLLOUT_URL"
    nohup python -m polar.cli serve_rollout \
      --config "$POLAR_TOPOLOGY_PATH" \
      >"$LOG_DIR/polar_rollout.out" 2>&1 &
    wait_port "$POLAR_ROLLOUT_HOST" "$POLAR_ROLLOUT_PORT" "Polar rollout" 300
  else
    echo "[6/8] Polar rollout already listening on $POLAR_ROLLOUT_URL"
  fi

  if [[ "$RESTART_POLAR_GATEWAY" == "1" ]]; then
    stop_port_processes "$POLAR_GATEWAY_HOST" "$POLAR_GATEWAY_PORT" "Polar gateway"
  fi
  if ! nc -z "$POLAR_GATEWAY_HOST" "$POLAR_GATEWAY_PORT" >/dev/null 2>&1; then
    echo "[6/8] starting Polar gateway on $POLAR_GATEWAY_URL"
    nohup python -m polar.cli serve_gateway \
      --config "$POLAR_TOPOLOGY_PATH" \
      --node-id "$POLAR_NODE_ID" \
      >"$LOG_DIR/polar_gateway.out" 2>&1 &
    wait_port "$POLAR_GATEWAY_HOST" "$POLAR_GATEWAY_PORT" "Polar gateway" 300
  else
    echo "[6/8] Polar gateway already listening on $POLAR_GATEWAY_URL"
  fi

  echo "[6/8] checking gateway upstream update API"
  curl -fsS -X POST "$POLAR_GATEWAY_URL/admin/sglang/upstream" \
    -H 'Content-Type: application/json' \
    -d '{"base_url":"http://127.0.0.1:12345/v1","timeout_seconds":1}' \
    >"$LOG_DIR/polar_gateway_upstream_check.json"
  cat "$LOG_DIR/polar_gateway_upstream_check.json"
  echo
}

run_train() {
  if [[ "$RUN_TRAIN" != "1" ]]; then
    echo "[7/8] RUN_TRAIN=$RUN_TRAIN, skip"
    return
  fi
  echo "[7/8] starting Polar long training"
  export CONFIG_PATH CONFIG_NAME MODEL_PATH TRAIN_DATA VAL_DATA TOOL_CONFIG VERL_ROOT
  export SEARCH_RETRIEVAL_URL POLAR_ROLLOUT_URL POLAR_GATEWAY_URL POLAR_TOPOLOGY_PATH POLAR_CALLBACK_HOST
  export POLAR_SEARCH_TOOL_CONFIG_PATH STANDALONE_TOOL_CONFIG_PATH SEARCH_MAX_MODEL_LEN SEARCH_TOP_K SEARCH_REPETITION_PENALTY
  CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
    examples/search_verl_polar/train_polar_long.sh \
    2>&1 | tee "$LOG_DIR/train_polar_long.out"
}

print_summary() {
  echo "[8/8] Done. Logs: $LOG_DIR"
  echo "Useful tails:"
  echo "  tail -200 $LOG_DIR/summary_sglang.out"
  echo "  tail -200 $LOG_DIR/retrieval.out"
  echo "  tail -200 $LOG_DIR/polar_rollout.out"
  echo "  tail -200 $LOG_DIR/polar_gateway.out"
  echo "  tail -200 $LOG_DIR/train_polar_long.out"
  echo "  grep -RInE 'step:[0-9]+ - .*training/global_step|polar/fanout_training|polar/dropped/no_trainable' $LOG_DIR /tmp/ray/session_latest/logs/worker-*.out 2>/dev/null | tail -20"
  echo "  find $LOG_DIR/artifacts -maxdepth 2 -type f | sort"
}

install_deps
maybe_patch_verl
apply_search_baseline_patches
prepare_long_data
start_summary_sglang
start_retrieval_server
start_polar_services
run_train
print_summary
