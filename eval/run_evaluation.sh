#!/bin/bash
# ============================================================================
# run_evaluation.sh - Complete evaluation pipeline
# 1. Start SGLang server
# 2. Run EvalScope evaluation
# 3. Stop SGLang server
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Configuration (can be overridden via environment variables or arguments)
# ============================================================================

# Model configuration
MODEL_PATH="${1:-/path/to/model}"
MODEL_NAME="${MODEL_NAME:-llm}"

# Server configuration
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31011}"
TP_SIZE="${TP_SIZE:-8}"  # Tensor parallel size (number of GPUs)

# GPU configuration
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Log configuration
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/log}"
mkdir -p "$LOG_DIR"
SERVER_LOG="${SERVER_LOG:-$LOG_DIR/sglang_server_${PORT}.log}"
EVAL_LOG="${EVAL_LOG:-$LOG_DIR/evalscope_$(date +%Y%m%d_%H%M%S).log}"

# Evaluation configuration
EVAL_DATASETS="${EVAL_DATASETS:-math_500 aime24 aime25 amc gpqa_diamond}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-32768}"

# Server wait timeout (in seconds)
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-300000}"

# ============================================================================
# Export environment variables
# ============================================================================
export MODEL_NAME
export MODEL_PATH
export CUDA_VISIBLE_DEVICES
export EVAL_API_URL="http://$HOST:$PORT/v1/"

echo "========================================================================"
echo "Evaluation Pipeline Configuration"
echo "========================================================================"
echo "Model Path:          $MODEL_PATH"
echo "Model Name:          $MODEL_NAME"
echo "Server:              http://$HOST:$PORT"
echo "TP Size:             $TP_SIZE"
echo "CUDA Devices:        $CUDA_VISIBLE_DEVICES"
echo "API URL:             $EVAL_API_URL"
echo "========================================================================"
echo "Eval Datasets:       $EVAL_DATASETS"
echo "Eval Batch Size:     $EVAL_BATCH_SIZE"
echo "Eval N Samples:      $EVAL_N_SAMPLES"
echo "Eval Temperature:    $EVAL_TEMPERATURE"
echo "Eval Max Tokens:     $EVAL_MAX_TOKENS"
echo "========================================================================"
echo "Server Log:          $SERVER_LOG"
echo "Eval Log:            $EVAL_LOG"
echo "========================================================================"

# ============================================================================
# Step 1: Start SGLang Server
# ============================================================================
echo ""
echo "[Step 1/3] Starting SGLang server..."
echo "Server log will be saved to: $SERVER_LOG"

# Kill any existing process on the port
echo "Checking if port $PORT is already in use..."
EXISTING_PID=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
  echo "Found existing process on port $PORT (PID: $EXISTING_PID), killing it..."
  kill -9 $EXISTING_PID 2>/dev/null || true
  sleep 3
  echo "Existing process killed."
fi

# Start SGLang server in background using start_sglang.sh
(cd "$SCRIPT_DIR" && \
  source /path/to/envs/sglang/bin/activate && \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  LWS_LEADER_ADDRESS="${LWS_LEADER_ADDRESS:-localhost}" \
  MASTER_PORT="${MASTER_PORT:-29500}" \
  LWS_WORKER_INDEX="${LWS_WORKER_INDEX:-0}" \
  LWS_GROUP_SIZE="${LWS_GROUP_SIZE:-1}" \
  bash ./start_sglang.sh "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tp "$TP_SIZE" \
  >"$SERVER_LOG" 2>&1) &

SERVER_PID=$!
echo "SGLang server started with PID: $SERVER_PID"

# Cleanup function to stop server on exit
cleanup() {
  echo ""
  echo "========================================================================"
  echo "Cleaning up..."
  echo "========================================================================"
  
  # Kill any processes on the port
  echo "Stopping SGLang server on port $PORT..."
  EXISTING_PID=$(lsof -ti :$PORT 2>/dev/null || true)
  if [ -n "$EXISTING_PID" ]; then
    echo "Found processes on port $PORT (PIDs: $EXISTING_PID), killing..."
    # Kill all processes on the port
    for pid in $EXISTING_PID; do
      kill "$pid" 2>/dev/null || true
    done
    sleep 3
    
    # Force kill if still running
    EXISTING_PID=$(lsof -ti :$PORT 2>/dev/null || true)
    if [ -n "$EXISTING_PID" ]; then
      echo "Force killing remaining processes..."
      for pid in $EXISTING_PID; do
        kill -9 "$pid" 2>/dev/null || true
      done
      sleep 2
    fi
    
    echo "SGLang server stopped."
  else
    echo "No processes found on port $PORT."
  fi
  
  # Also kill the background job if exists
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  
  echo "Cleanup completed."
  echo "========================================================================"
}

# Register cleanup function
trap cleanup EXIT INT TERM

# ============================================================================
# Step 2: Wait for SGLang Server to be Ready
# ============================================================================
echo ""
echo "[Step 2/3] Waiting for SGLang server to be ready..."

WAIT_START_TIME=$(date +%s)
SERVER_READY=false

for i in $(seq 1 $((SERVER_WAIT_TIMEOUT / 5))); do
  if command -v curl >/dev/null 2>&1; then
    # Try multiple endpoints
    if curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1 || \
       curl -sf "http://$HOST:$PORT/v1/" >/dev/null 2>&1 || \
       curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      echo "SGLang server is ready! (took $(( $(date +%s) - WAIT_START_TIME ))s)"
      SERVER_READY=true
      break
    fi
  else
    echo "curl not found, waiting 60s..."
    sleep 60
    SERVER_READY=true
    break
  fi
  
  # Check if server process is still running
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "ERROR: SGLang server process died unexpectedly. Check log: $SERVER_LOG"
    tail -n 50 "$SERVER_LOG"
    exit 1
  fi
  
  echo "  Waiting... ($(( i * 5 ))s / ${SERVER_WAIT_TIMEOUT}s)"
  sleep 5
done

if [ "$SERVER_READY" = false ]; then
  echo "ERROR: SGLang server failed to start within ${SERVER_WAIT_TIMEOUT}s"
  echo "Last 50 lines of server log:"
  tail -n 50 "$SERVER_LOG"
  exit 1
fi

# Additional wait to ensure server is fully initialized
echo "Waiting additional 10s for server stabilization..."
sleep 10

# ============================================================================
# Step 3: Run EvalScope Evaluation
# ============================================================================
echo ""
echo "[Step 3/3] Running EvalScope evaluation..."
echo "Evaluation log will be saved to: $EVAL_LOG"

# Activate conda environment and run evaluation
(
  source /path/to/envs/evalscope/bin/activate
  
  cd "$SCRIPT_DIR"
  
  # Run evaluation with all parameters
  python3 run_evalscope.py \
    --model_name "$MODEL_NAME" \
    --api_url "$EVAL_API_URL" \
    --datasets $EVAL_DATASETS \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --n_samples "$EVAL_N_SAMPLES" \
    --temperature "$EVAL_TEMPERATURE" \
    --max_tokens "$EVAL_MAX_TOKENS" \
    >"$EVAL_LOG" 2>&1
)

EVAL_EXIT_CODE=$?

echo ""
if [ $EVAL_EXIT_CODE -eq 0 ]; then
  echo "========================================================================"
  echo "Evaluation completed successfully!"
  echo "========================================================================"
  echo "Results saved to: $EVAL_LOG"
  echo "Server log:       $SERVER_LOG"
  echo "========================================================================"
  echo ""
  echo "Summary of evaluation logs:"
  tail -n 20 "$EVAL_LOG"
else
  echo "========================================================================"
  echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
  echo "========================================================================"
  echo "Check evaluation log: $EVAL_LOG"
  echo "Last 50 lines:"
  tail -n 50 "$EVAL_LOG"
  echo "========================================================================"
fi

# Exit with the same code as evaluation
cleanup
exit $EVAL_EXIT_CODE
