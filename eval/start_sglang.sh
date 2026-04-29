#!/bin/bash

# start_sglang.sh
# Usage: bash start_sglang.sh <model_path> [--host HOST] [--port PORT] [--tp TP_SIZE]

# Read environment variables with defaults
MASTER_ADDR="${LWS_LEADER_ADDRESS:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANK="${LWS_WORKER_INDEX:-0}"
NNODES="${LWS_GROUP_SIZE:-1}"

# GPU configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPU=8  # Tensor parallel size

# Read configuration from arguments
MODEL="${1:-/path/to/models/qwen-32b}"
HOST="127.0.0.1"
PORT="30011"

# Parse optional arguments
shift || true
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --tp)
      GPU="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1" >&2
      shift
      ;;
  esac
done

echo "========================================"
echo "SGLang Server Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  RANK: $RANK"
echo "  NNODES: $NNODES"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  GPU Count: $GPU"
echo "  HOST: $HOST"
echo "  PORT: $PORT"
echo "  MODEL: $MODEL"
echo "========================================"

python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --dist-init-addr "${MASTER_ADDR}:5100" \
    --nnodes "${NNODES}" \
    --node-rank "${RANK}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp "${GPU}" \
    --trust-remote-code \
    --max-running-requests 256 \
    --mem-fraction-static 0.5 \
    --chunked-prefill-size 4096
