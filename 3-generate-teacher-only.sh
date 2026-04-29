#!/bin/bash

# 3-generate-teacher-only.sh
# Pure Teacher Model Trajectory Generation Script
# Only uses teacher model to generate step-by-step reasoning trajectories

# ─────────────────────────────────────────────────────────────────────────────
# Environment Variable Reading + Defaults
# ─────────────────────────────────────────────────────────────────────────────

# Read configuration from environment variables (supports distributed deployment)
MASTER_ADDR="${LWS_LEADER_ADDRESS:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANK="${LWS_WORKER_INDEX:-0}"
NNODES="${LWS_GROUP_SIZE:-1}"

# GPU count detection (for logging only)
GPU=$(nvidia-smi -L | wc -l)

echo "========================================"
echo "Environment Information:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  RANK: $RANK"
echo "  NNODES: $NNODES"
echo "  GPU: $GPU"
echo "========================================"

# ─────────────────────────────────────────────────────────────────────────────
# Parameter Definitions (can be overridden via command line)
# ─────────────────────────────────────────────────────────────────────────────

# Input/output file paths
INPUT_FILE="${INPUT_FILE:-/path/to/input.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-/path/to/teacher_output.jsonl}"

# API endpoints
TEACHER_URL="${TEACHER_URL:-http://127.0.0.1:30011/generate}"

# Teacher model path (for precise calculation of logprob_start_len)
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/path/to/models/Qwen3-32B}"

# Generation parameters
NUM_RESPONSES="${NUM_RESPONSES:-5}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_STEP_TOKENS="${MAX_STEP_TOKENS:-8192}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"

# Control parameters
# NUM_WORKERS: Maximum concurrent questions (upper limit of questions being processed simultaneously, no longer number of processes in async version)
NUM_WORKERS="${NUM_WORKERS:-64}"
BATCH_SIZE="${BATCH_SIZE:-10}"

# ─────────────────────────────────────────────────────────────────────────────
# Print Configuration
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "3-generate-teacher-only.py Configuration:"
echo "========================================"
echo "  Input File         : $INPUT_FILE"
echo "  Output File        : $OUTPUT_FILE"
echo "  Teacher API URL    : $TEACHER_URL"
echo "  Teacher Model Path : $TEACHER_MODEL_PATH"
echo "  Responses per Item : $NUM_RESPONSES"
echo "  Temperature        : $TEMPERATURE"
echo "  Max tokens/step    : $MAX_STEP_TOKENS"
echo "  Max tokens/response: $MAX_TOTAL_TOKENS"
echo "  Max Concurrent Qst : $NUM_WORKERS"
echo "  Disk Write Batch   : $BATCH_SIZE"
echo "========================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Execute Python Script
# ─────────────────────────────────────────────────────────────────────────────

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Directly call Python script (using argparse to parse command line arguments)
python 3-generate-teacher-only.py \
    --input_file "${INPUT_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --teacher_url "${TEACHER_URL}" \
    --teacher_model_path "${TEACHER_MODEL_PATH}" \
    --num_responses "${NUM_RESPONSES}" \
    --temperature "${TEMPERATURE}" \
    --max_step_tokens "${MAX_STEP_TOKENS}" \
    --max_total_tokens "${MAX_TOTAL_TOKENS}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}"

echo ""
echo "========================================"
echo "Execution Complete!"
echo "========================================"
