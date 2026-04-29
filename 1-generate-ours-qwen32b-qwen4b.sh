#!/bin/bash

# 1-generate-ours-qwen32b-qwen4b.sh
# MOTAB Data Synthesis Script

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
# Parameter Definitions (can be overridden via environment variables)
# ─────────────────────────────────────────────────────────────────────────────

# Input/output file paths
INPUT_FILE="${INPUT_FILE:-/path/to/input.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-/path/to/motab_output.jsonl}"

# API endpoints
TEACHER_URL="${TEACHER_URL:-http://127.0.0.1:30011/generate}"
STUDENT_URL="${STUDENT_URL:-http://127.0.0.1:30012/generate}"

# Teacher model path (for precise calculation of logprob_start_len)
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/path/to/models/Qwen3-32B}"

# MOTAB algorithm parameters
#
# --epsilon0: Base safety threshold ε₀ ∈ (0,1]
#   - LARGER ε₀ → stricter gate, more teacher corrections, higher validity but lower coverage
#   - SMALLER ε₀ → looser gate, more student steps accepted, better coverage but risk of flawed trajectories
#   - Recommended range: [0.1, 0.5]. Default: 0.2
#
# --theta: Entropy scaling factor θ > 0
#   - LARGER θ → teacher entropy has stronger effect on boundary; more permissive when teacher is uncertain
#   - SMALLER θ → boundary closer to static ε₀; entropy-aware adaptation is muted
#   - Recommended range: [0.5, 2.0]. Default: 1.0
#
# --top_k_entropy: Number of top tokens for teacher entropy estimation H(ε_T|s)
#   - LARGER top_k → more accurate entropy estimate, higher computational cost
#   - SMALLER top_k → faster but noisier entropy signal
#   - Recommended range: [10, 50]. Default: 20

EPSILON0="${EPSILON0:-0.2}"           # Base safety threshold ε₀
THETA="${THETA:-1.0}"                 # Entropy scaling factor θ
TOP_K_ENTROPY="${TOP_K_ENTROPY:-20}"  # Top-k tokens for entropy approximation
NUM_RESPONSES="${NUM_RESPONSES:-5}"

# Generation parameters
#
# --temperature: Sampling temperature for both student and teacher generation
# --max_step_tokens: Maximum tokens per student reasoning step (stop=".\n\n")
# --max_total_tokens: Maximum total tokens per complete SFT trajectory

TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_STEP_TOKENS="${MAX_STEP_TOKENS:-8192}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"

# Control parameters
#
# --num_workers: Number of parallel worker processes
# --batch_size: Number of items buffered in memory before writing to disk

NUM_WORKERS="${NUM_WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-10}"

# ─────────────────────────────────────────────────────────────────────────────
# Print Configuration
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "1-generate-ours.py Configuration:"
echo "========================================"
echo "  Input File           : $INPUT_FILE"
echo "  Output File          : $OUTPUT_FILE"
echo "  Student API URL      : $STUDENT_URL"
echo "  Teacher API URL      : $TEACHER_URL"
echo "  Teacher Model Path   : $TEACHER_MODEL_PATH"
echo "  Base Threshold ε₀    : $EPSILON0"
echo "  Entropy Scale θ      : $THETA"
echo "  Top-k for Entropy    : $TOP_K_ENTROPY"
echo "  Responses per Item   : $NUM_RESPONSES"
echo "  Temperature          : $TEMPERATURE"
echo "  Max tokens/step      : $MAX_STEP_TOKENS"
echo "  Max tokens/response  : $MAX_TOTAL_TOKENS"
echo "  Parallel Workers     : $NUM_WORKERS"
echo "  Disk Write Batch     : $BATCH_SIZE"
echo "========================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Execute Python Script
# ─────────────────────────────────────────────────────────────────────────────

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Directly call Python script (using argparse to parse command line arguments)
python 1-generate-ours.py \
    --input_file "${INPUT_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --student_url "${STUDENT_URL}" \
    --teacher_url "${TEACHER_URL}" \
    --teacher_model_path "${TEACHER_MODEL_PATH}" \
    --epsilon0 "${EPSILON0}" \
    --theta "${THETA}" \
    --top_k_entropy "${TOP_K_ENTROPY}" \
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
