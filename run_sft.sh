#!/bin/bash
# run_sft.sh
# LLaMAFactory Full-Parameter SFT Training Script

SCRIPT_PATH=$(dirname $(realpath "$BASH_SOURCE"))

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (override via environment variables)
# ─────────────────────────────────────────────────────────────────────────────

# Path to the base model to fine-tune
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/path/to/models/Qwen3-4B-Instruct}"

# Path to the LLaMAFactory YAML training config
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${SCRIPT_PATH}/sft_config.yaml}"

# Path to dataset_info.json (LLaMAFactory dataset registry)
DATASET_INFO="${DATASET_INFO:-/path/to/data/dataset_info.json}"

# Distributed training settings
NNODES="${WORLD_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANK="${RANK:-0}"
NGPUS="${NGPUS:-8}"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Preparation
#   LLaMAFactory requires dataset_info.json in the local data/ directory.
#   Copy your dataset registry file before training starts.
# ─────────────────────────────────────────────────────────────────────────────

function prepare_dataset() {
    cp -rf "${DATASET_INFO}" data/dataset_info.json
}

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

function train_full() {
    torchrun \
        --nproc_per_node ${NGPUS} \
        --nnodes ${NNODES} \
        --rdzv_id=3649 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        src/llamafactory/launcher.py ${EXPERIMENT_CONFIG}
}

function main() {
    prepare_dataset && train_full
}

main
