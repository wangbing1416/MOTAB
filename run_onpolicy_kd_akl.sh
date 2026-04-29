#!/bin/bash

# run_onpolicy_kd_akl.sh
# KDFlow On-Policy Knowledge Distillation Training Script — AKL Loss
#
# Adaptive KL (AKL) dynamically balances forward-KL and reverse-KL per token:
#   loss = (g_head / (g_head + g_tail)) * FKL + (g_tail / (g_head + g_tail)) * RKL
# where the weights are derived from the probability gap between teacher and student.
# Reference: https://arxiv.org/abs/2404.02657
#
# Compared to run_onpolicy_kd.sh (RKL), AKL adaptively chooses:
#   - FKL (mode-covering) for tokens where student strongly disagrees with teacher
#   - RKL (mode-seeking) for tokens where student and teacher are well-aligned
#
# Usage:
#   bash run_onpolicy_kd_akl.sh
#
# Or override any parameter via environment variables:
#   STUDENT_MODEL=/path/to/student TEACHER_MODEL=/path/to/teacher \
#   TRAIN_DATA=/path/to/data.jsonl bash run_onpolicy_kd_akl.sh

set -e
set -x

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Data Preparation
#   Convert raw MOTAB input JSONL (field: input) to KDFlow format (field: messages)
#   Run this once before training; skip if output already exists.
# ─────────────────────────────────────────────────────────────────────────────

# Path to the raw input JSONL used by 1-generate-ours.py
RAW_INPUT_FILE="${RAW_INPUT_FILE:-/path/to/input.jsonl}"

# Path to the converted JSONL (KDFlow-compatible, will be created if absent)
TRAIN_DATA="${TRAIN_DATA:-/path/to/kdflow_prompts.jsonl}"

# Script directory (locate prepare_onpolicy_data.py relative to this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "========================================"
    echo "Converted training data not found, running data preparation..."
    echo "  Raw input : ${RAW_INPUT_FILE}"
    echo "  Output    : ${TRAIN_DATA}"
    echo "========================================"
    python "${SCRIPT_DIR}/prepare_onpolicy_data.py" \
        --input_file  "${RAW_INPUT_FILE}" \
        --output_file "${TRAIN_DATA}" \
        --deduplicate
else
    echo "Training data already exists, skipping preparation: ${TRAIN_DATA}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Start Ray (run once before first training)
# ─────────────────────────────────────────────────────────────────────────────

# Uncomment the line below on first run if Ray is not already running:
# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Model & Infrastructure Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Student model: the small model to be trained
STUDENT_MODEL="${STUDENT_MODEL:-/path/to/models/Qwen3-1.7B}"

# Teacher model: the large model to distill from
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/models/Qwen3-8B}"

# Save path for checkpoints and rollout data
SAVE_PATH="${SAVE_PATH:-/path/to/checkpoints/onpolicy_akl}"

# Hardware: number of nodes and GPUs per node
NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS="${NUM_GPUS:-8}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build KDFlow Arguments
# ─────────────────────────────────────────────────────────────────────────────

OPTS=""

# ── Training Arguments ───────────────────────────────────────────────────────
OPTS+=" --num_nodes ${NUM_NODES}"
OPTS+=" --num_gpus_per_node ${NUM_GPUS}"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 16"
# micro_train_batch_size=1: process one sequence at a time.
OPTS+=" --micro_train_batch_size 1"
OPTS+=" --learning_rate 2e-6"
OPTS+=" --lr_warmup_ratio 0.05"
# num_epochs=10: 10 * 12 iters/epoch = 120 total rollout iters
OPTS+=" --num_epochs 10"
OPTS+=" --save_path ${SAVE_PATH}"
# Save a checkpoint every 100 global steps as insurance for long runs
OPTS+=" --save_steps 100"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
# Dynamic batch size packs tokens to fill each GPU to max_token_len_per_gpu,
# improving GPU utilization by ~60-100% (KDFlow built-in feature).
# NOTE: Qwen3 vocab=151936; logits per GPU = max_token_len_per_gpu × 151936 × 2 bytes.
# At 32768 tokens: ~10 GB just for logits → OOM. Keep ≤8192 to stay within budget.
# Dynamic bsz MUST stay enabled: it caps total tokens per micro-batch to
# max_token_len_per_gpu, limiting logits to 4096×151936×2=1.25 GB per GPU.
# Disabling it removes this cap; with micro_train_batch_size=2 and max_len=32768,
# two long sequences pack to 65536 tokens → 20 GB logits → guaranteed OOM.
# OPTS+=" --use_dynamic_bsz True"
# OPTS+=" --max_token_len_per_gpu 4096"
# Enable sleep mode: Teacher / Student / Rollout share the same GPUs via time-slicing
OPTS+=" --enable_sleep True"

# ── Model Arguments ──────────────────────────────────────────────────────────
OPTS+=" --student_name_or_path ${STUDENT_MODEL}"
OPTS+=" --teacher_name_or_path ${TEACHER_MODEL}"
# use_liger_kernel: fuses RMSNorm / SiLU / attention ops to reduce activation memory.
OPTS+=" --use_liger_kernel True"
# Set to True if training with Qwen3 thinking mode (<think>...</think>)
OPTS+=" --enable_thinking True"

# ── Rollout Arguments (On-Policy) ────────────────────────────────────────────
OPTS+=" --rollout_num_engines ${NUM_GPUS}"
OPTS+=" --rollout_tp_size 1"
# rollout_batch_size=32: 32 prompts × 5 = 160 sequences per rollout.
OPTS+=" --rollout_batch_size 32"
# 1.7B model weights ~3.4GB; 0.5 * 80GB = 40GB for SGLang per engine.
OPTS+=" --rollout_mem_fraction_static 0.5"
# n_samples_per_prompt=5: 32 prompts × 5 = 160 sequences per rollout.
OPTS+=" --n_samples_per_prompt 5"
OPTS+=" --generate_max_len 32768"
OPTS+=" --temperature 0.6"
OPTS+=" --top_p 0.95"

# ── Data Arguments ───────────────────────────────────────────────────────────
OPTS+=" --train_dataset_path ${TRAIN_DATA}"
OPTS+=" --input_key messages"
OPTS+=" --apply_chat_template True"
OPTS+=" --max_len 32768"
OPTS+=" --prompt_max_len 2048"
OPTS+=" --preprocess_num_workers 32"

# ── Distillation Arguments (AKL) ─────────────────────────────────────────────
# kd_ratio=1.0: pure KD loss, no CE loss
OPTS+=" --kd_ratio 1.0"
# AKL adaptively blends FKL and RKL per token based on probability gap.
# adaptive_alpha controls the boundary: tokens in the bottom-alpha cumulative
# probability mass are treated as "tail" (use RKL); rest use FKL.
# Default 0.5 means roughly 50% of probability mass boundary.
OPTS+=" --kd_loss_fn akl"
OPTS+=" --adaptive_alpha 0.5"
OPTS+=" --kd_algorithm vanilla_kd"
# Teacher parallel config for Qwen3-8B on 8×80G:
#   tp=1: 8B (16GB bf16) fits on a single GPU; no tensor parallelism needed
#   dp=8: 8 independent replicas → maximum teacher throughput for rollout
OPTS+=" --teacher_tp_size 1"
OPTS+=" --teacher_dp_size 8"
OPTS+=" --teacher_mem_fraction_static 0.5"

# ── Logging Arguments ────────────────────────────────────────────────────────
OPTS+=" --logging_steps 10"
# Uncomment to enable W&B logging:
# OPTS+=" --use_wandb True"
# OPTS+=" --wandb_project KDFlow"
# OPTS+=" --wandb_group onpolicy_kd"
# OPTS+=" --wandb_run_name onpolicy_kd_akl_$(date +%Y%m%d_%H%M%S)"
# OPTS+=" --wandb_mode offline"
# OPTS+=" --wandb_dir ${SAVE_PATH}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Print Configuration and Launch
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "KDFlow On-Policy KD Training Configuration (AKL):"
echo "  Student Model  : ${STUDENT_MODEL}"
echo "  Teacher Model  : ${TEACHER_MODEL}"
echo "  Train Data     : ${TRAIN_DATA}"
echo "  Save Path      : ${SAVE_PATH}"
echo "  Nodes x GPUs   : ${NUM_NODES} x ${NUM_GPUS}"
echo "  Loss Function  : AKL (adaptive_alpha=0.5)"
echo "========================================"
echo ""

cd "$(dirname "$0")/KDFlow-main"

python -m kdflow.cli.train_kd_on_policy ${OPTS}

echo ""
echo "========================================"
echo "On-Policy KD Training (AKL) Complete!"
echo "Checkpoints saved to: ${SAVE_PATH}"
echo "========================================"
