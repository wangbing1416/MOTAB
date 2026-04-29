#!/bin/bash

# run_onpolicy_kd.sh
# KDFlow On-Policy Knowledge Distillation Training Script
#
# Usage:
#   bash run_onpolicy_kd.sh
#
# Or override any parameter via environment variables:
#   STUDENT_MODEL=/path/to/student TEACHER_MODEL=/path/to/teacher \
#   TRAIN_DATA=/path/to/data.jsonl bash run_onpolicy_kd.sh

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

# Student model: the small model to be trained (e.g., Qwen3-4B)
STUDENT_MODEL="${STUDENT_MODEL:-/path/to/models/Qwen3-1.7B}"

# Teacher model: the large model to distill from (e.g., Qwen3-32B)
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/models/Qwen3-8B}"

# Save path for checkpoints and rollout data
SAVE_PATH="${SAVE_PATH:-/path/to/checkpoints/onpolicy_rkl}"

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
# With dynamic bsz ON, this prevents two long sequences being packed together.
OPTS+=" --micro_train_batch_size 1"
OPTS+=" --learning_rate 2e-6"
OPTS+=" --lr_warmup_ratio 0.05"
# num_epochs=10: 10 * 12 iters/epoch = 120 total rollout iters
# Estimated total training time: ~1-1.5 hours (vs original 6 epochs ≈ 10 min)
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
# Set to False only if you have enough GPU memory to hold all three simultaneously
OPTS+=" --enable_sleep True"

# ── Model Arguments ──────────────────────────────────────────────────────────
OPTS+=" --student_name_or_path ${STUDENT_MODEL}"
OPTS+=" --teacher_name_or_path ${TEACHER_MODEL}"
# use_liger_kernel: fuses RMSNorm / SiLU / attention ops to reduce activation memory.
# Helpful but cannot reduce the KD logits backward allocation (32K x 151936 x bf16 ~10 GB)
# because KL/RKL divergence requires the full student+teacher probability distributions.
OPTS+=" --use_liger_kernel True"
# Set to True if training with Qwen3 thinking mode (<think>...</think>)
OPTS+=" --enable_thinking True"

# ── Rollout Arguments (On-Policy) ────────────────────────────────────────────
# rollout_num_engines: number of SGLang rollout instances (usually = NUM_GPUS)
# rollout_tp_size=1: 1.7B model needs no tensor parallelism
OPTS+=" --rollout_num_engines ${NUM_GPUS}"
OPTS+=" --rollout_tp_size 1"
# rollout_batch_size=64: 800 prompts / 64 = ~12 rollout iters/epoch (was 3 with 256)
# More frequent gradient updates improves on-policy learning signal quality
OPTS+=" --rollout_batch_size 32"
# 1.7B model weights ~3.4GB; 0.5 * 80GB = 40GB for SGLang per engine.
# 40 - 3.4 = 36.6GB for KV cache, supporting ~327K concurrent tokens.
# With enable_sleep=True, sleeping overhead is ~4GB regardless of this value,
# so a higher fraction here does NOT hurt student training memory.
OPTS+=" --rollout_mem_fraction_static 0.5"
# n_samples_per_prompt=5: 32 prompts × 5 = 160 sequences per rollout.
OPTS+=" --n_samples_per_prompt 5"
OPTS+=" --generate_max_len 32768"
OPTS+=" --temperature 0.6"
OPTS+=" --top_p 0.95"

# ── Data Arguments ───────────────────────────────────────────────────────────
OPTS+=" --train_dataset_path ${TRAIN_DATA}"
# input_key must match the field name in the converted JSONL (see prepare_onpolicy_data.py)
OPTS+=" --input_key messages"
OPTS+=" --apply_chat_template True"
OPTS+=" --max_len 32768"
OPTS+=" --prompt_max_len 2048"
OPTS+=" --preprocess_num_workers 32"

# ── Distillation Arguments ───────────────────────────────────────────────────
# kd_ratio must be 1.0 for on-policy KD (no CE loss, only KD loss)
OPTS+=" --kd_ratio 1.0"
# kd_loss_fn options: kl | rkl | jsd | akl
# rkl (reverse KL) is recommended for on-policy distillation to avoid mode-averaging
OPTS+=" --kd_loss_fn rkl"
OPTS+=" --kd_algorithm vanilla_kd"
# Teacher parallel config for Qwen3-8B on 8×80G:
#   tp=1: 8B (16GB bf16) fits on a single GPU; no tensor parallelism needed
#   dp=8: 8 independent replicas → maximum teacher throughput for rollout
#   mem_fraction=0.4: SGLang gets 32GB/GPU; 32-16=16GB left for KV cache
#
# Teacher size reference:
#   Qwen3-8B  on 8×80G: teacher_tp_size=1, teacher_dp_size=8  ← current
#   Qwen3-32B on 8×80G: teacher_tp_size=4, teacher_dp_size=2
#   Qwen3-72B on 8×80G: teacher_tp_size=8, teacher_dp_size=1
# Note: fp8 quantization is NOT supported on A100 (requires H100/H200).
OPTS+=" --teacher_tp_size 1"
OPTS+=" --teacher_dp_size 8"
# 0.6 * 80GB = 48GB; 48-16GB(model) = 32GB KV cache.
# Teacher runs ONLY during teacher phase (rollout+student sleeping, ~8GB overhead).
# With 0.3, only 8GB KV cache → scheduler can't allocate blocks for 32K-token
# sequences after wakeup → requests queued forever → GPU utilization = 0.
OPTS+=" --teacher_mem_fraction_static 0.5"

# ── Logging Arguments ────────────────────────────────────────────────────────
OPTS+=" --logging_steps 10"
# Uncomment to enable W&B logging:
# OPTS+=" --use_wandb True"
# OPTS+=" --wandb_project KDFlow"
# OPTS+=" --wandb_group onpolicy_kd"
# OPTS+=" --wandb_run_name onpolicy_kd_$(date +%Y%m%d_%H%M%S)"
# OPTS+=" --wandb_mode offline"
# OPTS+=" --wandb_dir ${SAVE_PATH}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Print Configuration and Launch
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "KDFlow On-Policy KD Training Configuration:"
echo "  Student Model  : ${STUDENT_MODEL}"
echo "  Teacher Model  : ${TEACHER_MODEL}"
echo "  Train Data     : ${TRAIN_DATA}"
echo "  Save Path      : ${SAVE_PATH}"
echo "  Nodes × GPUs   : ${NUM_NODES} × ${NUM_GPUS}"
echo "========================================"
echo ""

cd "$(dirname "$0")/KDFlow-main"

python -m kdflow.cli.train_kd_on_policy ${OPTS}

echo ""
echo "========================================"
echo "On-Policy KD Training Complete!"
echo "Checkpoints saved to: ${SAVE_PATH}"
echo "========================================"
