#!/bin/bash

# run_onpolicy_kd_dskd.sh
# KDFlow On-Policy Knowledge Distillation Training Script — DSKD Algorithm
#
# DSKD (Dual-Space Knowledge Distillation) operates in two complementary spaces:
#   - t2s path: teacher hidden states are projected to student vocab space via a
#               learnable projector (t2s_projector), then distilled to the student.
#   - s2t path: student hidden states are projected to teacher vocab space via a
#               pseudo-inverse projection (s2t_projector), then aligned with teacher logits.
#
# Since Qwen3-32B and Qwen3-1.7B share the same tokenizer (vocab_identical=True),
# DSKD uses direct logit-space projection without token alignment (no eta/cma needed).
# The dual-space design provides richer supervision than vanilla logit matching.
#
# Key extra parameters vs. run_onpolicy_kd.sh:
#   --kd_algorithm dskd
#   --dskd_topk_vocab -1        use full vocabulary for projector initialization
#   --dskd_projector_lr 1e-4    separate learning rate for t2s projector weights
#
# Usage:
#   bash run_onpolicy_kd_dskd.sh
#
# Or override any parameter via environment variables:
#   STUDENT_MODEL=/path/to/student TEACHER_MODEL=/path/to/teacher \
#   TRAIN_DATA=/path/to/data.jsonl bash run_onpolicy_kd_dskd.sh

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
SAVE_PATH="${SAVE_PATH:-/path/to/checkpoints/onpolicy_dskd}"

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

# ── Distillation Arguments (DSKD) ────────────────────────────────────────────
# kd_ratio=1.0: pure KD loss, no CE loss
OPTS+=" --kd_ratio 1.0"
# DSKD uses kd_loss_fn internally for both t2s and s2t divergence calculations.
# rkl (reverse KL) is recommended for on-policy distillation.
OPTS+=" --kd_loss_fn rkl"
# Use DSKD dual-space algorithm:
#   - t2s_projector: projects teacher hidden states → student vocab space (learnable)
#   - s2t path: projects student hidden states → teacher vocab space (via pseudo-inverse)
# Since Qwen3-32B and Qwen3-1.7B share the same tokenizer (vocab_identical=True),
# the simple _compute_dskd_loss branch is used (no cross-tokenizer token alignment).
OPTS+=" --kd_algorithm dskd"
# dskd_topk_vocab=-1: use all vocabulary tokens for projector pseudo-inverse
# initialization. Set to a smaller value (e.g., 50000) to reduce init time/memory
# if full-vocab initialization is too slow.
OPTS+=" --dskd_topk_vocab -1"
# dskd_projector_lr: separate learning rate for the t2s projector weights,
# typically higher than the student model lr to allow fast projector adaptation.
OPTS+=" --dskd_projector_lr 1e-4"
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
# OPTS+=" --wandb_run_name onpolicy_kd_dskd_$(date +%Y%m%d_%H%M%S)"
# OPTS+=" --wandb_mode offline"
# OPTS+=" --wandb_dir ${SAVE_PATH}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Print Configuration and Launch
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "KDFlow On-Policy KD Training Configuration (DSKD):"
echo "  Student Model  : ${STUDENT_MODEL}"
echo "  Teacher Model  : ${TEACHER_MODEL}"
echo "  Train Data     : ${TRAIN_DATA}"
echo "  Save Path      : ${SAVE_PATH}"
echo "  Nodes x GPUs   : ${NUM_NODES} x ${NUM_GPUS}"
echo "  KD Algorithm   : DSKD (dual-space, same tokenizer branch)"
echo "  Loss Function  : RKL (used in both t2s and s2t paths)"
echo "  Projector LR   : 1e-4"
echo "========================================"
echo ""

cd "$(dirname "$0")/KDFlow-main"

python -m kdflow.cli.train_kd_on_policy ${OPTS}

echo ""
echo "========================================"
echo "On-Policy KD Training (DSKD) Complete!"
echo "Checkpoints saved to: ${SAVE_PATH}"
echo "========================================"
