#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EVAL_DATASETS="math_500 aime24 aime25 amc gpqa_diamond"
export EVAL_N_SAMPLES=8

# ── SKD baseline checkpoints (Qwen3-4B student) ──────────────────────────────
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/skd_4b_g06"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/skd_4b_g07"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/skd_4b_g08"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/skd_4b_g09"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/skd_4b_g10"

# ── MOTAB checkpoints (Qwen3-4B student) ─────────────────────────────────────
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/motab_4b_ep01"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/motab_4b_ep02"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/motab_4b_ep03"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/motab_4b_ep04"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/motab_4b_ep05"

# ── SKD baseline checkpoints (Qwen2-7B student) ──────────────────────────────
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/skd_7b_g06"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/skd_7b_g07"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/skd_7b_g08"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/skd_7b_g09"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/skd_7b_g10"

# ── MOTAB checkpoints (Qwen2-7B student) ─────────────────────────────────────
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/motab_7b_ep01"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/motab_7b_ep02"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/motab_7b_ep03"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/motab_7b_ep04"
bash "$SCRIPT_DIR/run_evaluation.sh" "/path/to/checkpoints/motab_7b_ep05"
