#!/bin/bash

INPUT_FILE="/path/to/workspace/distill_exposure_bias/preliminary/your_data.jsonl"
OUTPUT_PREFIX="/path/to/workspace/distill_exposure_bias/preliminary/visualization/logprob_analysis"
API_URL="http://127.0.0.1:30011/generate"
MODEL_PATH="/path/to/your/model"
WINDOW_SIZE=10
NUM_WORKERS=8

python /path/to/workspace/distill_exposure_bias/preliminary/compute_and_analyze_logprobs.py \
    --input_file "$INPUT_FILE" \
    --output_prefix "$OUTPUT_PREFIX" \
    --api_url "$API_URL" \
    --model_path "$MODEL_PATH" \
    --window_size "$WINDOW_SIZE" \
    --num_workers "$NUM_WORKERS"
