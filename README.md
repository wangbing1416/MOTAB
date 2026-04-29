# MOTAB for LLM Reasoning Distillation

This repository contains the full pipeline for **MOTAB** — a framework for distilling long-chain reasoning capabilities from large teacher models into small student models.

The pipeline covers four stages:
1. **Data Synthesis** — Generate MOTAB SFT training data
2. **SFT Training** — Fine-tune the student model on synthesized data using [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory)
3. **On-Policy KD Training** — Further distill the student model online using [KDFlow](https://github.com/songmzhang/KDFlow)
4. **Evaluation** — Evaluate on math reasoning benchmarks using [EvalScope](https://github.com/modelscope/evalscope)

---

## Table of Contents

- [Step 1: Data Synthesis](#step-1-data-synthesis)
- [Step 2: SFT Training](#step-2-sft-training)
- [Step 3: On-Policy KD Training](#step-3-on-policy-kd-training)
- [Step 4: Evaluation](#step-4-evaluation)
- [License](#license)

---

## Step 1: Data Synthesis

### Dependencies

```bash
pip install "sglang[all]>=0.5.9"
pip install aiohttp tqdm transformers
```

### Data Source

The synthesis scripts take a JSONL file as input where each line contains an `input` field with the question text:

```json
{"id": "problem_001", "input": "Find all integer solutions to x^2 + y^2 = z^2.", ...}
```

**You can use the [LIMO dataset](https://huggingface.co/datasets/GAIR/LIMO) directly** — its format is compatible with all synthesis scripts out of the box.

```python
from datasets import load_dataset
import json

ds = load_dataset("GAIR/LIMO", split="train")
with open("limo_train.jsonl", "w") as f:
    for item in ds:
        f.write(json.dumps(item) + "\n")
```

### Start SGLang Servers

All generation scripts require a teacher and student SGLang server running beforehand:

```bash
# Teacher model (e.g., Qwen3-32B) — adjust CUDA_VISIBLE_DEVICES and --tp as needed
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
    --model-path /path/to/models/Qwen3-32B \
    --port 30011 --tp 4

# Student model (e.g., Qwen3-4B)
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
    --model-path /path/to/models/Qwen3-4B \
    --port 30012 --tp 1
```

### MOTAB Data Synthesis (Core Method)

```bash
INPUT_FILE=/path/to/limo_train.jsonl \
OUTPUT_FILE=/path/to/motab_output.jsonl \
TEACHER_URL=http://127.0.0.1:30011/generate \
STUDENT_URL=http://127.0.0.1:30012/generate \
TEACHER_MODEL_PATH=/path/to/models/Qwen3-32B \
bash 1-generate-ours-qwen32b-qwen4b.sh
```

Key parameters (override via environment variables):

| Variable | Default | Description |
|---|---|---|
| `EPSILON0` | `0.2` | Base acceptance threshold ε₀ |
| `THETA` | `1.0` | Entropy scaling factor θ |
| `TOP_K_ENTROPY` | `20` | Top-k tokens for entropy estimation |
| `NUM_RESPONSES` | `5` | Trajectories per question |
| `TEMPERATURE` | `0.6` | Sampling temperature |
| `MAX_STEP_TOKENS` | `8192` | Max tokens per reasoning step |
| `MAX_TOTAL_TOKENS` | `32768` | Max tokens per full trajectory |
| `NUM_WORKERS` | `1` | Parallel worker processes |

### SKD Baseline Synthesis

```bash
INPUT_FILE=/path/to/limo_train.jsonl \
OUTPUT_FILE=/path/to/baseline_output.jsonl \
TEACHER_URL=http://127.0.0.1:30011/generate \
STUDENT_URL=http://127.0.0.1:30012/generate \
TEACHER_MODEL_PATH=/path/to/models/Qwen3-32B \
bash 2-generate-baseline-g09-qwen32b-qwen4b.sh
```

The provided script uses γ=0.9 with Qwen3-32B as teacher. The switching threshold γ can be tuned via the `GAMMA` environment variable.

For the full SKD (Speculative Knowledge Distillation) algorithm details, see `2-generate-baseline.py`.

### Teacher-Only Generation (Upper Bound)

```bash
INPUT_FILE=/path/to/limo_train.jsonl \
OUTPUT_FILE=/path/to/teacher_output.jsonl \
TEACHER_URL=http://127.0.0.1:30011/generate \
TEACHER_MODEL_PATH=/path/to/models/Qwen3-32B \
bash 3-generate-teacher-only.sh
```

### Reorganize Output

After synthesis, reorganize the raw output into a clean JSONL format for training:

```bash
python 4-reorganize-vgds-output.py \
    --input_file /path/to/motab_output.jsonl \
    --output_file /path/to/motab_train.jsonl

python 4-reorganize-baseline-output.py \
    --input_file /path/to/baseline_output.jsonl \
    --output_file /path/to/baseline_train.jsonl
```

---

## Step 2: SFT Training

SFT training uses [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory). Training scripts are located in the `SFT/` directory.

### Dependencies

Install LLaMAFactory following the [official guide](https://github.com/hiyouga/LLaMA-Factory#getting-started):

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

A DeepSpeed ZeRO-3 config (`ds_z3_config.json`) is required. LLaMAFactory provides one at `examples/deepspeed/ds_z3_config.json`.

### Dataset Registration

LLaMAFactory requires datasets to be registered in `data/dataset_info.json`. Add an entry for your synthesized dataset:

```json
{
  "motab_train": {
    "file_name": "/path/to/motab_train.jsonl",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  }
}
```

Refer to the [LLaMAFactory dataset documentation](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) for the full format specification.

### Training Configuration

Key parameters in `SFT/sft_config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `model_name_or_path` | — | Path to base model |
| `dataset` | `motab_train` | Dataset name (must match `dataset_info.json`) |
| `template` | `qwen` | Chat template for the model |
| `cutoff_len` | `32768` | Maximum sequence length |
| `output_dir` | — | Checkpoint output directory |
| `per_device_train_batch_size` | `4` | Batch size per GPU |
| `gradient_accumulation_steps` | `1` | Gradient accumulation steps |
| `learning_rate` | `5.0e-5` | Learning rate |
| `num_train_epochs` | `6.0` | Number of training epochs |
| `lr_scheduler_type` | `cosine_with_min_lr` | LR scheduler |
| `deepspeed` | — | Path to DeepSpeed ZeRO-3 config |

### Run Training

```bash
cd LLaMA-Factory/

BASE_MODEL_PATH=/path/to/models/Qwen3-4B-Instruct \
DATASET_INFO=/path/to/data/dataset_info.json \
NGPUS=8 \
bash /path/to/SFT/run_sft.sh
```

For multi-node training, set `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, and `RANK` accordingly.

---

## Step 3: On-Policy KD Training

On-policy KD training uses [KDFlow](https://github.com/songmzhang/KDFlow) with FSDP2 for student training and SGLang for teacher inference and on-policy rollout.

### Dependencies

Clone KDFlow into the project root and install:

```bash
git clone https://github.com/songmzhang/KDFlow.git KDFlow-main
cd KDFlow-main
pip install -e .
pip install flash_attn==2.8.3 --no-build-isolation
```


### Prepare Training Data

Convert the LIMO JSONL (`input` field) to KDFlow on-policy format (`messages` field):

```bash
python prepare_onpolicy_data.py \
    --input_file /path/to/limo_train.jsonl \
    --output_file /path/to/kdflow_prompts.jsonl \
    --deduplicate
```

Output format per line:
```json
{"id": "problem_001", "messages": [{"role": "user", "content": "Find all integer solutions..."}]}
```

### Start Ray (first run only)

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
```

### Run Training

```bash
STUDENT_MODEL=/path/to/models/Qwen3-4B \
TEACHER_MODEL=/path/to/models/Qwen3-32B \
TRAIN_DATA=/path/to/kdflow_prompts.jsonl \
SAVE_PATH=/path/to/checkpoints/run01 \
NUM_NODES=1 NUM_GPUS=8 \
bash run_onpolicy_kd.sh
```

Three loss function variants are provided:

| Script | Loss Function | Algorithm |
|---|---|---|
| `run_onpolicy_kd.sh` | Reverse KL (`rkl`) | `vanilla_kd` |
| `run_onpolicy_kd_akl.sh` | Adaptive KL (`akl`) | `vanilla_kd` |
| `run_onpolicy_kd_dskd.sh` | KL (`kl`) | `dskd` |

### Key Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `STUDENT_MODEL` | — | Path to student model |
| `TEACHER_MODEL` | — | Path to teacher model |
| `TRAIN_DATA` | — | KDFlow-format prompts JSONL |
| `SAVE_PATH` | — | Checkpoint output directory |
| `NUM_NODES` | `1` | Number of training nodes |
| `NUM_GPUS` | `8` | GPUs per node |
| `--num_epochs` | `10` | Training epochs |
| `--learning_rate` | `2e-6` | Learning rate |
| `--train_batch_size` | `16` | Global batch size |
| `--n_samples_per_prompt` | `5` | Rollout samples per prompt |
| `--generate_max_len` | `32768` | Max generation length |
| `--enable_sleep` | `True` | Share GPUs between teacher/student/rollout |

> **Note on GPU memory:** With `--enable_sleep True`, teacher, student, and rollout engines time-share the same GPUs. Keep `--max_token_len_per_gpu ≤ 8192` if enabling dynamic batching, as Qwen3's vocab size (151,936) causes ~10 GB logit memory per 32K tokens.

For the full KDFlow argument reference, see the [KDFlow README](https://github.com/songmzhang/KDFlow#️-arguments).

---

## Step 4: Evaluation

Evaluation uses [EvalScope](https://github.com/modelscope/evalscope) with a locally running SGLang server. Evaluation scripts are in the `eval/` directory.

### Dependencies

```bash
pip install evalscope
pip install "sglang[all]>=0.5.9"
```

### Run the Full Evaluation Pipeline

`eval/run_evaluation.sh` automatically starts a SGLang server, runs EvalScope evaluation, then shuts the server down:

```bash
cd eval/

bash run_evaluation.sh /path/to/your/checkpoint
```

Override configuration via environment variables:

```bash
MODEL_NAME=my_model \
TP_SIZE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
EVAL_DATASETS="math_500 aime24 aime25 gpqa_diamond" \
EVAL_N_SAMPLES=8 \
bash run_evaluation.sh /path/to/your/checkpoint
```

### Supported Benchmarks

| Dataset | Description |
|---|---|
| `math_500` | MATH-500 competition math |
| `aime24` | AIME 2024 |
| `aime25` | AIME 2025 |
| `gpqa_diamond` | GPQA Diamond (graduate-level science) |

### Run EvalScope Directly

If you already have a running SGLang server:

```bash
python eval/run_evalscope.py \
    --model_name my_model \
    --api_url http://127.0.0.1:31011/v1/ \
    --datasets math_500 aime24 aime25 amc gpqa_diamond \
    --n_samples 4 \
    --temperature 0.6 \
    --max_tokens 32768
```

EvalScope calls the OpenAI-compatible API exposed by SGLang and reports `mean` and `pass@k` accuracy per benchmark.

### Evaluation Parameters

| Variable | Default | Description |
|---|---|---|
| `EVAL_DATASETS` | `math_500 aime24 aime25 amc gpqa_diamond` | Benchmarks to run |
| `EVAL_N_SAMPLES` | `4` | Samples per question (pass@k / majority vote) |
| `EVAL_TEMPERATURE` | `0.6` | Sampling temperature |
| `EVAL_MAX_TOKENS` | `32768` | Max generation tokens |
| `EVAL_BATCH_SIZE` | `64` | Request batch size |
| `PORT` | `31011` | SGLang server port |
| `TP_SIZE` | `8` | Tensor parallel degree |

---

## License

This project is released under the Apache License 2.0.

The [KDFlow](https://github.com/songmzhang/KDFlow) library (cloned as `KDFlow-main/`) is a third-party library under the [MIT License](https://github.com/songmzhang/KDFlow/blob/main/LICENSE).
