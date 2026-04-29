#!/usr/bin/env python3
"""
3-generate-teacher-only.py

Pure Teacher Model Trajectory Generation Script.
Only uses teacher model to generate step-by-step reasoning trajectories.
No student model, no MOTAB/SKD decision logic.

Algorithm:
  1. Teacher model generates one step (using ".\n\n" as step separator)
  2. Record the step with its logprobs
  3. Continue generating until EOS or max total tokens
  4. Generate num_responses times for each question
"""

import json
import argparse
import requests
import math
import multiprocessing as mp
from typing import List, Dict, Any, Optional
import os
import sys
from tqdm import tqdm
from collections import Counter

# Global tokenizer (set via process pool initialization function to avoid reloading per item)
_tokenizer = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions: File I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    print(f"Loading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    print(f"Loading complete: {len(data)} items")
    return data


def save_jsonl_chunk(data: List[Dict], file_path: str, mode: str = 'a'):
    """Batch write data to JSONL file"""
    if not data:
        return
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions: Logprobs Processing
# ─────────────────────────────────────────────────────────────────────────────

def parse_logprobs(raw_logprobs: List) -> List[List]:
    """
    Parse logprobs returned from SGLang into unified format:
    [[logit, token_id, token_str], ...]
    """
    logprobs = []
    for item in raw_logprobs:
        if isinstance(item, list) and len(item) >= 3:
            # Handle None values, use 0.0 as default
            logit = item[0] if item[0] is not None else 0.0
            token_id = item[1] if item[1] is not None else -1
            token_str = item[2] if item[2] is not None else ''
            logprobs.append([logit, token_id, token_str])
        elif isinstance(item, dict):
            logprobs.append([
                item.get('logprob', 0.0) or 0.0,
                item.get('token_id', -1) or -1,
                item.get('decoded_token', '') or ''
            ])
    return logprobs


def avg_prob_from_logprobs(logprobs: List[List]) -> float:
    """Calculate average probability from logprobs (average logprob with exp)"""
    if not logprobs:
        return 0.0
    lp_values = [item[0] for item in logprobs]
    avg_lp = sum(lp_values) / len(lp_values)
    return math.exp(avg_lp)


def get_finish_type(meta_info: Dict) -> str:
    """Extract finish_reason type string from meta_info"""
    finish_reason = meta_info.get('finish_reason', {})
    if isinstance(finish_reason, dict):
        return finish_reason.get('type', 'unknown')
    elif isinstance(finish_reason, str):
        return finish_reason
    return 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# Core Logic: API Calls
# ─────────────────────────────────────────────────────────────────────────────

def generate_step(context: str, api_url: str, temperature: float, max_step_tokens: int) -> Dict:
    """
    Call SGLang API to generate a step with stop=[".\n\n"] as separator.

    Return format:
        {
            'success':     bool,
            'text':        str,        # Generated step text (excluding context)
            'logprobs':    List[List], # [[logit, token_id, token_str], ...]
            'finish_type': str,        # 'stop' | 'eos' | 'length' | ...
            'error':       str | None
        }
    """
    payload = {
        "text": context,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_step_tokens,
            "stop": [".\n\n"],
        }
    }
    try:
        response = requests.post(api_url, json=payload, timeout=10800)
        response.raise_for_status()
        result = response.json()

        generated_text = result.get('text', '')
        # SGLang /generate endpoint sometimes returns full text including prompt, need to truncate context part
        if generated_text.startswith(context):
            generated_text = generated_text[len(context):]

        meta_info  = result.get('meta_info', {})
        logprobs   = parse_logprobs(meta_info.get('output_token_logprobs', []))
        finish_type = get_finish_type(meta_info)

        return {
            'success':     True,
            'text':        generated_text,
            'logprobs':    logprobs,
            'finish_type': finish_type,
            'error':       None
        }
    except requests.exceptions.Timeout:
        return {
            'success':     False,
            'text':        '',
            'logprobs':    [],
            'finish_type': 'error',
            'error':       'Request timeout (>10800s)'
        }
    except Exception as e:
        return {
            'success':     False,
            'text':        '',
            'logprobs':    [],
            'finish_type': 'error',
            'error':       str(e)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pure Teacher Trajectory Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_teacher_response(
    question: str,
    teacher_url: str,
    temperature: float,
    max_step_tokens: int,
    max_total_tokens: int,
    item_id: str,
    response_idx: int
) -> Dict:
    """
    Generate one complete response using only the teacher model.

    Teacher generates step-by-step with stop=[".\n\n"] delimiter,
    accumulating steps until EOS or max_total_tokens is reached.

    Return format:
        {
            'response':     str,         # Complete response text
            'steps':        List[Dict],  # Details for each step
            'total_steps':  int,
            'total_tokens': int,
            'student_rate': float,       # Always 0.0 (no student used)
            'teacher_rate': float,       # Always 1.0 (all steps from teacher)
        }
    """
    context      = question
    steps        = []
    total_tokens = 0
    step_idx     = 0

    while total_tokens < max_total_tokens:

        # ── Teacher generates next step ─────────────────────────────────
        teacher_result = generate_step(context, teacher_url, temperature, max_step_tokens)

        if not teacher_result['success']:
            break

        teacher_step_text = teacher_result['text']
        teacher_logprobs  = teacher_result['logprobs']
        teacher_finish    = teacher_result['finish_type']
        teacher_avg_prob  = avg_prob_from_logprobs(teacher_logprobs)

        # If generated text is empty (e.g., direct EOS), treat as generation complete
        if not teacher_step_text:
            break

        # ── Record step information ─────────────────────────────────────
        step_info = {
            'step_index':                        step_idx,
            'source':                            'teacher',
            'step_text':                         teacher_step_text,
            'student_avg_prob':                  round(teacher_avg_prob, 8),
            'teacher_avg_prob_for_student_step': None,  # No separate student scoring
            'logprobs':                          teacher_logprobs,
            'finish_type':                       teacher_finish,
        }
        steps.append(step_info)

        # Update context (append step text + "\n\n" separator)
        context       = context + teacher_step_text + "\n\n"
        total_tokens += len(teacher_logprobs)
        step_idx     += 1

        # Debug output: print detailed info every 50 steps
        if step_idx % 50 == 0:
            print(f"  [DEBUG] Step {step_idx}: text_len={len(teacher_step_text)}, "
                  f"logprobs_count={len(teacher_logprobs)}, total_tokens={total_tokens}, "
                  f"finish={teacher_finish}")

        # ── Termination conditions ──────────────────────────────────────
        # 1. Reach max token count
        if total_tokens >= max_total_tokens:
            break

        # 2. finish_type is not 'stop' (e.g., 'eos', 'length'), indicates generation complete
        if teacher_finish != 'stop':
            break

    # ── Statistics ──────────────────────────────────────────────────────
    # All steps are from teacher
    student_count = 0
    teacher_count = len(steps)
    total_steps_count = len(steps)

    student_rate = 0.0
    teacher_rate = 1.0 if total_steps_count > 0 else 0.0

    # Debug: analyze step length distribution
    step_lengths = [len(s['step_text']) for s in steps]
    step_token_counts = [len(s['logprobs']) for s in steps]
    avg_step_len = sum(step_lengths) / len(step_lengths) if step_lengths else 0
    avg_step_tokens = sum(step_token_counts) / len(step_token_counts) if step_token_counts else 0
    empty_steps = sum(1 for sl in step_lengths if sl == 0)

    print(f"\n  [{'='*40}]")
    print(f"  [Stats] ID:{item_id} R:{response_idx} | Steps: {len(steps)}, Total Tokens: {total_tokens}")
    print(f"  [Stats] Average Step length: {avg_step_len:.1f} chars, Average Step tokens: {avg_step_tokens:.1f}")
    print(f"  [Stats] Empty step count: {empty_steps}/{len(steps)}")
    print(f"  [Stats] finish_type distribution:")
    if steps:
        finish_dist = Counter(s['finish_type'] for s in steps)
        for ft, count in finish_dist.most_common():
            print(f"    - '{ft}': {count} ({count/len(steps)*100:.1f}%)")
    print(f"  [{'='*40}]\n")

    full_response = "\n\n".join(s['step_text'] for s in steps)

    return {
        'response':        full_response,
        'steps':           steps,
        'total_steps':     len(steps),
        'total_tokens':    total_tokens,
        'student_rate':    round(student_rate, 4),
        'teacher_rate':    round(teacher_rate, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Process Worker Functions
# ─────────────────────────────────────────────────────────────────────────────

def init_worker(teacher_model_path: str):
    """
    Process pool initialization function: called once when each worker process starts,
    loads tokenizer to global variable. Avoids reloading tokenizer for each data item.
    """
    global _tokenizer
    if teacher_model_path:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
        print(f"[Worker PID:{os.getpid()}] Tokenizer loading complete: {teacher_model_path}")
    else:
        _tokenizer = None
        print(f"[Worker PID:{os.getpid()}] teacher_model_path not provided, "
              f"using character count approximation for logprob_start_len")


def process_item_worker(args_tuple):
    """Multi-process worker function: process single data item (generate num_responses responses)"""
    global _tokenizer

    (data_item, teacher_url, temperature,
     max_step_tokens, max_total_tokens, num_responses) = args_tuple

    item_id  = data_item.get('id', 'Unknown')
    question = data_item.get('input', '')

    if not question:
        print(f"[SKIP] ID:{item_id} | 'input' field is empty")
        return {
            'id':        item_id,
            'input':     question,
            'responses': [],
            'status':    'skipped_no_input'
        }

    print(f"\n{'='*50}")
    print(f"[START] ID:{item_id} | Will generate {num_responses} responses (Teacher-only)")

    responses_list = []

    for r_idx in range(num_responses):
        resp = generate_teacher_response(
            question         = question,
            teacher_url      = teacher_url,
            temperature      = temperature,
            max_step_tokens  = max_step_tokens,
            max_total_tokens = max_total_tokens,
            item_id          = item_id,
            response_idx     = r_idx
        )
        responses_list.append(resp)
        t_rate = resp.get('teacher_rate', 0)
        print(f"[R{r_idx+1}/{num_responses}] ID:{item_id} | "
              f"steps={resp['total_steps']} tokens={resp['total_tokens']} | "
              f"Teacher: {t_rate:.2%}")

    # All steps are from teacher
    avg_student_rate = 0.0
    avg_teacher_rate = 1.0 if responses_list else 0.0
    print(f"[DONE] ID:{item_id} | Teacher-only generation complete | Teacher rate: {avg_teacher_rate:.2%}")

    return {
        'id':             item_id,
        'input':          question,
        'responses':      responses_list,
        'student_rate':   round(avg_student_rate, 4),
        'teacher_rate':   round(avg_teacher_rate, 4),
        'status':         'success'
    }


# ─────────────────────────────────────────────────────────────────────────────
# Process Control
# ─────────────────────────────────────────────────────────────────────────────

def run_process(
    data: List[Dict],
    teacher_url: str,
    temperature: float,
    max_step_tokens: int,
    max_total_tokens: int,
    num_responses: int,
    output_file: str,
    num_workers: int,
    batch_size: int,
    teacher_model_path: str
):
    total = len(data)
    print(f"Starting processing (Teacher-only), parallel workers: {num_workers}, total {total} items")
    save_jsonl_chunk([], output_file, mode='w')  # Clear output file
    print("-" * 60)

    tasks = [
        (item, teacher_url, temperature,
         max_step_tokens, max_total_tokens, num_responses)
        for item in data
    ]
    buffer = []

    if num_workers <= 1:
        # ── Single Process Mode ────────────────────────────────────────────────
        init_worker(teacher_model_path)
        for i, task in enumerate(tqdm(tasks, desc="Processing progress", unit="items")):
            res = process_item_worker(task)
            buffer.append(res)
            if len(buffer) >= batch_size:
                save_jsonl_chunk(buffer, output_file, mode='a')
                buffer = []
    else:
        # ── Multi-Process Mode ────────────────────────────────────────────────
        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(teacher_model_path,)
        ) as pool:
            iterator = pool.imap_unordered(process_item_worker, tasks, chunksize=1)
            for res in tqdm(iterator, total=total, desc="Processing progress", unit="items"):
                buffer.append(res)
                if len(buffer) >= batch_size:
                    save_jsonl_chunk(buffer, output_file, mode='a')
                    buffer = []

    if buffer:
        save_jsonl_chunk(buffer, output_file, mode='a')

    print("-" * 60)
    print("All processing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pure Teacher model trajectory generation script: step-by-step generation with teacher only"
    )

    # ── File Paths ──────────────────────────────────────────────────────
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Input JSONL file path (each line contains 'input' field)"
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Output JSONL file path"
    )

    # ── API Endpoints ──────────────────────────────────────────────────────
    parser.add_argument(
        "--teacher_url", type=str,
        default="http://127.0.0.1:30011/generate",
        help="Teacher model SGLang API endpoint (default: http://127.0.0.1:30011/generate)"
    )

    # ── Model Path (for loading tokenizer, precise calculation of logprob_start_len) ────
    parser.add_argument(
        "--teacher_model_path", type=str,
        default="/path/to/models/Qwen3-32B",
        help="Teacher model path, used to load tokenizer for precise calculation of logprob_start_len. "
             "(default: /path/to/models/Qwen3-32B)"
    )

    # ── Generation Parameters ────────────────────────────────────────────────
    parser.add_argument(
        "--num_responses", type=int, default=5,
        help="Number of responses to generate per question (default: 5)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Generation temperature (default: 0.6)"
    )
    parser.add_argument(
        "--max_step_tokens", type=int, default=8192,
        help="Maximum tokens per step (default: 8192)"
    )
    parser.add_argument(
        "--max_total_tokens", type=int, default=32768,
        help="Maximum total tokens per complete response (default: 32768)"
    )

    # ── Control Parameters ───────────────────────────────────────────────────
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel processes (1=single process, >1=multi-process) (default: 1)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Number of items to process before writing to disk (default: 10)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Pure Teacher-Only Trajectory Generation Configuration:")
    print(f"  Input File         : {args.input_file}")
    print(f"  Output File        : {args.output_file}")
    print(f"  Teacher API URL    : {args.teacher_url}")
    print(f"  Teacher Model Path : {args.teacher_model_path or 'Not specified (character count approximation)'}")
    print(f"  Responses per Item : {args.num_responses}")
    print(f"  Temperature        : {args.temperature}")
    print(f"  Max tokens/step    : {args.max_step_tokens}")
    print(f"  Max tokens/response: {args.max_total_tokens}")
    print(f"  Parallel Workers   : {args.num_workers}")
    print(f"  Disk Write Batch   : {args.batch_size}")
    print(f"  Request Timeout    : 10800s (3 hours)")
    print("=" * 60)

    data = load_jsonl(args.input_file)
    if not data:
        print("Error: Input file is empty or cannot be read.")
        return

    run_process(
        data              = data,
        teacher_url       = args.teacher_url,
        temperature       = args.temperature,
        max_step_tokens   = args.max_step_tokens,
        max_total_tokens  = args.max_total_tokens,
        num_responses     = args.num_responses,
        output_file       = args.output_file,
        num_workers       = args.num_workers,
        batch_size        = args.batch_size,
        teacher_model_path= args.teacher_model_path
    )


if __name__ == "__main__":
    # Ensure multi-processing works correctly on macOS/Windows
    mp.set_start_method('spawn', force=True)
    main()
