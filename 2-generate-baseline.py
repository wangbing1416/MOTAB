#!/usr/bin/env python3
"""
2-generate-baseline.py

SKD (Speculative Knowledge Distillation) Generation Script (Async High-Concurrency Version).

Algorithm:
  1. Student model generates one step (using ".\n\n" as step separator)
  2. Teacher model scores the step (via input_token_logprobs)
  3. If teacher_avg_prob > γ * student_avg_prob, accept student output;
     otherwise, teacher regenerates the step
  4. Repeat until the entire response is generated
  5. Generate num_responses times for each question

Optimizations (vs. original sequential version):
  - Async I/O: aiohttp replaces requests; other coroutines run during I/O waits
  - Concurrent response generation: num_responses replies via asyncio.gather
  - Student speculative prefetch: while Teacher scores step N, Student pre-generates step N+1
  - Multi-question concurrency: asyncio.Semaphore controls max concurrent questions
"""

import json
import argparse
import asyncio
import aiohttp
import math
from typing import List, Dict, Any, Optional
import os
from tqdm import tqdm
from collections import Counter


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
    print(f"Loaded: {len(data)} items")
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
            logit    = item[0] if item[0] is not None else 0.0
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
    """Compute geometric mean probability from logprobs (average logprob with exp)"""
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
# Core Logic: Async API Calls
# ─────────────────────────────────────────────────────────────────────────────

async def generate_step_async(
    context: str,
    api_url: str,
    temperature: float,
    max_step_tokens: int,
    session: aiohttp.ClientSession
) -> Dict:
    """
    Async call to SGLang API to generate one step with stop=[".\n\n"] separator.

    Return format:
        {
            'success':     bool,
            'text':        str,
            'logprobs':    List[List],
            'finish_type': str,
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
        async with session.post(api_url, json=payload) as response:
            response.raise_for_status()
            result = await response.json()

        generated_text = result.get('text', '')
        # SGLang /generate may return full text including prompt; strip context prefix
        if generated_text.startswith(context):
            generated_text = generated_text[len(context):]

        meta_info   = result.get('meta_info', {})
        logprobs    = parse_logprobs(meta_info.get('output_token_logprobs', []))
        finish_type = get_finish_type(meta_info)

        return {
            'success':     True,
            'text':        generated_text,
            'logprobs':    logprobs,
            'finish_type': finish_type,
            'error':       None
        }
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return {
            'success':     False,
            'text':        '',
            'logprobs':    [],
            'finish_type': 'error',
            'error':       str(e)
        }


async def score_step_async(
    context: str,
    step_text: str,
    api_url: str,
    tokenizer,
    session: aiohttp.ClientSession
) -> Dict:
    """
    Async: compute per-token logprobs for step_text using the teacher model.

    Sends full_text = context + step_text with logprob_start_len to skip the context
    prefix, returning only input_token_logprobs for the step_text tokens.

    Return format:
        {
            'success':  bool,
            'logprobs': List[List],
            'error':    str | None
        }
    """
    full_text = context + step_text

    if tokenizer is not None:
        prefix_tokens = tokenizer.encode(context, add_special_tokens=False)
        start_len = len(prefix_tokens)
    else:
        # Fallback: character-count approximation (imprecise; provide teacher_model_path for accuracy)
        start_len = len(context)

    payload = {
        "text": full_text,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "logprob_start_len": start_len,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,
        }
    }
    try:
        async with session.post(api_url, json=payload) as response:
            response.raise_for_status()
            result = await response.json()

        meta_info    = result.get('meta_info', {})
        raw_logprobs = meta_info.get('input_token_logprobs', [])

        if not raw_logprobs:
            return {
                'success':  False,
                'logprobs': [],
                'error':    'No input_token_logprobs returned from SGLang'
            }

        return {
            'success':  True,
            'logprobs': parse_logprobs(raw_logprobs),
            'error':    None
        }
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return {
            'success':  False,
            'logprobs': [],
            'error':    str(e)
        }


# ─────────────────────────────────────────────────────────────────────────────
# SKD Generation (Async + Student Speculative Prefetch Pipeline)
# ─────────────────────────────────────────────────────────────────────────────

async def generate_skd_response_async(
    question: str,
    student_url: str,
    teacher_url: str,
    gamma: float,
    temperature: float,
    max_step_tokens: int,
    max_total_tokens: int,
    tokenizer,
    item_id: str,
    response_idx: int,
    session: aiohttp.ClientSession
) -> Dict:
    """
    Async generation of one complete response using SKD with speculative prefetch.

    Speculative Prefetch Strategy:
        While Teacher scores Step N, Student concurrently pre-generates Step N+1.
        - If Step N accepted: Step N+1 is already ready or in progress (saves one full wait)
        - If Step N rejected: cancel prefetch, Teacher regenerates Step N from scratch

    Timing comparison:
        Sequential: [Student N] → wait → [Teacher Score N] → wait → [Student N+1] → ...
        Optimized:  [Student N] → wait → [Teacher Score N] + [Student N+1 prefetch] → ...
                                         ↑ Both run concurrently, effective wait ≈ max(T_score, T_gen)
    """
    context      = question
    steps        = []
    total_tokens = 0
    step_idx     = 0
    prefetch_task: Optional[asyncio.Task] = None  # Student speculative prefetch task handle

    while total_tokens < max_total_tokens:

        # ── Step 1: Get Student result (use speculative prefetch if available) ────────────────
        if prefetch_task is not None:
            student_result = await prefetch_task
            prefetch_task  = None
        else:
            student_result = await generate_step_async(
                context, student_url, temperature, max_step_tokens, session
            )

        if not student_result['success']:
            break

        student_step_text = student_result['text']
        student_logprobs  = student_result['logprobs']
        student_finish    = student_result['finish_type']
        student_avg_prob  = avg_prob_from_logprobs(student_logprobs)

        # Empty generation (e.g., direct EOS) → generation complete
        if not student_step_text:
            break

        # ── Step 2: Concurrently run Teacher scoring + Student speculative prefetch ──
        #   Teacher scores current step while Student pre-generates next step (assuming accepted)
        next_context_if_accepted = context + student_step_text + "\n\n"

        teacher_score_task = asyncio.create_task(
            score_step_async(context, student_step_text, teacher_url, tokenizer, session)
        )

        # Prefetch only when current step may not be the final one (avoid unnecessary requests)
        tokens_so_far   = total_tokens + len(student_logprobs)
        should_prefetch = (student_finish == 'stop' and tokens_so_far < max_total_tokens)
        if should_prefetch:
            speculative_task = asyncio.create_task(
                generate_step_async(
                    next_context_if_accepted, student_url, temperature, max_step_tokens, session
                )
            )
        else:
            speculative_task = None

        # Wait for Teacher scoring (speculative_task runs concurrently in background)
        teacher_score    = await teacher_score_task
        teacher_avg_prob = 0.0
        if teacher_score['success']:
            teacher_avg_prob = avg_prob_from_logprobs(teacher_score['logprobs'])

        # ── Step 3: Decision — gamma threshold comparison ───────────────────────
        #   teacher_avg_prob > γ × student_avg_prob → accept student output
        #   otherwise → teacher regenerates the step
        if teacher_score['success'] and teacher_avg_prob > gamma * student_avg_prob:
            # Accept Student output, reuse speculative prefetch (if available)
            accepted_text     = student_step_text
            accepted_logprobs = student_logprobs
            accepted_finish   = student_finish
            source            = 'student'
            context           = next_context_if_accepted
            prefetch_task     = speculative_task  # directly await speculative result next iteration
        else:
            # Reject: cancel speculative prefetch (based on wrong context), Teacher regenerates
            if speculative_task is not None:
                speculative_task.cancel()
                try:
                    await speculative_task
                except asyncio.CancelledError:
                    pass

            teacher_gen = await generate_step_async(
                context, teacher_url, temperature, max_step_tokens, session
            )

            if teacher_gen['success'] and teacher_gen['text']:
                accepted_text     = teacher_gen['text']
                accepted_logprobs = teacher_gen['logprobs']
                accepted_finish   = teacher_gen['finish_type']
            else:
                # Fallback: teacher generation failed, use student output
                accepted_text     = student_step_text
                accepted_logprobs = student_logprobs
                accepted_finish   = student_finish

            source        = 'teacher'
            context       = context + accepted_text + "\n\n"
            prefetch_task = None  # restart prefetch from next iteration

        # ── Record step information ───────────────────────────────────────────────────
        step_info = {
            'step_index':                        step_idx,
            'source':                            source,
            'step_text':                         accepted_text,
            'student_avg_prob':                  round(student_avg_prob, 8),
            'teacher_avg_prob_for_student_step': round(teacher_avg_prob, 8) if teacher_score['success'] else None,
            'logprobs':                          accepted_logprobs,
            'finish_type':                       accepted_finish,
        }
        steps.append(step_info)

        total_tokens += len(accepted_logprobs)
        step_idx     += 1

        # Debug output: print info every 50 steps
        if step_idx % 50 == 0:
            print(f"  [DEBUG] ID:{item_id} R:{response_idx} Step {step_idx}: "
                  f"text_len={len(accepted_text)}, total_tokens={total_tokens}, finish={accepted_finish}")

        # ── Termination conditions ────────────────────────────────────────────
        if total_tokens >= max_total_tokens:
            break
        # finish_type is not 'stop' (e.g., 'eos', 'length') → generation complete
        if accepted_finish != 'stop':
            break

    # Clean up residual prefetch task (in case loop exited while prefetch still running)
    if prefetch_task is not None:
        prefetch_task.cancel()
        try:
            await prefetch_task
        except asyncio.CancelledError:
            pass

    # ── Statistics ─────────────────────────────────────────────────────────────
    student_count     = sum(1 for s in steps if s['source'] == 'student')
    teacher_count     = len(steps) - student_count
    total_steps_count = len(steps)

    student_rate = student_count / total_steps_count if total_steps_count > 0 else 0.0
    teacher_rate = teacher_count / total_steps_count if total_steps_count > 0 else 0.0

    step_lengths      = [len(s['step_text']) for s in steps]
    step_token_counts = [len(s['logprobs']) for s in steps]
    avg_step_len      = sum(step_lengths) / len(step_lengths) if step_lengths else 0
    avg_step_tokens   = sum(step_token_counts) / len(step_token_counts) if step_token_counts else 0
    empty_steps       = sum(1 for sl in step_lengths if sl == 0)

    print(f"\n  [{'='*40}]")
    print(f"  [Stats] ID:{item_id} R:{response_idx} | Steps: {len(steps)}, Total Tokens: {total_tokens}")
    print(f"  [Stats] Avg step length: {avg_step_len:.1f} chars, Avg step tokens: {avg_step_tokens:.1f}")
    print(f"  [Stats] Empty steps: {empty_steps}/{len(steps)}")
    print(f"  [Stats] finish_type distribution:")
    if steps:
        finish_dist = Counter(s['finish_type'] for s in steps)
        for ft, count in finish_dist.most_common():
            print(f"    - '{ft}': {count} ({count / len(steps) * 100:.1f}%)")
    else:
        print("    - (no steps)")
    print(f"  [{'='*40}]\n")

    full_response = "\n\n".join(s['step_text'] for s in steps)

    return {
        'response':     full_response,
        'steps':        steps,
        'total_steps':  len(steps),
        'total_tokens': total_tokens,
        'student_rate': round(student_rate, 4),
        'teacher_rate': round(teacher_rate, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Async Worker: Single Item (Concurrent Generation of num_responses)
# ─────────────────────────────────────────────────────────────────────────────

async def process_item_async(
    data_item: Dict,
    student_url: str,
    teacher_url: str,
    gamma: float,
    temperature: float,
    max_step_tokens: int,
    max_total_tokens: int,
    num_responses: int,
    tokenizer,
    session: aiohttp.ClientSession
) -> Dict:
    """
    Async: process one data item by concurrently generating num_responses complete responses.

    Unlike a sequential loop (for r_idx in range(num_responses)),
    asyncio.gather runs all replies simultaneously, making total time ≈ single-reply time.
    """
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
    print(f"[START] ID:{item_id} | Concurrently generating {num_responses} responses")

    # Concurrently execute all num_responses generations
    tasks = [
        generate_skd_response_async(
            question         = question,
            student_url      = student_url,
            teacher_url      = teacher_url,
            gamma            = gamma,
            temperature      = temperature,
            max_step_tokens  = max_step_tokens,
            max_total_tokens = max_total_tokens,
            tokenizer        = tokenizer,
            item_id          = item_id,
            response_idx     = r_idx,
            session          = session
        )
        for r_idx in range(num_responses)
    ]
    responses_list = await asyncio.gather(*tasks)

    all_student_rate = [r.get('student_rate', 0) for r in responses_list]
    all_teacher_rate = [r.get('teacher_rate', 0) for r in responses_list]
    avg_student_rate = sum(all_student_rate) / len(all_student_rate) if all_student_rate else 0.0
    avg_teacher_rate = sum(all_teacher_rate) / len(all_teacher_rate) if all_teacher_rate else 0.0

    for r_idx, resp in enumerate(responses_list):
        s_rate = resp.get('student_rate', 0)
        t_rate = resp.get('teacher_rate', 0)
        print(f"  [R{r_idx+1}/{num_responses}] ID:{item_id} | "
              f"steps={resp['total_steps']} tokens={resp['total_tokens']} | "
              f"Student: {s_rate:.2%} Teacher: {t_rate:.2%}")

    print(f"[DONE] ID:{item_id} | Avg Student: {avg_student_rate:.2%} | Avg Teacher: {avg_teacher_rate:.2%}")

    return {
        'id':           item_id,
        'input':        question,
        'responses':    list(responses_list),
        'student_rate': round(avg_student_rate, 4),
        'teacher_rate': round(avg_teacher_rate, 4),
        'status':       'success'
    }


# ─────────────────────────────────────────────────────────────────────────────
# Process Control — asyncio.Semaphore Multi-Question Concurrency
# ─────────────────────────────────────────────────────────────────────────────

async def run_all_async(
    data: List[Dict],
    student_url: str,
    teacher_url: str,
    gamma: float,
    temperature: float,
    max_step_tokens: int,
    max_total_tokens: int,
    num_responses: int,
    output_file: str,
    max_concurrent: int,
    batch_size: int,
    tokenizer
):
    """
    Async concurrent processing of all data.

    asyncio.Semaphore controls the maximum number of concurrently processed questions
    to avoid overloading the SGLang server with too many simultaneous requests.
    All questions share a single aiohttp.ClientSession for automatic connection pooling.
    """
    total = len(data)
    print(f"Starting processing | max_concurrent={max_concurrent} | total={total} items")
    save_jsonl_chunk([], output_file, mode='w')  # Clear output file
    print("-" * 60)

    sem        = asyncio.Semaphore(max_concurrent)
    write_lock = asyncio.Lock()
    buffer     = []
    pbar       = tqdm(total=total, desc="Processing", unit="items")

    # Configure aiohttp connection pool: no limit, keepalive enabled, 3h timeout
    connector = aiohttp.TCPConnector(limit=0, force_close=False, enable_cleanup_closed=True)
    timeout   = aiohttp.ClientTimeout(total=10800)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def process_with_sem(item: Dict):
            async with sem:
                return await process_item_async(
                    data_item        = item,
                    student_url      = student_url,
                    teacher_url      = teacher_url,
                    gamma            = gamma,
                    temperature      = temperature,
                    max_step_tokens  = max_step_tokens,
                    max_total_tokens = max_total_tokens,
                    num_responses    = num_responses,
                    tokenizer        = tokenizer,
                    session          = session
                )

        tasks = [asyncio.create_task(process_with_sem(item)) for item in data]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            async with write_lock:
                buffer.append(result)
                if len(buffer) >= batch_size:
                    save_jsonl_chunk(buffer, output_file, mode='a')
                    buffer = []
            pbar.update(1)

    # Flush remaining buffer
    if buffer:
        save_jsonl_chunk(buffer, output_file, mode='a')

    pbar.close()
    print("-" * 60)
    print("All processing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SKD (Speculative Knowledge Distillation): step-by-step Student/Teacher selection (async high-concurrency)"
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
        "--student_url", type=str,
        default="http://127.0.0.1:30012/generate",
        help="Student model SGLang API endpoint (default: http://127.0.0.1:30012/generate)"
    )
    parser.add_argument(
        "--teacher_url", type=str,
        default="http://127.0.0.1:30011/generate",
        help="Teacher model SGLang API endpoint (default: http://127.0.0.1:30011/generate)"
    )

    # ── Model Path (for loading tokenizer, precise logprob_start_len) ────
    parser.add_argument(
        "--teacher_model_path", type=str, default="/path/to/teacher/model",
        help="Teacher model path for loading tokenizer (for precise logprob_start_len)"
    )

    # ── SKD Parameters ──────────────────────────────────────────────────────
    parser.add_argument(
        "--gamma", type=float, default=0.9,
        help="Acceptance threshold γ: accept student if teacher_prob > γ × student_prob (default: 0.9)"
    )
    parser.add_argument(
        "--num_responses", type=int, default=5,
        help="Number of responses to generate per question (default: 5)"
    )

    # ── Generation Parameters ──────────────────────────────────────────────
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature (default: 0.6)"
    )
    parser.add_argument(
        "--max_step_tokens", type=int, default=8192,
        help="Maximum tokens per step (default: 8192)"
    )
    parser.add_argument(
        "--max_total_tokens", type=int, default=32768,
        help="Maximum total tokens per complete response (default: 32768)"
    )

    # ── Control Parameters ──────────────────────────────────────────────
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Maximum concurrent questions (default: 8)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Number of items to buffer before writing to disk (default: 10)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SKD (Speculative Knowledge Distillation) Configuration (Async High-Concurrency):")
    print(f"  Input File           : {args.input_file}")
    print(f"  Output File          : {args.output_file}")
    print(f"  Student API URL    : {args.student_url}")
    print(f"  Teacher API URL    : {args.teacher_url}")
    print(f"  Teacher Model Path   : {args.teacher_model_path or 'Not specified (char-count approx)'}")
    print(f"  Gamma (γ)          : {args.gamma}")
    print(f"  Responses per Item   : {args.num_responses}")
    print(f"  Temperature          : {args.temperature}")
    print(f"  Max tokens/step      : {args.max_step_tokens}")
    print(f"  Max tokens/response  : {args.max_total_tokens}")
    print(f"  Max Concurrent Qst   : {args.num_workers}")
    print(f"  Disk Write Batch     : {args.batch_size}")
    print(f"  HTTP Timeout         : 10800s (3 hours)")
    print("=" * 60)

    # Load tokenizer once in main process; shared across all async coroutines
    tokenizer = None
    if args.teacher_model_path:
        print(f"Loading tokenizer: {args.teacher_model_path}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_model_path, trust_remote_code=True
        )
        print("Tokenizer loaded.")
    else:
        print("No teacher_model_path provided, using character-count approximation for logprob_start_len")

    data = load_jsonl(args.input_file)
    if not data:
        print("Error: Input file is empty or cannot be read.")
        return

    asyncio.run(
        run_all_async(
            data             = data,
            student_url      = args.student_url,
            teacher_url      = args.teacher_url,
            gamma            = args.gamma,
            temperature      = args.temperature,
            max_step_tokens  = args.max_step_tokens,
            max_total_tokens = args.max_total_tokens,
            num_responses    = args.num_responses,
            output_file      = args.output_file,
            max_concurrent   = args.num_workers,
            batch_size       = args.batch_size,
            tokenizer        = tokenizer
        )
    )


if __name__ == "__main__":
    main()
