#!/usr/bin/env python3
"""
Comprehensive script:
1. Read a JSONL file with input/output fields
2. Call SGLang API to compute log probabilities for the output portion
3. Compute windowed average logprob per token position (window size=10)
4. Output a line chart (PNG) and corresponding data as an Excel file
"""

import json
import argparse
import requests
import os
import sys
import threading
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[WARNING] transformers not installed; using character-length estimation for prefix offset (less precise)")

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("[WARNING] openpyxl not installed; falling back to CSV output. Run: pip install openpyxl")


# =============================================================================
# Field Extraction
# =============================================================================

def get_input_field(item: Dict[str, Any]) -> str:
    """Extract input (question) field, compatible with multiple common field names."""
    for key in ('input', 'question', 'prompt', 'instruction'):
        if key in item and item[key]:
            return str(item[key])
    return ""


def get_output_field(item: Dict[str, Any]) -> str:
    """Extract output (response) field, compatible with multiple common field names."""
    for key in ('output', 'response', 'answer', 'text', 'completion'):
        if key in item and item[key]:
            return str(item[key])
    return ""


# =============================================================================
# SGLang API Call
# =============================================================================

def call_sglang_logprobs(
    prefix_text: str,
    target_text: str,
    api_url: str,
    tokenizer=None,
    item_id: str = "?",
) -> Dict[str, Any]:
    """
    Compute per-token log probability for target_text using prefix_text as context.
    Uses logprob_start_len to return only the target portion's logprobs directly.
    """
    separator = "\n"
    full_prompt = f"{prefix_text}{separator}{target_text}" if prefix_text else target_text

    # Precise prefix token count
    if tokenizer and prefix_text:
        prefix_tokens = tokenizer.encode(prefix_text + separator, add_special_tokens=False)
        start_len = len(prefix_tokens)
    elif prefix_text:
        # Rough approximation
        start_len = len((prefix_text + separator).split())
    else:
        start_len = 0

    payload = {
        "text": full_prompt,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "logprob_start_len": start_len,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,
        },
    }

    try:
        resp = requests.post(api_url, json=payload, timeout=10800)
        resp.raise_for_status()
        result = resp.json()

        meta_info = result.get('meta_info', {})
        raw_logprobs = meta_info.get('input_token_logprobs', [])

        if not raw_logprobs:
            return {'success': False, 'error': 'No input_token_logprobs returned', 'logits': []}

        # Unified format: [logprob, token_id, token_str]
        logits = []
        for item in raw_logprobs:
            if isinstance(item, list) and len(item) >= 3:
                logits.append([item[0], item[1], item[2]])
            elif isinstance(item, dict):
                logits.append([
                    item.get('logprob', 0.0),
                    item.get('token_id', -1),
                    item.get('decoded_token', ''),
                ])

        print(f"  [✓] ID={item_id} | prefix_tokens={start_len} | returned_logprobs={len(logits)}")
        return {'success': True, 'error': None, 'logits': logits, 'start_len': start_len}

    except Exception as e:
        print(f"  [✗] ID={item_id} | Error: {e}")
        return {'success': False, 'error': str(e), 'logits': []}


# =============================================================================
# Batch Processing
# =============================================================================

def _fetch_one(task: tuple):
    """
    Single-item worker for parallel execution.
    task = (idx, item, api_url, tokenizer)
    Returns (idx, item_id, logits_or_None)
    """
    idx, item, api_url, tokenizer = task
    item_id = item.get('id', str(idx))
    prefix = get_input_field(item)
    target = get_output_field(item)

    if not target:
        print(f"  [!] Row {idx}: no output field found, skipping")
        return idx, item_id, None

    result = call_sglang_logprobs(prefix, target, api_url, tokenizer, item_id)

    if result['success'] and result['logits']:
        return idx, item_id, result['logits']
    else:
        print(f"  [!] Row {idx}: logprobs empty or failed, skipping")
        return idx, item_id, None


def process_jsonl(
    input_file: str,
    api_url: str,
    tokenizer=None,
    logits_output_file: str = None,
    num_workers: int = 1,
) -> List[List[Any]]:
    """
    Iterate over JSONL, call API to fetch logprobs.
    When num_workers > 1, uses a thread pool for parallel I/O-bound API calls.
    If logits_output_file is specified, raw logits are also written to that JSONL file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    total = len(lines)
    print(f"{total} items total, num_workers={num_workers}, calling API...\n")

    # Clear/create output file if needed
    file_lock = threading.Lock()
    if logits_output_file:
        ensure_dir(logits_output_file)
        open(logits_output_file, 'w', encoding='utf-8').close()

    # Parse all lines into items
    tasks = []
    for idx, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
            tasks.append((idx, item, api_url, tokenizer))
        except json.JSONDecodeError:
            print(f"  [!] Row {idx}: JSON parse failed, skipping")

    # Collect results (preserve original order)
    results_map: Dict[int, tuple] = {}

    def _save_record(idx, item_id, logits):
        """Thread-safe append write to JSONL."""
        if logits_output_file:
            record = {'id': item_id, 'calculated_logits': logits}
            with file_lock:
                with open(logits_output_file, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

    if num_workers <= 1:
        for task in tqdm(tasks, desc="Fetching logprobs"):
            idx, item_id, logits = _fetch_one(task)
            if logits is not None:
                results_map[idx] = (item_id, logits)
                _save_record(idx, item_id, logits)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_fetch_one, task): task[0] for task in tasks}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Fetching logprobs",
            ):
                idx, item_id, logits = future.result()
                if logits is not None:
                    results_map[idx] = (item_id, logits)
                    _save_record(idx, item_id, logits)

    # Return logits in original order
    all_logits = [results_map[i][1] for i in sorted(results_map.keys())]

    print(f"\nSuccessfully fetched {len(all_logits)} logprobs records.")
    if logits_output_file:
        print(f"[JSONL] Raw logprobs saved: {logits_output_file}")
    return all_logits


# =============================================================================
# Windowed Average Calculation (from 3-analysis-logits.py)
# =============================================================================

def calculate_windowed_averages(
    all_logits: List[List[Any]],
    window_size: int = 10,
) -> List[tuple]:
    """
    Group logits by token position and compute average logprob per window.
    Returns [(start_pos, avg_logprob), ...] list.
    """
    position_values = defaultdict(list)

    for row_logits in all_logits:
        for i, item in enumerate(row_logits):
            if isinstance(item, list) and len(item) >= 1 and item[0] is not None:
                position_values[i].append(item[0])

    if not position_values:
        return []

    max_pos = max(position_values.keys())
    window_averages = []

    for start_pos in range(0, max_pos + 1, window_size):
        end_pos = min(start_pos + window_size, max_pos + 1)
        window_logits = []
        for pos in range(start_pos, end_pos):
            if pos in position_values:
                window_logits.extend(position_values[pos])
        if window_logits:
            avg = sum(window_logits) / len(window_logits)
            window_averages.append((start_pos, avg))

    return window_averages


# =============================================================================
# Output: Charts + Excel/CSV
# =============================================================================

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_chart(window_averages: List[tuple], output_prefix: str, window_size: int):
    positions, averages = zip(*window_averages)

    plt.figure(figsize=(14, 6))
    plt.plot(positions, averages, marker='o', linestyle='-', markersize=5, linewidth=1.5)
    plt.title(f'Token Position vs Average Log Probability  (Window Size = {window_size})')
    plt.xlabel('Token Position')
    plt.ylabel('Average Log Probability')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    chart_path = f"{output_prefix}.png"
    ensure_dir(chart_path)
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[Chart] Saved: {chart_path}")


def save_excel(window_averages: List[tuple], output_prefix: str):
    excel_path = f"{output_prefix}.xlsx"
    ensure_dir(excel_path)

    if HAS_OPENPYXL:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Windowed LogProbs"

        # Column headers
        ws.append(["Token_Position_Start", "Average_LogProb"])

        for pos, avg in window_averages:
            ws.append([pos, avg])

        # Simple column width adjustment
        ws.column_dimensions['A'].width = 22
        ws.column_dimensions['B'].width = 22

        wb.save(excel_path)
        print(f"[Excel] Saved: {excel_path}")
    else:
        # Fallback to CSV
        import csv
        csv_path = f"{output_prefix}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Token_Position_Start', 'Average_LogProb'])
            for pos, avg in window_averages:
                writer.writerow([pos, avg])
        print(f"[CSV fallback] Saved: {csv_path}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute log probabilities for JSONL responses and generate windowed average chart + Excel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compute_and_analyze_logprobs.py \\
      --input_file data.jsonl \\
      --output_prefix ./results/my_model \\
      --api_url http://127.0.0.1:30011/generate \\
      --model_path /path/to/model

JSONL format (multiple field names supported):
  {"input": "question...", "output": "answer..."}
  {"question": "...", "response": "..."}
  {"prompt": "...", "answer": "..."}
        """
    )

    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Input JSONL file path (each line must have input/output fields)"
    )
    parser.add_argument(
        "--output_prefix", type=str, default="./output/logprob_analysis",
        help="Output path prefix (no extension); generates .png, .xlsx and _logits.jsonl"
    )
    parser.add_argument(
        "--save_logits", action="store_true", default=True,
        help="Whether to save raw logprobs to a JSONL file (default: enabled)"
    )
    parser.add_argument(
        "--no_save_logits", dest="save_logits", action="store_false",
        help="Disable saving raw logprobs JSONL file"
    )
    parser.add_argument(
        "--api_url", type=str, default="http://127.0.0.1:30011/generate",
        help="SGLang generate API endpoint"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Tokenizer path (for precise prefix token count); uses character-length estimation if omitted"
    )
    parser.add_argument(
        "--window_size", type=int, default=10,
        help="Window size (default: 10)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=64,
        help="Parallel thread count (default: 1=sequential). Increase to speed up API calls; recommend 4-16"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_prefix = args.output_prefix
    ensure_dir(output_prefix + ".png")  # Use ensure_dir to create parent directory

    # Load tokenizer (optional)
    tokenizer = None
    if args.model_path and HAS_TRANSFORMERS:
        print(f"[INFO] Loading tokenizer: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    elif args.model_path and not HAS_TRANSFORMERS:
        print("[WARNING] transformers not installed; cannot load tokenizer, using character estimation.")

    print("=" * 60)
    print(f"  Input File     : {args.input_file}")
    print(f"  Output Prefix  : {output_prefix}")
    print(f"  Save Logits    : {'yes -> ' + output_prefix + '_logits.jsonl' if args.save_logits else 'no'}")
    print(f"  API URL        : {args.api_url}")
    print(f"  Window Size    : {args.window_size}")
    print(f"  Parallel Workers: {args.num_workers}")
    print(f"  Tokenizer      : {args.model_path or 'not provided (char estimation)'}")
    print("=" * 60)

    # Step 1: Batch-fetch logprobs
    logits_jsonl = f"{output_prefix}_logits.jsonl" if args.save_logits else None
    all_logits = process_jsonl(args.input_file, args.api_url, tokenizer, logits_jsonl, args.num_workers)

    if not all_logits:
        print("[ERROR] Failed to fetch any logprobs, exiting.")
        sys.exit(1)

    # Step 2: Compute windowed averages
    print(f"\nComputing windowed averages (window_size={args.window_size})...")
    window_averages = calculate_windowed_averages(all_logits, args.window_size)

    if not window_averages:
        print("[ERROR] Failed to compute window averages: no valid data.")
        sys.exit(1)

    positions, averages = zip(*window_averages)
    print(f"  Token position range : {min(positions)} ~ {max(positions)}")
    print(f"  LogProb range        : {min(averages):.4f} ~ {max(averages):.4f}")
    print(f"  Window count         : {len(window_averages)}")

    # Step 3: Save charts + Excel
    save_chart(window_averages, output_prefix, args.window_size)
    save_excel(window_averages, output_prefix)

    print("\nAll done!")


if __name__ == "__main__":
    main()
