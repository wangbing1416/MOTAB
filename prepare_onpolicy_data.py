#!/usr/bin/env python3
"""
prepare_onpolicy_data.py

Convert MOTAB raw input JSONL (fields: id, input, ...) to KDFlow on-policy
training format (fields: id, messages).

Input format  (same as 1-generate-ours.py input):
    {"id": "xxx", "input": "question text", ...}

Output format (KDFlow PromptDataset with --input_key messages):
    {"id": "xxx", "messages": [{"role": "user", "content": "question text"}]}

Usage:
    python prepare_onpolicy_data.py \
        --input_file  /path/to/raw_data.jsonl \
        --output_file /path/to/kdflow_prompts.jsonl \
        [--deduplicate]  \
        [--max_samples 10000]
"""

import json
import argparse
from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file, skip blank lines."""
    data = []
    print(f"Loading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {lineno} JSON parse error, skipped: {e}")
    print(f"Loaded: {len(data)} items")
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save a list of dicts to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved: {len(data)} items → {file_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_item(item: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Convert one raw item to KDFlow on-policy prompt format.

    Input:  {"id": "xxx", "input": "question text", ...}
    Output: {"id": "xxx", "messages": [{"role": "user", "content": "question text"}]}

    Returns None if the item has no valid 'input' field.
    """
    question = item.get('input', '').strip()
    if not question:
        return None

    result = {
        "id": item.get('id', ''),
        "messages": [
            {"role": "user", "content": question}
        ]
    }
    return result


def convert_dataset(
    data: List[Dict[str, Any]],
    deduplicate: bool = False,
    max_samples: int = None,
) -> List[Dict[str, Any]]:
    """
    Convert a list of raw items to KDFlow format.

    Args:
        data:        Raw items loaded from JSONL.
        deduplicate: If True, remove duplicate questions (by content).
        max_samples: If set, truncate to at most this many items after dedup.

    Returns:
        List of converted items ready for KDFlow training.
    """
    converted = []
    skipped_empty = 0

    seen_questions = set()

    for item in data:
        out = convert_item(item)
        if out is None:
            skipped_empty += 1
            continue

        question = out['messages'][0]['content']

        if deduplicate:
            if question in seen_questions:
                continue
            seen_questions.add(question)

        converted.append(out)

    print(f"Conversion complete:")
    print(f"  Raw items        : {len(data)}")
    print(f"  Skipped (empty)  : {skipped_empty}")
    if deduplicate:
        print(f"  After dedup      : {len(converted)}")

    if max_samples is not None and len(converted) > max_samples:
        converted = converted[:max_samples]
        print(f"  After truncation : {len(converted)} (max_samples={max_samples})")

    return converted


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert MOTAB raw JSONL to KDFlow on-policy training format"
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Input JSONL file path (fields: id, input, ...)"
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Output JSONL file path for KDFlow (fields: id, messages)"
    )
    parser.add_argument(
        "--deduplicate", action="store_true", default=False,
        help="Remove duplicate questions (by content) before saving"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to output (applied after dedup)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("prepare_onpolicy_data.py Configuration:")
    print(f"  Input File    : {args.input_file}")
    print(f"  Output File   : {args.output_file}")
    print(f"  Deduplicate   : {args.deduplicate}")
    print(f"  Max Samples   : {args.max_samples if args.max_samples else 'unlimited'}")
    print("=" * 60)

    raw_data = load_jsonl(args.input_file)
    if not raw_data:
        print("Error: Input file is empty or cannot be read.")
        return

    converted = convert_dataset(
        data=raw_data,
        deduplicate=args.deduplicate,
        max_samples=args.max_samples,
    )

    if not converted:
        print("Error: No valid items after conversion.")
        return

    save_jsonl(converted, args.output_file)

    # Print a preview of the first item
    print("\nSample output item:")
    print(json.dumps(converted[0], ensure_ascii=False, indent=2))
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
