#!/usr/bin/env python3
"""
1-generate-ours.py

MOTAB Data Synthesis Pipeline.
Implements the MOTAB methodology for LLM reasoning distillation.

Algorithm (MOTAB):
  For each question, generate SFT trajectories as follows:
  1. Student generates reasoning steps one by one (stop=".\n\n")
  2. For each student step:
     a. Compute Oracle Value:
            V_T = exp(1/L * sum_i log P_teacher(y_i | context, y_{<i}))
     b. Compute Teacher Predictive Entropy (top-k approximation):
            H(ε_T | s) ≈ -sum_i p̃_i * log(p̃_i)   [renormalized top-k]
     c. Compute State-Adaptive Safety Boundary:
            ε(s) = ε₀ · exp(-θ · H(ε_T | s))
     d. If V_T >= ε(s): accept student step, append to trajectory, continue
     e. If V_T < ε(s): BOUNDARY BREACH DETECTED
        i.   TD-error backtracking to identify root-cause bifurcation node t*:
                 δ_k = V_T(s_{k-1}, a^S_k) - V_T(s_{k-2}, a^S_{k-1})
                 t* = argmin_k(δ_k)  s.t. V_T(s_{t*-2}, a^S_{t*-1}) >= ε(s_{t*-2})
        ii.  Rewind to pristine context s_{t*-1}
        iii. Teacher generates full corrective suffix a^T_{t*:T} from s_{t*-1}
        iv.  Asymmetric stitching:
                 ω_SFT = [student flawed exploration 1:t] + <REVISE> + [teacher correction t*:T]
  3. If no breach detected: ω_SFT = complete student trajectory
"""

import json
import argparse
import requests
import math
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
import os
from tqdm import tqdm
from collections import Counter

# Discrete transition token for asymmetric stitching (Section 3.3, Phase 3)
REVISE_TOKEN = "However,"

# Global tokenizer (loaded once per worker process)
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
            logit     = item[0] if item[0] is not None else 0.0
            token_id  = item[1] if item[1] is not None else -1
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
    """
    Compute geometric mean probability from logprobs.
    This implements: exp(1/L * sum_i logprob_i)  [Eq. 5 in paper]
    """
    if not logprobs:
        return 0.0
    lp_values = [item[0] for item in logprobs if item[0] is not None]
    if not lp_values:
        return 0.0
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
# MOTAB Core: Oracle Value Function  (Eq. 5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_oracle_value(
    context: str,
    step_text: str,
    teacher_url: str,
    tokenizer
) -> Tuple[float, bool, Optional[str]]:
    """
    Compute Oracle Value V_T for a student-generated step under the teacher model.

    Implements Eq. 5 from the paper:
        V_T(s_{t-1}, a^S_t) = exp( 1/L * sum_{i=1}^{L} log ε_T(y_i | s_{t-1}, y_{<i}) )

    This is the conditional log-likelihood (CLL) of the student step under the teacher,
    computed via input_token_logprobs with logprob_start_len to skip the context prefix.

    Returns:
        (oracle_value: float, success: bool, error_msg: str | None)
    """
    full_text = context + step_text

    if tokenizer is not None:
        prefix_tokens = tokenizer.encode(context, add_special_tokens=False)
        start_len = len(prefix_tokens)
    else:
        # Fallback: character-count approximation (imprecise)
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
        response = requests.post(teacher_url, json=payload, timeout=10800)
        response.raise_for_status()
        result = response.json()

        meta_info    = result.get('meta_info', {})
        raw_logprobs = meta_info.get('input_token_logprobs', [])

        if not raw_logprobs:
            return 0.0, False, 'No input_token_logprobs returned from SGLang'

        logprobs     = parse_logprobs(raw_logprobs)
        oracle_value = avg_prob_from_logprobs(logprobs)
        return oracle_value, True, None

    except requests.exceptions.Timeout:
        return 0.0, False, 'Request timeout (>10800s)'
    except Exception as e:
        return 0.0, False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# MOTAB Core: Teacher Entropy & Adaptive Boundary  (Eq. 6, 7)
# ─────────────────────────────────────────────────────────────────────────────

def compute_teacher_entropy(
    context: str,
    teacher_url: str,
    top_k: int = 20
) -> Tuple[float, bool]:
    """
    Approximate teacher's predictive sequence entropy H(ε_T | s) at given context.

    Implements Eq. 6 from the paper:
        H(ε_T | s_{t-1}) = -E_{y ~ ω_T(·|s_{t-1})}[log ε_T(y | s_{t-1})]

    Approximation: use the renormalized entropy over top-k next-token logprobs:
        H ≈ -sum_{i=1}^{k} p̃_i · log(p̃_i)   where p̃_i = p_i / sum_j(p_j)

    High entropy → teacher is uncertain (multi-modal reasoning manifold)
                 → lower adaptive boundary → more permissive for student
    Low entropy  → teacher is deterministic (axiomatic logic)
                 → higher adaptive boundary → stricter quality gate

    Returns:
        (entropy: float, success: bool)
    """
    payload = {
        "text": context,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "top_logprobs_num": top_k,   # top-level field, NOT inside sampling_params
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,
        }
    }
    try:
        response = requests.post(teacher_url, json=payload, timeout=10800)
        response.raise_for_status()
        result = response.json()

        meta_info      = result.get('meta_info', {})
        output_logprobs = meta_info.get('output_token_logprobs', [])

        if not output_logprobs:
            return 0.0, False

        # output_token_logprobs[0]: top-k entries at position 0
        pos0 = output_logprobs[0]

        # Parse top-k logprobs from different possible SGLang response formats
        topk_lps = []
        if isinstance(pos0, list):
            if len(pos0) > 0 and isinstance(pos0[0], (list, tuple)):
                # Format: [[logprob, token_id, token_str], ...]  (top-k list)
                topk_lps = [item[0] for item in pos0
                            if isinstance(item, (list, tuple)) and len(item) >= 1
                            and item[0] is not None]
            elif len(pos0) >= 1 and isinstance(pos0[0], (int, float)):
                # Format: [logprob, token_id, token_str]  (single token, no top-k)
                topk_lps = [pos0[0]]
        elif isinstance(pos0, dict):
            topk_lps = [pos0.get('logprob', 0.0)]

        if not topk_lps:
            return 0.0, False

        # Compute renormalized top-k entropy
        probs     = [math.exp(lp) for lp in topk_lps]
        total_p   = sum(probs)
        if total_p <= 1e-12:
            return 0.0, False

        probs_norm = [p / total_p for p in probs]
        entropy    = -sum(p * math.log(p + 1e-12) for p in probs_norm if p > 0)
        return entropy, True

    except requests.exceptions.Timeout:
        return 0.0, False
    except Exception:
        return 0.0, False


def compute_adaptive_boundary(
    base_epsilon: float,
    theta: float,
    entropy: float
) -> float:
    """
    Compute state-adaptive safety boundary.

    Implements Eq. 7 from the paper:
        ε(s_{t-1}) = ε₀ · exp(-θ · H(ε_T | s_{t-1}))

    where ε₀ ∈ (0,1] is the base threshold and θ > 0 is the entropy scaling factor.
    """
    return base_epsilon * math.exp(-theta * entropy)


# ─────────────────────────────────────────────────────────────────────────────
# MOTAB Core: TD-Error Root-Cause Attribution  (Eq. 8, 9)
# ─────────────────────────────────────────────────────────────────────────────

def find_bifurcation_node(step_history: List[Dict]) -> int:
    """
    Identify the root-cause bifurcation node t* via TD-error backtracking.

    Implements Eq. 8-9 from the paper:
        TD error:  δ_k = V_T(s_{k-1}, a^S_k) - V_T(s_{k-2}, a^S_{k-1})
                       = oracle_value[k] - oracle_value[k-1]

        t* = argmin_{k <= t}(δ_k)
             s.t. V_T(s_{t*-2}, a^S_{t*-1}) >= ε(s_{t*-2})
             (the step just before t* was still above the safety threshold)

    This finds the step exhibiting the most severe degradation in oracle value,
    ensuring we find the exact moment the student deviated from the safe manifold.

    Args:
        step_history: list of step dicts with 'oracle_value' and 'adaptive_boundary'

    Returns:
        t* (0-based index of the root-cause bifurcation step)
    """
    n = len(step_history)
    if n <= 1:
        return 0

    # Compute TD errors δ_k for k >= 1
    # Constraint: the step at k-1 was above its adaptive boundary
    candidates = []
    for k in range(1, n):
        v_k      = step_history[k].get('oracle_value', 0.0)
        v_km1    = step_history[k - 1].get('oracle_value', 0.0)
        delta    = v_k - v_km1

        # Constraint: previous step (k-1) was still above its adaptive boundary
        prev_boundary = step_history[k - 1].get('adaptive_boundary', 0.0)
        if v_km1 >= prev_boundary:
            candidates.append((k, delta))

    if candidates:
        # t* = argmin δ_k  (most severe oracle value degradation)
        t_star = min(candidates, key=lambda x: x[1])[0]
        return t_star

    # Fallback: if no valid candidate found (all steps below threshold from the start),
    # return the step with the globally minimum oracle value
    return min(range(n), key=lambda i: step_history[i].get('oracle_value', 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# API Calls: Student Step Generation & Teacher Continuation
# ─────────────────────────────────────────────────────────────────────────────

def generate_step(
    context: str,
    api_url: str,
    temperature: float,
    max_step_tokens: int
) -> Dict:
    """
    Call SGLang API to generate one reasoning step.
    Uses stop=[".\n\n"] as the step boundary separator.

    Return format:
        {
            'success':     bool,
            'text':        str,        # Generated step text
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
    except requests.exceptions.Timeout:
        return {'success': False, 'text': '', 'logprobs': [], 'finish_type': 'error',
                'error': 'Request timeout (>10800s)'}
    except Exception as e:
        return {'success': False, 'text': '', 'logprobs': [], 'finish_type': 'error', 'error': str(e)}


def generate_teacher_continuation(
    context: str,
    teacher_url: str,
    temperature: float,
    max_tokens: int
) -> Dict:
    """
    Teacher generates a complete corrective suffix from pristine context s_{t*-1}.

    This implements the Decoupled Oracle Generation phase (Section 3.3, Phase 2):
    "The teacher is queried to generate the valid corrective suffix a^T_{t*:T}"

    Unlike student step generation, NO step-level stop sequences are used —
    teacher generates until EOS or max_tokens, producing a full continuation.

    Return format:
        {
            'success':     bool,
            'text':        str,        # Full teacher continuation
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
            "max_new_tokens": max_tokens,
        }
    }
    try:
        response = requests.post(teacher_url, json=payload, timeout=10800)
        response.raise_for_status()
        result = response.json()

        generated_text = result.get('text', '')
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
    except requests.exceptions.Timeout:
        return {'success': False, 'text': '', 'logprobs': [], 'finish_type': 'error',
                'error': 'Request timeout (>10800s)'}
    except Exception as e:
        return {'success': False, 'text': '', 'logprobs': [], 'finish_type': 'error', 'error': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# MOTAB Main Generation Logic
# ─────────────────────────────────────────────────────────────────────────────

def generate_motab_response(
    question: str,
    student_url: str,
    teacher_url: str,
    epsilon0: float,
    theta: float,
    temperature: float,
    max_step_tokens: int,
    max_total_tokens: int,
    tokenizer,
    top_k_entropy: int,
    item_id: str,
    response_idx: int
) -> Dict:
    """
    Generate one complete SFT trajectory using the MOTAB algorithm.

    Output ω_SFT (Eq. 11):
        - If no breach:  [student full trajectory]
        - If breach:     [student flawed exploration 1:t]
                         + "\n\n<REVISE>\n\n"
                         + [teacher root-corrected target t*:T]

    Return format:
        {
            'sft_text':           str,        # Final SFT trajectory for training
            'student_steps':      List[Dict], # All student step records
            'breach_detected':    bool,
            'bifurcation_step':   int | None, # t* index (0-based)
            'teacher_correction': str | None, # Teacher continuation text
            'teacher_correction_logprobs': List[List] | None,
            'total_student_steps': int,
            'total_tokens':        int,
        }
    """
    context      = question
    step_history = []   # Records all student steps with oracle values and boundaries
    total_tokens = 0
    step_idx     = 0

    breach_detected    = False
    bifurcation_step   = None
    teacher_correction = None
    teacher_correction_logprobs = None

    # Pre-compute teacher entropy at initial context (used for step 0's boundary)
    h_entropy, h_success = compute_teacher_entropy(context, teacher_url, top_k_entropy)
    if not h_success:
        h_entropy = 0.0  # Fallback: entropy=0 → ε(s) = ε₀ (static threshold)

    while total_tokens < max_total_tokens:

        # ── Phase 1: Student generates next reasoning step ──────────────────
        student_result = generate_step(context, student_url, temperature, max_step_tokens)

        if not student_result['success']:
            break

        student_step_text = student_result['text']
        student_logprobs  = student_result['logprobs']
        student_finish    = student_result['finish_type']

        # Empty generation (e.g., direct EOS) → generation complete
        if not student_step_text:
            break

        # ── Phase 2: Compute Oracle Value V_T  (Eq. 5) ───────────────────────
        oracle_value, ov_success, ov_error = compute_oracle_value(
            context, student_step_text, teacher_url, tokenizer
        )
        if not ov_success:
            # If teacher scoring fails, treat step as safe to avoid false breach
            oracle_value = 1.0
            print(f"  [WARN] Step {step_idx}: Oracle value failed: {ov_error}, defaulting to 1.0")

        # ── Phase 3: Compute Adaptive Safety Boundary ε(s)  (Eq. 7) ─────────
        adaptive_boundary = compute_adaptive_boundary(epsilon0, theta, h_entropy)

        # ── Record step information ──────────────────────────────────────────
        step_record = {
            'step_index':        step_idx,
            'context_before':    context,   # Pristine context before this step (for backtracking)
            'step_text':         student_step_text,
            'logprobs':          student_logprobs,
            'finish_type':       student_finish,
            'oracle_value':      round(oracle_value, 8),      # V_T
            'adaptive_boundary': round(adaptive_boundary, 8), # ε(s)
            'teacher_entropy':   round(h_entropy, 8),         # H(ε_T | s)
        }
        step_history.append(step_record)
        total_tokens += len(student_logprobs)
        step_idx     += 1

        if step_idx % 50 == 0:
            print(f"  [DEBUG] Step {step_idx}: V_T={oracle_value:.4f}, "
                  f"ε={adaptive_boundary:.4f}, H={h_entropy:.4f}, tokens={total_tokens}")

        # ── Phase 4: Boundary Check (Exploration & Adaptive Boundary Detection)
        if oracle_value < adaptive_boundary:
            # ── BOUNDARY BREACH: V_T < ε(s) ──────────────────────────────────
            breach_detected = True
            print(f"  [BREACH] Step {step_idx - 1}: "
                  f"V_T={oracle_value:.4f} < ε={adaptive_boundary:.4f} | "
                  f"Triggering root-cause backtracking...")

            # ── Phase 5: TD-Error Root-Cause Attribution  (Eq. 8-9) ──────────
            bifurcation_step = find_bifurcation_node(step_history)
            print(f"  [BACKTRACK] Root-cause bifurcation at t*={bifurcation_step} "
                  f"(V_T at t*={step_history[bifurcation_step]['oracle_value']:.4f})")

            # ── Phase 6: Decoupled Oracle Generation  (Eq. 10) ───────────────
            # Rewind to the pristine context s_{t*-1} (strictly before bifurcation step)
            pristine_context = step_history[bifurcation_step]['context_before']

            # Budget: remaining tokens after the safe prefix
            safe_tokens = sum(len(s['logprobs']) for s in step_history[:bifurcation_step])
            remaining_tokens = max(256, min(max_total_tokens - safe_tokens, max_total_tokens))

            teacher_gen = generate_teacher_continuation(
                pristine_context, teacher_url, temperature, remaining_tokens
            )

            if teacher_gen['success'] and teacher_gen['text']:
                teacher_correction         = teacher_gen['text']
                teacher_correction_logprobs = teacher_gen['logprobs']
            else:
                # Fallback: concatenate remaining student steps as-is
                teacher_correction = "\n\n".join(
                    s['step_text'] for s in step_history[bifurcation_step:]
                )
                teacher_correction_logprobs = []
                print(f"  [WARN] Teacher continuation failed "
                      f"({teacher_gen.get('error', 'unknown')}), using student fallback")

            break  # End student exploration phase after breach handling

        # ── Phase 4b: Accept step, update context ────────────────────────────
        context       = context + student_step_text + "\n\n"
        if total_tokens >= max_total_tokens:
            break
        if student_finish != 'stop':
            # EOS or length limit hit → generation naturally complete
            break

        # Pre-compute teacher entropy at new context for next iteration
        h_entropy, h_success = compute_teacher_entropy(context, teacher_url, top_k_entropy)
        if not h_success:
            h_entropy = 0.0

    # ── Phase 7: Asymmetric Semantic Stitching  (Eq. 11) ─────────────────────
    if breach_detected and teacher_correction is not None:
        # ω_SFT = [student flawed exploration 1:t] ⊕ <REVISE> ⊕ [teacher correction t*:T]
        student_prefix_text = "\n\n".join(s['step_text'] for s in step_history)
        sft_text = student_prefix_text + "\n\n" + REVISE_TOKEN + "\n\n" + teacher_correction
    else:
        # No breach: pure student trajectory accepted as SFT data
        sft_text = "\n\n".join(s['step_text'] for s in step_history)

    # ── Statistics & Debug Output ─────────────────────────────────────────────
    n_steps      = len(step_history)
    step_lengths = [len(s['step_text']) for s in step_history]
    oracle_vals  = [s['oracle_value'] for s in step_history]
    boundaries   = [s['adaptive_boundary'] for s in step_history]
    avg_step_len = sum(step_lengths) / n_steps if n_steps > 0 else 0.0
    avg_oracle   = sum(oracle_vals)  / n_steps if n_steps > 0 else 0.0
    avg_boundary = sum(boundaries)   / n_steps if n_steps > 0 else 0.0
    empty_steps  = sum(1 for sl in step_lengths if sl == 0)

    print(f"\n  [{'=' * 40}]")
    print(f"  [MOTAB Stats] ID:{item_id} R:{response_idx}")
    print(f"  Student steps: {n_steps}, Total tokens: {total_tokens}")
    print(f"  Breach detected: {breach_detected}, Bifurcation at: {bifurcation_step}")
    print(f"  Avg oracle value: {avg_oracle:.4f}, Avg adaptive boundary: {avg_boundary:.4f}")
    print(f"  Avg step length: {avg_step_len:.1f} chars, Empty steps: {empty_steps}/{n_steps}")
    if step_history:
        print(f"  finish_type distribution (student steps):")
        finish_dist = Counter(s['finish_type'] for s in step_history)
        for ft, cnt in finish_dist.most_common():
            print(f"    - '{ft}': {cnt} ({cnt / n_steps * 100:.1f}%)")
    if breach_detected and teacher_correction:
        print(f"  Teacher correction length: {len(teacher_correction)} chars, "
              f"tokens: {len(teacher_correction_logprobs) if teacher_correction_logprobs else 'N/A'}")
    print(f"  [{'=' * 40}]\n")

    return {
        'sft_text':                   sft_text,
        'student_steps':              step_history,
        'breach_detected':            breach_detected,
        'bifurcation_step':           bifurcation_step,
        'teacher_correction':         teacher_correction,
        'teacher_correction_logprobs': teacher_correction_logprobs,
        'total_student_steps':        n_steps,
        'total_tokens':               total_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Process Worker Functions
# ─────────────────────────────────────────────────────────────────────────────

def init_worker(teacher_model_path: str):
    """
    Process pool initialization: load tokenizer once per worker process.
    Avoids repeated tokenizer loading for each data item.
    """
    global _tokenizer
    if teacher_model_path:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
        print(f"[Worker PID:{os.getpid()}] Tokenizer loaded: {teacher_model_path}")
    else:
        _tokenizer = None
        print(f"[Worker PID:{os.getpid()}] No teacher_model_path provided, "
              f"using character-count approximation for logprob_start_len")


def process_item_worker(args_tuple) -> Dict:
    """Multi-process worker: generate num_responses MOTAB trajectories for one data item."""
    global _tokenizer

    (data_item, student_url, teacher_url, epsilon0, theta, temperature,
     max_step_tokens, max_total_tokens, num_responses, top_k_entropy) = args_tuple

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

    print(f"\n{'=' * 50}")
    print(f"[START] ID:{item_id} | Generating {num_responses} MOTAB SFT trajectories")

    responses_list = []
    breach_count   = 0

    for r_idx in range(num_responses):
        resp = generate_motab_response(
            question         = question,
            student_url      = student_url,
            teacher_url      = teacher_url,
            epsilon0         = epsilon0,
            theta            = theta,
            temperature      = temperature,
            max_step_tokens  = max_step_tokens,
            max_total_tokens = max_total_tokens,
            tokenizer        = _tokenizer,
            top_k_entropy    = top_k_entropy,
            item_id          = item_id,
            response_idx     = r_idx
        )
        responses_list.append(resp)

        if resp.get('breach_detected', False):
            breach_count += 1

        breach_info = (f"YES (t*={resp['bifurcation_step']})"
                       if resp['breach_detected'] else "NO")
        print(f"[R{r_idx + 1}/{num_responses}] ID:{item_id} | "
              f"steps={resp['total_student_steps']} tokens={resp['total_tokens']} | "
              f"breach={breach_info}")

    breach_rate = breach_count / num_responses if num_responses > 0 else 0.0
    print(f"[DONE] ID:{item_id} | Breach rate: {breach_rate:.2%} ({breach_count}/{num_responses})")

    return {
        'id':          item_id,
        'input':       question,
        'responses':   responses_list,
        'breach_rate': round(breach_rate, 4),
        'status':      'success'
    }


# ─────────────────────────────────────────────────────────────────────────────
# Process Control
# ─────────────────────────────────────────────────────────────────────────────

def run_process(
    data: List[Dict],
    student_url: str,
    teacher_url: str,
    epsilon0: float,
    theta: float,
    temperature: float,
    max_step_tokens: int,
    max_total_tokens: int,
    num_responses: int,
    output_file: str,
    num_workers: int,
    batch_size: int,
    teacher_model_path: str,
    top_k_entropy: int
):
    total = len(data)
    print(f"Starting MOTAB processing | workers={num_workers} | total={total} items")
    save_jsonl_chunk([], output_file, mode='w')  # Clear output file
    print("-" * 60)

    tasks = [
        (item, student_url, teacher_url, epsilon0, theta, temperature,
         max_step_tokens, max_total_tokens, num_responses, top_k_entropy)
        for item in data
    ]
    buffer = []

    if num_workers <= 1:
        # ── Single Process Mode ──────────────────────────────────────────────
        init_worker(teacher_model_path)
        for task in tqdm(tasks, desc="MOTAB Progress", unit="items"):
            res = process_item_worker(task)
            buffer.append(res)
            if len(buffer) >= batch_size:
                save_jsonl_chunk(buffer, output_file, mode='a')
                buffer = []
    else:
        # ── Multi-Process Mode ───────────────────────────────────────────────
        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(teacher_model_path,)
        ) as pool:
            iterator = pool.imap_unordered(process_item_worker, tasks, chunksize=1)
            for res in tqdm(iterator, total=total, desc="MOTAB Progress", unit="items"):
                buffer.append(res)
                if len(buffer) >= batch_size:
                    save_jsonl_chunk(buffer, output_file, mode='a')
                    buffer = []

    if buffer:
        save_jsonl_chunk(buffer, output_file, mode='a')

    print("-" * 60)
    print("All MOTAB processing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MOTAB data synthesis"
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

    # ── Model Path ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--teacher_model_path", type=str,
        default="/path/to/models/qwen-32b",
        help="Teacher model path for loading tokenizer "
             "(for precise logprob_start_len; default: /path/to/models/qwen-32b)"
    )

    # ── MOTAB Algorithm Parameters ────────────────────────────────────────────
    parser.add_argument(
        "--epsilon0", type=float, default=0.2,
        help="Base safety threshold ε₀ ∈ (0,1]: oracle value must exceed ε₀·exp(-θ·H) "
             "to accept student step (default: 0.2)"
    )
    parser.add_argument(
        "--theta", type=float, default=1.0,
        help="Entropy scaling factor θ > 0: controls how much teacher uncertainty "
             "relaxes the boundary (default: 1.0)"
    )
    parser.add_argument(
        "--top_k_entropy", type=int, default=20,
        help="Top-k tokens used for teacher entropy approximation H(ε_T|s) (default: 20)"
    )
    parser.add_argument(
        "--num_responses", type=int, default=5,
        help="Number of SFT trajectories to generate per question (default: 5)"
    )

    # ── Generation Parameters ────────────────────────────────────────────────
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature for student and teacher generation (default: 0.6)"
    )
    parser.add_argument(
        "--max_step_tokens", type=int, default=8192,
        help="Maximum tokens per student reasoning step (default: 8192)"
    )
    parser.add_argument(
        "--max_total_tokens", type=int, default=32768,
        help="Maximum total tokens per complete trajectory (default: 32768)"
    )

    # ── Control Parameters ───────────────────────────────────────────────────
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel worker processes (1=single process, >1=multi-process) (default: 1)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Number of items to buffer before writing to disk (default: 10)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MOTAB Configuration:")
    print(f"  Input File           : {args.input_file}")
    print(f"  Output File          : {args.output_file}")
    print(f"  Student API URL      : {args.student_url}")
    print(f"  Teacher API URL      : {args.teacher_url}")
    print(f"  Teacher Model Path   : {args.teacher_model_path or 'Not specified (char-count approx)'}")
    print(f"  Base Threshold ε₀    : {args.epsilon0}")
    print(f"  Entropy Scale θ      : {args.theta}")
    print(f"  Top-k for Entropy    : {args.top_k_entropy}")
    print(f"  Responses per Item   : {args.num_responses}")
    print(f"  Temperature          : {args.temperature}")
    print(f"  Max tokens/step      : {args.max_step_tokens}")
    print(f"  Max tokens/response  : {args.max_total_tokens}")
    print(f"  Parallel Workers     : {args.num_workers}")
    print(f"  Disk Write Batch     : {args.batch_size}")
    print(f"  Request Timeout      : 10800s (3 hours)")
    print("=" * 60)

    data = load_jsonl(args.input_file)
    if not data:
        print("Error: Input file is empty or cannot be read.")
        return

    run_process(
        data               = data,
        student_url        = args.student_url,
        teacher_url        = args.teacher_url,
        epsilon0           = args.epsilon0,
        theta              = args.theta,
        temperature        = args.temperature,
        max_step_tokens    = args.max_step_tokens,
        max_total_tokens   = args.max_total_tokens,
        num_responses      = args.num_responses,
        output_file        = args.output_file,
        num_workers        = args.num_workers,
        batch_size         = args.batch_size,
        teacher_model_path = args.teacher_model_path,
        top_k_entropy      = args.top_k_entropy
    )


if __name__ == "__main__":
    # Ensure multi-processing works correctly on macOS/Windows
    mp.set_start_method('spawn', force=True)
    main()
