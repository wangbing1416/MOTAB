# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluate responses from a pre-generated dataset using deepscaler metrics.
This script does NOT perform inference, only computes evaluation metrics.
"""
import csv
import numpy as np
import hydra
import os
import time
from tabulate import tabulate
from collections import Counter

import pandas as pd

from transformers import AutoTokenizer

from deepscaler.rewards.math_reward import deepscaler_reward_fn
from deepscaler.rewards.math_utils.utils import extract_answer


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    start_time = time.time()
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Load tokenizer for response length calculation
    local_path = config.model.path
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # Load the pre-generated dataset
    input_path = config.data.input_path
    if input_path.endswith('.parquet'):
        dataset = pd.read_parquet(input_path)
    elif input_path.endswith('.json'):
        dataset = pd.read_json(input_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    print(f"Loaded dataset with {len(dataset)} samples from {input_path}")

    # Compute correctness if not present
    if 'correctness' not in dataset.columns:
        print("Computing correctness field...")
        total_lst = compute_correctness(dataset)
        dataset['correctness'] = total_lst
    else:
        print("Correctness field already exists, skipping computation.")

    # Save the evaluated dataset
    output_path = config.data.output_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    dataset.to_json(output_path, orient='records', force_ascii=False, lines=True)
    print(f"Saved evaluated dataset to {output_path}")

    # Compute evaluation metrics
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    # Calculate response length statistics
    output_lst = [str(r) for responses_this in list(responses) for r in responses_this]
    print(f"Total responses to evaluate: {len(output_lst)}")
    
    unpad_tokenized = tokenizer(output_lst, add_special_tokens=False).input_ids
    len_response_tokens = [len(tokens) for tokens in unpad_tokenized]
    len_mean = np.mean(len_response_tokens)
    
    # Calculate cutoff ratio if response_length is specified in config
    if hasattr(config, 'rollout') and hasattr(config.rollout, 'response_length'):
        cutoff_ratio = sum([l == config.rollout.response_length for l in len_response_tokens]) / len(unpad_tokenized)
    else:
        cutoff_ratio = 0.0
    print(f'Mean response tokens: {len_mean:.2f}')
    print(f'Length cutoff ratio: {cutoff_ratio:.4f}')

    # Compute pass@n and cons@n metrics
    passes = 0
    total = len(dataset)
    total_scores = []
    conses = 0

    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        
        score_lst = []
        for r in response_lst:
            try:
                if config.data.skip_format_reward:
                    score = reward_fn(r, ground_truth, skip_format_reward=True)
                else:
                    score = reward_fn(r, ground_truth, skip_format_reward=False)
            except:
                score = reward_fn(r, ground_truth, skip_format_reward=True)
            score_lst.append(score)
        
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1

        # Compute consensus metrics
        extracted_lst = [extract_answer(r) for r in response_lst]
        extracted_lst = [r for r in extracted_lst if r is not None]
        cons_answers = find_mode(extracted_lst)
        cons_response_lst = [r for r in response_lst if extract_answer(r) in cons_answers]
        is_cons_correct_list = list()
        for r in cons_response_lst:
            try:
                if config.data.skip_format_reward:
                    score = reward_fn(r, ground_truth, skip_format_reward=True)
                else:
                    score = reward_fn(r, ground_truth, skip_format_reward=False)
            except:
                score = reward_fn(r, ground_truth, skip_format_reward=True)
            is_cons_correct_list.append(score)
        if any(is_cons_correct_list):
            conses += np.mean(is_cons_correct_list)

    n_samples = config.data.n_samples
    pass_at_n = passes / total
    pass_at_1 = np.mean(total_scores)
    cons_at_n = conses / total

    spent_time = time.time() - start_time
    spent_hours = spent_time / 60 / 60
    
    # Save metrics to CSV
    csv_path = os.path.join(output_dir, f'eval_{spent_hours:.2f}h.csv')

    # Prepare the row data
    dataset_name = os.path.basename(input_path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@{n_samples}': pass_at_n,
        f'cons@{n_samples}': cons_at_n,
        'cutoff_ratio': cutoff_ratio,
        'mean_response_tokens': len_mean,
        'run_hours': spent_hours
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)

    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]

    # Print table
    print("\nEvaluation Results:")
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))


def compute_correctness(dataset):
    """Compute correctness for each response in the dataset."""
    total_lst = list()
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        gt = row['reward_model']['ground_truth']
        responses_this = row['responses']

        true_false = [int(deepscaler_reward_fn(response, gt, skip_format_reward=True)) 
                      for response in responses_this]
        total_lst.append(true_false)
    return total_lst


def find_mode(lst):
    """Find the mode(s) of a list."""
    if len(lst) == 0:
        return list()
    counter = Counter(lst)
    max_count = max(counter.values())
    mode = [k for k, v in counter.items() if v == max_count]
    return mode


def select_reward_fn(data_source):
    """Select the appropriate reward function based on data source."""
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from deepscaler.rewards.math_reward import deepscaler_reward_fn
        return deepscaler_reward_fn


if __name__ == '__main__':
    main()
