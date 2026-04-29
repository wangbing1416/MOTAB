"""
EvalScope evaluation script with configurable parameters via argparse.
Supports multiple math benchmarks: math_500, aime24, aime25, amc, gpqa_diamond.
"""
import argparse
import os
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run EvalScope evaluation on math benchmarks')
    
    # Model and API configuration
    parser.add_argument('--model_name', type=str, default=os.getenv('MODEL_NAME', 'llm'),
                        help='Model name (default: env var MODEL_NAME or "llm")')
    parser.add_argument('--api_url', type=str, default=os.getenv('EVAL_API_URL', 'http://127.0.0.1:31011/v1/'),
                        help='API endpoint URL (default: env var EVAL_API_URL)')
    
    # Dataset configuration
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['math_500', 'aime24', 'aime25', 'ifeval', 'gpqa_diamond'],
                        help='List of datasets to evaluate')
    
    # Generation parameters
    parser.add_argument('--max_tokens', type=int, default=32768,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=-1,
                        help='Top-k sampling parameter (used for math_500 and gpqa_diamond)')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of responses to generate per request')
    
    # Evaluation configuration
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--timeout', type=int, default=6000000,
                        help='Timeout in milliseconds')
    parser.add_argument('--stream', action='store_true', default=False,
                        help='Enable streaming mode')
    
    # Optional: limit number of samples for testing
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of samples to evaluate (for testing)')
    parser.add_argument('--use_cache', type=str, default=None,
                        help='Path to cached outputs to reuse')
    
    return parser.parse_args()


def create_task_config(args, dataset_name):
    """
    Create TaskConfig for a specific dataset.
    
    Args:
        args: Parsed command line arguments
        dataset_name: Name of the dataset to configure
        
    Returns:
        TaskConfig object for the specified dataset
    """
    # Determine top_p based on dataset
    if dataset_name in ['math_500', 'gpqa_diamond']:
        top_p = 0.95
        use_top_k = True
    else:
        top_p = 0.95
        use_top_k = False
    
    api_url = args.api_url
    
    # Determine n_samples based on dataset
    # aime24, aime25, gpqa_diamond use n_samples from args, others use 1
    if dataset_name in ['aime24', 'aime25', 'amc', 'gpqa_diamond']:
        n_samples = args.n_samples
    else:
        n_samples = 1
    
    # Build generation config
    generation_config = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': top_p,
        'n': n_samples,
    }
    
    # Add top_k for specific datasets
    if use_top_k:
        generation_config['top_k'] = args.top_k
    
    # Build dataset args
    dataset_args = {
        dataset_name: {
            'filters': {'remove_until': '</think>'},
            'aggregation': 'mean_and_pass_at_k'
        }
    }
    
    # Create task configuration
    task_cfg = TaskConfig(
        model=args.model_name,
        api_url=api_url,
        eval_type=EvalType.SERVICE,
        datasets=[dataset_name],
        dataset_args=dataset_args,
        eval_batch_size=args.eval_batch_size,
        generation_config=generation_config,
        timeout=args.timeout,
        stream=args.stream,
    )
    
    # Add optional parameters if specified
    if args.limit is not None:
        task_cfg.limit = args.limit
    if args.use_cache is not None:
        task_cfg.use_cache = args.use_cache
    
    return task_cfg


def main():
    """Main function to run evaluations on specified datasets."""
    args = parse_args()
    
    print(f"Starting EvalScope evaluation")
    print(f"Model: {args.model_name}")
    print(f"API URL: {args.api_url}")
    print(f"Datasets: {args.datasets}")
    print(f"Generation config: max_tokens={args.max_tokens}, temperature={args.temperature}, "
          f"n_samples={args.n_samples}")
    print("=" * 60)
    
    # Run evaluation for each dataset
    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name}...")
        print(f"{'=' * 60}")
        
        task_cfg = create_task_config(args, dataset_name)
        run_task(task_cfg=task_cfg)
        
        print(f"\nCompleted evaluation on {dataset_name}")
    
    print(f"\n{'=' * 60}")
    print(f"All evaluations completed!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()