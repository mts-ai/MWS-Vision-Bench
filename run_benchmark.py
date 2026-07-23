"""MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This file: Main benchmark runner script for executing inference and evaluation pipeline

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

# Standard library imports
import argparse
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import List, Dict, Any

# Third-party imports
from prettytable import PrettyTable

# Local application imports
from src.evaluation.get_score_ru import get_metrics
from src.utils.dataset_loader import load_benchmark_datasets

# Setup command line argument parser
parser = argparse.ArgumentParser(
    description='MWSVisionBench - Run OCR benchmark inference and evaluation pipeline'
)
parser.add_argument('--model_name', default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help='Model name to use for inference')
parser.add_argument('--api_url', default=None,
                    help='API endpoint URL for inference (if not provided, uses model-specific default)')
parser.add_argument('--base_path', default=None,
                    help='Base path for image files (required only for local data files)')
parser.add_argument('--api_key', default=None,
                    help='API key for authentication')
parser.add_argument('--data_path', nargs='+', default=None,
                    help='Local JSON files to process (if not set, downloads from HuggingFace)')
parser.add_argument('--hf_token', default=None,
                    help='HuggingFace token for private test dataset (or set HF_TOKEN env var)')
parser.add_argument('--hf_revision', default=None,
                    help='HuggingFace dataset revision (branch name or commit hash)')
parser.add_argument('--cache_dir', default=None,
                    help='Cache directory for HuggingFace datasets')
parser.add_argument('--sample', type=int, default=None,
                    help='Number of samples to process (for quick testing)')
parser.add_argument('--start_index', type=int, default=0,
                    help='Starting index for processing')
parser.add_argument('--end_index', type=int, default=-1,
                    help='Ending index for processing (-1 for all)')
parser.add_argument('--raw_output_path', default=None,
                    help='(DEPRECATED) Path to existing raw output file')
parser.add_argument('--test_name', default=None,
                    help='Name for the test run (used as subfolder in results/)')
parser.add_argument('--use_base_prompt', action='store_true',
                    help='Whether to use base prompt for inference')
parser.add_argument('--max_workers', type=int, default=None,
                    help='Number of parallel workers for inference (if not set, uses model-specific default)')
args = parser.parse_args()

# Configuration constants
INFERENCE_SCRIPT = 'src/inference/inference_unified.py'

def get_short_model_name(full_name: str) -> str:
    """Extract short model name from full model path.
    
    Args:
        full_name: Full model name or path (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct')
        
    Returns:
        Short model name (e.g., 'Qwen2.5-VL-7B-Instruct')
    """
    if '/' in full_name:
        return full_name.split('/')[-1]
    return full_name

# Initialize test environment and logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
test_name = args.test_name if args.test_name else f"{get_short_model_name(args.model_name)}_{timestamp}"

# Create output directories
results_dir = os.path.join("results", test_name)
logs_dir = os.path.join("logs")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Setup logging configuration
log_path = os.path.join(logs_dir, f"{test_name}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logging.info(f"Starting benchmark pipeline for test: {test_name}")
logging.info(f"Model: {args.model_name}")
logging.info(f"Results will be saved to: {results_dir}")

# Load benchmark datasets with smart fallback
logging.info("Loading benchmark datasets...")
try:
    datasets, split_names = load_benchmark_datasets(
        data_paths=args.data_path,
        base_path=args.base_path,
        hf_token=args.hf_token or os.environ.get('HF_TOKEN'),
        hf_revision=args.hf_revision,
        cache_dir=args.cache_dir,
        sample=args.sample,
        silent=False
    )
    logging.info(f"Loaded {len(datasets)} dataset(s): {split_names}")
except Exception as e:
    logging.error(f"Failed to load datasets: {e}")
    raise

def run_inference_and_eval(dataset: List[Dict[str, Any]], output_prefix: str, 
                           temp_data_path: str) -> str:
    """Run inference and evaluation for a single dataset.
    
    Args:
        dataset: List of dataset entries (dicts)
        output_prefix: Prefix for output filenames
        temp_data_path: Temporary path to save dataset JSON
        
    Returns:
        Path to the evaluation output file
    """
    # Define paths for this run (no timestamp in filenames)
    raw_output_path = os.path.join(results_dir, f"{output_prefix}_raw.json")
    eval_output_path = os.path.join(results_dir, f"{output_prefix}_eval.json")
    
    # Save dataset to temporary JSON file for inference script
    with open(temp_data_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # Run inference
    inference_cmd = [
        "python", INFERENCE_SCRIPT,
        "--model_name", args.model_name,
        "--data_path", temp_data_path,
        "--start_index", str(args.start_index),
        "--end_index", str(args.end_index),
        "--output_path", raw_output_path
    ]
    
    # Add base_path only if provided (for local images)
    if args.base_path:
        inference_cmd.extend(["--base_path", args.base_path])

    # Add API-specific arguments for unified inference script
    if args.api_url:
        inference_cmd.extend(["--api_url", args.api_url])
    if args.api_key:
        inference_cmd.extend(["--api_key", args.api_key])
    if args.use_base_prompt:
        inference_cmd.append("--use_base_prompt")
    if args.max_workers is not None:
        inference_cmd.extend(["--max_workers", str(args.max_workers)])

    # Run inference up to 3 times - in case some questions will require one more attempt
    for attempt in range(2):
        logging.info(f"Running {INFERENCE_SCRIPT} for {temp_data_path} (attempt {attempt + 1})...")
        subprocess.run(inference_cmd, check=True)
        
        # Check if there are any errors in the output
        if not os.path.exists(raw_output_path):
            logging.error(f"Raw output file {raw_output_path} not found after inference run!")
            continue
        with open(raw_output_path, 'r') as f:
            results = json.load(f)
            error_count = sum(1 for item in results if item.get("predict") == "ERROR in getting response")
            if error_count == 0:
                break
            logging.info(f"Found {error_count} errors, retrying...")

    # Run eval with updated version
    eval_cmd = [
        "python", "src/evaluation/eval_parallel.py",
        "--input_path", raw_output_path,
        "--output_path", eval_output_path
    ]
    subprocess.run(eval_cmd, check=True)
    
    return eval_output_path

# Main execution: process all datasets
eval_paths: List[str] = []
logging.info("Starting inference and evaluation for all datasets...")

# Create temp directory for dataset files
temp_dir = os.path.join(results_dir, "temp_datasets")
os.makedirs(temp_dir, exist_ok=True)

for i, (dataset, split_name) in enumerate(zip(datasets, split_names)):
    prefix = f"{get_short_model_name(args.model_name)}_{split_name}"
    temp_data_path = os.path.join(temp_dir, f"{split_name}.json")
    
    logging.info(f"Processing dataset {i+1}/{len(datasets)}: {split_name} ({len(dataset)} samples)")
    eval_path = run_inference_and_eval(dataset, prefix, temp_data_path)
    eval_paths.append(eval_path)

# Calculate metrics for each processed part
logging.info("Calculating metrics for each part...")
metrics_list: List[Dict[str, Any]] = []
for i, eval_path in enumerate(eval_paths):
    logging.info(f"Calculating metrics for part {i+1}: {eval_path}")
    metrics, _ = get_metrics(eval_path)
    metrics_list.append(metrics)

# Combine results if multiple datasets were processed
if len(datasets) == 2:
    logging.info("Combining evaluation results from both parts...")
    combined_eval_output = os.path.join(
        results_dir, 
        f"{get_short_model_name(args.model_name)}_combined_eval.json"
    )
    
    # Merge evaluation results from all parts
    combined_data = []
    for eval_path in eval_paths:
        with open(eval_path, 'r', encoding='utf-8') as f:
            combined_data.extend(json.load(f))
    
    with open(combined_eval_output, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    # Calculate metrics for combined results
    logging.info("Calculating combined metrics...")
    combined_metrics, _ = get_metrics(combined_eval_output)
    metrics_list.append(combined_metrics)

# Generate and display results summary table
logging.info("Generating results summary table...")
table = PrettyTable()
# Use actual split names for column headers
column_headers = split_names.copy()
if len(datasets) == 2:
    column_headers.append("combined")
table.field_names = ["Metric"] + column_headers

# Collect all unique metrics across all results
all_metrics = set()
for metrics in metrics_list:
    all_metrics.update(metrics.keys())

# Populate table with metric scores
for metric in sorted(all_metrics):
    row = [metric]
    for metrics in metrics_list:
        score = metrics.get(metric, "-")
        if isinstance(score, float):
            row.append(f"{score:.3f}")
        else:
            row.append(str(score))
    table.add_row(row)

# Calculate and add average scores row
average_row = ["average"]
for metrics in metrics_list:
    numeric_scores = [
        score for metric in sorted(all_metrics)
        if (score := metrics.get(metric)) is not None and isinstance(score, float)
    ]
    avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
    average_row.append(f"{avg_score:.3f}")
table.add_row(average_row)

# Display final results
logging.info("BENCHMARK RESULTS SUMMARY")
logging.info("\n" + str(table))

logging.info("Benchmark pipeline completed successfully!")
logging.info(f"Results saved to: {results_dir}")
logging.info(f"Log file: {log_path}") 