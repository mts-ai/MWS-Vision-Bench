"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This file: Simplified evaluation script for 5 task types:
- text grounding ru
- reasoning VQA ru  
- full-page OCR ru
- document parsing ru
- key information extraction ru

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

# Standard library imports
import argparse
import json
import time
import warnings
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List

# Suppress specific warnings that clutter output
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*BLEU score.*')
warnings.filterwarnings('ignore', category=UserWarning, module='jieba')
warnings.filterwarnings('ignore', category=UserWarning, module='nltk')

# Third-party imports
from tqdm import tqdm

# Local application imports
from metrics.iou_metric import calculate_iou, extract_coordinates
from metrics.page_ocr_metric import cal_per_metrics
from metrics.teds_metric import (
    TEDS, 
    compute_f1_score, 
    convert_str_to_dict, 
    doc_parsing_evaluation, 
    generate_combinations
)
from metrics.vqa_metric import vqa_evaluation


def is_nan_value(value: Any) -> bool:
    """Check if value is NaN/None.
    
    Args:
        value: Value to check for NaN/None status
        
    Returns:
        True if value is NaN or None, False otherwise
    """
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == 'nan':
        return True
    return False


def process_single_item(data_item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single evaluation item for our 5 task types.
    
    Args:
        data_item: Dictionary containing item data with 'type', 'predict', etc.
        
    Returns:
        Updated data_item with calculated score
    """
    start_time = time.time()
    
    # Check if prediction is an error - always assign score 0.0
    predict = data_item.get("predict", "")
    if isinstance(predict, str) and predict == "ERROR in getting response":
        data_item["score"] = 0.0
        data_item["eval_time"] = time.time() - start_time
        return data_item
    
    # Truncate overly long predictions
    MAX_PREDICT_LEN = 18000
    if isinstance(data_item.get("predict"), str) and len(data_item["predict"]) > MAX_PREDICT_LEN:
        data_item["predict"] = data_item["predict"][:MAX_PREDICT_LEN] + " [TRUNCATED]"
        data_item["predict_truncated"] = True
    
    # Process based on task type
    task_type = data_item["type"]
    
    if task_type == "text grounding ru":
        # Text grounding - coordinate extraction
        if not isinstance(data_item["predict"], str):
            data_item["score"] = 0
        else:
            try:
                pred_coords = extract_coordinates(data_item["predict"])
                if pred_coords is not None and len(pred_coords) == 4:  # x1, y1, x2, y2
                    gt_coords = data_item["answers"]  # Should be [x1, y1, x2, y2]
                    if len(gt_coords) == 4:
                        iou = calculate_iou(pred_coords, gt_coords)
                        data_item["score"] = iou  # Save exact IoU value like old code
                    else:
                        data_item["score"] = 0
                else:
                    data_item["score"] = 0
            except Exception as e:
                print(f"Error in text grounding for item {data_item.get('id', 'unknown')}: {e}")
                data_item["score"] = 0
    
    elif task_type == "reasoning VQA ru":
        # VQA reasoning task
        data_item["score"] = vqa_evaluation(data_item["predict"], data_item["answers"])
    
    elif task_type == "full-page OCR ru":
        # Full page OCR
        if not isinstance(data_item["predict"], str) or not data_item["predict"]:
            data_item["score"] = 0
        else:
            assert len(data_item["answers"]) == 1
            # Use the same 4-metric combination as old code
            ocr_metrics = cal_per_metrics(data_item["predict"], data_item["answers"][0])
            if isinstance(ocr_metrics, dict):
                # Same formula as old code: (bleu + meteor + f_measure + (1 - edit_dist)) / 4
                def get_value_or_zero(value):
                    return 0.0 if value is None else value
                
                data_item["score"] = (
                    get_value_or_zero(ocr_metrics.get("bleu", 0)) + 
                    get_value_or_zero(ocr_metrics.get("meteor", 0)) + 
                    get_value_or_zero(ocr_metrics.get("f_measure", 0)) + 
                    get_value_or_zero((1 - ocr_metrics.get("edit_dist", 1)))
                ) / 4
            else:
                data_item["score"] = 0
    
    elif task_type == "document parsing ru":
        # Document parsing (markdown)
        if not isinstance(data_item["predict"], str):
            data_item["score"] = 0
        else:
            assert len(data_item["answers"]) == 1
            data_item["score"] = doc_parsing_evaluation(data_item["predict"], data_item["answers"][0])
    
    elif task_type == "key information extraction ru":
        # Key information extraction
        if not isinstance(data_item["predict"], str):
            data_item["score"] = 0
        else:
            assert len(data_item["answers"]) == 1
            try:
                # Use the same logic as old code
                answers = generate_combinations(data_item["answers"][0])
                
                if isinstance(answers, list) and len(answers) == 1:
                    if not isinstance(data_item["predict"], str):
                        data_item["score"] = 0
                    else:
                        try:
                            pred_dict = convert_str_to_dict(data_item["predict"])
                            data_item["score"] = compute_f1_score(pred_dict, answers[0])
                        except Exception as e:
                            print(f"Error converting predict to dict for item {data_item.get('id', 'unknown')}: {e}")
                            data_item["score"] = 0
                else:
                    # Multiple valid answers - take max score like old code
                    max_score = 0
                    for answer in answers:
                        try:
                            pred_dict = convert_str_to_dict(data_item["predict"])
                            score = compute_f1_score(pred_dict, answer)
                            if score > max_score:
                                max_score = score
                        except Exception as e:
                            print(f"Error in multi-answer key extraction for item {data_item.get('id', 'unknown')}: {e}")
                            continue
                    data_item["score"] = max_score
            except Exception as e:
                print(f"Error in key extraction for item {data_item.get('id', 'unknown')}: {e}")
                data_item["score"] = 0
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Ensure score is valid and numeric
    score = data_item.get("score", 0)
    if is_nan_value(score):
        data_item["score"] = 0
    elif not isinstance(score, (int, float)):
        print(f"WARNING: Non-numeric score for item {data_item.get('id', 'unknown')}: {score}")
        data_item["score"] = 0
    else:
        data_item["score"] = float(score)  # Ensure it's a number
    
    # Add timing info
    end_time = time.time()
    data_item["eval_time"] = end_time - start_time
    
    return data_item


def evaluate_parallel(data_list, num_processes=None):
    """Evaluate all items in parallel"""
    if num_processes is None:
        num_processes = min(cpu_count(), 8)  # Limit to reasonable number
    
    print(f"Evaluating {len(data_list)} items using {num_processes} processes...")
    
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_item, data_list),
            total=len(data_list),
            desc="Evaluating"
        ))
    
    return results


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate metrics for each task type.
    
    Args:
        results: List of processed evaluation items with scores
        
    Returns:
        Dictionary containing calculated metrics and summary statistics
    """
    task_metrics = {}
    
    # Group by task type
    for item in results:
        task_type = item["type"]
        if task_type not in task_metrics:
            task_metrics[task_type] = []
        
        # Ensure score is a number, not dict or other type
        score = item.get("score", 0)
        if isinstance(score, (int, float)):
            task_metrics[task_type].append(score)
        else:
            print(f"WARNING: Non-numeric score in item {item.get('id', 'unknown')}: {score}")
            task_metrics[task_type].append(0)
    
    # Calculate average for each type
    summary = {}
    total_score = 0
    total_count = 0
    
    for task_type, scores in task_metrics.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        summary[task_type] = {
            "avg_score": avg_score,
            "count": len(scores),
            "total_score": sum(scores)
        }
        total_score += sum(scores)
        total_count += len(scores)
        
        # Don't print here - leave it for run_benchmark.py
        pass
    
    # Overall average
    overall_avg = total_score / total_count if total_count > 0 else 0
    summary["overall"] = {
        "avg_score": overall_avg,
        "count": total_count,
        "total_score": total_score
    }
    
    return summary


def main() -> None:
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="MWSVisionBench evaluation for 5 Russian OCR task types"
    )
    parser.add_argument("--input_path", required=True, help="Path to input JSON file")
    parser.add_argument("--output_path", required=True, help="Path to output JSON file")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_path}")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"Loaded {len(data_list)} items")
    
    # Check task types
    task_types = set(item["type"] for item in data_list)
    print(f"Found task types: {task_types}")
    
    # Validate that we only have supported types
    supported_types = {
        "text grounding ru", 
        "reasoning VQA ru", 
        "full-page OCR ru", 
        "document parsing ru", 
        "key information extraction ru"
    }
    
    unsupported = task_types - supported_types
    if unsupported:
        print(f"WARNING: Found unsupported task types: {unsupported}")
        # Filter to only supported types
        original_count = len(data_list)
        data_list = [item for item in data_list if item["type"] in supported_types]
        print(f"Filtered to {len(data_list)} items (removed {original_count - len(data_list)} unsupported)")
    
    # Evaluate
    results = evaluate_parallel(data_list, args.num_processes)
    
    # Calculate metrics (no printing here)
    metrics = calculate_metrics(results)
    
    # Save results
    print(f"\nSaving results to {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
