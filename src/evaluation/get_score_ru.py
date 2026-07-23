"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This file: Simplified metrics calculation for 5 task types only

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

# Standard library imports
import argparse
import json
from typing import Any, Dict, List, Tuple


def get_metrics(json_path: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Calculate metrics for our 5 task types.
    
    Args:
        json_path: Path to the evaluation JSON file
        
    Returns:
        Tuple of (metrics_dict, detailed_scores)
        - metrics_dict: Dictionary with metric names and scores
        - detailed_scores: Detailed breakdown with counts and averages
    """
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    # Initialize score lists for each task type
    text_grounding_scores = []
    reasoning_vqa_scores = []
    full_page_ocr_scores = []
    document_parsing_scores = []
    key_extraction_scores = []
    
    # Collect scores by task type (handle both ru and en versions)
    for item in data_list:
        task_type = item["type"]
        score = item.get("score", 0)
        
        if task_type in ["text grounding ru", "text grounding en"]:
            text_grounding_scores.append(score)
        elif task_type in ["reasoning VQA ru", "reasoning VQA en"]:
            reasoning_vqa_scores.append(score)
        elif task_type in ["full-page OCR ru", "full-page OCR en"]:
            full_page_ocr_scores.append(score)
        elif task_type in ["document parsing ru", "document parsing en"]:
            document_parsing_scores.append(score)
        elif task_type in ["key information extraction ru", "key information extraction en"]:
            key_extraction_scores.append(score)
        else:
            # Skip unknown types silently or just ignore them
            pass
    
    # Calculate averages
    def safe_average(scores: List[float]) -> float:
        """Calculate safe average of scores, returning 0.0 for empty lists."""
        return sum(scores) / len(scores) if scores else 0.0
    
    # Calculate individual metric averages
    text_grounding_avg = safe_average(text_grounding_scores)
    reasoning_vqa_avg = safe_average(reasoning_vqa_scores)
    full_page_ocr_avg = safe_average(full_page_ocr_scores)
    document_parsing_avg = safe_average(document_parsing_scores)
    key_extraction_avg = safe_average(key_extraction_scores)
    
    metrics = {
        "image 2 text (text_recognition)": full_page_ocr_avg,
        "text_grounding_basic": text_grounding_avg,
        "keymap (relationship_extraction)": key_extraction_avg,
        "image 2 markdown (element_parsing)": document_parsing_avg,
        "vqa (knowledge_reasoning)": reasoning_vqa_avg
    }
    
    # Calculate overall average as mean of metric averages (like old code)
    metric_averages = []
    if text_grounding_scores:
        metric_averages.append(text_grounding_avg)
    if reasoning_vqa_scores:
        metric_averages.append(reasoning_vqa_avg)
    if full_page_ocr_scores:
        metric_averages.append(full_page_ocr_avg)
    if document_parsing_scores:
        metric_averages.append(document_parsing_avg)
    if key_extraction_scores:
        metric_averages.append(key_extraction_avg)
    
    overall_avg = safe_average(metric_averages)
    
    # Detailed breakdown - use simple structure
    total_count = (len(text_grounding_scores) + len(reasoning_vqa_scores) + 
                   len(full_page_ocr_scores) + len(document_parsing_scores) + 
                   len(key_extraction_scores))
    
    detailed = {
        "overall": {
            "count": total_count,
            "average": overall_avg
        }
    }
    
    return metrics, detailed


def main() -> None:
    """Command line interface for metrics calculation."""
    parser = argparse.ArgumentParser(
        description="MWSVisionBench - Calculate simplified metrics for Russian OCR tasks"
    )
    parser.add_argument("--input_path", required=True, help="Path to evaluation JSON file")
    parser.add_argument("--output_path", help="Path to save detailed metrics (optional)")
    
    args = parser.parse_args()
    
    # Calculate metrics
    metrics, detailed = get_metrics(args.input_path)
    
    # Print results in old format
    print("Russian Scores:")
    for metric_name, score in metrics.items():
        print(f"{metric_name}: {score:.3f}")
    
    print(f"\nOverall Scores:")
    print(f"Russian Overall Score: {detailed['overall']['average']:.3f}")
    print("End of Code!")
    
    # Save detailed metrics if requested
    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed metrics saved to {args.output_path}")


if __name__ == "__main__":
    main()
