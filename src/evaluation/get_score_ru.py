"""
MWSVisionBench - Russian document benchmark for multimodal LLMs

This file: Leaderboard metrics for document understanding and anti-fraud

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

# Standard library imports
import argparse
from collections import Counter
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.evaluation.bootstrap import (
        bootstrap_category_means,
        bootstrap_stratified_score,
        validate_bootstrap_options,
    )
except ModuleNotFoundError:  # Support direct script execution.
    from bootstrap import (  # type: ignore[no-redef]
        bootstrap_category_means,
        bootstrap_stratified_score,
        validate_bootstrap_options,
    )


VISION_CATEGORIES = (
    "text_grounding",
    "reasoning_vqa",
    "full_page_ocr",
    "document_parsing",
    "key_extraction",
)

VISION_CATEGORY_METRICS = {
    "text_grounding": "text_grounding_basic",
    "reasoning_vqa": "vqa (knowledge_reasoning)",
    "full_page_ocr": "image 2 text (text_recognition)",
    "document_parsing": "image 2 markdown (element_parsing)",
    "key_extraction": "keymap (relationship_extraction)",
}


def _compute_antifraud_details(
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate AF v0.1 and its auditable intermediate values."""
    labels = ("ai_gen", "edited", "original")
    correct = {label: 0 for label in labels}
    total = {label: 0 for label in labels}
    edited_reason_scores: List[float] = []

    for item in items:
        label = item.get("dataset_name", "")
        if label not in labels:
            continue
        total[label] += 1
        if item.get("correct"):
            correct[label] += 1
        if label == "edited":
            reason_score = float(item.get("reason_score") or 0.0)
            edited_reason_scores.append(min(1.0, max(0.0, reason_score)))

    recalls = {
        label: correct[label] / total[label] if total[label] else 0.0
        for label in labels
    }
    # AF is defined for exactly three classes. Missing classes contribute zero
    # recall instead of inflating a sampled run by changing the denominator.
    balanced_accuracy = sum(recalls.values()) / len(labels)
    edited_reason_score = (
        sum(edited_reason_scores) / len(edited_reason_scores)
        if edited_reason_scores
        else 0.0
    )
    score = (
        0.75 * max(0.0, balanced_accuracy - 1 / 3)
        + 0.5 * edited_reason_score
    )
    return {
        "score": min(1.0, max(0.0, score)),
        "balanced_accuracy": balanced_accuracy,
        "edited_reason_score": edited_reason_score,
        "recall_by_class": recalls,
        "correct_by_class": correct,
        "count_by_class": total,
        "count": sum(total.values()),
    }


def _compute_antifraud_score(items: List[Dict[str, Any]]) -> float:
    """Calculate the combined anti-fraud classification/explanation score."""
    return float(_compute_antifraud_details(items)["score"])


def get_metrics(
    json_path: str,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
    confidence_level: float = 0.95,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Calculate metrics for all supported task types.
    
    Args:
        json_path: Path to the evaluation JSON file
        bootstrap_samples: Number of bootstrap replicates. Zero disables CIs.
        bootstrap_seed: Random seed used for deterministic resampling.
        confidence_level: Confidence level for percentile intervals.
        
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
    antifraud_items = []
    present_vision_categories = set()
    unknown_task_types: Counter[str] = Counter()
    
    # Collect scores by task type (handle both ru and en versions)
    for item in data_list:
        task_type = item["type"]
        score = item.get("score", 0)
        
        if task_type in ["text grounding ru", "text grounding en"]:
            text_grounding_scores.append(score)
            present_vision_categories.add("text_grounding")
        elif task_type in ["reasoning VQA ru", "reasoning VQA en"]:
            reasoning_vqa_scores.append(score)
            present_vision_categories.add("reasoning_vqa")
        elif task_type in ["full-page OCR ru", "full-page OCR en"]:
            full_page_ocr_scores.append(score)
            present_vision_categories.add("full_page_ocr")
        elif task_type in ["document parsing ru", "document parsing en"]:
            document_parsing_scores.append(score)
            present_vision_categories.add("document_parsing")
        elif task_type in ["key information extraction ru", "key information extraction en"]:
            key_extraction_scores.append(score)
            present_vision_categories.add("key_extraction")
        elif task_type == "antifraud ru":
            antifraud_items.append(item)
        else:
            unknown_task_types[str(task_type)] += 1
    
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
    antifraud_avg = (
        _compute_antifraud_score(antifraud_items)
        if antifraud_items
        else 0.0
    )
    
    vision_count = (
        len(text_grounding_scores)
        + len(reasoning_vqa_scores)
        + len(full_page_ocr_scores)
        + len(document_parsing_scores)
        + len(key_extraction_scores)
    )

    metrics: Dict[str, float] = {}
    if vision_count:
        metrics.update(
            {
                "image 2 text (text_recognition)": full_page_ocr_avg,
                "text_grounding_basic": text_grounding_avg,
                "keymap (relationship_extraction)": key_extraction_avg,
                "image 2 markdown (element_parsing)": document_parsing_avg,
                "vqa (knowledge_reasoning)": reasoning_vqa_avg,
            }
        )
    if antifraud_items:
        metrics["antifraud (document_verification)"] = antifraud_avg

    # Leaderboard Overall is the macro-average of the five original category
    # scores. Anti-fraud is deliberately excluded and reported separately.
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
    missing_vision_categories = [
        category
        for category in VISION_CATEGORIES
        if category not in present_vision_categories
    ]
    vision_coverage_complete = not missing_vision_categories
    
    # Detailed breakdown - use simple structure
    detailed = {
        "overall": {
            "count": vision_count,
            "average": overall_avg,
            "includes_antifraud": False,
            "coverage_complete": vision_coverage_complete,
            "present_categories": [
                category
                for category in VISION_CATEGORIES
                if category in present_vision_categories
            ],
            "missing_categories": missing_vision_categories,
        },
        "total_count": vision_count + len(antifraud_items),
        "unknown_task_types": dict(sorted(unknown_task_types.items())),
    }
    if antifraud_items:
        detailed["antifraud"] = _compute_antifraud_details(antifraud_items)

    if bootstrap_samples:
        validate_bootstrap_options(bootstrap_samples, confidence_level)
        bootstrap_details: Dict[str, Any] = {
            "method": "percentile",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "metrics": {},
            "stratification": {},
        }
        category_scores = {
            "text_grounding": text_grounding_scores,
            "reasoning_vqa": reasoning_vqa_scores,
            "full_page_ocr": full_page_ocr_scores,
            "document_parsing": document_parsing_scores,
            "key_extraction": key_extraction_scores,
        }
        if vision_count:
            vision_bootstrap = bootstrap_category_means(
                category_scores,
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )
            bootstrap_details["metrics"].update(
                {
                    VISION_CATEGORY_METRICS[category]: interval
                    for category, interval
                    in vision_bootstrap["categories"].items()
                }
            )
            bootstrap_details["overall"] = vision_bootstrap["overall"]
            bootstrap_details["stratification"]["vision"] = "task_category"
        if antifraud_items:
            antifraud_interval = bootstrap_stratified_score(
                antifraud_items,
                strata_key="dataset_name",
                score_fn=_compute_antifraud_score,
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )
            bootstrap_details["metrics"][
                "antifraud (document_verification)"
            ] = antifraud_interval
            bootstrap_details["antifraud"] = antifraud_interval
            bootstrap_details["stratification"]["antifraud"] = (
                "dataset_name"
            )
        detailed["bootstrap"] = bootstrap_details
    
    return metrics, detailed


def format_score(
    score: float,
    interval: Optional[Dict[str, float]] = None,
    confidence_level: float = 0.95,
) -> str:
    """Format a point estimate and an optional confidence interval."""
    formatted = f"{score:.3f}"
    if interval is not None:
        confidence_percent = f"{confidence_level * 100:g}%"
        formatted += (
            f" [{confidence_percent} CI: {interval['low']:.3f}-"
            f"{interval['high']:.3f}]"
        )
    return formatted


def get_summary_score(
    metrics: Dict[str, float],
    detailed: Dict[str, Any],
    dataset_family: str,
) -> float:
    """Return the score displayed in the benchmark summary table."""
    if dataset_family == "antifraud":
        return metrics.get("antifraud (document_verification)", 0.0)
    return float(detailed["overall"]["average"])


def main() -> None:
    """Command line interface for metrics calculation."""
    parser = argparse.ArgumentParser(
        description="MWSVisionBench - Calculate simplified metrics for Russian OCR tasks"
    )
    parser.add_argument("--input_path", required=True, help="Path to evaluation JSON file")
    parser.add_argument("--output_path", help="Path to save detailed metrics (optional)")
    parser.add_argument(
        "--bootstrap_samples",
        "--bootstrap-samples",
        dest="bootstrap_samples",
        type=int,
        default=0,
        help="Number of bootstrap replicates for confidence intervals",
    )
    parser.add_argument(
        "--bootstrap_seed",
        "--bootstrap-seed",
        dest="bootstrap_seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling (default: 42)",
    )
    parser.add_argument(
        "--confidence_level",
        "--confidence-level",
        dest="confidence_level",
        type=float,
        default=0.95,
        help="Bootstrap confidence level (default: 0.95)",
    )
    args = parser.parse_args()
    
    # Calculate metrics
    metrics, detailed = get_metrics(
        args.input_path,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    bootstrap_details = detailed.get("bootstrap", {})
    metric_intervals = bootstrap_details.get("metrics", {})

    if detailed["unknown_task_types"]:
        print(
            "WARNING: ignored unknown task types: "
            f"{detailed['unknown_task_types']}"
        )
    
    # Print results in old format
    print("Russian Scores:")
    for metric_name, score in metrics.items():
        print(
            f"{metric_name}: "
            f"{format_score(score, metric_intervals.get(metric_name), args.confidence_level)}"
        )
    
    if detailed["overall"]["count"]:
        print("\nOverall Scores:")
        overall_label = (
            "Russian Overall Score"
            if detailed["overall"]["coverage_complete"]
            else "Russian Partial Overall Score"
        )
        overall_score = format_score(
            detailed["overall"]["average"],
            bootstrap_details.get("overall"),
            args.confidence_level,
        )
        print(f"{overall_label}: {overall_score}")
        if not detailed["overall"]["coverage_complete"]:
            print(
                "WARNING: result is not leaderboard-comparable; missing "
                f"categories: {detailed['overall']['missing_categories']}"
            )
    if "antifraud (document_verification)" in metrics:
        antifraud_score = format_score(
            metrics["antifraud (document_verification)"],
            bootstrap_details.get("antifraud"),
            args.confidence_level,
        )
        print(f"\nAnti-fraud Score: {antifraud_score}")
    print("End of Code!")
    
    # Save detailed metrics if requested
    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed metrics saved to {args.output_path}")


if __name__ == "__main__":
    main()
