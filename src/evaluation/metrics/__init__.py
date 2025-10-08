# Evaluation metrics adapted from OCRBench_v2
# Original source: https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2/eval_scripts

from .vqa_metric import vqa_evaluation, levenshtein_distance
from .iou_metric import calculate_iou, vqa_with_position_evaluation, extract_coordinates
from .teds_metric import TEDS, doc_parsing_evaluation, convert_str_to_dict, generate_combinations, compute_f1_score
from .page_ocr_metric import cal_per_metrics, contain_chinese_string
from .spotting_metric import extract_bounding_boxes_robust

__all__ = [
    'vqa_evaluation',
    'levenshtein_distance',
    'calculate_iou', 
    'vqa_with_position_evaluation',
    'extract_coordinates',
    'TEDS',
    'doc_parsing_evaluation',
    'convert_str_to_dict',
    'generate_combinations',
    'compute_f1_score',
    'cal_per_metrics',
    'contain_chinese_string',
    'extract_bounding_boxes_robust'
]