# Text spotting evaluation metrics adapted from OCRBench_v2
# Original source: https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2/eval_scripts

import re
import os
import ast


def extract_bounding_boxes_robust(predict_str):
    """
    Extract coordinates and text content from the given prediction string, 
    handling potential format issues.

    Args:
    predict_str (str): Model prediction output as a string.

    Returns:
    list: Extracted data in the format [[x1, y1, x2, y2, text_content], ...].
          Returns None if no valid data is extracted.
    """
    results = []
    seen = set()

    # try parsing with ast.literal_eval
    try:
        data = ast.literal_eval(predict_str)
    except Exception:
        data = None

    if data is None:
        # Try to match individual bounding box patterns
        # Pattern to match coordinates and text in various formats
        pattern = r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*["\']([^"\']*)["\']?\s*\]'
        matches = re.findall(pattern, predict_str)
        
        for match in matches:
            try:
                x1, y1, x2, y2, text = match
                coords = [float(x1), float(y1), float(x2), float(y2), text.strip()]
                coord_key = (coords[0], coords[1], coords[2], coords[3])
                if coord_key not in seen:
                    results.append(coords)
                    seen.add(coord_key)
            except (ValueError, IndexError):
                continue
    else:
        # Process parsed data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 5:
                    try:
                        coords = [float(item[0]), float(item[1]), float(item[2]), float(item[3]), str(item[4]).strip()]
                        coord_key = (coords[0], coords[1], coords[2], coords[3])
                        if coord_key not in seen:
                            results.append(coords)
                            seen.add(coord_key)
                    except (ValueError, IndexError):
                        continue

    return results if results else None


def spotting_evaluation(prediction_list, img_metas):
    """
    Simplified spotting evaluation function.
    Returns a basic score based on successful bounding box extraction.
    
    Note: This is a simplified version that doesn't require the complex
    spotting_eval dependencies that were removed.
    """
    total_predictions = len(prediction_list)
    successful_extractions = 0
    
    for prediction in prediction_list:
        if isinstance(prediction, dict) and 'predict' in prediction:
            predict_str = prediction['predict']
            extracted_boxes = extract_bounding_boxes_robust(predict_str)
            if extracted_boxes:
                successful_extractions += 1
    
    # Simple metric: percentage of successful extractions
    if total_predictions > 0:
        score = successful_extractions / total_predictions
    else:
        score = 0.0
    
    return {
        'spotting_score': score,
        'total_predictions': total_predictions,
        'successful_extractions': successful_extractions
    }