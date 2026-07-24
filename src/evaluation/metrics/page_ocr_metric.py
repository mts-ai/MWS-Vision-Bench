# Page OCR evaluation metrics adapted from OCRBench_v2
# Original source: https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2/eval_scripts

import hashlib
import os
import re
from pathlib import Path

import jieba
import nltk
import requests
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score

_WORDNET_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/"
    "gh-pages/packages/corpora/wordnet.zip"
)
_WORDNET_SHA256 = "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"


def _wordnet_available() -> bool:
    for resource in ("corpora/wordnet", "corpora/wordnet.zip"):
        try:
            nltk.data.find(resource)
            return True
        except LookupError:
            continue
    return False


def _wordnet_data_root() -> Path:
    configured = os.environ.get("NLTK_DATA")
    if configured:
        return Path(configured.split(os.pathsep)[0]).expanduser()
    return Path.home() / "nltk_data"


def ensure_wordnet() -> None:
    """Install the pinned WordNet corpus without changing metric semantics."""
    if _wordnet_available():
        return

    data_root = _wordnet_data_root()
    corpora_dir = data_root / "corpora"
    corpora_dir.mkdir(parents=True, exist_ok=True)
    destination = corpora_dir / "wordnet.zip"
    temporary = corpora_dir / f".wordnet-{os.getpid()}.part"

    digest = hashlib.sha256()
    try:
        with requests.get(
            _WORDNET_URL,
            stream=True,
            timeout=(10, 120),
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    digest.update(chunk)

        if digest.hexdigest() != _WORDNET_SHA256:
            raise RuntimeError(
                "Downloaded NLTK WordNet corpus failed its SHA-256 check"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    if str(data_root) not in nltk.data.path:
        nltk.data.path.insert(0, str(data_root))
    if not _wordnet_available():
        raise RuntimeError(
            "The OCR METEOR metric requires the NLTK WordNet corpus, "
            "but the verified download could not be loaded"
        )


def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))

def cal_per_metrics(pred, gt):
    ensure_wordnet()
    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return metrics


if __name__ == "__main__":

    # Examples for region text recognition and read all text tasks
    predict_text = "metrics['edit_dist'] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"
    true_text = "metrics = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"

    scores = cal_per_metrics(predict_text, true_text)

    predict_text = "metrics['edit_dist'] len(gt))"
    true_text = "metrics = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"

    scores = cal_per_metrics(predict_text, true_text)
    # Metrics calculation completed
