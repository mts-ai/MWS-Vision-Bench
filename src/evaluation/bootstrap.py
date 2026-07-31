"""Deterministic bootstrap utilities for benchmark uncertainty estimates."""

import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Sequence


Interval = Dict[str, float]


def validate_bootstrap_options(
    samples: int,
    confidence_level: float,
) -> None:
    """Validate shared bootstrap options."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be greater than zero")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile from sorted values."""
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _confidence_interval(
    values: Sequence[float],
    confidence_level: float,
) -> Interval:
    ordered = sorted(float(value) for value in values)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "low": _percentile(ordered, tail),
        "high": _percentile(ordered, 1.0 - tail),
    }


def bootstrap_category_means(
    category_scores: Mapping[str, Sequence[float]],
    *,
    samples: int,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap category means and their macro-average.

    Every category is resampled independently at its original size. The
    overall distribution is the macro-average of the category means in each
    bootstrap replicate, matching the benchmark's Overall definition.
    """
    validate_bootstrap_options(samples, confidence_level)
    non_empty_scores = {
        category: [float(score) for score in scores]
        for category, scores in category_scores.items()
        if scores
    }
    if not non_empty_scores:
        raise ValueError("at least one non-empty category is required")

    rng = random.Random(seed)
    category_distributions: Dict[str, List[float]] = {
        category: [] for category in non_empty_scores
    }
    overall_distribution: List[float] = []

    for _ in range(samples):
        replicate_means = []
        for category, scores in non_empty_scores.items():
            replicate = rng.choices(scores, k=len(scores))
            replicate_mean = sum(replicate) / len(replicate)
            category_distributions[category].append(replicate_mean)
            replicate_means.append(replicate_mean)
        overall_distribution.append(
            sum(replicate_means) / len(replicate_means)
        )

    return {
        "categories": {
            category: _confidence_interval(values, confidence_level)
            for category, values in category_distributions.items()
        },
        "overall": _confidence_interval(
            overall_distribution,
            confidence_level,
        ),
    }


def bootstrap_stratified_score(
    items: Sequence[Dict[str, Any]],
    *,
    strata_key: str,
    score_fn: Callable[[List[Dict[str, Any]]], float],
    samples: int,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Interval:
    """Bootstrap a non-decomposable score while preserving class counts."""
    validate_bootstrap_options(samples, confidence_level)
    strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        strata[str(item.get(strata_key, ""))].append(item)
    if not strata:
        raise ValueError("at least one item is required")

    rng = random.Random(seed)
    distribution = []
    for _ in range(samples):
        replicate = []
        for stratum_items in strata.values():
            replicate.extend(
                rng.choices(stratum_items, k=len(stratum_items))
            )
        distribution.append(float(score_fn(replicate)))

    return _confidence_interval(distribution, confidence_level)
