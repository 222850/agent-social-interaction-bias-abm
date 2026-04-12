from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_first_available(
    obj: Any,
    candidates: list[str],
    default: Any = None,
) -> Any:
    """
    Robustly extract a field from:
      - pandas DataFrame
      - pandas Series
      - dict
      - generic objects with attributes
    """
    if obj is None:
        return default

    if isinstance(obj, pd.DataFrame):
        for key in candidates:
            if key in obj.columns and len(obj[key]) > 0:
                return obj[key].iloc[0]
        return default

    if isinstance(obj, pd.Series):
        for key in candidates:
            if key in obj.index:
                return obj[key]
        return default

    if isinstance(obj, dict):
        for key in candidates:
            if key in obj:
                return obj[key]
        return default

    for key in candidates:
        if hasattr(obj, key):
            return getattr(obj, key)

    return default


def permutation_test(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    n_permutations: int = 5000,
    random_state: int = 42,
) -> dict:
    """
    Two-sided permutation test for difference in means.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return {
            "observed_diff": None,
            "p_value": None,
            "significant": False,
            "n_x": len(x),
            "n_y": len(y),
        }

    observed = float(np.mean(x) - np.mean(y))
    combined = np.concatenate([x, y])
    rng = np.random.default_rng(random_state)

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        x_perm = perm[: len(x)]
        y_perm = perm[len(x):]
        diff = np.mean(x_perm) - np.mean(y_perm)
        if abs(diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return {
        "observed_diff": observed,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "n_x": len(x),
        "n_y": len(y),
    }


def bayesian_ttest(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> dict:
    """
    Version-robust wrapper around a t-test style output.

    We return a stable schema regardless of backend quirks:
      - t_value
      - p_value
      - bf10
      - mean_x
      - mean_y

    This implementation does not depend on a fragile column name such as 'p-val'.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or len(y) < 2:
        return {
            "t_value": None,
            "p_value": None,
            "bf10": None,
            "mean_x": float(np.mean(x)) if len(x) else None,
            "mean_y": float(np.mean(y)) if len(y) else None,
            "error": "too few samples",
        }

    t_stat, p_value = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")

    # Lightweight approximate BF10 from t-statistic magnitude.
    # This is a pragmatic fallback for pipeline stability, not a publication-grade Bayes factor.
    if p_value is None or np.isnan(p_value):
        bf10 = None
    else:
        if p_value <= 1e-12:
            bf10 = 1e12
        else:
            bf10 = max((1.0 / p_value) - 1.0, 0.0)

    return {
        "t_value": _safe_float(t_stat),
        "p_value": _safe_float(p_value),
        "bf10": _safe_float(bf10),
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
    }


def impact_ratio(
    selected_group_a: int,
    total_group_a: int,
    selected_group_b: int,
    total_group_b: int,
) -> float:
    """
    4/5ths rule style impact ratio.
    """
    rate_a = selected_group_a / total_group_a if total_group_a > 0 else 0.0
    rate_b = selected_group_b / total_group_b if total_group_b > 0 else 0.0

    max_rate = max(rate_a, rate_b)
    min_rate = min(rate_a, rate_b)

    if max_rate == 0:
        return 1.0
    return float(min_rate / max_rate)


def rank_after_scoring(scores: list[float]) -> list[float]:
    """
    Average ranks for descending scores.
    Highest score gets rank 1.
    Ties get average rank.
    """
    s = pd.Series(scores, dtype=float)
    return s.rank(method="average", ascending=False).tolist()


def clt_confidence_interval(
    values: np.ndarray | list[float],
    confidence: float = 0.95,
) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": None, "lower": None, "upper": None, "n": 0}

    mean = float(np.mean(values))
    if len(values) == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "n": 1}

    se = stats.sem(values, nan_policy="omit")
    z = stats.norm.ppf((1 + confidence) / 2)
    lower = mean - z * se
    upper = mean + z * se

    return {
        "mean": mean,
        "lower": float(lower),
        "upper": float(upper),
        "n": int(len(values)),
    }


def vacuous_neutrality_check(
    bias_score: float | None,
    utility_score: float | None,
    bias_threshold: float = 0.10,
    utility_threshold: float = 0.40,
) -> bool:
    """
    Low observed bias + low utility = possible vacuous neutrality.
    """
    if bias_score is None or utility_score is None:
        return False

    return abs(float(bias_score)) < bias_threshold and float(utility_score) < utility_threshold