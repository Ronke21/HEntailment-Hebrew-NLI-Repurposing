#!/usr/bin/env python3
"""
metrics_utils.py — macro-F1 / accuracy / per-label breakdown for NLI evaluation.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import accuracy_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def macro_f1(y_true: list, y_pred: list, labels: list[str] | None = None) -> float:
    if not y_true:
        return 0.0
    if HAS_SKLEARN:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0,
                               labels=labels))
    classes = sorted(set(y_true)) if labels is None else list(labels)
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    if HAS_SKLEARN:
        return float(accuracy_score(y_true, y_pred))
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def per_label_f1(y_true: list, y_pred: list, labels: list[str]) -> dict[str, float]:
    out = {}
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out[c] = round(2 * pr * rc / (pr + rc), 4) if (pr + rc) > 0 else 0.0
    return out


def full_report(y_true: list, y_pred: list, labels: list[str]) -> dict:
    """Rich metrics dict for one test set."""
    return {
        "n"           : len(y_true),
        "macro_f1"    : round(macro_f1(y_true, y_pred, labels), 4),
        "accuracy"    : round(accuracy(y_true, y_pred), 4),
        "per_label_f1": per_label_f1(y_true, y_pred, labels),
    }


def average_reports(reports: dict[str, dict]) -> dict:
    """Mean of macro_f1/accuracy/per_label_f1 across multiple per-test-set
    full_report() outputs — a statistical average of already-computed scores.
    Distinct from a POOLED score (call full_report on the concatenated
    y_true/y_pred across all test sets for that instead) — mean-of-scores
    and score-over-the-union are different statistics, especially for
    macro-F1, which is sensitive to the combined label distribution."""
    macro_f1s = [r["macro_f1"] for r in reports.values()]
    accs = [r["accuracy"] for r in reports.values()]
    labels = list(next(iter(reports.values()))["per_label_f1"].keys())
    avg_per_label = {l: round(float(np.mean([r["per_label_f1"][l] for r in reports.values()])), 4)
                      for l in labels}
    return {
        "n_test_sets" : len(reports),
        "macro_f1"    : round(float(np.mean(macro_f1s)), 4),
        "accuracy"    : round(float(np.mean(accs)), 4),
        "per_label_f1": avg_per_label,
    }
