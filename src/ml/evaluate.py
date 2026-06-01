"""
src/ml/evaluate.py
------------------
Compute and persist evaluation metrics for the trained model.

Run:
    python -m src.ml.evaluate
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve,
)
from sklearn.model_selection import train_test_split

from src.utils.config import (
    EVALUATION_METRICS_PATH, FEATURE_COLUMNS_PATH, MODEL_PATH, settings,
)
from src.data.preprocessor import load_features

logger = structlog.get_logger(__name__)


def _gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(verbose: bool = True) -> dict:
    """Evaluate the saved model on the held-out validation split.

    Returns
    -------
    dict  with ROC-AUC, PR-AUC, Brier, Gini, KS-statistic.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_cols: list[str] = json.load(f)

    df = load_features()
    target = settings.target_column
    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(int)

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=settings.test_size,
        random_state=settings.random_state, stratify=y,
    )

    probs = model.predict_proba(X_val)[:, 1]

    metrics = {
        "roc_auc":       round(float(roc_auc_score(y_val, probs)), 4),
        "pr_auc":        round(float(average_precision_score(y_val, probs)), 4),
        "brier_score":   round(float(brier_score_loss(y_val, probs)), 4),
        "gini":          round(float(_gini(y_val, probs)), 4),
        "ks_statistic":  round(float(_ks_statistic(y_val, probs)), 4),
        "n_val_samples": int(len(y_val)),
        "n_val_pos":     int(y_val.sum()),
        "val_default_rate_pct": round(float(y_val.mean()) * 100, 2),
    }

    EVALUATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        logger.info("evaluation_complete", **metrics)

    return metrics


def load_metrics() -> dict:
    """Load persisted evaluation metrics."""
    if not EVALUATION_METRICS_PATH.exists():
        return {}
    with open(EVALUATION_METRICS_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    m = evaluate()
    print("\n=== Evaluation Metrics ===")
    for k, v in m.items():
        print(f"  {k}: {v}")
