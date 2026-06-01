"""
src/ml/explain.py
-----------------
SHAP-based explainability for individual and batch predictions.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap
import structlog

from src.utils.config import FEATURE_COLUMNS_PATH, MODEL_PATH

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _load_explainer():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_cols: list[str] = json.load(f)
    explainer = shap.TreeExplainer(model)
    return explainer, feature_cols


def explain_single(row: dict) -> pd.DataFrame:
    """Return a sorted DataFrame of SHAP feature contributions for one applicant.

    Parameters
    ----------
    row : dict   Feature dictionary keyed by column name.

    Returns
    -------
    pd.DataFrame  columns: feature, value, shap_value, abs_shap
                  Sorted by abs_shap descending (most impactful first).
    """
    explainer, feature_cols = _load_explainer()
    x = np.array([[float(row.get(col, 0)) for col in feature_cols]], dtype=np.float32)
    shap_values = explainer.shap_values(x)

    # LightGBM binary: shap_values may be list[array] or single array
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # positive class
    else:
        sv = shap_values[0]

    result = pd.DataFrame({
        "feature": feature_cols,
        "value": x[0],
        "shap_value": sv,
        "abs_shap": np.abs(sv),
    }).sort_values("abs_shap", ascending=False).reset_index(drop=True)

    return result


def explain_batch(df: pd.DataFrame, max_rows: int = 500) -> tuple[np.ndarray, list[str]]:
    """Return SHAP values for a batch of applicants.

    Parameters
    ----------
    df : pd.DataFrame
    max_rows : int   Cap rows for performance.

    Returns
    -------
    (shap_values_array, feature_cols)
    """
    explainer, feature_cols = _load_explainer()
    sample = df[feature_cols].fillna(0).head(max_rows).values.astype(np.float32)
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    return shap_vals, feature_cols


def top_features_global(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Compute mean absolute SHAP values across the dataset.

    Returns
    -------
    pd.DataFrame  columns: feature, mean_abs_shap   (top_n rows)
    """
    sv, feature_cols = explain_batch(df)
    mean_abs = np.abs(sv).mean(axis=0)
    result = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    return result.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
