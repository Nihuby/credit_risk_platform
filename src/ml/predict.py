"""
src/ml/predict.py
-----------------
Predict risk probability, score (0-100), and band for new applicants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog

from src.utils.config import FEATURE_COLUMNS_PATH, MODEL_PATH, settings

logger = structlog.get_logger(__name__)


@dataclass
class PredictionResult:
    probability: float      # raw default probability [0, 1]
    risk_score: int         # mapped score [0, 100]
    risk_band: str          # "Low" | "Medium" | "High"
    risk_color: str         # hex color for UI
    input_features: dict[str, Any]


def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run src/ml/train.py first.")
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_cols: list[str] = json.load(f)
    return model, feature_cols


_model = None
_feature_cols: list[str] | None = None


def _get_model():
    global _model, _feature_cols
    if _model is None:
        _model, _feature_cols = _load_artifacts()
    return _model, _feature_cols


def _prob_to_score(prob: float) -> int:
    """Map default probability [0,1] → risk score [0,100]."""
    return int(round(prob * 100))


def _score_to_band(score: int) -> tuple[str, str]:
    """Return (band_label, hex_color) for a given risk score."""
    lo = settings.low_risk_threshold
    hi = settings.high_risk_threshold
    if score < lo:
        return "Low", "#00D4FF"
    elif score < hi:
        return "Medium", "#FFD700"
    else:
        return "High", "#FF6B6B"


def predict_single(row: dict[str, Any]) -> PredictionResult:
    """Generate a risk prediction for one applicant.

    Parameters
    ----------
    row : dict
        Feature dictionary keyed by column name. Missing keys are filled with 0.

    Returns
    -------
    PredictionResult
    """
    model, feature_cols = _get_model()

    # Build feature vector — fill unknowns with 0
    x = np.array([[float(row.get(col, 0)) for col in feature_cols]], dtype=np.float32)
    prob = float(model.predict_proba(x)[0, 1])
    score = _prob_to_score(prob)
    band, color = _score_to_band(score)

    return PredictionResult(
        probability=round(prob, 4),
        risk_score=score,
        risk_band=band,
        risk_color=color,
        input_features=row,
    )


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict risk scores for a DataFrame of applicants.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  with columns: probability, risk_score, risk_band added.
    """
    model, feature_cols = _get_model()
    missing = [c for c in feature_cols if c not in df.columns]
    for col in missing:
        df[col] = 0.0

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    probs = model.predict_proba(X)[:, 1]
    scores = (probs * 100).round().astype(int)

    out = df.copy()
    out["probability"] = probs.round(4)
    out["risk_score"] = scores
    out["risk_band"] = [_score_to_band(s)[0] for s in scores]
    return out
