"""
src/ml/train.py
---------------
Train a LightGBM classifier on the merged feature matrix.

Run:
    python -m src.ml.train
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from src.utils.config import (
    FEATURE_COLUMNS_PATH, FEATURES_PARQUET_PATH,
    MODEL_PATH, MODELS_DIR, settings,
)
from src.data.preprocessor import load_features, get_feature_columns

logger = structlog.get_logger(__name__)


def train(verbose: bool = True) -> dict:
    """Train LightGBM with stratified CV + early stopping.

    Returns
    -------
    dict  with keys: model_path, feature_columns_path, cv_roc_auc, val_roc_auc
    """
    logger.info("train_start")
    t0 = time.perf_counter()

    df = load_features()
    target = settings.target_column
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(int)

    # Class imbalance weight
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    spw = float(n_neg / max(n_pos, 1))

    logger.info("dataset_loaded", n_samples=len(y), n_features=len(feature_cols),
                n_pos=int(n_pos), n_neg=int(n_neg), scale_pos_weight=round(spw, 2))

    # ── Stratified 80/20 split ────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=settings.test_size,
        random_state=settings.random_state, stratify=y,
    )

    # ── 5-Fold CV to estimate generalisation ─────────────────────────────────
    cv = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state)
    cv_scores: list[float] = []

    params = dict(
        objective="binary",
        metric="auc",
        n_estimators=settings.lgbm_n_estimators,
        learning_rate=settings.lgbm_learning_rate,
        scale_pos_weight=spw,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=settings.random_state,
        n_jobs=-1,
        verbose=-1,
    )

    logger.info("cross_validation_start", folds=settings.cv_folds)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), 1):
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]

        model_cv = lgb.LGBMClassifier(**params)
        model_cv.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[
                lgb.early_stopping(settings.lgbm_early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        preds = model_cv.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, preds)
        cv_scores.append(auc)
        if verbose:
            logger.info("fold_done", fold=fold, roc_auc=round(auc, 4))

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    logger.info("cv_complete", mean_auc=round(cv_mean, 4), std_auc=round(cv_std, 4))

    # ── Final model trained on full train split ───────────────────────────────
    logger.info("training_final_model")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(settings.lgbm_early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    val_preds = final_model.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_preds))
    logger.info("val_roc_auc", val_roc_auc=round(val_auc, 4))

    # ── Persist artifacts ─────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    elapsed = round(time.perf_counter() - t0, 1)
    result = {
        "model_path": str(MODEL_PATH),
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        "cv_roc_auc_mean": round(cv_mean, 4),
        "cv_roc_auc_std": round(cv_std, 4),
        "val_roc_auc": round(val_auc, 4),
        "n_features": len(feature_cols),
        "n_estimators_used": final_model.best_iteration_ or settings.lgbm_n_estimators,
        "scale_pos_weight": round(spw, 2),
        "elapsed_s": elapsed,
    }
    logger.info("training_complete", **result)
    return result


if __name__ == "__main__":
    result = train()
    print("\n=== Training Complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
