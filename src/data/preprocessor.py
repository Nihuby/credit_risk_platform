"""
src/data/preprocessor.py
------------------------
Full feature engineering pipeline for the Credit Risk Platform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from src.utils.config import FEATURES_PARQUET_PATH, settings
from src.data.loader import (
    load_application_data, load_bureau,
    load_installments_payments, load_previous_application,
)

logger = structlog.get_logger(__name__)


def _engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365_243, np.nan)
    df["DAYS_EMPLOYED_ANOM"] = df["DAYS_EMPLOYED"].isna().astype(int)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.0).round(1)
    df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.0).clip(lower=0)
    df["EMPLOYED_RATIO"] = (df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]).fillna(0)
    inc = df["AMT_INCOME_TOTAL"] + 1
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / inc
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / inc
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1)
    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext:
        df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext].min(axis=1)
        df["EXT_SOURCE_PROD"] = df[ext].fillna(0).prod(axis=1)
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCS_PROVIDED"] = df[doc_cols].sum(axis=1)
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        if col in df.columns:
            df[col] = df[col].map({"Y": 1, "N": 0})
    if "CODE_GENDER" in df.columns:
        df["IS_MALE"] = (df["CODE_GENDER"].astype(str) == "M").astype(int)
    return df


def _aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    agg = bureau.groupby("SK_ID_CURR").agg(
        bureau_loan_count=("SK_ID_BUREAU", "count"),
        bureau_active_count=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        bureau_total_credit=("AMT_CREDIT_SUM", "sum"),
        bureau_mean_credit=("AMT_CREDIT_SUM", "mean"),
        bureau_total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_mean_debt=("AMT_CREDIT_SUM_DEBT", "mean"),
        bureau_max_overdue_days=("CREDIT_DAY_OVERDUE", "max"),
        bureau_mean_overdue_days=("CREDIT_DAY_OVERDUE", "mean"),
        bureau_total_overdue_amt=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_prolong_count=("CNT_CREDIT_PROLONG", "sum"),
    ).reset_index()
    agg["bureau_debt_credit_ratio"] = agg["bureau_total_debt"] / (agg["bureau_total_credit"] + 1)
    return agg


def _aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    agg = prev.groupby("SK_ID_CURR").agg(
        prev_app_count=("SK_ID_PREV", "count"),
        prev_approved_count=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        prev_refused_count=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        prev_mean_application=("AMT_APPLICATION", "mean"),
        prev_mean_credit=("AMT_CREDIT", "mean"),
        prev_total_credit=("AMT_CREDIT", "sum"),
        prev_max_credit=("AMT_CREDIT", "max"),
        prev_mean_annuity=("AMT_ANNUITY", "mean"),
    ).reset_index()
    agg["prev_approval_rate"] = agg["prev_approved_count"] / (agg["prev_app_count"] + 1)
    agg["prev_refusal_rate"] = agg["prev_refused_count"] / (agg["prev_app_count"] + 1)
    return agg


def _aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    inst = inst.copy()
    inst["DPD"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["DPD_POS"] = inst["DPD"].clip(lower=0)
    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"] + 1)
    agg = inst.groupby("SK_ID_CURR").agg(
        inst_count=("SK_ID_PREV", "count"),
        inst_dpd_mean=("DPD", "mean"),
        inst_dpd_max=("DPD_POS", "max"),
        inst_dpd_sum=("DPD_POS", "sum"),
        inst_payment_ratio_mean=("PAYMENT_RATIO", "mean"),
        inst_payment_ratio_min=("PAYMENT_RATIO", "min"),
        inst_total_paid=("AMT_PAYMENT", "sum"),
        inst_total_due=("AMT_INSTALMENT", "sum"),
        inst_late_count=("DPD_POS", lambda x: (x > 0).sum()),
    ).reset_index()
    agg["inst_underpaid_ratio"] = 1 - (agg["inst_total_paid"] / (agg["inst_total_due"] + 1))
    agg["inst_late_rate"] = agg["inst_late_count"] / (agg["inst_count"] + 1)
    return agg


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    for col in cat_cols:
        if col == settings.target_column:
            continue
        if df[col].nunique(dropna=True) <= 20:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            df = df.drop(columns=[col])
    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


_DROP_COLS = ["SK_ID_CURR", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]


def build_features(save: bool = True, nrows: int | None = None, verbose: bool = True) -> tuple[pd.DataFrame, float]:
    """Build master feature matrix from all 4 source tables.

    Returns (DataFrame, scale_pos_weight) where scale_pos_weight = n_neg/n_pos.
    """
    logger.info("build_features_start")
    app = load_application_data("train", nrows=nrows, verbose=verbose)
    bureau = load_bureau(nrows=None, verbose=verbose)
    prev = load_previous_application(nrows=None, verbose=verbose)
    inst = load_installments_payments(nrows=None, verbose=verbose)

    app = _engineer_application_features(app)
    bureau_agg = _aggregate_bureau(bureau)
    prev_agg = _aggregate_previous_application(prev)
    inst_agg = _aggregate_installments(inst)

    df = (
        app
        .merge(bureau_agg, on="SK_ID_CURR", how="left")
        .merge(prev_agg, on="SK_ID_CURR", how="left")
        .merge(inst_agg, on="SK_ID_CURR", how="left")
    )
    logger.info("merge_done", shape=df.shape)

    df = _encode_categoricals(df)
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns], errors="ignore")
    non_num = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != settings.target_column]
    df = df.drop(columns=non_num, errors="ignore")
    df = _impute_missing(df)

    target = settings.target_column
    n_neg = int((df[target] == 0).sum())
    n_pos = int((df[target] == 1).sum())
    spw = float(n_neg / max(n_pos, 1))

    logger.info("class_balance", n_neg=n_neg, n_pos=n_pos, scale_pos_weight=round(spw, 2))
    logger.info("feature_matrix_ready", shape=df.shape)

    if save:
        FEATURES_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FEATURES_PARQUET_PATH, index=False)
        logger.info("features_saved", path=str(FEATURES_PARQUET_PATH))

    return df, spw


def load_features() -> pd.DataFrame:
    """Load the pre-built feature parquet."""
    if not FEATURES_PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {FEATURES_PARQUET_PATH}.\n"
            "Run: from src.data.preprocessor import build_features; build_features()"
        )
    return pd.read_parquet(FEATURES_PARQUET_PATH)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns excluding target."""
    return [c for c in df.columns if c != settings.target_column]
