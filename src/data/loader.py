"""
src/data/loader.py
------------------
Data ingestion helpers for the AI-Powered Credit Risk Intelligence Platform.

Key entry points
----------------
download_dataset()          – pull all CSVs from Kaggle via kagglehub
load_application_data()     – application_train / application_test
load_bureau()               – bureau.csv
load_previous_application() – previous_application.csv
load_installments_payments()– installments_payments.csv
quick_sample()              – fast random sample for prototyping
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import structlog

from src.utils.config import DATA_DIR, settings

logger = structlog.get_logger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────
Split = Literal["train", "test"]

# ── CSV files to copy from the Kaggle download cache ─────────────────────────
_REQUIRED_FILES: list[str] = [
    settings.application_train_file,
    settings.application_test_file,
    settings.bureau_file,
    settings.bureau_balance_file,
    settings.previous_application_file,
    settings.installments_payments_file,
    settings.pos_cash_balance_file,
    settings.credit_card_balance_file,
]

# ── Memory-efficient dtype overrides ─────────────────────────────────────────
_APP_DTYPE_OVERRIDES: dict[str, str] = {
    "TARGET": "Int8",
    "SK_ID_CURR": "int32",
    "FLAG_OWN_CAR": "category",
    "FLAG_OWN_REALTY": "category",
    "NAME_CONTRACT_TYPE": "category",
    "NAME_INCOME_TYPE": "category",
    "NAME_EDUCATION_TYPE": "category",
    "NAME_FAMILY_STATUS": "category",
    "NAME_HOUSING_TYPE": "category",
    "OCCUPATION_TYPE": "category",
    "ORGANIZATION_TYPE": "category",
    "CODE_GENDER": "category",
    "WEEKDAY_APPR_PROCESS_START": "category",
    "NAME_TYPE_SUITE": "category",
}

_BUREAU_DTYPE_OVERRIDES: dict[str, str] = {
    "SK_ID_CURR": "int32",
    "SK_ID_BUREAU": "int32",
    "CREDIT_ACTIVE": "category",
    "CREDIT_CURRENCY": "category",
    "CREDIT_TYPE": "category",
}

_PREV_APP_DTYPE_OVERRIDES: dict[str, str] = {
    "SK_ID_CURR": "int32",
    "SK_ID_PREV": "int32",
    "NAME_CONTRACT_TYPE": "category",
    "NAME_CONTRACT_STATUS": "category",
    "NAME_PAYMENT_TYPE": "category",
    "NAME_TYPE_SUITE": "category",
    "NAME_CLIENT_TYPE": "category",
    "NAME_GOODS_CATEGORY": "category",
    "NAME_PORTFOLIO": "category",
    "NAME_PRODUCT_TYPE": "category",
    "CHANNEL_TYPE": "category",
    "NAME_SELLER_INDUSTRY": "category",
    "NAME_YIELD_GROUP": "category",
    "PRODUCT_COMBINATION": "category",
    "WEEKDAY_APPR_PROCESS_START": "category",
}

_INSTALL_DTYPE_OVERRIDES: dict[str, str] = {
    "SK_ID_CURR": "int32",
    "SK_ID_PREV": "int32",
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Dataset Download
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(force: bool = False) -> dict[str, Path]:
    """Download the Home Credit Default Risk dataset via kagglehub.

    Uses the authenticated Kaggle API to pull all competition files into the
    kagglehub local cache, then copies the required CSVs into ``data/``.

    Parameters
    ----------
    force : bool, default False
        If True, re-download even if the files already exist locally.

    Returns
    -------
    dict[str, Path]
        Mapping of filename → absolute path for each copied CSV.

    Raises
    ------
    ImportError
        If ``kagglehub`` is not installed.
    RuntimeError
        If the download or file copy fails.
    """
    try:
        import kagglehub  # local import — optional dependency
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for automated download. "
            "Install it with: pip install kagglehub"
        ) from exc

    # ── Check which files already exist ──────────────────────────────────────
    existing = {f: (DATA_DIR / f) for f in _REQUIRED_FILES if (DATA_DIR / f).exists()}
    if existing and not force:
        logger.info(
            "dataset_already_present",
            count=len(existing),
            files=list(existing.keys()),
        )
        # Still attempt to fill in any missing ones
        missing = [f for f in _REQUIRED_FILES if f not in existing]
        if not missing:
            return existing

    logger.info(
        "dataset_download_start",
        competition=settings.kaggle_competition,
    )
    t0 = time.perf_counter()

    try:
        cache_path: str = kagglehub.competition_download(settings.kaggle_competition)
        cache_dir = Path(cache_path)
        logger.info("download_complete", cache_dir=str(cache_dir), elapsed_s=round(time.perf_counter() - t0, 1))
    except Exception as exc:
        zip_path = DATA_DIR / f"{settings.kaggle_competition}.zip"
        if zip_path.exists():
            logger.info("cli_zip_fallback", zip=str(zip_path))
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATA_DIR)
            cache_dir = DATA_DIR
        else:
            raise RuntimeError(
            f"kagglehub download failed: {exc}\n"
            "Make sure your Kaggle credentials are in place:\n"
            "  • ~/.kaggle/kaggle.json, OR\n"
            "  • KAGGLE_USERNAME / KAGGLE_KEY environment variables"
        ) from exc

    # ── Copy CSVs into data/ ──────────────────────────────────────────────────
    copied: dict[str, Path] = {}
    for filename in _REQUIRED_FILES:
        # Search recursively — kagglehub may unzip into subdirectories
        candidates = list(cache_dir.rglob(filename))
        if not candidates:
            logger.warning("file_not_found_in_cache", filename=filename)
            continue

        src = candidates[0]  # take the first match
        dst = DATA_DIR / filename

        if dst.exists() and not force:
            logger.debug("file_already_exists_skipping", filename=filename)
            copied[filename] = dst
            continue

        shutil.copy2(src, dst)
        copied[filename] = dst
        size_mb = dst.stat().st_size / 1024**2
        logger.info("file_copied", filename=filename, size_mb=round(size_mb, 1))

    total_elapsed = round(time.perf_counter() - t0, 1)
    logger.info(
        "dataset_ready",
        files_copied=len(copied),
        elapsed_s=total_elapsed,
        data_dir=str(DATA_DIR),
    )
    return copied


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Generic CSV loader helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(
    filename: str,
    dtype_overrides: dict[str, str],
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    verbose: bool = True,
    label: str | None = None,
) -> pd.DataFrame:
    """Generic CSV reader with dtype casting, logging, and validation."""
    resolved_dir = data_dir if data_dir is not None else DATA_DIR
    csv_path = resolved_dir / filename
    tag = label or filename

    if not csv_path.exists():
        raise FileNotFoundError(
            f"[{tag}] File not found: {csv_path}\n"
            "Run `from src.data.loader import download_dataset; download_dataset()` first."
        )

    log = logger.bind(file=tag, path=str(csv_path), nrows=nrows)
    if verbose:
        log.info("load_start")

    t0 = time.perf_counter()
    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)

    for col, dtype in dtype_overrides.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as exc:
                logger.warning("dtype_cast_failed", column=col, dtype=dtype, error=str(exc))

    elapsed = round(time.perf_counter() - t0, 3)
    if verbose:
        log.info(
            "load_done",
            rows=len(df),
            cols=len(df.columns),
            memory_mb=round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            elapsed_s=elapsed,
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Public loader functions
# ─────────────────────────────────────────────────────────────────────────────

def load_application_data(
    split: Split = "train",
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    drop_duplicates: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load and validate application_train.csv or application_test.csv.

    Parameters
    ----------
    split : {"train", "test"}, default "train"
    data_dir : Path | None
        Override default data directory (useful for testing).
    nrows : int | None
        Read only first N rows for quick sampling.
    drop_duplicates : bool
        Drop rows with duplicate SK_ID_CURR.
    verbose : bool
        Emit structured log events.

    Returns
    -------
    pd.DataFrame
    """
    filename = (
        settings.application_train_file if split == "train"
        else settings.application_test_file
    )
    df = _load_csv(
        filename,
        _APP_DTYPE_OVERRIDES,
        data_dir=data_dir,
        nrows=nrows,
        verbose=verbose,
        label=f"application_{split}",
    )

    if drop_duplicates and settings.id_column in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=[settings.id_column], keep="first")
        n_dupes = n_before - len(df)
        if n_dupes and verbose:
            logger.warning("duplicate_ids_dropped", count=n_dupes, file=filename)

    # Integrity checks
    required = {settings.id_column}
    if split == "train":
        required.add(settings.target_column)
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if split == "train" and verbose:
        default_rate = df[settings.target_column].mean()
        logger.info(
            "class_distribution",
            default_rate_pct=round(float(default_rate) * 100, 2),
            n_defaulted=int(df[settings.target_column].sum()),
            n_repaid=int((df[settings.target_column] == 0).sum()),
        )

    return df


def load_bureau(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load bureau.csv — credit history from the Credit Bureau.

    Each row is one previous credit per applicant. One applicant may have
    multiple rows (one per credit line).

    Returns
    -------
    pd.DataFrame  with columns including SK_ID_CURR, SK_ID_BUREAU,
                  CREDIT_ACTIVE, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT,
                  CREDIT_DAY_OVERDUE, etc.
    """
    return _load_csv(
        settings.bureau_file,
        _BUREAU_DTYPE_OVERRIDES,
        data_dir=data_dir,
        nrows=nrows,
        verbose=verbose,
        label="bureau",
    )


def load_previous_application(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load previous_application.csv — prior loan applications at Home Credit.

    Each row is one previous loan application. Applicants may have multiple
    prior applications.

    Returns
    -------
    pd.DataFrame  with columns including SK_ID_CURR, SK_ID_PREV,
                  NAME_CONTRACT_STATUS, AMT_APPLICATION, AMT_CREDIT, etc.
    """
    return _load_csv(
        settings.previous_application_file,
        _PREV_APP_DTYPE_OVERRIDES,
        data_dir=data_dir,
        nrows=nrows,
        verbose=verbose,
        label="previous_application",
    )


def load_installments_payments(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load installments_payments.csv — historical repayment behaviour.

    Each row is one scheduled installment payment for a previous loan.
    Columns include DAYS_INSTALMENT (scheduled day), DAYS_ENTRY_PAYMENT
    (actual payment day), AMT_INSTALMENT, AMT_PAYMENT.

    Returns
    -------
    pd.DataFrame
    """
    return _load_csv(
        settings.installments_payments_file,
        _INSTALL_DTYPE_OVERRIDES,
        data_dir=data_dir,
        nrows=nrows,
        verbose=verbose,
        label="installments_payments",
    )


def load_pos_cash_balance(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load POS_CASH_balance.csv — monthly POS/cash loan snapshots."""
    return _load_csv(
        settings.pos_cash_balance_file,
        {"SK_ID_CURR": "int32", "SK_ID_PREV": "int32", "NAME_CONTRACT_STATUS": "category"},
        data_dir=data_dir,
        nrows=nrows,
        verbose=verbose,
        label="pos_cash_balance",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

def quick_sample(n: int = 5_000, random_state: int | None = None) -> pd.DataFrame:
    """Return a random sample of the training set for rapid prototyping.

    Parameters
    ----------
    n : int, default 5_000
    random_state : int | None
    """
    df = load_application_data(split="train", verbose=False)
    return df.sample(
        n=min(n, len(df)),
        random_state=random_state or settings.random_state,
    )


def data_files_present() -> dict[str, bool]:
    """Check which dataset CSVs are already in data/.

    Returns
    -------
    dict  filename → bool (True if file exists and is non-empty)
    """
    return {
        f: (DATA_DIR / f).exists() and (DATA_DIR / f).stat().st_size > 0
        for f in _REQUIRED_FILES
    }
