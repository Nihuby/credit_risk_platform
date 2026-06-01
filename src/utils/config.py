"""
src/utils/config.py
-------------------
Centralised configuration for the AI-Powered Credit Risk Intelligence Platform.

All path constants are resolved relative to the project root so the code works
regardless of the working directory from which the application is launched.

Usage
-----
    from src.utils.config import DATA_DIR, MODELS_DIR, DUCKDB_PATH, settings
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import structlog

# ── Project root (two levels up from this file) ───────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

# ── Load .env (silently skip if not present) ──────────────────────────────────
load_dotenv(ROOT_DIR / ".env")

# ── Directory constants ───────────────────────────────────────────────────────
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
DOCUMENTS_DIR: Path = ROOT_DIR / "documents"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
SQL_DIR: Path = ROOT_DIR / "sql"

# ── Database paths ────────────────────────────────────────────────────────────
_duckdb_override: str | None = os.getenv("DUCKDB_PATH")
DUCKDB_PATH: Path = (
    Path(_duckdb_override) if _duckdb_override else DATA_DIR / "credit_risk.duckdb"
)
SQLITE_DB_PATH: Path = DATA_DIR / "credit_risk.db"

# ── Model artifact paths ──────────────────────────────────────────────────────
MODEL_PATH: Path = MODELS_DIR / "credit_risk_model.pkl"
FEATURE_COLUMNS_PATH: Path = MODELS_DIR / "feature_columns.json"
EVALUATION_METRICS_PATH: Path = MODELS_DIR / "evaluation_metrics.json"

# ── Feature store path ────────────────────────────────────────────────────────
FEATURES_PARQUET_PATH: Path = DATA_DIR / "features.parquet"

# ── Ensure critical directories exist ────────────────────────────────────────
for _dir in (DATA_DIR, MODELS_DIR, DOCUMENTS_DIR, NOTEBOOKS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ── Application settings (typed convenience wrapper) ─────────────────────────
class Settings:
    """Lightweight settings object populated from environment variables."""

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

    # Kaggle
    kaggle_competition: str = "home-credit-default-risk"
    kaggle_username: str = os.getenv("KAGGLE_USERNAME", "")
    kaggle_key: str = os.getenv("KAGGLE_KEY", "")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # ── Dataset filenames ─────────────────────────────────────────────────────
    application_train_file: str = "application_train.csv"
    application_test_file: str = "application_test.csv"
    bureau_file: str = "bureau.csv"
    bureau_balance_file: str = "bureau_balance.csv"
    previous_application_file: str = "previous_application.csv"
    installments_payments_file: str = "installments_payments.csv"
    pos_cash_balance_file: str = "POS_CASH_balance.csv"
    credit_card_balance_file: str = "credit_card_balance.csv"

    # ── ML defaults ───────────────────────────────────────────────────────────
    random_state: int = 42
    test_size: float = 0.20
    cv_folds: int = 5
    lgbm_n_estimators: int = 1000
    lgbm_early_stopping_rounds: int = 50
    lgbm_learning_rate: float = 0.05

    # ── Target / ID columns ───────────────────────────────────────────────────
    target_column: str = "TARGET"
    id_column: str = "SK_ID_CURR"

    # ── Risk score bands ──────────────────────────────────────────────────────
    low_risk_threshold: int = 35
    high_risk_threshold: int = 65


settings = Settings()

# ── Structured logger ─────────────────────────────────────────────────────────
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(__import__("logging"), settings.log_level, 20)
    ),
)
logger = structlog.get_logger(__name__)
logger.debug(
    "config_loaded",
    root=str(ROOT_DIR),
    data_dir=str(DATA_DIR),
    duckdb=str(DUCKDB_PATH),
    model=str(MODEL_PATH),
)
