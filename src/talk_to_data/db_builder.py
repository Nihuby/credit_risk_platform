"""
src/talk_to_data/db_builder.py
-------------------------------
Ingest CSV files and the feature parquet into a DuckDB database.

Creates views: applications, bureau_summary, prev_app_summary,
               installments_summary, features
"""

from __future__ import annotations

import structlog
import duckdb

from src.utils.config import DATA_DIR, DUCKDB_PATH, FEATURES_PARQUET_PATH, settings

logger = structlog.get_logger(__name__)

_VIEWS: dict[str, str] = {
    "applications": settings.application_train_file,
    "bureau":       settings.bureau_file,
    "prev_app":     settings.previous_application_file,
    "installments": settings.installments_payments_file,
}


def build_db(force: bool = False) -> str:
    """Load CSV files into DuckDB and create analytical views.

    Parameters
    ----------
    force : bool   Drop and re-create existing tables.

    Returns
    -------
    str   Path to the DuckDB file.
    """
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))

    # ── Raw CSV tables ────────────────────────────────────────────────────────
    for table_name, filename in _VIEWS.items():
        csv_path = DATA_DIR / filename
        if not csv_path.exists():
            logger.warning("csv_missing_skipping", table=table_name, file=str(csv_path))
            continue
        if force:
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
        existing = con.execute(
            f"SELECT count(*) FROM information_schema.tables WHERE table_name='{table_name}'"
        ).fetchone()[0]
        if existing and not force:
            logger.info("table_exists_skipping", table=table_name)
            continue
        logger.info("loading_table", table=table_name, csv=filename)
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{csv_path.as_posix()}', sample_size=-1)
        """)
        n = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        logger.info("table_loaded", table=table_name, rows=n)

    # ── Features parquet ──────────────────────────────────────────────────────
    if FEATURES_PARQUET_PATH.exists():
        con.execute(f"""
            CREATE OR REPLACE VIEW features AS
            SELECT * FROM read_parquet('{FEATURES_PARQUET_PATH.as_posix()}')
        """)
        logger.info("features_view_created")

    # ── Analytical views ──────────────────────────────────────────────────────
    _create_views(con)

    con.close()
    logger.info("duckdb_ready", path=str(DUCKDB_PATH))
    return str(DUCKDB_PATH)


def _create_views(con: duckdb.DuckDBPyConnection) -> None:
    """Create pre-aggregated analytical views for the chatbot."""

    # Check which raw tables exist
    existing_tables = {
        r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }

    if "bureau" in existing_tables:
        con.execute("""
            CREATE OR REPLACE VIEW bureau_summary AS
            SELECT
                SK_ID_CURR,
                count(*) AS bureau_loan_count,
                sum(CASE WHEN CREDIT_ACTIVE='Active' THEN 1 ELSE 0 END) AS active_loans,
                sum(AMT_CREDIT_SUM) AS total_credit,
                sum(AMT_CREDIT_SUM_DEBT) AS total_debt,
                max(CREDIT_DAY_OVERDUE) AS max_overdue_days,
                avg(CREDIT_DAY_OVERDUE) AS avg_overdue_days,
                sum(AMT_CREDIT_SUM_OVERDUE) AS total_overdue_amount
            FROM bureau
            GROUP BY SK_ID_CURR
        """)

    if "prev_app" in existing_tables:
        con.execute("""
            CREATE OR REPLACE VIEW prev_app_summary AS
            SELECT
                SK_ID_CURR,
                count(*) AS total_applications,
                sum(CASE WHEN NAME_CONTRACT_STATUS='Approved' THEN 1 ELSE 0 END) AS approved,
                sum(CASE WHEN NAME_CONTRACT_STATUS='Refused' THEN 1 ELSE 0 END) AS refused,
                avg(AMT_APPLICATION) AS avg_application_amount,
                avg(AMT_CREDIT) AS avg_credit_amount,
                max(AMT_CREDIT) AS max_credit_amount
            FROM prev_app
            GROUP BY SK_ID_CURR
        """)

    if "installments" in existing_tables:
        con.execute("""
            CREATE OR REPLACE VIEW installments_summary AS
            SELECT
                SK_ID_CURR,
                count(*) AS total_installments,
                avg(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS avg_days_late,
                max(GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)) AS max_days_late,
                avg(AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0)) AS avg_payment_ratio,
                sum(AMT_PAYMENT) AS total_paid
            FROM installments
            GROUP BY SK_ID_CURR
        """)


def get_schema() -> str:
    """Return a string description of all tables/views for LLM prompting."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    tables = con.execute(
        "SELECT table_name, table_type FROM information_schema.tables ORDER BY table_name"
    ).fetchall()

    lines = ["## DuckDB Schema\n"]
    for (tname, ttype) in tables:
        try:
            cols = con.execute(f"DESCRIBE {tname}").fetchall()
            lines.append(f"### {tname} ({ttype})")
            for col in cols:
                lines.append(f"  - {col[0]}: {col[1]}")
            lines.append("")
        except Exception:
            pass
    con.close()
    return "\n".join(lines)
