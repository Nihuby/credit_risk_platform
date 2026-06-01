"""
src/talk_to_data/query_runner.py
---------------------------------
Execute validated SQL against DuckDB and optionally summarise results via LLM.
"""

from __future__ import annotations

import structlog
import duckdb
import pandas as pd

from src.utils.config import DUCKDB_PATH, settings

logger = structlog.get_logger(__name__)

_MAX_RESULT_ROWS = 500
_SUMMARY_MAX_ROWS = 10  # rows sent to LLM for summarisation


def run_query(sql: str) -> tuple[pd.DataFrame, str]:
    """Execute a SQL query against the DuckDB database.

    Parameters
    ----------
    sql : str   A validated SELECT statement.

    Returns
    -------
    (DataFrame, error_message)   error_message is "" on success.
    """
    if not DUCKDB_PATH.exists():
        return pd.DataFrame(), (
            f"Database not found at {DUCKDB_PATH}. "
            "Run: from src.talk_to_data.db_builder import build_db; build_db()"
        )
    try:
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        df = con.execute(sql).df()
        con.close()
        if len(df) > _MAX_RESULT_ROWS:
            df = df.head(_MAX_RESULT_ROWS)
            logger.info("result_truncated", max_rows=_MAX_RESULT_ROWS)
        logger.info("query_success", rows=len(df), cols=len(df.columns))
        return df, ""
    except Exception as exc:
        logger.error("query_failed", sql=sql[:120], error=str(exc))
        return pd.DataFrame(), str(exc)


def summarise_results(
    user_query: str,
    sql: str,
    df: pd.DataFrame,
) -> str:
    """Ask the LLM to summarise the query result in plain English.

    Falls back to a template if no API key is set.

    Returns
    -------
    str  Plain-English summary.
    """
    if df.empty:
        return "The query returned no results."

    if not settings.openai_api_key:
        # Template summary (no LLM)
        n_rows, n_cols = df.shape
        col_list = ", ".join(df.columns[:5])
        sample = df.head(3).to_string(index=False)
        return (
            f"Query returned **{n_rows} row(s)** with {n_cols} column(s) "
            f"({col_list}{'...' if n_cols > 5 else ''}).\n\n"
            f"Sample results:\n```\n{sample}\n```\n\n"
            "_Set OPENAI_API_KEY for an AI-generated summary._"
        )

    try:
        from openai import OpenAI
        client_kwargs: dict = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        client = OpenAI(**client_kwargs)

        sample_str = df.head(_SUMMARY_MAX_ROWS).to_markdown(index=False)
        prompt = (
            f"A user asked: \"{user_query}\"\n\n"
            f"SQL executed:\n```sql\n{sql}\n```\n\n"
            f"Result ({len(df)} rows, showing up to {_SUMMARY_MAX_ROWS}):\n{sample_str}\n\n"
            "Provide a concise, business-friendly 2-4 sentence summary of the key findings. "
            "Mention specific numbers. Use plain language, no technical jargon."
        )
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a credit risk data analyst presenting findings to business stakeholders."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=250,
        )
        return response.choices[0].message.content or "Summary unavailable."
    except Exception as exc:
        logger.error("summarise_error", error=str(exc))
        return f"_(LLM summarisation failed: {exc})_"
