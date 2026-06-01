"""
src/talk_to_data/nl_to_sql.py
------------------------------
Convert natural-language questions to validated SQL via OpenAI.
Falls back to demo mode (keyword matching) if no API key is set.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import structlog

from src.utils.config import settings
from src.talk_to_data.prompt_templates import build_messages

logger = structlog.get_logger(__name__)

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)


@dataclass
class NLtoSQLResult:
    sql: str
    explanation: str
    source: str   # "openai" | "demo"
    raw_response: str = ""


# ── Demo mode fallbacks ───────────────────────────────────────────────────────
_DEMO_QUERIES: list[tuple[list[str], str, str]] = [
    (
        ["default rate", "overall", "how many"],
        "SELECT round(avg(TARGET)*100, 2) AS default_rate_pct, count(*) AS total FROM applications",
        "Overall default rate across all applicants.",
    ),
    (
        ["income type", "income"],
        "SELECT NAME_INCOME_TYPE, round(avg(TARGET)*100,2) AS default_rate_pct, count(*) AS n FROM applications GROUP BY 1 ORDER BY 2 DESC",
        "Default rate broken down by income type.",
    ),
    (
        ["education", "degree"],
        "SELECT NAME_EDUCATION_TYPE, round(avg(TARGET)*100,2) AS default_rate_pct, count(*) AS n FROM applications GROUP BY 1 ORDER BY 2 DESC",
        "Default rate by education level.",
    ),
    (
        ["age", "young", "old"],
        "SELECT round(-DAYS_BIRTH/365.0) AS age, round(avg(TARGET)*100,2) AS default_rate_pct FROM applications GROUP BY 1 ORDER BY 1 LIMIT 60",
        "Default rate by applicant age.",
    ),
    (
        ["overdue", "bureau", "delinquent"],
        "SELECT b.max_overdue_days, round(avg(a.TARGET)*100,2) AS default_rate_pct, count(*) AS n FROM applications a JOIN bureau_summary b ON a.SK_ID_CURR=b.SK_ID_CURR GROUP BY 1 ORDER BY 1 LIMIT 30",
        "Default rate grouped by max bureau overdue days.",
    ),
    (
        ["credit", "amount", "loan size"],
        "SELECT round(AMT_CREDIT/50000)*50000 AS credit_bucket, round(avg(TARGET)*100,2) AS default_rate_pct, count(*) AS n FROM applications GROUP BY 1 ORDER BY 1 LIMIT 30",
        "Default rate by credit amount bucket.",
    ),
    (
        ["gender", "male", "female"],
        "SELECT CODE_GENDER, round(avg(TARGET)*100,2) AS default_rate_pct, count(*) AS n FROM applications GROUP BY 1 ORDER BY 2 DESC",
        "Default rate by gender.",
    ),
]


def _demo_query(user_query: str) -> NLtoSQLResult:
    q = user_query.lower()
    for keywords, sql, explanation in _DEMO_QUERIES:
        if any(kw in q for kw in keywords):
            return NLtoSQLResult(sql=sql, explanation=explanation, source="demo")
    return NLtoSQLResult(
        sql="SELECT round(avg(TARGET)*100,2) AS default_rate_pct, count(*) AS total FROM applications",
        explanation="Demo fallback: overall default rate (no API key set).",
        source="demo",
    )


def _validate_sql(sql: str) -> str:
    """Ensure the SQL is a safe SELECT statement."""
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError(f"Only SELECT statements are allowed. Got: {stripped[:60]}")
    if _FORBIDDEN.search(stripped):
        raise ValueError(f"Potentially destructive SQL keyword detected: {stripped[:80]}")
    return stripped


def nl_to_sql(user_query: str, schema: str | None = None) -> NLtoSQLResult:
    """Convert a natural-language question to a validated SQL query.

    Parameters
    ----------
    user_query : str
    schema : str | None   Live DuckDB schema string for context injection.

    Returns
    -------
    NLtoSQLResult
    """
    if not settings.openai_api_key:
        logger.info("nl_to_sql_demo_mode", reason="no_api_key")
        return _demo_query(user_query)

    try:
        from openai import OpenAI
        client_kwargs: dict = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        client = OpenAI(**client_kwargs)

        messages = build_messages(user_query, schema_override=schema)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.0,
            max_tokens=600,
        )
        raw = response.choices[0].message.content or ""
        logger.debug("llm_raw_response", content=raw[:200])

        parsed = json.loads(raw)
        sql = _validate_sql(parsed["sql"])
        explanation = parsed.get("explanation", "")
        return NLtoSQLResult(sql=sql, explanation=explanation, source="openai", raw_response=raw)

    except json.JSONDecodeError as exc:
        logger.error("llm_json_parse_error", error=str(exc))
        return NLtoSQLResult(
            sql="SELECT 'LLM returned invalid JSON — please rephrase' AS error",
            explanation="Parse error. Try rephrasing your question.",
            source="error",
        )
    except Exception as exc:
        logger.error("nl_to_sql_error", error=str(exc))
        return NLtoSQLResult(
            sql="SELECT 'API error — see logs' AS error",
            explanation=f"API error: {exc}",
            source="error",
        )
