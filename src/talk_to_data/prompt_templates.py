"""
src/talk_to_data/prompt_templates.py
--------------------------------------
System prompt and few-shot examples for the NL→SQL LLM.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an expert SQL analyst for a Credit Risk Intelligence Platform.
You have access to a DuckDB database with the following tables and views:

TABLE: applications
  Key columns: SK_ID_CURR (int), TARGET (0=repaid, 1=defaulted),
  AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
  NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS,
  CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, AGE (via -DAYS_BIRTH/365),
  EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3,
  OCCUPATION_TYPE, ORGANIZATION_TYPE, CNT_CHILDREN, CNT_FAM_MEMBERS

VIEW: bureau_summary (aggregated per SK_ID_CURR)
  Columns: SK_ID_CURR, bureau_loan_count, active_loans, total_credit,
  total_debt, max_overdue_days, avg_overdue_days, total_overdue_amount

VIEW: prev_app_summary (aggregated per SK_ID_CURR)
  Columns: SK_ID_CURR, total_applications, approved, refused,
  avg_application_amount, avg_credit_amount, max_credit_amount

VIEW: installments_summary (aggregated per SK_ID_CURR)
  Columns: SK_ID_CURR, total_installments, avg_days_late, max_days_late,
  avg_payment_ratio, total_paid

VIEW: features (engineered feature matrix, all numeric)

RULES you MUST follow:
1. Only write SELECT statements — never INSERT, UPDATE, DELETE, DROP, CREATE.
2. Always LIMIT results to 500 rows unless the user asks for aggregations.
3. Use DuckDB-compatible SQL syntax (e.g., epoch() not YEAR(), QUALIFY for window filters).
4. For default rate queries, use: avg(TARGET) * 100 AS default_rate_pct
5. When joining tables, always join on SK_ID_CURR.
6. Return ONLY valid JSON in this exact format — no markdown, no commentary:
   {"sql": "SELECT ...", "explanation": "Plain English description of what this query does."}
"""

FEW_SHOT_EXAMPLES = [
    {
        "user": "What is the overall default rate?",
        "assistant": '{"sql": "SELECT round(avg(TARGET) * 100, 2) AS default_rate_pct, count(*) AS total_applicants FROM applications", "explanation": "Calculates the overall loan default rate as a percentage and total applicant count."}'
    },
    {
        "user": "Show default rate by income type",
        "assistant": '{"sql": "SELECT NAME_INCOME_TYPE, round(avg(TARGET) * 100, 2) AS default_rate_pct, count(*) AS applicant_count FROM applications GROUP BY NAME_INCOME_TYPE ORDER BY default_rate_pct DESC LIMIT 20", "explanation": "Groups applicants by income type and calculates default rate for each group, sorted highest to lowest."}'
    },
    {
        "user": "Who are the top 10 applicants by credit amount?",
        "assistant": '{"sql": "SELECT SK_ID_CURR, AMT_CREDIT, AMT_INCOME_TOTAL, round(AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0), 2) AS credit_income_ratio, TARGET FROM applications ORDER BY AMT_CREDIT DESC LIMIT 10", "explanation": "Returns the 10 applicants with the highest credit amounts, along with their income and default status."}'
    },
    {
        "user": "How many applicants have bureau overdue days greater than 90?",
        "assistant": '{"sql": "SELECT count(*) AS high_risk_count, round(avg(a.TARGET) * 100, 2) AS default_rate_pct FROM applications a JOIN bureau_summary b ON a.SK_ID_CURR = b.SK_ID_CURR WHERE b.max_overdue_days > 90", "explanation": "Counts applicants with any bureau credit line overdue by more than 90 days and shows their default rate."}'
    },
    {
        "user": "Average credit amount by education level for defaulters vs repaid",
        "assistant": '{"sql": "SELECT NAME_EDUCATION_TYPE, TARGET, round(avg(AMT_CREDIT), 0) AS avg_credit, count(*) AS count FROM applications GROUP BY NAME_EDUCATION_TYPE, TARGET ORDER BY NAME_EDUCATION_TYPE, TARGET", "explanation": "Compares average credit amounts across education levels, split by default outcome."}'
    },
    {
        "user": "Show me applicants with late installment payments on average",
        "assistant": '{"sql": "SELECT a.SK_ID_CURR, a.TARGET, i.avg_days_late, i.max_days_late, i.avg_payment_ratio FROM applications a JOIN installments_summary i ON a.SK_ID_CURR = i.SK_ID_CURR WHERE i.avg_days_late > 5 ORDER BY i.avg_days_late DESC LIMIT 100", "explanation": "Finds applicants who pay their installments late on average (>5 days), showing their default status and payment metrics."}'
    },
]


def build_messages(user_query: str, schema_override: str | None = None) -> list[dict]:
    """Build the messages list for the OpenAI Chat API.

    Parameters
    ----------
    user_query : str
    schema_override : str | None   Inject a live schema if available.

    Returns
    -------
    list[dict]  OpenAI messages format.
    """
    system = SYSTEM_PROMPT
    if schema_override:
        system += f"\n\n## Live Schema\n{schema_override}"

    messages = [{"role": "system", "content": system}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": user_query})
    return messages
