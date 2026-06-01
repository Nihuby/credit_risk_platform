"""
src/ml/rules.py
---------------
Interpretable credit decision rules derived from domain knowledge
and top SHAP feature importance.

Decision framework:
  APPROVE  → risk_score < 35
  REVIEW   → 35 ≤ risk_score < 65
  DECLINE  → risk_score ≥ 65
             OR hard-stop rule triggered (see HARD_STOP_RULES below)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuleResult:
    decision: str                          # "APPROVE" | "REVIEW" | "DECLINE"
    decision_color: str                    # hex for UI badge
    risk_score: int
    risk_band: str
    triggered_rules: list[str] = field(default_factory=list)
    explanation: str = ""


# ── Hard-stop thresholds (domain knowledge + SHAP-derived) ────────────────────
_HARD_STOP_RULES: list[tuple[str, str, float, str]] = [
    # (feature_key, operator, threshold, human_readable_label)
    ("CREDIT_INCOME_RATIO",   ">",  0.60,  "Credit-to-income ratio exceeds 60% — extreme debt burden"),
    ("EXT_SOURCE_MEAN",       "<",  0.15,  "External bureau score mean < 0.15 — very poor credit history"),
    ("EXT_SOURCE_MIN",        "<",  0.05,  "Lowest external bureau score < 0.05 — critical credit flag"),
    ("bureau_max_overdue_days", ">", 180,  "Bureau overdue days > 180 — active delinquency"),
    ("bureau_debt_credit_ratio", ">", 0.90, "Bureau debt/credit ratio > 90% — near-maxed credit lines"),
    ("inst_dpd_mean",         ">",  30.0,  "Mean installment days past due > 30 — chronic late payment"),
    ("inst_late_rate",        ">",  0.50,  "Over 50% of installments were late — poor repayment discipline"),
    ("ANNUITY_INCOME_RATIO",  ">",  0.40,  "Annuity/income ratio > 40% — unaffordable repayment burden"),
    ("AGE_YEARS",             "<",  21.0,  "Applicant under 21 — insufficient credit history"),
    ("prev_refusal_rate",     ">",  0.80,  "Prior refusal rate > 80% — serial application rejections"),
]

_POSITIVE_SIGNALS: list[tuple[str, str, float, str]] = [
    ("EXT_SOURCE_MEAN",       ">",  0.60,  "Strong external credit bureau scores"),
    ("CREDIT_INCOME_RATIO",   "<",  0.30,  "Conservative debt-to-income ratio"),
    ("inst_late_rate",        "<",  0.05,  "Near-perfect installment repayment record"),
    ("prev_approval_rate",    ">",  0.70,  "High prior loan approval rate"),
    ("bureau_active_count",   "<",  3.0,   "Low number of active credit lines"),
]


def _eval_rule(val: float, op: str, threshold: float) -> bool:
    if op == ">":
        return val > threshold
    if op == "<":
        return val < threshold
    if op == ">=":
        return val >= threshold
    if op == "<=":
        return val <= threshold
    return False


def evaluate_rules(features: dict[str, Any], risk_score: int, risk_band: str) -> RuleResult:
    """Apply the credit policy rule engine to a feature vector.

    Parameters
    ----------
    features : dict   Feature dictionary from predict_single().input_features or similar.
    risk_score : int  0-100 risk score from predict.py.
    risk_band : str   "Low" | "Medium" | "High".

    Returns
    -------
    RuleResult
    """
    triggered: list[str] = []

    # ── Check hard-stop rules ─────────────────────────────────────────────────
    for feat_key, op, threshold, label in _HARD_STOP_RULES:
        val = features.get(feat_key)
        if val is not None and _eval_rule(float(val), op, threshold):
            triggered.append(f"🚫 HARD STOP: {label} (value={float(val):.2f}, threshold{op}{threshold})")

    # ── Determine decision ────────────────────────────────────────────────────
    if risk_score >= settings_thresholds["high"] or triggered:
        decision = "DECLINE"
        color = "#FF6B6B"
    elif risk_score >= settings_thresholds["medium"]:
        decision = "REVIEW"
        color = "#FFD700"
    else:
        decision = "APPROVE"
        color = "#00FF87"

    # ── Positive signals ──────────────────────────────────────────────────────
    positives: list[str] = []
    for feat_key, op, threshold, label in _POSITIVE_SIGNALS:
        val = features.get(feat_key)
        if val is not None and _eval_rule(float(val), op, threshold):
            positives.append(f"✅ {label}")

    # ── Build explanation ─────────────────────────────────────────────────────
    parts = [f"Risk Score: {risk_score}/100 → Band: {risk_band} → Decision: {decision}"]
    if triggered:
        parts.append("\nPolicy Violations:")
        parts.extend(f"  • {r}" for r in triggered)
    if positives:
        parts.append("\nPositive Signals:")
        parts.extend(f"  • {p}" for p in positives)
    if not triggered and not positives:
        parts.append("\nNo exceptional risk factors or positive signals detected.")

    return RuleResult(
        decision=decision,
        decision_color=color,
        risk_score=risk_score,
        risk_band=risk_band,
        triggered_rules=triggered,
        explanation="\n".join(parts),
    )


# Module-level threshold constants (avoids importing config to keep rules independent)
settings_thresholds = {"medium": 35, "high": 65}


def get_all_rules_display() -> list[dict]:
    """Return all rules as a list of dicts for the Streamlit UI."""
    rules = []
    for feat, op, thresh, label in _HARD_STOP_RULES:
        rules.append({
            "type": "Hard Stop",
            "feature": feat,
            "condition": f"{feat} {op} {thresh}",
            "description": label,
            "action": "DECLINE",
            "color": "#FF6B6B",
        })
    rules.append({"type": "Score Band", "condition": "risk_score < 35",
                  "description": "Low risk score — automatic approval", "action": "APPROVE", "color": "#00FF87"})
    rules.append({"type": "Score Band", "condition": "35 ≤ risk_score < 65",
                  "description": "Medium risk — manual underwriter review", "action": "REVIEW", "color": "#FFD700"})
    rules.append({"type": "Score Band", "condition": "risk_score ≥ 65",
                  "description": "High risk score — automatic decline", "action": "DECLINE", "color": "#FF6B6B"})
    return rules
