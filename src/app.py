"""
src/app.py
----------
AI-Powered Credit Risk Intelligence Platform — 4-tab Streamlit dashboard.

Run: streamlit run src/app.py
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%); }
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px; padding: 20px; margin: 6px 0;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: rgba(0,212,255,0.6); }
.risk-badge {
    display: inline-block; padding: 6px 18px; border-radius: 20px;
    font-weight: 700; font-size: 1.1em; letter-spacing: 1px;
}
.badge-low    { background: rgba(0,255,135,0.15); color: #00FF87; border: 1px solid #00FF87; }
.badge-medium { background: rgba(255,215,0,0.15);  color: #FFD700; border: 1px solid #FFD700; }
.badge-high   { background: rgba(255,107,107,0.15);color: #FF6B6B; border: 1px solid #FF6B6B; }
.chat-msg-user { background: rgba(0,212,255,0.1); border-left: 3px solid #00D4FF;
                 border-radius: 8px; padding: 10px 14px; margin: 6px 0; }
.chat-msg-bot  { background: rgba(199,125,255,0.1); border-left: 3px solid #C77DFF;
                 border-radius: 8px; padding: 10px 14px; margin: 6px 0; }
.rule-card { background: rgba(255,255,255,0.03); border-radius: 10px;
             padding: 12px 16px; margin: 5px 0; border-left: 4px solid; }
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers / caching ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_app_data():
    try:
        from src.data.loader import load_application_data
        return load_application_data(verbose=False)
    except FileNotFoundError:
        return None

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    try:
        import joblib
        from src.utils.config import MODEL_PATH, FEATURE_COLUMNS_PATH
        if not MODEL_PATH.exists():
            return None, None
        model = joblib.load(MODEL_PATH)
        with open(FEATURE_COLUMNS_PATH) as f:
            cols = json.load(f)
        return model, cols
    except Exception:
        return None, None

@st.cache_data(show_spinner=False)
def load_metrics():
    try:
        from src.ml.evaluate import load_metrics as _lm
        return _lm()
    except Exception:
        return {}

def _plotly_dark():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        xaxis=dict(gridcolor="#1e2535", linecolor="#333"),
        yaxis=dict(gridcolor="#1e2535", linecolor="#333"),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk\n**Intelligence Platform**")
    st.markdown("---")
    st.markdown("**Data Status**")
    app_df = load_app_data()
    if app_df is not None:
        st.success(f"✅ {len(app_df):,} applicants loaded")
    else:
        st.error("❌ Dataset not found\nPlace CSVs in `data/`")

    model, feat_cols = load_model_artifacts()
    if model is not None:
        st.success(f"✅ Model loaded ({len(feat_cols)} features)")
    else:
        st.warning("⚠️ Model not trained yet")

    from src.utils.config import DUCKDB_PATH
    if DUCKDB_PATH.exists():
        st.success("✅ DuckDB ready")
    else:
        st.warning("⚠️ DuckDB not built")

    st.markdown("---")
    st.caption("v1.0 | Home Credit Dataset")

TAB_EDA, TAB_UW, TAB_RULES, TAB_CHAT = st.tabs([
    "📊 EDA & Insights",
    "🔍 Underwriting & XAI",
    "📋 Credit Policy Rules",
    "💬 Talk-to-Data",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA & Insights
# ════════════════════════════════════════════════════════════════════════════
with TAB_EDA:
    st.title("📊 Executive EDA & Business Insights")

    if app_df is None:
        st.error("Dataset not found. Download it first — see the README.")
        st.stop()

    df = app_df.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365_243, np.nan)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(0)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    ext = [c for c in ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"] if c in df.columns]
    if ext:
        df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)

    # KPI row
    default_rate = df["TARGET"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applicants", f"{len(df):,}")
    c2.metric("Default Rate", f"{default_rate*100:.2f}%", delta="Class imbalance", delta_color="inverse")
    c3.metric("Avg Credit Amount", f"${df['AMT_CREDIT'].mean():,.0f}")
    c4.metric("Avg Income", f"${df['AMT_INCOME_TOTAL'].mean():,.0f}")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # Chart 1: Default rate by income type
    with col_l:
        inc = (df.groupby("NAME_INCOME_TYPE")["TARGET"]
               .agg(rate="mean", count="count").reset_index()
               .sort_values("rate", ascending=True))
        inc["rate_pct"] = (inc["rate"] * 100).round(2)
        fig = px.bar(inc, x="rate_pct", y="NAME_INCOME_TYPE", orientation="h",
                     title="Default Rate by Income Type (%)",
                     color="rate_pct", color_continuous_scale="RdYlGn_r",
                     text="rate_pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(**_plotly_dark(), coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Credit-income ratio distribution
    with col_r:
        fig2 = go.Figure()
        for val, color, name in [(0,"#00D4FF","Repaid"), (1,"#FF6B6B","Defaulted")]:
            data = df[df["TARGET"]==val]["CREDIT_INCOME_RATIO"].clip(0,8)
            fig2.add_trace(go.Histogram(x=data, name=name, opacity=0.7,
                                        marker_color=color, histnorm="probability density", nbinsx=60))
        fig2.update_layout(**_plotly_dark(), title="Credit/Income Ratio Distribution",
                           barmode="overlay", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    # Chart 3: Default rate by age bracket
    with col_l2:
        df["AGE_BRACKET"] = pd.cut(df["AGE_YEARS"], bins=[18,25,35,45,55,65,100],
                                    labels=["18-25","25-35","35-45","45-55","55-65","65+"])
        age = (df.groupby("AGE_BRACKET", observed=True)["TARGET"]
               .agg(rate="mean", count="count").reset_index())
        age["rate_pct"] = (age["rate"]*100).round(2)
        fig3 = px.bar(age, x="AGE_BRACKET", y="rate_pct",
                      title="Default Rate by Age Bracket",
                      color="rate_pct", color_continuous_scale="RdYlGn_r", text="rate_pct")
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.update_layout(**_plotly_dark(), coloraxis_showscale=False, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: EXT_SOURCE scores
    with col_r2:
        if ext:
            fig4 = go.Figure()
            colors_ext = ["#00D4FF","#FFD700","#C77DFF"]
            for col_e, color in zip(ext, colors_ext):
                fig4.add_trace(go.Box(y=df[col_e], name=col_e, marker_color=color, boxmean=True))
            fig4.update_layout(**_plotly_dark(), title="External Credit Bureau Score Distribution", height=350)
            st.plotly_chart(fig4, use_container_width=True)

    # Chart 5: Missing value heatmap (top 20 cols)
    st.markdown("#### Data Quality — Top 20 Columns by Missing Value Rate")
    miss = df.isnull().mean().sort_values(ascending=False).head(20) * 100
    fig5 = px.bar(x=miss.index.tolist(), y=miss.values,
                  labels={"x":"Column","y":"Missing (%)"},
                  color=miss.values, color_continuous_scale="Reds")
    fig5.update_layout(**_plotly_dark(), height=320, coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Underwriting & XAI
# ════════════════════════════════════════════════════════════════════════════
with TAB_UW:
    st.title("🔍 Underwriting Decision & Explainability")

    if model is None:
        st.warning("⚠️ Model not trained yet. Run `python -m src.ml.train` first.")

        # Show evaluation metrics if available
        metrics = load_metrics()
        if metrics:
            st.markdown("#### 📈 Model Evaluation Metrics")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("ROC-AUC", metrics.get("roc_auc","—"))
            mc2.metric("PR-AUC",  metrics.get("pr_auc","—"))
            mc3.metric("Gini",    metrics.get("gini","—"))
            mc4.metric("KS-Stat", metrics.get("ks_statistic","—"))
            mc5.metric("Brier",   metrics.get("brier_score","—"))
    else:
        # ── Metrics banner ────────────────────────────────────────────────────
        metrics = load_metrics()
        if metrics:
            st.markdown("#### 📈 Model Evaluation Metrics")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("ROC-AUC", metrics.get("roc_auc","—"))
            mc2.metric("PR-AUC",  metrics.get("pr_auc","—"))
            mc3.metric("Gini",    metrics.get("gini","—"))
            mc4.metric("KS-Stat", metrics.get("ks_statistic","—"))
            mc5.metric("Brier",   metrics.get("brier_score","—"))
            st.markdown("---")

        st.markdown("#### Enter Applicant Features")
        input_mode = st.radio("Input mode", ["Manual Entry", "Use Dataset Row"], horizontal=True)

        if input_mode == "Use Dataset Row" and app_df is not None:
            sk_id = st.number_input("SK_ID_CURR", min_value=int(app_df["SK_ID_CURR"].min()),
                                    max_value=int(app_df["SK_ID_CURR"].max()),
                                    value=int(app_df["SK_ID_CURR"].iloc[0]))
            row_df = app_df[app_df["SK_ID_CURR"] == sk_id]
            if row_df.empty:
                st.warning("ID not found."); row_dict = {}
            else:
                row_dict = row_df.iloc[0].to_dict()
                st.dataframe(pd.DataFrame([row_dict]).T.rename(columns={0:"Value"}), height=200)
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                income      = st.number_input("Annual Income ($)", 10000, 1000000, 150000, step=5000)
                credit_amt  = st.number_input("Credit Amount ($)", 10000, 2000000, 300000, step=10000)
                annuity     = st.number_input("Annuity ($)", 1000, 100000, 15000, step=500)
            with c2:
                ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, 0.01)
                ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.55, 0.01)
                ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01)
            with c3:
                age         = st.slider("Age (years)", 20, 70, 35)
                emp_years   = st.slider("Years Employed", 0, 40, 5)
                overdue     = st.slider("Bureau Max Overdue Days", 0, 365, 0)

            row_dict = {
                "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit_amt,
                "AMT_ANNUITY": annuity, "EXT_SOURCE_1": ext1,
                "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
                "AGE_YEARS": age, "EMPLOYED_YEARS": emp_years,
                "bureau_max_overdue_days": overdue,
                "CREDIT_INCOME_RATIO": credit_amt / (income + 1),
                "ANNUITY_INCOME_RATIO": annuity / (income + 1),
                "EXT_SOURCE_MEAN": (ext1 + ext2 + ext3) / 3,
                "EXT_SOURCE_MIN": min(ext1, ext2, ext3),
            }

        if st.button("🔮 Predict Risk", type="primary") and row_dict:
            from src.ml.predict import predict_single
            from src.ml.rules import evaluate_rules

            with st.spinner("Analysing applicant..."):
                pred = predict_single(row_dict)
                rule_result = evaluate_rules(row_dict, pred.risk_score, pred.risk_band)

            # ── Risk gauge ────────────────────────────────────────────────────
            col_gauge, col_info = st.columns([1, 1])
            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred.risk_score,
                    title={"text": "Risk Score", "font": {"size": 20, "color": "white"}},
                    number={"font": {"size": 48, "color": pred.risk_color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "white"},
                        "bar": {"color": pred.risk_color},
                        "steps": [
                            {"range": [0, 35], "color": "rgba(0,255,135,0.15)"},
                            {"range": [35, 65], "color": "rgba(255,215,0,0.15)"},
                            {"range": [65, 100], "color": "rgba(255,107,107,0.15)"},
                        ],
                        "threshold": {"line": {"color": pred.risk_color, "width": 3},
                                      "thickness": 0.75, "value": pred.risk_score},
                        "bgcolor": "rgba(0,0,0,0)",
                        "bordercolor": "#333",
                    },
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=280)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_info:
                band_class = pred.risk_band.lower()
                st.markdown(f"""
                <div class="metric-card">
                  <p style="color:#888;margin:0">Risk Band</p>
                  <span class="risk-badge badge-{band_class}">{pred.risk_band.upper()}</span>
                  <br><br>
                  <p style="color:#888;margin:0">Decision</p>
                  <span class="risk-badge" style="background:rgba(0,0,0,0.2);
                        color:{rule_result.decision_color};border:1px solid {rule_result.decision_color}">
                    {rule_result.decision}
                  </span>
                  <br><br>
                  <p style="color:#888;margin:0">Default Probability</p>
                  <h2 style="color:white;margin:0">{pred.probability*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 📋 Policy Assessment")
            st.code(rule_result.explanation)

            # ── SHAP waterfall ────────────────────────────────────────────────
            try:
                from src.ml.explain import explain_single
                shap_df = explain_single(row_dict).head(15)
                colors = ["#FF6B6B" if v > 0 else "#00D4FF" for v in shap_df["shap_value"]]
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["shap_value"], y=shap_df["feature"],
                    orientation="h", marker_color=colors,
                    text=[f"{v:.4f}" for v in shap_df["shap_value"]],
                    textposition="outside",
                ))
                fig_shap.update_layout(
                    **_plotly_dark(),
                    title="SHAP Feature Contributions (Top 15)",
                    height=450,
                    xaxis_title="SHAP Value (contribution to default probability)",
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.info(f"SHAP plot unavailable: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Credit Policy Rules
# ════════════════════════════════════════════════════════════════════════════
with TAB_RULES:
    st.title("📋 Credit Policy Rule Engine")
    st.markdown("All rules are evaluated in order. A single hard-stop triggers **DECLINE** regardless of score.")

    from src.ml.rules import get_all_rules_display
    rules = get_all_rules_display()

    hard_stops = [r for r in rules if r["type"] == "Hard Stop"]
    score_bands = [r for r in rules if r["type"] == "Score Band"]

    st.markdown("#### 🚫 Hard-Stop Rules (Automatic DECLINE)")
    for r in hard_stops:
        st.markdown(f"""
        <div class="rule-card" style="border-left-color:{r['color']}">
          <strong style="color:{r['color']}">{r['condition']}</strong><br>
          <span style="color:#ccc">{r['description']}</span>
          <span style="float:right;color:{r['color']};font-weight:700">{r['action']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 🎯 Score Band Rules")
    for r in score_bands:
        st.markdown(f"""
        <div class="rule-card" style="border-left-color:{r['color']}">
          <strong style="color:{r['color']}">{r['condition']}</strong><br>
          <span style="color:#ccc">{r['description']}</span>
          <span style="float:right;color:{r['color']};font-weight:700">{r['action']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🧪 Rule Simulator")
    sim_score = st.slider("Simulate Risk Score", 0, 100, 45)
    sim_overdue = st.number_input("bureau_max_overdue_days", 0, 365, 0)
    sim_cir = st.number_input("CREDIT_INCOME_RATIO", 0.0, 5.0, 0.4, 0.05)
    sim_ext = st.number_input("EXT_SOURCE_MEAN", 0.0, 1.0, 0.5, 0.01)

    if st.button("▶ Evaluate Rules"):
        from src.ml.rules import evaluate_rules
        sim_features = {
            "bureau_max_overdue_days": sim_overdue,
            "CREDIT_INCOME_RATIO": sim_cir,
            "EXT_SOURCE_MEAN": sim_ext,
        }
        band = "Low" if sim_score < 35 else ("Medium" if sim_score < 65 else "High")
        result = evaluate_rules(sim_features, sim_score, band)
        decision_map = {"APPROVE": "badge-low", "REVIEW": "badge-medium", "DECLINE": "badge-high"}
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;padding:20px">
          <span class="risk-badge {decision_map[result.decision]}">{result.decision}</span>
        </div>
        """, unsafe_allow_html=True)
        st.code(result.explanation)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Talk-to-Data Chatbot
# ════════════════════════════════════════════════════════════════════════════
with TAB_CHAT:
    st.title("💬 Talk-to-Data — Natural Language SQL")

    from src.utils.config import DUCKDB_PATH
    if not DUCKDB_PATH.exists():
        st.warning("⚠️ DuckDB not built. Run `from src.talk_to_data.db_builder import build_db; build_db()`")

    if not hasattr(st.session_state, "chat_history"):
        st.session_state["chat_history"] = []

    api_status = "✅ OpenAI connected" if __import__("os").getenv("OPENAI_API_KEY") else "🔶 Demo mode (no API key)"
    st.caption(api_status)

    # Display history
    for msg in st.session_state["chat_history"]:
        role_class = "chat-msg-user" if msg["role"] == "user" else "chat-msg-bot"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{role_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)
        if "df" in msg and msg["df"] is not None and not msg["df"].empty:
            st.dataframe(msg["df"], use_container_width=True, height=200)
        if "sql" in msg:
            with st.expander("SQL Query", expanded=False):
                st.code(msg["sql"], language="sql")

    # Suggested queries
    suggestions = [
        "What is the overall default rate?",
        "Show default rate by income type",
        "Which education level has the highest default rate?",
        "Top 10 applicants by credit amount",
        "Default rate by gender",
    ]
    st.markdown("**Quick queries:**")
    cols = st.columns(len(suggestions))
    chosen = None
    for i, sug in enumerate(suggestions):
        if cols[i].button(sug, key=f"sug_{i}"):
            chosen = sug

    user_input = st.chat_input("Ask anything about the credit data...") or chosen

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        with st.spinner("🔍 Generating SQL and fetching results..."):
            from src.talk_to_data.nl_to_sql import nl_to_sql
            from src.talk_to_data.query_runner import run_query, summarise_results

            nl_result = nl_to_sql(user_input)
            result_df, error = run_query(nl_result.sql)
            if error:
                summary = f"❌ Query error: {error}"
                result_df = None
            else:
                summary = summarise_results(user_input, nl_result.sql, result_df)

        response_content = f"**{nl_result.explanation}**\n\n{summary}"
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": response_content,
            "sql": nl_result.sql,
            "df": result_df,
        })
        st.rerun()
