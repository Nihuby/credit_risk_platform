# 🏦 AI-Powered Credit Risk Intelligence Platform

A full-stack credit risk decisioning system built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset. It combines multi-table feature engineering, a LightGBM classifier with SHAP explainability, an NL→SQL chatbot powered by **Meta Llama-3.1-8b-instant via Groq**, and a 4-tab Streamlit dashboard.

---

## Architecture

```
credit_risk_platform/
├── data/                        # Raw CSVs + DuckDB + feature parquet
├── models/                      # Trained model + feature columns + metrics
├── documents/eda_charts/        # EDA chart PNGs
├── notebooks/eda.py             # Standalone EDA script (5 insights)
├── src/
│   ├── utils/config.py          # Centralised paths & settings
│   ├── data/
│   │   ├── loader.py            # kagglehub download + CSV loaders
│   │   └── preprocessor.py     # 4-table merge, feature engineering
│   ├── ml/
│   │   ├── train.py             # LightGBM training + CV
│   │   ├── evaluate.py          # ROC-AUC, PR-AUC, Gini, KS
│   │   ├── predict.py           # Risk score (0-100) + bands
│   │   ├── explain.py           # SHAP TreeExplainer
│   │   └── rules.py             # Credit policy rule engine
│   ├── talk_to_data/
│   │   ├── db_builder.py        # DuckDB ingestion
│   │   ├── prompt_templates.py  # LLM system prompt + few-shots
│   │   ├── nl_to_sql.py         # NL → validated SQL (via Groq API)
│   │   └── query_runner.py      # SQL executor + LLM summary
│   └── app.py                   # Streamlit 4-tab dashboard
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Start (Local)

### 1. Prerequisites
- Python 3.11+
- Kaggle account with `~/.kaggle/kaggle.json` set up
- Competition rules accepted at [kaggle.com/competitions/home-credit-default-risk/rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules)
- Active **Groq API Key** for the GenAI Talk-to-Data engine
### 2. Setup environment
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install tabulate
cp .env.example .env   # then edit .env with your keys
```
Configure your .env file using the Groq OpenAI-compatible redirection endpoints:
OPENAI_API_KEY=gsk_your_groq_api_key_here
OPENAI_BASE_URL=[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)
### 3. Download dataset
```python
from src.data.loader import download_dataset
download_dataset()
```

### 4. Build features & train model
```python
from src.data.preprocessor import build_features
df, spw = build_features(save=True)   # ~5-10 min

python -m src.ml.train                # trains LightGBM, saves model
python -m src.ml.evaluate             # prints metrics, saves JSON
```

### 5. Build DuckDB & run EDA
```python
from src.talk_to_data.db_builder import build_db
build_db()

python notebooks/eda.py              # saves 5 charts to documents/eda_charts/
```

### 6. Launch dashboard
```powershell
streamlit run src/app.py
```
Open [http://localhost:8501](http://localhost:8501)

---

## Quick Start (Docker)
```bash
cp .env.example .env    # add your OPENAI_API_KEY
# Place your CSVs in data/ first (download separately)
docker compose up --build
```

---

## Model Selection Rationale

**LightGBM** was chosen over logistic regression, XGBoost, and neural networks because:
- **Speed**: Gradient boosting on histogram bins is 10–20× faster than XGBoost on tabular data
- **Missing value handling**: Native support avoids pre-imputation during training
- **Feature importance**: Compatible with SHAP `TreeExplainer` for exact, fast explanations
- **Imbalanced data**: `scale_pos_weight` parameter directly corrects the class ratio

---

## Class Imbalance Strategy

The dataset has ~8% default rate (≈11.8:1 class ratio). We use:

| Strategy | Implementation |
|---|---|
| `scale_pos_weight` | `n_negative / n_positive` passed to LightGBM |
| Stratified splits | `StratifiedKFold` + `train_test_split(stratify=y)` |
| PR-AUC metric | Prioritised over accuracy — measures precision/recall for the minority class |
| No oversampling | Preserves distributional integrity for the real-world imbalance |

---

## Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| **ROC-AUC** | Area under ROC curve. Competition primary metric. | > 0.76 |
| **PR-AUC** | Area under Precision-Recall curve. Better for imbalanced data. | > 0.35 |
| **Gini** | `2 × ROC-AUC − 1`. Industry standard in credit scoring. | > 0.50 |
| **KS-Statistic** | Max separation between TPR and FPR curves. | > 0.35 |
| **Brier Score** | Mean squared error of probability estimates. Lower is better. | < 0.07 |

---

Prompt Framework (Talk-to-Data via Groq)
The NL→SQL system uses an ultra-fast, open-source LLM layer utilizing the Meta Llama-3.1-8b-instant model routed through the Groq Inference Gateway. This setup achieves near-zero latency execution on a structured 3-layer prompt architecture:

System prompt — Embeds the full DuckDB database layout schema, hard SQL safety boundaries (forcing SELECT parameters only, strictly blocking DDL/DML vectors), and enforces a clean output payload schema layout structure (response_format={"type": "json_object"}).

Few-shot examples — 6 hand-crafted user→assistant interaction pairs mapping advanced logic structures like joins, sub-aggregations, filter arrays, and mathematical window operations.

Token Optimization & Schema Injection — Rather than passing raw table contents or deep text rows, only highly compressed, metadata-only data types and relationship definitions are exposed. This preserves Groq’s token limits and guarantees deterministic inference parameters at zero operational cost.

Safety guardrails: Regex parsing engines completely block execution attempts containing dangerous SQL command tokens (INSERT, UPDATE, DELETE, DROP, CREATE, ALTER). A default cap limits query output size arrays to a maximum height of 500 records.

Offline/demo mode: Simple local keyword matching fallback triggers clean sample outputs if the OPENAI_API_KEY environment token variable is blank or unconfigured.

---

## Credit Policy Rule Engine

Rules are evaluated in priority order:

1. **Hard-stop rules** (immediate DECLINE):
   - `CREDIT_INCOME_RATIO > 0.60` — extreme debt burden
   - `EXT_SOURCE_MEAN < 0.15` — critically poor bureau scores
   - `bureau_max_overdue_days > 180` — active delinquency
   - `inst_dpd_mean > 30` — chronic late payments
   - `ANNUITY_INCOME_RATIO > 0.40` — unaffordable repayments
   - *(6 additional rules — see `src/ml/rules.py`)*

2. **Score band rules**:
   - Score < 35 → **APPROVE**
   - 35 ≤ Score < 65 → **REVIEW** (manual underwriter)
   - Score ≥ 65 → **DECLINE**

---

## System Limitations

| Limitation | Detail |
|---|---|
| **Temporal leakage risk** | No time-based train/test split — model may overfit to historical patterns |
| **External bureau data** | Aggregated to simple stats; richer time-series features (e.g., delinquency trends) not used |
| **No calibration** | Probabilities are raw LightGBM outputs — add `CalibratedClassifierCV` for deployment |
| **Offline fallback SQL** | Demo NL→SQL uses simple keyword matching, not semantic understanding |
| **Single model** | No ensemble or model stacking; adding XGBoost/CatBoost would improve AUC by ~0.01–0.02 |
| **Feature drift** | No monitoring for production input distribution shifts |
