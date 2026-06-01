"""
notebooks/eda.py
----------------
5 Major Business Insights — EDA script for the Credit Risk Platform.
Run: python notebooks/eda.py
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import DOCUMENTS_DIR
from src.data.loader import load_application_data

SAVE_DIR = DOCUMENTS_DIR / "eda_charts"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BG = "#0E1117"
C1, C2, C3, C4, C5 = "#00D4FF", "#FF6B6B", "#FFD700", "#00FF87", "#C77DFF"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "axes.edgecolor": "#333", "grid.color": "#222",
})

TARGET = "TARGET"

print("Loading application_train.csv ...")
app = load_application_data(split="train", verbose=False)
app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(365_243, np.nan)
app["AGE_YEARS"] = (-app["DAYS_BIRTH"] / 365.0)
app["CREDIT_INCOME_RATIO"] = app["AMT_CREDIT"] / (app["AMT_INCOME_TOTAL"] + 1)
ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in app.columns]
if ext:
    app["EXT_SOURCE_MEAN"] = app[ext].mean(axis=1)
overall_rate = app[TARGET].mean()

def savefig(name: str):
    path = SAVE_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  → saved {path.name}")


# ── Insight 1: Default Rate by Income Type ────────────────────────────────────
print("\n[1/5] Default rate by income type")
inc = (
    app.groupby("NAME_INCOME_TYPE")[TARGET]
    .agg(rate="mean", count="count")
    .reset_index()
    .sort_values("rate", ascending=False)
)
inc["rate_pct"] = inc["rate"] * 100
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(inc["NAME_INCOME_TYPE"], inc["rate_pct"], color=C1, alpha=0.85, edgecolor="white", lw=0.4)
ax.axvline(overall_rate * 100, color=C2, lw=2, ls="--", label=f"Overall: {overall_rate*100:.1f}%")
for bar, (_, row) in zip(bars, inc.iterrows()):
    ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            f"{row['rate_pct']:.1f}% (n={int(row['count']):,})", va="center", fontsize=8)
ax.set_xlabel("Default Rate (%)")
ax.set_title("Insight 1: Default Rate by Income Type", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
savefig("insight_1_default_by_income")

# ── Insight 2: Credit-to-Income Ratio Distribution ───────────────────────────
print("[2/5] Credit-to-income ratio distribution")
fig, ax = plt.subplots(figsize=(10, 5))
for val, color, label in [(0, C1, "Repaid (0)"), (1, C2, "Defaulted (1)")]:
    data = app[app[TARGET] == val]["CREDIT_INCOME_RATIO"].clip(0, 10)
    ax.hist(data, bins=60, alpha=0.65, color=color, label=label, density=True)
ax.set_xlabel("Credit / Income Ratio (clipped at 10)")
ax.set_ylabel("Density")
ax.set_title("Insight 2: Credit-to-Income Ratio by Loan Outcome", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
savefig("insight_2_credit_income_ratio")

# ── Insight 3: DAYS_EMPLOYED Anomaly ─────────────────────────────────────────
print("[3/5] Days-employed anomaly")
n_anom = app["DAYS_EMPLOYED"].isna().sum()
total = len(app)
emp_years = (-app["DAYS_EMPLOYED"].dropna() / 365).clip(0, 50)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(emp_years, bins=50, color=C4, alpha=0.85, edgecolor="white", lw=0.3)
ax1.set_title("Employment Duration (years)", fontweight="bold")
ax1.set_xlabel("Years")
ax2.pie(
    [n_anom, total - n_anom],
    labels=[f"Anomaly\n{n_anom:,} ({n_anom/total*100:.1f}%)", f"Valid\n{total-n_anom:,}"],
    colors=[C2, C1], autopct="%1.1f%%", textprops={"color": "white"},
)
ax2.set_title("Insight 3: DAYS_EMPLOYED Anomaly Share", fontweight="bold")
plt.tight_layout()
savefig("insight_3_employment_anomaly")

# ── Insight 4: External Credit Scores ────────────────────────────────────────
print("[4/5] External credit scores")
colors_ext = [C1, C2, C3]
fig, axes = plt.subplots(1, len(ext), figsize=(5 * len(ext), 5), squeeze=False)
for i, col in enumerate(ext):
    for val, color, label in [(0, C1, "Repaid"), (1, C2, "Defaulted")]:
        axes[0][i].hist(app[app[TARGET] == val][col].dropna(), bins=40, alpha=0.65,
                        color=color, label=label, density=True)
    axes[0][i].set_title(f"Insight 4: {col}", fontweight="bold")
    axes[0][i].set_xlabel("Score")
    axes[0][i].legend(fontsize=8)
plt.suptitle("External Bureau Scores — Lower = Higher Default Risk", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("insight_4_ext_source_scores")

# ── Insight 5: Default Rate by Age Bracket ───────────────────────────────────
print("[5/5] Default rate by age bracket")
app["AGE_BRACKET"] = pd.cut(
    app["AGE_YEARS"], bins=[18, 25, 35, 45, 55, 65, 100],
    labels=["18-25", "25-35", "35-45", "45-55", "55-65", "65+"],
)
age = (
    app.groupby("AGE_BRACKET", observed=True)[TARGET]
    .agg(rate="mean", count="count")
    .reset_index()
)
age["rate_pct"] = age["rate"] * 100
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(age["AGE_BRACKET"].astype(str), age["rate_pct"], color=[C1, C3, C4, C5, C2, "#FF9F40"], alpha=0.9, edgecolor="white", lw=0.5)
for bar, (_, row) in zip(bars, age.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{row['rate_pct']:.1f}%\n(n={int(row['count']):,})", ha="center", fontsize=8)
ax.axhline(overall_rate * 100, color=C2, lw=2, ls="--", label=f"Overall: {overall_rate*100:.1f}%")
ax.set_xlabel("Age Bracket")
ax.set_ylabel("Default Rate (%)")
ax.set_title("Insight 5: Default Rate by Applicant Age Bracket", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
savefig("insight_5_default_by_age")

print(f"\n✅ All 5 EDA charts saved to: {SAVE_DIR}")

# ── Print summary metrics ─────────────────────────────────────────────────────
print("\n=== Business Insight Summary ===")
print(f"  Dataset: {len(app):,} applicants | {app.shape[1]} columns")
print(f"  Overall default rate: {overall_rate*100:.2f}%")
print(f"  Class ratio (repaid:default): {(1-overall_rate)/overall_rate:.1f}:1  → severe imbalance")
print(f"  Missing value rate: {app.isnull().mean().mean()*100:.1f}%")
if ext:
    print(f"  EXT_SOURCE mean for defaulters: {app[app[TARGET]==1]['EXT_SOURCE_MEAN'].mean():.3f}")
    print(f"  EXT_SOURCE mean for repaid:     {app[app[TARGET]==0]['EXT_SOURCE_MEAN'].mean():.3f}")
