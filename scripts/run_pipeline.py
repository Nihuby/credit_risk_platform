"""
scripts/run_pipeline.py
-----------------------
Full post-download pipeline: extract → features → train → DuckDB → evaluate.
Polls for the zip file, then chains all steps automatically.

Run from the project root:
    $env:PYTHONPATH="."; .\\venv\\Scripts\\python.exe scripts\\run_pipeline.py
"""

from __future__ import annotations
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DATA_DIR, FEATURES_PARQUET_PATH, MODEL_PATH, DUCKDB_PATH

COMPETITION_ZIP = DATA_DIR / "home-credit-default-risk.zip"
REQUIRED_CSVS = [
    "application_train.csv",
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
]

def banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ─── STEP 1: Wait for zip + extract ──────────────────────────────────────────
banner("STEP 1 / 5 — Waiting for download & extracting CSVs")

csvs_present = all((DATA_DIR / f).exists() for f in REQUIRED_CSVS)

if csvs_present:
    print("All required CSVs already present in data/ — skipping extraction.")
elif COMPETITION_ZIP.exists():
    print(f"Found {COMPETITION_ZIP.name} ({COMPETITION_ZIP.stat().st_size/1024**2:.0f} MB)")
    print("Extracting...")
    with zipfile.ZipFile(COMPETITION_ZIP, "r") as z:
        members = z.namelist()
        print(f"  Archive contains {len(members)} files")
        z.extractall(DATA_DIR)
    print("Extraction complete.")
else:
    print(f"Zip not found at {COMPETITION_ZIP}")
    print("Waiting for download to complete (polling every 60s)...")
    while not COMPETITION_ZIP.exists():
        time.sleep(60)
        if COMPETITION_ZIP.exists():
            size_mb = COMPETITION_ZIP.stat().st_size / 1024**2
            print(f"Zip appeared: {size_mb:.0f} MB — waiting for write to finish...")
            time.sleep(10)
            break
    print(f"Extracting {COMPETITION_ZIP.name}...")
    with zipfile.ZipFile(COMPETITION_ZIP, "r") as z:
        z.extractall(DATA_DIR)
    print("Extraction complete.")

print("\nCSVs in data/:")
for csv in sorted(DATA_DIR.glob("*.csv")):
    print(f"  {csv.name}  ({csv.stat().st_size/1024**2:.1f} MB)")

# ─── STEP 2: Build features ───────────────────────────────────────────────────
banner("STEP 2 / 5 — Building feature matrix (preprocessor.py)")

if FEATURES_PARQUET_PATH.exists():
    print(f"features.parquet already exists ({FEATURES_PARQUET_PATH.stat().st_size/1024**2:.1f} MB) — skipping.")
    import pandas as pd
    df_check = pd.read_parquet(FEATURES_PARQUET_PATH)
    print(f"  Shape: {df_check.shape}")
else:
    t0 = time.perf_counter()
    from src.data.preprocessor import build_features
    df, spw = build_features(save=True, verbose=True)
    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\nFeature matrix: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"scale_pos_weight: {spw:.2f}")
    print(f"Elapsed: {elapsed}s")

# ─── STEP 3: Train LightGBM ───────────────────────────────────────────────────
banner("STEP 3 / 5 — Training LightGBM model (train.py)")

if MODEL_PATH.exists():
    print(f"Model already exists at {MODEL_PATH} — skipping training.")
    print("Delete models/credit_risk_model.pkl to retrain.")
else:
    t0 = time.perf_counter()
    from src.ml.train import train
    result = train(verbose=True)
    print("\n--- Training Results ---")
    for k, v in result.items():
        print(f"  {k}: {v}")

# ─── STEP 4: Build DuckDB ────────────────────────────────────────────────────
banner("STEP 4 / 5 — Building DuckDB database (db_builder.py)")

if DUCKDB_PATH.exists():
    print(f"DuckDB already exists ({DUCKDB_PATH.stat().st_size/1024**2:.1f} MB) — rebuilding views.")

from src.talk_to_data.db_builder import build_db
build_db(force=True)
print(f"DuckDB ready at {DUCKDB_PATH}")

# ─── STEP 5: Evaluate model ───────────────────────────────────────────────────
banner("STEP 5 / 5 — Evaluating model metrics (evaluate.py)")

from src.ml.evaluate import evaluate
metrics = evaluate(verbose=True)

print("\n" + "=" * 60)
print("  FINAL MODEL METRICS")
print("=" * 60)
print(f"  ROC-AUC  : {metrics.get('roc_auc', 'N/A')}")
print(f"  PR-AUC   : {metrics.get('pr_auc', 'N/A')}")
print(f"  Gini     : {metrics.get('gini', 'N/A')}")
print(f"  KS-Stat  : {metrics.get('ks_statistic', 'N/A')}")
print(f"  Brier    : {metrics.get('brier_score', 'N/A')}")
print("=" * 60)
print("\n✅ Pipeline complete. Launch the dashboard with:")
print("   $env:PYTHONPATH='.'; .\\venv\\Scripts\\streamlit.exe run src/app.py")
