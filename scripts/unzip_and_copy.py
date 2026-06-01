"""
scripts/unzip_and_copy.py
-------------------------
Unzip the Kaggle competition zip and place CSVs into data/.
Run after: kaggle competitions download -c home-credit-default-risk -p data
"""
import zipfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DATA_DIR

zip_path = DATA_DIR / "home-credit-default-risk.zip"

if not zip_path.exists():
    print(f"ERROR: {zip_path} not found. Run the kaggle CLI download first.")
    sys.exit(1)

print(f"Unzipping {zip_path.name} ({zip_path.stat().st_size/1024**2:.0f} MB)...")
with zipfile.ZipFile(zip_path, "r") as z:
    members = z.namelist()
    print(f"  {len(members)} files in archive")
    z.extractall(DATA_DIR)

print("\n=== CSV files now in data/ ===")
for f in sorted(DATA_DIR.glob("*.csv")):
    print(f"  {f.name}  ({f.stat().st_size/1024**2:.1f} MB)")

print("\nDone. You can now run: python -m src.data.preprocessor")
