from src.data.loader import download_dataset, data_files_present

print("Downloading Home Credit dataset via kagglehub...")
print("This may take 5-20 min depending on connection speed.")
print()

try:
    copied = download_dataset()
    print()
    print("=== data/ contents ===")
    for fname, present in data_files_present().items():
        status = "OK  " if present else "MISS"
        print(f"  [{status}] {fname}")
    print()
    print(f"Download complete — {len(copied)} files ready in data/")
except Exception as e:
    print(f"ERROR: {e}")
