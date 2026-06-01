"""
scripts/patch_loader.py — inject zip fallback into loader.py
"""
from pathlib import Path

loader = Path("src/data/loader.py")
text = loader.read_text(encoding="utf-8")

old = '''    try:
        # kagglehub downloads to a local cache and returns the cache path
        cache_path: str = kagglehub.competition_download(
            settings.kaggle_competition
        )
        cache_dir = Path(cache_path)
        logger.info("download_complete", cache_dir=str(cache_dir), elapsed_s=round(time.perf_counter() - t0, 1))
    except Exception as exc:
        raise RuntimeError('''

new = '''    try:
        cache_path: str = kagglehub.competition_download(settings.kaggle_competition)
        cache_dir = Path(cache_path)
        logger.info("download_complete", cache_dir=str(cache_dir), elapsed_s=round(time.perf_counter() - t0, 1))
    except Exception as exc:
        zip_path = DATA_DIR / f"{settings.kaggle_competition}.zip"
        if zip_path.exists():
            logger.info("cli_zip_fallback", zip=str(zip_path))
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATA_DIR)
            cache_dir = DATA_DIR
        else:
            raise RuntimeError('''

if old in text:
    loader.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("Patch applied.")
else:
    print("Pattern not found — skipping patch (loader already patched or text differs).")
