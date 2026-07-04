"""
scripts/ingest_data.py
Phase 1 data ingestion pipeline.

Usage:
    python3 scripts/ingest_data.py --dataset nslkdd
    python3 scripts/ingest_data.py --dataset cicids
    python3 scripts/ingest_data.py --dataset both
    python3 scripts/ingest_data.py --dataset nslkdd --no-cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from config.logger import configure_logging, get_logger
from config.settings import settings
from processor.dataset_loader import NSLKDDLoader, CICIDS2017Loader
from processor.preprocessor import DataPreprocessor
from processor.feature_schema import FlowFeatures

configure_logging()
log = get_logger("ingest")


def banner(text: str) -> None:
    print(f"\n{'─'*52}")
    print(f"  {text}")
    print(f"{'─'*52}")


def chk(ok: bool) -> str:
    return "✓" if ok else "✗"


def ingest_nslkdd(use_cache: bool = True) -> None:
    banner("NSL-KDD ingestion")

    print("\nStep 1 — Verifying raw files")
    train_path = settings.data_raw_dir / "nslkdd" / "KDDTrain+.txt"
    test_path = settings.data_raw_dir / "nslkdd" / "KDDTest+.txt"
    train_ok = train_path.exists()
    test_ok = test_path.exists()
    print(
        f"  {chk(train_ok)}  KDDTrain+.txt  ({train_path.stat().st_size/1e6:.1f} MB)"
        if train_ok
        else f"  ✗  KDDTrain+.txt NOT FOUND"
    )
    print(
        f"  {chk(test_ok)}  KDDTest+.txt   ({test_path.stat().st_size/1e6:.1f} MB)"
        if test_ok
        else f"  ✗  KDDTest+.txt NOT FOUND"
    )
    if not (train_ok and test_ok):
        print("\n  Run: bash scripts/download_dataset.sh nslkdd")
        sys.exit(1)

    print("\nStep 2 — Loading dataset")
    t0 = time.time()
    loader = NSLKDDLoader()
    df_train = loader.load_train(use_cache=use_cache)
    df_test = loader.load_test(use_cache=use_cache)
    print(f"  ✓  Train: {len(df_train):,} rows")
    print(f"  ✓  Test:  {len(df_test):,} rows")
    print(f"  ✓  Done in {time.time()-t0:.1f}s")

    print("\nStep 3 — Label distribution (train)")
    counts = df_train["label_encoded"].value_counts().sort_index()
    total = len(df_train)
    for code, cnt in counts.items():
        name = FlowFeatures.LABEL_NAMES.get(int(code), str(code))
        pct = cnt / total
        bar = "█" * int(pct * 28)
        print(f"  {name:<10} {cnt:>7,}  ({pct*100:5.1f}%)  {bar}")

    print("\nStep 4 — Preprocessing (scale + split)")
    pp = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = pp.fit_transform(df_train)
    print(f"  ✓  X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"  ✓  X_val:   {X_val.shape}     y_val:   {y_val.shape}")
    print(f"  ✓  X_test:  {X_test.shape}    y_test:  {y_test.shape}")
    print(f"  ✓  Features: {X_train.shape[1]}")
    print(f"  ✓  No NaN:   {not np.isnan(X_train).any()}")

    print("\nStep 5 — Saving artifacts")
    out = settings.data_processed_dir
    np.save(out / "X_train_nslkdd.npy", X_train)
    np.save(out / "X_val_nslkdd.npy", X_val)
    np.save(out / "X_test_nslkdd.npy", X_test)
    np.save(out / "y_train_nslkdd.npy", y_train)
    np.save(out / "y_val_nslkdd.npy", y_val)
    np.save(out / "y_test_nslkdd.npy", y_test)
    print(f"  ✓  NumPy arrays saved → {out}/")
    pp.save_scaler(out / "scaler_nslkdd.joblib")
    pp.save_scaler(out / "scaler.joblib")
    print(f"  ✓  Scaler saved → {out}/scaler.joblib")

    banner("NSL-KDD ingestion complete")
    print(f"\n  Files in {out}:")
    for f in sorted(out.iterdir()):
        print(f"    {f.name:<42} {f.stat().st_size/1e6:.2f} MB")
    print("\n  Next: python3 scripts/ingest_data.py --dataset cicids")
    print("  Or:   proceed to Phase 2 model training\n")


def ingest_cicids(use_cache: bool = True) -> None:
    banner("CICIDS2017 ingestion")
    csv_dir = settings.data_raw_dir / "cicids2017"
    csvs = list(csv_dir.glob("*.csv"))

    print(f"\nStep 1 — Verifying raw files in {csv_dir}")
    if not csvs:
        print("  ✗  No CSV files found.")
        print("     Download: https://www.kaggle.com/datasets/cicdataset/cicids2017")
        print("     Place the 8 CSVs in: data/raw/cicids2017/")
        sys.exit(1)
    for c in sorted(csvs):
        print(f"  ✓  {c.name}  ({c.stat().st_size/1e6:.0f} MB)")

    print(f"\nStep 2 — Loading {len(csvs)} CSV files (~5 min)")
    t0 = time.time()
    loader = CICIDS2017Loader()
    df = loader.load(use_cache=use_cache)
    print(f"  ✓  Combined: {len(df):,} rows in {time.time()-t0:.1f}s")

    print("\nStep 3 — Label distribution")
    counts = df["label_encoded"].value_counts().sort_index()
    total = len(df)
    for code, cnt in counts.items():
        name = FlowFeatures.LABEL_NAMES.get(int(code), str(code))
        pct = cnt / total
        bar = "█" * int(pct * 28)
        print(f"  {name:<10} {cnt:>8,}  ({pct*100:5.1f}%)  {bar}")

    print("\nStep 4 — Preprocessing")
    pp = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = pp.fit_transform(df)
    print(f"  ✓  X_train: {X_train.shape}")
    print(f"  ✓  X_val:   {X_val.shape}")
    print(f"  ✓  X_test:  {X_test.shape}")

    print("\nStep 5 — Saving artifacts")
    out = settings.data_processed_dir
    np.save(out / "X_train_cicids.npy", X_train)
    np.save(out / "X_val_cicids.npy", X_val)
    np.save(out / "X_test_cicids.npy", X_test)
    np.save(out / "y_train_cicids.npy", y_train)
    np.save(out / "y_val_cicids.npy", y_val)
    np.save(out / "y_test_cicids.npy", y_test)
    pp.save_scaler(out / "scaler_cicids.joblib")
    pp.save_scaler(out / "scaler.joblib")
    print(f"  ✓  All artifacts saved → {out}/")
    banner("CICIDS2017 ingestion complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="NetGuard Phase 1 — data ingestion")
    parser.add_argument(
        "--dataset", choices=["nslkdd", "cicids", "both"], default="nslkdd"
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    use_cache = not args.no_cache

    print("╔══════════════════════════════════════════════════════╗")
    print("║   NetGuard — Phase 1 Data Ingestion Pipeline         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Cache:    {'disabled' if args.no_cache else 'enabled'}")
    print(f"  Raw dir:  {settings.data_raw_dir}")
    print(f"  Out dir:  {settings.data_processed_dir}")

    if args.dataset in ("nslkdd", "both"):
        ingest_nslkdd(use_cache)
    if args.dataset in ("cicids", "both"):
        ingest_cicids(use_cache)


if __name__ == "__main__":
    main()
