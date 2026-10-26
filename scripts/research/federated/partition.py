"""
scripts/research/federated/partition.py
Splits the CICIDS training set across simulated federated clients.
Saves row INDICES only (the data files are not copied).

  python scripts/research/federated/partition.py --mode iid
  python scripts/research/federated/partition.py --mode noniid --alpha 0.5

iid     every client gets a random slice.
noniid  per class, a Dirichlet(alpha) share goes to each client.
        Small alpha = clients see very different attack types.
Each client keeps 10% of its rows as a local validation set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from processor.feature_schema import FlowFeatures  # noqa: E402

OUT_DIR = ROOT / "data" / "research" / "federated" / "partitions"
MIN_ROWS = 1000


def iid_split(y: np.ndarray, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [np.sort(p) for p in np.array_split(rng.permutation(len(y)), n)]


def dirichlet_split(
    y: np.ndarray, n: int, alpha: float, rng: np.random.Generator
) -> list[np.ndarray]:
    parts: list[list[int]] = [[] for _ in range(n)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        share = rng.dirichlet([alpha] * n)
        cuts = (np.cumsum(share) * len(idx)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(idx, cuts)):
            parts[i].extend(chunk.tolist())
    return [np.sort(np.array(p, dtype=np.int64)) for p in parts]


def split_val(
    idx: np.ndarray, rng: np.random.Generator, frac: float = 0.10
) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(idx)
    k = max(1, int(len(idx) * frac))
    return np.sort(idx[k:]), np.sort(idx[:k])


def main() -> int:
    ap = argparse.ArgumentParser(description="Partition CICIDS for federated runs")
    ap.add_argument("--mode", choices=["iid", "noniid"], required=True)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="cicids")
    args = ap.parse_args()

    path = settings.data_processed_dir / f"y_train_{args.dataset}.npy"
    if not path.is_file():
        print(f"{path} not found")
        return 1
    y = np.load(path)

    parts: list[np.ndarray] = []
    for attempt in range(50):
        rng = np.random.default_rng(args.seed + attempt)
        if args.mode == "iid":
            parts = iid_split(y, args.clients, rng)
        else:
            parts = dirichlet_split(y, args.clients, args.alpha, rng)
        if min(len(p) for p in parts) >= MIN_ROWS:
            break
    else:
        print("could not build partitions with enough rows per client")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    names = [FlowFeatures.LABEL_NAMES[i] for i in range(5)]
    print(f"mode={args.mode} clients={args.clients} seed={args.seed + attempt}")
    print(f"{'client':<8}{'train':>9}{'val':>8}" + "".join(f"{n:>9}" for n in names))
    for i, p in enumerate(parts):
        tr, va = split_val(p, np.random.default_rng(args.seed + 1000 + i))
        arrays[f"train_{i}"] = tr
        arrays[f"val_{i}"] = va
        counts = np.bincount(y[tr], minlength=5)
        print(
            f"{i:<8}{len(tr):>9,}{len(va):>8,}"
            + "".join(f"{int(c):>9,}" for c in counts)
        )

    out = OUT_DIR / f"{args.mode}_{args.clients}.npz"
    np.savez(out, **arrays)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
