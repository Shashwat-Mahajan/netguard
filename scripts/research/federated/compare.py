"""
scripts/research/federated/compare.py
Central vs federated (IID, non-IID, with and without sample weights) on the
same CICIDS test set.

  python scripts/research/federated/compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from common import FED_DIR, MODELS_DIR, load_test  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import settings  # noqa: E402
from models.xgboost_classifier import IntrusionClassifier  # noqa: E402
from processor.feature_schema import FlowFeatures  # noqa: E402

NAMES = [FlowFeatures.LABEL_NAMES[i] for i in range(5)]
MODELS = ("iid", "iid_noweights", "noniid", "noniid_noweights")


def metrics(y, pred) -> dict:
    labels = list(range(5))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision": precision_score(
            y, pred, labels=labels, average=None, zero_division=0
        ).tolist(),
        "recall": recall_score(
            y, pred, labels=labels, average=None, zero_division=0
        ).tolist(),
    }


def main() -> int:
    X, y = load_test()
    results: dict[str, dict] = {}

    clf = IntrusionClassifier()
    clf.load(settings.data_processed_dir / "xgboost_cicids.joblib")
    results["central"] = metrics(y, clf._model.predict(X))

    for name in MODELS:
        path = MODELS_DIR / f"{name}.json"
        if not path.is_file():
            print(f"skipping {name}: {path} not found")
            continue
        bst = xgb.Booster()
        bst.load_model(str(path))
        probs = bst.predict(xgb.DMatrix(X))
        results[f"fed_{name}"] = metrics(y, np.argmax(probs, axis=1))
        results[f"fed_{name}"]["trees"] = bst.num_boosted_rounds()

    print(f"\n{'model':<20}{'accuracy':>10}{'macroF1':>10}")
    for k, m in results.items():
        print(f"{k:<20}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}")

    for title, key in (("RECALL", "recall"), ("PRECISION", "precision")):
        print(f"\n{title} per class")
        print(f"{'model':<20}" + "".join(f"{n:>9}" for n in NAMES))
        for k, m in results.items():
            print(f"{k:<20}" + "".join(f"{v:>9.3f}" for v in m[key]))

    FED_DIR.mkdir(parents=True, exist_ok=True)
    out = FED_DIR / "comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
