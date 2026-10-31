"""
scripts/research/log_mlflow.py
Logs the EXISTING central/federated results to MLflow. Nothing is retrained.

  python scripts/research/log_mlflow.py
  mlflow ui --backend-store-uri sqlite:///mlflow.db
then open http://localhost:5000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402

FED = ROOT / "data" / "research" / "federated"
CLASSES = ["Benign", "DoS", "Probe", "R2L", "U2R"]


def load(path: Path):
    if not path.is_file():
        print(f"skipping {path.name}: not found")
        return None
    return json.loads(path.read_text())


def log_scores(m: dict, prefix: str = "") -> None:
    mlflow.log_metric(f"{prefix}accuracy", m["accuracy"])
    mlflow.log_metric(f"{prefix}macro_f1", m["macro_f1"])
    for i, c in enumerate(CLASSES):
        mlflow.log_metric(f"{prefix}precision_{c}", m["precision"][i])
        mlflow.log_metric(f"{prefix}recall_{c}", m["recall"][i])


def main() -> int:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("netguard-federated")

    comp = load(FED / "comparison.json") or {}
    calib = load(FED / "calibration.json") or {}

    for name, m in comp.items():
        run_name = "central" if name == "central" else name
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("kind", "central" if name == "central" else "federated")
            if "trees" in m:
                mlflow.log_param("trees", m["trees"])
            mlflow.log_param("sample_weights", "noweights" not in name)
            if name.startswith("fed_"):
                mlflow.log_param("split", "noniid" if "noniid" in name else "iid")
                mlflow.log_param("clients", 5)
            log_scores(m)

            # per-round history written by server.py
            key = name.replace("fed_", "")
            res = load(FED / key / "results.json")
            if res:
                for h in res.get("history", []):
                    rnd = int(h["round"])
                    mlflow.log_metric("round_accuracy", h["accuracy"], step=rnd)
                    mlflow.log_metric("round_macro_f1", h["macro_f1"], step=rnd)
                    for c, v in h.get("recall", {}).items():
                        mlflow.log_metric(f"round_recall_{c}", v, step=rnd)

            cal = calib.get(key)
            if cal:
                mlflow.log_metric("calibrated_macro_f1", cal["macro_f1_after"])
                for i, b in enumerate(cal["bias"]):
                    mlflow.log_param(f"bias_{CLASSES[i]}", b)

    print("logged. Start the UI with:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
