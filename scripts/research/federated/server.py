"""
scripts/research/federated/server.py
Flower server: FedXgbBagging, plus a central test-set score after every round.
Normally started by run_federated.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flwr as fl  # noqa: E402
import xgboost as xgb  # noqa: E402
from common import FED_DIR, MODELS_DIR, load_test, score  # noqa: E402
from flwr.server.strategy import FedXgbBagging  # noqa: E402


class ScoredBagging(FedXgbBagging):
    """FedXgbBagging that scores the merged global model after every round."""

    def __init__(self, X_test, y_test, **kwargs) -> None:
        super().__init__(**kwargs)
        self.X_test = X_test
        self.y_test = y_test
        self.history: list[dict] = []
        self.final_model: bytes = b""

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if failures:
            print(f"[server] round {server_round}: {len(failures)} client failure(s)")
        if params is not None and params.tensors:
            raw = bytes(params.tensors[0])
            self.final_model = raw
            bst = xgb.Booster()
            bst.load_model(bytearray(raw))
            probs = bst.predict(xgb.DMatrix(self.X_test))
            s = score(self.y_test, probs)
            s["round"] = server_round
            s["boosting_rounds"] = bst.num_boosted_rounds()
            self.history.append(s)
            rec = {k: round(v, 3) for k, v in s["recall"].items()}
            print(
                f"[server] round {server_round}: acc={s['accuracy']:.4f} "
                f"macroF1={s['macro_f1']:.4f} recall={rec}"
            )
        return params, metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["iid", "noniid"], required=True)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-rounds", type=int, default=5)
    ap.add_argument("--address", default="127.0.0.1:8080")
    ap.add_argument("--tag", default="", help="suffix for output names")
    a = ap.parse_args()

    X_test, y_test = load_test()
    strategy = ScoredBagging(
        X_test,
        y_test,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=a.clients,
        min_evaluate_clients=0,
        min_available_clients=a.clients,
        on_fit_config_fn=lambda rnd: {"global_round": str(rnd)},
    )
    fl.server.start_server(
        server_address=a.address,
        config=fl.server.ServerConfig(num_rounds=a.rounds),
        strategy=strategy,
    )

    name = f"{a.mode}{a.tag}"
    FED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = FED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": a.mode,
        "tag": a.tag,
        "clients": a.clients,
        "rounds": a.rounds,
        "local_rounds": a.local_rounds,
        "history": strategy.history,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    if strategy.final_model:
        (MODELS_DIR / f"{name}.json").write_bytes(strategy.final_model)
    print(f"[server] saved {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
