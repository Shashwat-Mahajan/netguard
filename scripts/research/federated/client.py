"""
scripts/research/federated/client.py
One federated client. Normally started by run_federated.py.

  python scripts/research/federated/client.py --mode iid --cid 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flwr as fl  # noqa: E402
import xgboost as xgb  # noqa: E402
from common import PARAMS, load_partition, sample_weights  # noqa: E402
from flwr.common import (  # noqa: E402
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)

OK = Status(code=Code.OK, message="OK")


class XgbClient(fl.client.Client):
    def __init__(self, mode: str, clients: int, cid: int, local_rounds: int) -> None:
        X_tr, y_tr, X_va, y_va = load_partition(mode, clients, cid)
        self.cid = cid
        self.local_rounds = local_rounds
        self.n_train = len(y_tr)
        self.n_val = len(y_va)
        self.dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weights(y_tr))
        print(f"[client {cid}] {self.n_train:,} train rows, {self.n_val:,} val rows")

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status=OK, parameters=Parameters(tensor_type="", tensors=[])
        )

    def fit(self, ins: FitIns) -> FitRes:
        rnd = int(ins.config["global_round"])
        if rnd == 1:
            bst = xgb.train(PARAMS, self.dtrain, num_boost_round=self.local_rounds)
        else:
            bst = xgb.Booster(params=PARAMS)
            bst.load_model(bytearray(ins.parameters.tensors[0]))
            for _ in range(self.local_rounds):
                bst.update(self.dtrain, bst.num_boosted_rounds())
        n = bst.num_boosted_rounds()
        new_trees = bst[n - self.local_rounds : n]  # only this round's trees
        raw = bytes(new_trees.save_raw("json"))
        print(f"[client {self.cid}] round {rnd}: sent {len(raw) / 1e3:.0f} kB of trees")
        return FitRes(
            status=OK,
            parameters=Parameters(tensor_type="", tensors=[raw]),
            num_examples=self.n_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Not used: the server scores the global model on the central test set.
        return EvaluateRes(status=OK, loss=0.0, num_examples=self.n_val, metrics={})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["iid", "noniid"], required=True)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--local-rounds", type=int, default=5)
    ap.add_argument("--address", default="127.0.0.1:8080")
    a = ap.parse_args()
    client = XgbClient(a.mode, a.clients, a.cid, a.local_rounds)
    fl.client.start_client(server_address=a.address, client=client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
