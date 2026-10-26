"""
scripts/research/federated/run_federated.py
Runs one full federated experiment on this machine: 1 server + N client
processes talking over gRPC (127.0.0.1).

  python scripts/research/federated/run_federated.py --mode iid --rounds 10
  python scripts/research/federated/run_federated.py --mode iid --no-weights
Run partition.py for the same mode first.
--no-weights trains clients without sample weights; results are saved as
<mode>_noweights.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["iid", "noniid"], required=True)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-rounds", type=int, default=5)
    ap.add_argument("--address", default="127.0.0.1:8080")
    ap.add_argument("--no-weights", action="store_true")
    a = ap.parse_args()

    common = ["--mode", a.mode, "--clients", str(a.clients), "--address", a.address]
    server_extra = ["--tag", "_noweights"] if a.no_weights else []
    client_extra = ["--no-weights"] if a.no_weights else []
    server = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "server.py"),
            *common,
            "--rounds",
            str(a.rounds),
            "--local-rounds",
            str(a.local_rounds),
            *server_extra,
        ]
    )
    time.sleep(8)  # let the server load the test set and start listening
    clients = [
        subprocess.Popen(
            [
                sys.executable,
                str(HERE / "client.py"),
                *common,
                "--cid",
                str(i),
                "--local-rounds",
                str(a.local_rounds),
                *client_extra,
            ]
        )
        for i in range(a.clients)
    ]
    code = server.wait()
    for c in clients:
        try:
            c.wait(timeout=30)
        except subprocess.TimeoutExpired:
            c.terminate()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
