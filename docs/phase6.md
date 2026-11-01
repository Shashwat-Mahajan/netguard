# Phase 6: real-world proof lab, MLflow, k3s, SOAR

## 1. Real-attack lab
See `docs/lab_results.md` for the setup, tables and limitations.
Summary: ML (CICIDS-trained) labelled every real nmap/hping3 flow Benign;
signature rules caught scans and floods after two rule fixes; ET Open raised
nothing on these captures. One run per attack, no detection rates.

## 2. MLflow
`python scripts/research/log_mlflow.py` logs the existing central and federated
results (4 runs, per-round curves, calibration biases). Nothing is retrained.
View: `mlflow ui --backend-store-uri sqlite:///D:/netguard/mlflow.db --port 5001`.
`mlruns/` and `mlflow.db` are git-ignored.

## 3. k3s (single node, WSL2)
- Image: `docker build -f docker/api.Dockerfile -t netguard-api:local .`
  (Python 3.12, 1 worker, no models or secrets inside the image).
- Import: `docker save netguard-api:local -o data/netguard-api.tar`, then
  `k3s ctr images import` inside WSL (the tar is git-ignored).
- Secret (never committed), created from the command line:
  `kubectl -n netguard create secret generic netguard-secrets
  --from-literal=clickhouse-password=... --from-literal=secret-key=...`
- Apply: `kubectl apply -f deploy/k3s/netguard.yaml`.
- Models are mounted from the Windows folder `data/processed` (hostPath).
- Bug found: Kubernetes injects `CLICKHOUSE_PORT=tcp://...` for a Service named
  `clickhouse`, which broke the integer setting. Fixed with
  `enableServiceLinks: false` and an explicit `CLICKHOUSE_PORT`.
- Status: the image starts and `/ready` returned ready under Docker. The user
  reported the pods working in k3s; the final `kubectl get pods` output was not
  reviewed.
- Limits: local single-node lab, not production; no ingress, TLS or autoscaling.

## 4. SOAR
Full Shuffle was not run: the machine had about 1.3 GB free RAM and Shuffle
needs about 4 GB. Instead:
- `streaming/shuffle.py` (`SoarClient`) posts high-severity rule alerts as JSON
  to `SOAR_WEBHOOK_URL` (off when unset, deduplicated, fails soft).
- `scripts/soar_receiver.py` is a small local playbook: receive the alert,
  set a priority from threat-intel hits, create a TheHive alert.
- Tested with `data/lab/scan.pcap`: `'soar': {'sent': 1, 'failed': 0}`.
- The sender only posts JSON, so it should work with a Shuffle webhook trigger,
  but that was not tested.
- TheHive case creation failed in the last run (stale key); retest after
  rotating the key.

## Known issues
- Direction heuristic in `port_scan` can miss scanners using low source ports.
- Rule fixes were tuned on lab captures; only three flood variants were
  out-of-sample.
- Secrets were pasted in chat during development (NVIDIA, TheHive keys); both
  must be revoked or rotated.