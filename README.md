# NetGuard: Explainable Hybrid Network Intrusion Detection

Flow features from packet captures (Scapy, Zeek) go into a fixed 39-feature
schema, then an XGBoost 5-class classifier and an Isolation Forest, with SHAP
explanations and LLM incident reports. Signature rules, Suricata, Zeek notices,
JA3 blocklists and threat intel run alongside the ML. Served with FastAPI,
stored in ClickHouse, monitored with Prometheus and Grafana.

## Results (measured)

| Experiment | Result |
|---|---|
| Central XGBoost, CICIDS2017 test set | accuracy 0.9987, macro-F1 0.927 |
| Federated (Flower, 5 clients, IID), raw | macro-F1 0.586 |
| Federated, non-IID (Dirichlet 0.5), raw | macro-F1 0.591 |
| Federated, no sample weights + per-class bias calibration | macro-F1 0.852 (best case, bias fitted on pooled validation data) |

Real attacks in an isolated two-VM lab (nmap, hping3), one run each:

| Attack | ML | Signature rules | Suricata (ET Open) |
|---|---|---|---|
| SYN scan | missed (all Benign) | detected | no alert |
| SYN flood, 3 ports | missed (all Benign) | detected | no alert |

The CICIDS-trained model does not transfer to real lab traffic. See the
documents below for method and limitations.

## Documents

- [Federated experiment](scripts/research/federated/README.md)
- [Real-attack lab](docs/lab_results.md)
- [Phase 6: MLflow, k3s, SOAR](docs/phase6.md)

## Quick start (Windows)

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
docker compose -f docker-compose.phase4.yml up -d
python scripts\train_model.py --dataset cicids
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Set `CLICKHOUSE_PASSWORD` and `GRAFANA_ADMIN_PASSWORD` in `.env` first.

## Layout

capture/ (sniffer, rules, Zeek parser), processor/, models/, streaming/
(ClickHouse, Suricata, JA3, intel, TheHive, SOAR webhook), api/ (FastAPI,
JWT/API-key auth), config/, docker/, deploy/k3s/, scripts/, scripts/research/,
docs/, tests/.

## Honest limitations

- ML trained on CICIDS2017 misses real nmap/hping3 attacks; the rules catch them.
- Lab results are one run per attack, no detection rates.
- Full Shuffle was not run; SOAR is a webhook sender plus a small local receiver.
- Runs locally; the k3s setup is a single-node lab.

## License

MIT, see [LICENSE](LICENSE).