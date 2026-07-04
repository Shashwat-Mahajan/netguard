# NetGuard — AI-Powered Network Intrusion Detection System

[![Paper](https://img.shields.io/badge/arXiv-preprint-red)](https://arxiv.org/abs/PLACEHOLDER)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://python.org)

> Explainable Federated Network Intrusion Detection using XGBoost, Isolation Forest, SHAP, and NVIDIA NIM LLM Triage

## Key Results

| Method | Dataset | Macro F1 |
|--------|---------|----------|
| XGBoost (proposed) | NSL-KDD | **0.9717** |
| XGBoost (proposed) | CICIDS2017 | **0.9159** |
| Federated XGBoost (3 nodes) | NSL-KDD | **0.9730** |
| Random Forest (baseline) | NSL-KDD | 0.9574 |
| MLP (baseline) | NSL-KDD | 0.8765 |

## Architecture
Scapy/Zeek → Feature Extraction (39 features)
→ XGBoost Classifier (5 attack classes)
→ Isolation Forest (zero-day detection)
→ SHAP Explainability (11ms per flow)
→ NVIDIA NIM LLM (incident report in 2.5s)
→ FastAPI (50 req/sec) → ClickHouse → Grafana

## Project Structure
netguard/
├── capture/          # Scapy packet capture + Zeek log parser
├── processor/        # Dataset loaders (NSL-KDD, CICIDS2017) + preprocessor
├── models/           # XGBoost classifier + Isolation Forest + SHAP
├── streaming/        # Kafka producer/consumer + ClickHouse writer
├── api/              # FastAPI REST + WebSocket endpoints
├── scripts/
│   ├── ingest_data.py        # Phase 1: data ingestion
│   ├── train_model.py        # Phase 2: model training
│   ├── test_phase3.py        # Phase 3: API tests
│   └── research/
│       ├── r1_baselines.py   # Baseline comparisons
│       ├── r2_cicids.py      # CICIDS2017 evaluation
│       ├── r3_federated.py   # Federated learning
│       ├── r4_nim_reporter.py # NVIDIA NIM incident reporter
│       └── r5_ablation.py    # Ablation study
├── config/           # Settings, logging, Prometheus
├── docker/           # Dockerfiles for API and Zeek
├── paper/            # Research paper (LaTeX)
└── docker-compose.phase4.yml

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/netguard
cd netguard
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download dataset
bash scripts/download_dataset.sh nslkdd

# 3. Ingest data
python3 scripts/ingest_data.py --dataset nslkdd

# 4. Train models
python3 scripts/train_model.py --dataset nslkdd

# 5. Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 6. Start dashboard (Docker required)
docker compose -f docker-compose.phase4.yml up -d
# Open http://localhost:3000 (admin/netguard)
```

## Research Phases

| Phase | Script | Output |
|-------|--------|--------|
| R1 — Baseline comparisons | `r1_baselines.py` | Table 2, Figures 2-6 |
| R2 — CICIDS2017 + cross-dataset | `r2_cicids.py` | Table 3, Figures 7-8 |
| R3 — Federated learning | `r3_federated.py` | Table 4, Figures 9-10 |
| R4 — NVIDIA NIM reporter | `r4_nim_reporter.py` | Table 5, Figure 11 |
| R5 — Ablation study | `r5_ablation.py` | Tables 6-7, Figures 12-13 |

## Environment Variables

```bash
cp .env.example .env
# Edit .env:
CAPTURE_INTERFACE=eth0
NVIDIA_NIM_API_KEY=nvapi-...
NVIDIA_NIM_MODEL=meta/llama-3.1-8b-instruct
```

## Tech Stack

- **ML**: XGBoost, Isolation Forest, SHAP, Flower (federated)
- **API**: FastAPI, Uvicorn, WebSocket
- **Storage**: ClickHouse, Prometheus
- **Dashboard**: Grafana OSS
- **Capture**: Scapy, Zeek
- **LLM**: NVIDIA NIM (Llama-3.1-8b-instruct)
- **Infra**: Docker, GitHub Actions

## Citation

If you use this work, please cite:

```bibtex
@article{mahajan2025netguard,
  title={NetGuard: Explainable Federated Network Intrusion Detection
         Using Gradient Boosting, Isolation Forest, and LLM-Augmented Alert Triage},
  author={Mahajan, Shashwat},
  journal={arXiv preprint arXiv:PLACEHOLDER},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
