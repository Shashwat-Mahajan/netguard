"""
scripts/test_phase3.py
Tests the full Phase 3 API pipeline without needing Redpanda running.

Sends real flow records to the /predict endpoint and verifies:
  - XGBoost classifies correctly
  - Isolation Forest anomaly scores are returned
  - /stats endpoint shows live counts
  - /alerts endpoint returns threats
  - Inference time is under 50ms

Usage:
    # Terminal 1 — start the API
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    # Terminal 2 — run tests
    python3 scripts/test_phase3.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

BASE = "http://localhost:8000"


def banner(text: str) -> None:
    print(f"\n{'─'*52}")
    print(f"  {text}")
    print(f"{'─'*52}")


# ── Sample flows (from NSL-KDD characteristics) ───────────────────────────────

BENIGN_FLOW = {
    "src_ip": "192.168.1.10",
    "dst_ip": "8.8.8.8",
    "src_port": 54321,
    "dst_port": 443,
    "protocol": 6,
    "duration": 2.5,
    "fwd_packet_count": 8,
    "bwd_packet_count": 6,
    "total_fwd_bytes": 1200,
    "total_bwd_bytes": 8400,
    "flow_bytes_per_sec": 3840.0,
    "flow_packets_per_sec": 5.6,
    "syn_flag_count": 1,
    "fin_flag_count": 1,
    "rst_flag_count": 0,
    "psh_flag_count": 3,
}

DOS_FLOW = {
    "src_ip": "10.0.0.1",
    "dst_ip": "192.168.1.100",
    "src_port": 12345,
    "dst_port": 80,
    "protocol": 6,
    "duration": 0.001,        # very short duration
    "fwd_packet_count": 1000, # massive packet count
    "bwd_packet_count": 0,    # no response (server overwhelmed)
    "total_fwd_bytes": 60000,
    "total_bwd_bytes": 0,
    "flow_bytes_per_sec": 60_000_000.0,  # 60 MB/s — classic DoS
    "flow_packets_per_sec": 1_000_000.0,
    "syn_flag_count": 1000,   # SYN flood
    "fin_flag_count": 0,
    "rst_flag_count": 0,
    "psh_flag_count": 0,
}

PROBE_FLOW = {
    "src_ip": "172.16.0.5",
    "dst_ip": "192.168.1.1",
    "src_port": 44444,
    "dst_port": 22,
    "protocol": 6,
    "duration": 0.5,
    "fwd_packet_count": 3,
    "bwd_packet_count": 0,    # no reply — port may be closed
    "total_fwd_bytes": 180,
    "total_bwd_bytes": 0,
    "flow_bytes_per_sec": 360.0,
    "flow_packets_per_sec": 6.0,
    "syn_flag_count": 1,
    "fin_flag_count": 0,
    "rst_flag_count": 1,      # RST — refused connection (port scan)
    "psh_flag_count": 0,
}


def check_health(client: httpx.Client) -> bool:
    banner("Test 1 — Health check")
    resp = client.get(f"{BASE}/health")
    ok = resp.status_code == 200
    print(f"  {'✓' if ok else '✗'}  GET /health → {resp.status_code}")
    if ok:
        print(f"       {resp.json()}")
    return ok


def check_ready(client: httpx.Client) -> bool:
    banner("Test 2 — Readiness check")
    resp = client.get(f"{BASE}/ready")
    print(f"  {'✓' if resp.status_code == 200 else '⚠'}  GET /ready → {resp.status_code}")
    print(f"       {resp.json()}")
    return True  # 503 is OK if models not loaded yet


def test_predict(client: httpx.Client, flow: dict, expected_label: str, name: str) -> bool:
    resp = client.post(f"{BASE}/api/v1/predict", json=flow, timeout=30)
    if resp.status_code != 200:
        print(f"  ✗  {name} → HTTP {resp.status_code}: {resp.text}")
        return False

    r = resp.json()
    correct = r["label"] == expected_label
    threat  = r["is_threat"]
    ms      = r["inference_ms"]
    fast    = ms < 50

    print(f"  {'✓' if correct else '✗'}  {name}")
    print(f"       Label:      {r['label']} (expected {expected_label})  {'✓' if correct else '✗'}")
    print(f"       Confidence: {r['confidence']:.4f}")
    print(f"       Anomaly:    {r['is_anomaly']} (score={r['anomaly_score']})")
    print(f"       Threat:     {threat}")
    print(f"       Inference:  {ms}ms  {'✓ fast' if fast else '⚠ slow'}")
    print(f"       Probs:      {r['class_probs']}")

    return correct


def test_all_predictions(client: httpx.Client) -> None:
    banner("Test 3 — Flow predictions")
    test_predict(client, BENIGN_FLOW, "Benign", "Normal HTTPS flow")
    test_predict(client, DOS_FLOW,    "DoS",    "SYN flood attack")
    test_predict(client, PROBE_FLOW,  "Probe",  "Port scan probe")


def test_stats(client: httpx.Client) -> None:
    banner("Test 4 — Dashboard stats")
    resp = client.get(f"{BASE}/api/v1/stats")
    if resp.status_code == 200:
        s = resp.json()
        print("  ✓  GET /stats")
        print(f"       Total flows:      {s['total_flows']}")
        print(f"       Threats detected: {s['alert_count']}")
        print(f"       Threat rate:      {s['threat_rate']:.1%}")
        print(f"       Label counts:     {s['label_counts']}")
        print(f"       Models loaded:    {s['inference_engine']['models_loaded']}")
    else:
        print(f"  ✗  GET /stats → {resp.status_code}")


def test_alerts(client: httpx.Client) -> None:
    banner("Test 5 — Alert feed")
    resp = client.get(f"{BASE}/api/v1/alerts")
    if resp.status_code == 200:
        alerts = resp.json()["alerts"]
        print(f"  ✓  GET /alerts → {len(alerts)} alerts")
        for a in alerts[:3]:
            print(f"       {a['src_ip']}:{a['src_port']} → {a['dst_ip']}:{a['dst_port']}"
                  f"  [{a['label']}]  confidence={a['confidence']}")
    else:
        print(f"  ✗  GET /alerts → {resp.status_code}")


def throughput_test(client: httpx.Client, n: int = 50) -> None:
    banner(f"Test 6 — Throughput ({n} predictions)")
    t0 = time.perf_counter()
    ok = 0
    for _ in range(n):
        r = client.post(f"{BASE}/api/v1/predict", json=BENIGN_FLOW, timeout=10)
        if r.status_code == 200:
            ok += 1
    elapsed = time.perf_counter() - t0
    rps = ok / elapsed
    print(f"  ✓  {ok}/{n} requests succeeded")
    print(f"       Total time: {elapsed:.2f}s")
    print(f"       Throughput: {rps:.1f} requests/sec")
    print(f"       Avg latency: {elapsed/ok*1000:.1f}ms per prediction")


def main() -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print("║   NetGuard — Phase 3 API Tests                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Target: {BASE}")

    with httpx.Client() as client:
        # Check server is up
        try:
            client.get(f"{BASE}/health", timeout=3)
        except Exception:
            print(f"\n  ✗  Cannot connect to {BASE}")
            print("     Start the API first:")
            print("     uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
            sys.exit(1)

        check_health(client)
        check_ready(client)
        test_all_predictions(client)
        test_stats(client)
        test_alerts(client)
        throughput_test(client, n=20)

    banner("All tests complete")
    print("""
  If all labels matched expected values:
    ✓  Phase 3 is working correctly
    ✓  XGBoost is classifying live flows
    ✓  API is production-ready

  Next steps:
    - Open http://localhost:8000/docs  to explore the interactive API
    - Start Phase 4: Grafana dashboard + ClickHouse storage
""")


if __name__ == "__main__":
    main()
