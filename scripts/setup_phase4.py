"""
scripts/setup_phase4.py
Sets up Phase 4 — starts Docker services and verifies everything works.

Usage:
    python3 scripts/setup_phase4.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)


def banner(text: str) -> None:
    print(f"\n{'─'*52}")
    print(f"  {text}")
    print(f"{'─'*52}")


def check_docker() -> bool:
    result = run("docker info", check=False)
    if result.returncode != 0:
        print("  ✗  Docker is not running")
        print("     Start Docker Desktop or run: sudo service docker start")
        return False
    print("  ✓  Docker is running")
    return True


def start_services() -> bool:
    banner("Starting Phase 4 Docker services")
    print("  Starting ClickHouse + Prometheus + Grafana...")
    result = run(
        "docker compose -f docker-compose.phase4.yml up -d",
        check=False,
    )
    if result.returncode != 0:
        print(f"  ✗  Failed: {result.stderr}")
        return False
    print("  ✓  Containers started")
    return True


def wait_for_clickhouse(timeout: int = 60) -> bool:
    print(f"\n  Waiting for ClickHouse to be ready (up to {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        result = run(
            'docker exec netguard-clickhouse clickhouse-client --query "SELECT 1" 2>/dev/null',
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == "1":
            print("  ✓  ClickHouse is ready")
            return True
        time.sleep(3)
        print("    waiting...", end="\r")
    print("  ✗  ClickHouse did not start in time")
    return False


def verify_schema() -> bool:
    banner("Verifying ClickHouse schema")
    checks = {
        "netguard.flows table":    "SELECT count() FROM netguard.flows",
        "netguard.alerts view":    "SELECT count() FROM netguard.alerts",
        "netguard.hourly_stats":   "SELECT count() FROM netguard.hourly_stats",
    }
    all_ok = True
    for name, query in checks.items():
        result = run(
            f'docker exec netguard-clickhouse clickhouse-client '
            f'--user netguard --password netguard123 '
            f'--query "{query}" 2>/dev/null',
            check=False,
        )
        ok = result.returncode == 0
        print(f"  {'✓' if ok else '✗'}  {name}")
        if not ok:
            print(f"       Error: {result.stderr.strip()}")
            all_ok = False
    return all_ok


def install_clickhouse_driver() -> bool:
    banner("Installing clickhouse-driver")
    result = run("pip install clickhouse-driver==0.2.7", check=False)
    if result.returncode == 0:
        print("  ✓  clickhouse-driver installed")
        return True
    print(f"  ✗  Failed: {result.stderr}")
    return False


def test_clickhouse_write() -> bool:
    banner("Testing ClickHouse write from Python")
    try:
        from clickhouse_driver import Client
        client = Client(
            host="localhost", port=9000,
            database="netguard",
            user="netguard", password="netguard123",
            connect_timeout=5,
        )
        from datetime import datetime
        client.execute(
            "INSERT INTO netguard.flows "
            "(flow_id, src_ip, dst_ip, src_port, dst_port, protocol, "
            "timestamp, label, label_encoded, confidence, is_threat) VALUES",
            [("test-001", "192.168.1.1", "8.8.8.8", 12345, 443, 6,
              datetime.now(), "Benign", 0, 0.99, 0)],
        )
        count = client.execute("SELECT count() FROM netguard.flows")[0][0]
        print(f"  ✓  Write test passed — {count} row(s) in flows table")
        return True
    except Exception as exc:
        print(f"  ✗  Write failed: {exc}")
        return False


def show_summary() -> None:
    banner("Phase 4 setup complete")
    print("""
  Services running:
    ClickHouse   → localhost:8123  (HTTP) / localhost:9000 (TCP)
    Prometheus   → http://localhost:9090
    Grafana      → http://localhost:3000

  Grafana login:
    Username: admin
    Password: netguard

  What to do now:
    1. Open http://localhost:3000 in your browser
    2. Go to Dashboards → Import
    3. Upload:  config/grafana/dashboards/netguard_soc.json
    4. Start the API:  uvicorn api.main:app --host 0.0.0.0 --port 8000
    5. Send flows:     python3 scripts/test_phase3.py
    6. Watch Grafana update in real time

  Stop all services:
    docker compose -f docker-compose.phase4.yml down
""")


def main() -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print("║   NetGuard — Phase 4 Setup                           ║")
    print("╚══════════════════════════════════════════════════════╝")

    banner("Step 1 — Docker check")
    if not check_docker():
        sys.exit(1)

    banner("Step 2 — Start services")
    if not start_services():
        sys.exit(1)

    banner("Step 3 — Wait for ClickHouse")
    if not wait_for_clickhouse():
        print("  Try: docker compose -f docker-compose.phase4.yml logs clickhouse")
        sys.exit(1)

    if not verify_schema():
        sys.exit(1)

    banner("Step 4 — Python driver")
    install_clickhouse_driver()
    test_clickhouse_write()

    show_summary()


if __name__ == "__main__":
    main()
