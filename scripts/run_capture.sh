#!/usr/bin/env bash
# scripts/run_capture.sh
# Start live packet capture on a given interface.
# Requires root (or CAP_NET_RAW) to open a raw socket.
#
# Usage:
#   sudo bash scripts/run_capture.sh eth0
#   sudo bash scripts/run_capture.sh eth0 --max-packets 10000

set -euo pipefail

IFACE="${1:-eth0}"
MAX_PKTS="${2:-0}"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$EUID" -ne 0 ]]; then
    echo "[error] Packet capture requires root. Run with: sudo bash $0 $*"
    exit 1
fi

export PYTHONPATH="${BASE_DIR}"
export CAPTURE_INTERFACE="${IFACE}"
export CAPTURE_MAX_PACKETS="${MAX_PKTS}"

echo "[netguard] Starting live capture on interface: ${IFACE}"
echo "[netguard] Press Ctrl+C to stop and flush remaining flows."

python3 - <<'PYEOF'
import signal, sys, time
from config.logger import configure_logging
from capture.sniffer import PacketCapture
from config.logger import get_logger

configure_logging()
log = get_logger("run_capture")

def handle_flow(f):
    log.info(
        "flow",
        src=f"{f.src_ip}:{f.src_port}",
        dst=f"{f.dst_ip}:{f.dst_port}",
        proto=f.protocol,
        duration=round(f.duration, 3),
        fwd_pkts=f.fwd_packet_count,
        bwd_pkts=f.bwd_packet_count,
        bytes_fwd=f.total_fwd_bytes,
        bytes_bwd=f.total_bwd_bytes,
    )

cap = PacketCapture(on_flow=handle_flow)
cap.start()

def shutdown(sig, frame):
    log.info("capture.shutdown_signal", packets=cap.packet_count)
    cap.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

while cap.is_running():
    time.sleep(1)
PYEOF