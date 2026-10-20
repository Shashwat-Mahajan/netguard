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

exec python3 scripts/run_capture.py --interface "${IFACE}" --max-packets "${MAX_PKTS}"