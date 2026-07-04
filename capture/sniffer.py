"""
capture/sniffer.py
Live packet capture using Scapy.

Architecture:
  PacketCapture  — raw packet sniffing, runs in a dedicated thread
  FlowTracker    — assembles packets into bidirectional flows
  FlowTable      — thread-safe store of active flows with timeout eviction

Design decisions:
  - Bidirectional flows: (src_ip, src_port, dst_ip, dst_port, proto) is
    normalised so the smaller (ip, port) tuple is always "src", making
    A→B and B→A the same flow key.
  - Flow eviction: flows are evicted after `flow_timeout` seconds of
    inactivity and yielded to the consumer via a callback.
  - Thread safety: FlowTable uses a single RLock — capture thread writes,
    eviction timer reads and deletes.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from capture.feature_extractor import FlowFeatureExtractor
from config.logger import get_logger
from config.settings import settings
from processor.feature_schema import FlowFeatures

logger = get_logger(__name__)

# Flow key: (src_ip, src_port, dst_ip, dst_port, protocol)
FlowKey = tuple[str, int, str, int, int]


@dataclass
class RawFlow:
    """Accumulated raw packet data for one bidirectional flow."""

    key: FlowKey
    start_time: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Forward direction (first packet direction)
    fwd_packets: list[dict] = field(default_factory=list)
    # Backward direction
    bwd_packets: list[dict] = field(default_factory=list)

    # TCP flag accumulators
    fin_flags: int = 0
    syn_flags: int = 0
    rst_flags: int = 0
    psh_flags: int = 0
    ack_flags: int = 0
    urg_flags: int = 0

    # Directional flags
    fwd_psh: int = 0
    bwd_psh: int = 0
    fwd_urg: int = 0
    bwd_urg: int = 0

    # Header sizes
    fwd_header_len: int = 0
    bwd_header_len: int = 0

    # Window sizes (first seen in each direction)
    fwd_win: int = 0
    bwd_win: int = 0


class FlowTable:
    """Thread-safe store of active flows."""

    def __init__(self, timeout: int = settings.capture_flow_timeout) -> None:
        self._flows: dict[FlowKey, RawFlow] = {}
        self._lock = threading.RLock()
        self._timeout = timeout

    def get_or_create(self, key: FlowKey) -> RawFlow:
        with self._lock:
            if key not in self._flows:
                self._flows[key] = RawFlow(key=key)
            return self._flows[key]

    def update_last_seen(self, key: FlowKey) -> None:
        with self._lock:
            if key in self._flows:
                self._flows[key].last_seen = time.time()

    def evict_timed_out(self) -> list[RawFlow]:
        """Return and remove flows that have been idle past timeout."""
        now = time.time()
        evicted: list[RawFlow] = []
        with self._lock:
            stale = [
                k for k, v in self._flows.items() if now - v.last_seen >= self._timeout
            ]
            for k in stale:
                evicted.append(self._flows.pop(k))
        return evicted

    def drain_all(self) -> list[RawFlow]:
        """Evict all remaining flows — call on shutdown."""
        with self._lock:
            flows = list(self._flows.values())
            self._flows.clear()
        return flows

    def __len__(self) -> int:
        with self._lock:
            return len(self._flows)


def _normalise_key(
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
    proto: int,
) -> tuple[FlowKey, bool]:
    """
    Return a canonical flow key and whether this packet is forward-direction.
    Forward = whichever (ip, port) pair compares smaller lexicographically.
    """
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a <= b:
        return (src_ip, src_port, dst_ip, dst_port, proto), True
    else:
        return (dst_ip, dst_port, src_ip, src_port, proto), False


class PacketCapture:
    """
    Captures packets from a network interface using Scapy.

    Usage:
        cap = PacketCapture(on_flow=my_callback)
        cap.start()
        ...
        cap.stop()

    `on_flow` is called from the eviction timer thread whenever a flow
    is completed (timeout) or on shutdown.
    """

    def __init__(
        self,
        interface: str = settings.capture_interface,
        on_flow: Callable[[FlowFeatures], None] | None = None,
        max_packets: int = settings.capture_max_packets,
    ) -> None:
        self._interface = interface
        self._on_flow = on_flow or (lambda f: None)
        self._max_packets = max_packets
        self._table = FlowTable()
        self._extractor = FlowFeatureExtractor()
        self._running = False
        self._sniff_thread: threading.Thread | None = None
        self._evict_thread: threading.Thread | None = None
        self._packet_count = 0

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        logger.info("capture.start", interface=self._interface)

        self._evict_thread = threading.Thread(
            target=self._eviction_loop, daemon=True, name="flow-eviction"
        )
        self._evict_thread.start()

        self._sniff_thread = threading.Thread(
            target=self._sniff_loop, daemon=True, name="packet-sniff"
        )
        self._sniff_thread.start()

    def stop(self) -> None:
        logger.info("capture.stop", active_flows=len(self._table))
        self._running = False
        # Drain remaining flows
        for raw in self._table.drain_all():
            self._emit(raw)

    def is_running(self) -> bool:
        return self._running

    @property
    def packet_count(self) -> int:
        return self._packet_count

    # ── Internal ──────────────────────────────────────────────────────────

    def _sniff_loop(self) -> None:
        """Import Scapy here to avoid import-time side effects."""
        try:
            from scapy.all import sniff  # type: ignore

            sniff(
                iface=self._interface,
                prn=self._handle_packet,
                store=False,
                stop_filter=lambda _: not self._running,
                count=self._max_packets if self._max_packets > 0 else 0,
            )
        except PermissionError:
            logger.error(
                "capture.permission_denied",
                hint="Run with sudo or grant CAP_NET_RAW to the process",
            )
            self._running = False
        except Exception as exc:
            logger.exception("capture.sniff_error", error=str(exc))
            self._running = False

    def _eviction_loop(self) -> None:
        while self._running:
            time.sleep(10)
            for raw in self._table.evict_timed_out():
                self._emit(raw)

    def _handle_packet(self, pkt) -> None:
        try:
            from scapy.layers.inet import IP, TCP, UDP, ICMP  # type: ignore

            if not pkt.haslayer(IP):
                return

            ip = pkt[IP]
            proto = ip.proto
            src_ip, dst_ip = ip.src, ip.dst
            src_port = dst_port = 0

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                src_port, dst_port = tcp.sport, tcp.dport
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                src_port, dst_port = udp.sport, udp.dport

            key, is_fwd = _normalise_key(src_ip, src_port, dst_ip, dst_port, proto)
            flow = self._table.get_or_create(key)
            flow.last_seen = time.time()

            payload_len = len(pkt.payload)
            header_len = len(pkt) - payload_len
            pkt_info = {
                "len": payload_len,
                "time": float(pkt.time),
                "header": header_len,
            }

            if is_fwd:
                flow.fwd_packets.append(pkt_info)
                flow.fwd_header_len += header_len
            else:
                flow.bwd_packets.append(pkt_info)
                flow.bwd_header_len += header_len

            # TCP flags
            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                flags = tcp.flags
                if flags & 0x01:  # FIN
                    flow.fin_flags += 1
                if flags & 0x02:  # SYN
                    flow.syn_flags += 1
                if flags & 0x04:  # RST
                    flow.rst_flags += 1
                if flags & 0x08:  # PSH
                    flow.psh_flags += 1
                    if is_fwd:
                        flow.fwd_psh += 1
                    else:
                        flow.bwd_psh += 1
                if flags & 0x10:  # ACK
                    flow.ack_flags += 1
                if flags & 0x20:  # URG
                    flow.urg_flags += 1
                    if is_fwd:
                        flow.fwd_urg += 1
                    else:
                        flow.bwd_urg += 1

                # Capture first window size in each direction
                if is_fwd and flow.fwd_win == 0:
                    flow.fwd_win = tcp.window
                elif not is_fwd and flow.bwd_win == 0:
                    flow.bwd_win = tcp.window

            self._packet_count += 1

        except Exception as exc:
            logger.debug("capture.packet_error", error=str(exc))

    def _emit(self, raw: RawFlow) -> None:
        """Convert a raw flow to FlowFeatures and call the consumer callback."""
        try:
            features = self._extractor.extract(raw)
            self._on_flow(features)
        except Exception as exc:
            logger.warning("capture.emit_error", flow_key=str(raw.key), error=str(exc))
