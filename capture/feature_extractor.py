"""
capture/feature_extractor.py
Converts a RawFlow (accumulated packet data) into a FlowFeatures object.

All 40 features in FlowFeatures.ML_FEATURES are computed here.
Values match the definitions used in CICIDS2017 so features are
directly comparable between live capture and the training dataset.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np

from processor.feature_schema import FlowFeatures

if TYPE_CHECKING:
    from capture.sniffer import RawFlow


def _safe_stats(values: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, max, min) — safe for empty lists."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), float(arr.max()), float(arr.min())


def _iats(packet_times: list[float]) -> list[float]:
    """Compute inter-arrival times from a sorted list of packet timestamps."""
    if len(packet_times) < 2:
        return []
    times = sorted(packet_times)
    return [times[i + 1] - times[i] for i in range(len(times) - 1)]


class FlowFeatureExtractor:
    """Stateless extractor — can be reused across many flows."""

    def extract(self, raw: "RawFlow") -> FlowFeatures:
        fwd = raw.fwd_packets
        bwd = raw.bwd_packets
        all_pkts = fwd + bwd

        duration = max(raw.last_seen - raw.start_time, 1e-6)
        total_packets = len(all_pkts)
        total_bytes = sum(p["len"] for p in all_pkts)

        # ── Volume ────────────────────────────────────────────────
        fwd_lens = [p["len"] for p in fwd]
        bwd_lens = [p["len"] for p in bwd]
        total_fwd_bytes = sum(fwd_lens)
        total_bwd_bytes = sum(bwd_lens)

        fwd_mean, fwd_std, fwd_max, fwd_min = _safe_stats(fwd_lens)
        bwd_mean, bwd_std, bwd_max, bwd_min = _safe_stats(bwd_lens)

        # ── Rates ─────────────────────────────────────────────────
        flow_bps = total_bytes / duration
        flow_pps = total_packets / duration
        fwd_pps = len(fwd) / duration
        bwd_pps = len(bwd) / duration

        # ── IATs ──────────────────────────────────────────────────
        all_times = [p["time"] for p in all_pkts]
        fwd_times = [p["time"] for p in fwd]
        bwd_times = [p["time"] for p in bwd]

        iat_all = _iats(all_times)
        iat_fwd = _iats(fwd_times)
        iat_bwd = _iats(bwd_times)

        iat_mean, iat_std, iat_max, iat_min = _safe_stats(iat_all)
        fiat_mean, fiat_std, _, _ = _safe_stats(iat_fwd)
        biat_mean, biat_std, _, _ = _safe_stats(iat_bwd)

        src_ip, src_port, dst_ip, dst_port, proto = raw.key
        if not getattr(raw, "initiator_is_canon", True):
            # report the initiator (first sender) as the source, as in CICIDS
            src_ip, src_port, dst_ip, dst_port = dst_ip, dst_port, src_ip, src_port
        flow_id = str(uuid.uuid4())

        f = FlowFeatures(
            flow_id=flow_id,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=proto,
            timestamp=raw.start_time,
            # Volume
            duration=duration,
            fwd_packet_count=len(fwd),
            bwd_packet_count=len(bwd),
            total_fwd_bytes=total_fwd_bytes,
            total_bwd_bytes=total_bwd_bytes,
            fwd_packet_len_max=int(fwd_max),
            fwd_packet_len_min=int(fwd_min),
            fwd_packet_len_mean=fwd_mean,
            fwd_packet_len_std=fwd_std,
            bwd_packet_len_max=int(bwd_max),
            bwd_packet_len_min=int(bwd_min),
            bwd_packet_len_mean=bwd_mean,
            bwd_packet_len_std=bwd_std,
            # Rates
            flow_bytes_per_sec=flow_bps,
            flow_packets_per_sec=flow_pps,
            fwd_packets_per_sec=fwd_pps,
            bwd_packets_per_sec=bwd_pps,
            # IATs
            flow_iat_mean=iat_mean,
            flow_iat_std=iat_std,
            flow_iat_max=iat_max,
            flow_iat_min=iat_min,
            fwd_iat_mean=fiat_mean,
            fwd_iat_std=fiat_std,
            bwd_iat_mean=biat_mean,
            bwd_iat_std=biat_std,
            # Flags
            fwd_psh_flags=raw.fwd_psh,
            bwd_psh_flags=raw.bwd_psh,
            fwd_urg_flags=raw.fwd_urg,
            bwd_urg_flags=raw.bwd_urg,
            fin_flag_count=raw.fin_flags,
            syn_flag_count=raw.syn_flags,
            rst_flag_count=raw.rst_flags,
            psh_flag_count=raw.psh_flags,
            ack_flag_count=raw.ack_flags,
            # Headers / window
            fwd_header_len=raw.fwd_header_len,
            bwd_header_len=raw.bwd_header_len,
            fwd_win_bytes=raw.fwd_win,
            bwd_win_bytes=raw.bwd_win,
        )

        f.encode_label()
        return f
