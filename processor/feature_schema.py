"""
processor/feature_schema.py
Canonical feature schema used across the entire pipeline.

Every flow — whether extracted live by Scapy or parsed from CICIDS2017/NSL-KDD —
is normalised to these 40 features before any ML model touches the data.
Adding a new feature means adding it here first, then updating extractors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(slots=True)
class FlowFeatures:
    """One record per completed network flow."""

    # ── Identifiers (dropped before ML, kept for logging) ───────────────
    flow_id: str = ""
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 0  # 6=TCP, 17=UDP, 1=ICMP
    timestamp: float = 0.0

    # ── Volume features ──────────────────────────────────────────────────
    duration: float = 0.0  # Flow duration in seconds
    fwd_packet_count: int = 0  # Packets sent src→dst
    bwd_packet_count: int = 0  # Packets sent dst→src
    total_fwd_bytes: int = 0
    total_bwd_bytes: int = 0
    fwd_packet_len_max: int = 0
    fwd_packet_len_min: int = 0
    fwd_packet_len_mean: float = 0.0
    fwd_packet_len_std: float = 0.0
    bwd_packet_len_max: int = 0
    bwd_packet_len_min: int = 0
    bwd_packet_len_mean: float = 0.0
    bwd_packet_len_std: float = 0.0

    # ── Rate features ────────────────────────────────────────────────────
    flow_bytes_per_sec: float = 0.0
    flow_packets_per_sec: float = 0.0
    fwd_packets_per_sec: float = 0.0
    bwd_packets_per_sec: float = 0.0

    # ── Inter-arrival time (IAT) features ────────────────────────────────
    flow_iat_mean: float = 0.0
    flow_iat_std: float = 0.0
    flow_iat_max: float = 0.0
    flow_iat_min: float = 0.0
    fwd_iat_mean: float = 0.0
    fwd_iat_std: float = 0.0
    bwd_iat_mean: float = 0.0
    bwd_iat_std: float = 0.0

    # ── TCP flag counts ───────────────────────────────────────────────────
    fwd_psh_flags: int = 0
    bwd_psh_flags: int = 0
    fwd_urg_flags: int = 0
    bwd_urg_flags: int = 0
    fin_flag_count: int = 0
    syn_flag_count: int = 0
    rst_flag_count: int = 0
    psh_flag_count: int = 0
    ack_flag_count: int = 0

    # ── Window / header features ──────────────────────────────────────────
    fwd_header_len: int = 0
    bwd_header_len: int = 0
    fwd_win_bytes: int = 0
    bwd_win_bytes: int = 0

    # ── Label (ground truth for training, "unknown" for live capture) ─────
    label: str = "unknown"
    label_encoded: int = -1

    # ── Class variable: ordered list of ML feature names ─────────────────
    ML_FEATURES: ClassVar[list[str]] = [
        "duration",
        "fwd_packet_count",
        "bwd_packet_count",
        "total_fwd_bytes",
        "total_bwd_bytes",
        "fwd_packet_len_max",
        "fwd_packet_len_min",
        "fwd_packet_len_mean",
        "fwd_packet_len_std",
        "bwd_packet_len_max",
        "bwd_packet_len_min",
        "bwd_packet_len_mean",
        "bwd_packet_len_std",
        "flow_bytes_per_sec",
        "flow_packets_per_sec",
        "fwd_packets_per_sec",
        "bwd_packets_per_sec",
        "flow_iat_mean",
        "flow_iat_std",
        "flow_iat_max",
        "flow_iat_min",
        "fwd_iat_mean",
        "fwd_iat_std",
        "bwd_iat_mean",
        "bwd_iat_std",
        "fwd_psh_flags",
        "bwd_psh_flags",
        "fwd_urg_flags",
        "bwd_urg_flags",
        "fin_flag_count",
        "syn_flag_count",
        "rst_flag_count",
        "psh_flag_count",
        "ack_flag_count",
        "fwd_header_len",
        "bwd_header_len",
        "fwd_win_bytes",
        "bwd_win_bytes",
        "protocol",
    ]

    # ── Label encoding map (matches CICIDS2017 class names) ──────────────
    LABEL_MAP: ClassVar[dict[str, int]] = {
        "benign": 0,
        "normal": 0,
        "dos": 1,
        "dos hulk": 1,
        "dos goldeneye": 1,
        "dos slowloris": 1,
        "dos slowhttptest": 1,
        "heartbleed": 1,
        "probe": 2,
        "portscan": 2,
        "ftp-patator": 3,
        "ssh-patator": 3,
        "r2l": 3,
        "u2r": 4,
        "bot": 4,
        "infiltration": 4,
        "web attack": 4,
        "web attack – brute force": 4,
        "web attack – xss": 4,
        "web attack – sql injection": 4,
        "unknown": -1,
    }

    LABEL_NAMES: ClassVar[dict[int, str]] = {
        0: "Benign",
        1: "DoS",
        2: "Probe",
        3: "R2L",
        4: "U2R",
        -1: "Unknown",
    }

    def to_vector(self) -> np.ndarray:
        """Return a 1-D numpy array of ML features in canonical order."""
        return np.array([getattr(self, f) for f in self.ML_FEATURES], dtype=np.float32)

    def encode_label(self) -> None:
        """Populate label_encoded from label string."""
        key = self.label.lower().strip()
        self.label_encoded = self.LABEL_MAP.get(key, -1)
