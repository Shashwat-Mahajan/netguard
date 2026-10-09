"""
capture/zeek_parser.py
Parses Zeek conn.log files into FlowFeatures objects.

Zeek writes one line per connection with tab-separated fields.
We read these files in two modes:
  1. Batch: parse a completed log file (for historical analysis / offline training).
  2. Watch: tail a live log file using watchdog, emitting flows as they are written.

Zeek conn.log field reference (subset used):
  ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p, proto,
  duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts,
  orig_ip_bytes, resp_ip_bytes, conn_state, missed_bytes, ...
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Generator

from config.logger import get_logger
from processor.feature_schema import FlowFeatures

logger = get_logger(__name__)

# Zeek conn.log uses '-' to represent null/missing values.
_ZEEK_NULL = {"-", "(empty)", ""}


def _zeek_float(val: str, default: float = 0.0) -> float:
    return float(val) if val not in _ZEEK_NULL else default


def _zeek_int(val: str, default: int = 0) -> int:
    return int(float(val)) if val not in _ZEEK_NULL else default


def _proto_to_int(proto: str) -> int:
    return {"tcp": 6, "udp": 17, "icmp": 1}.get(proto.lower(), 0)


def _row_to_features(row: dict[str, str]) -> FlowFeatures | None:
    """
    Convert one Zeek conn.log row to a FlowFeatures object.

    Zeek provides connection-level aggregates, not per-packet data, so
    some fine-grained features (IAT, per-packet lengths) are not available.
    We compute what we can; the rest remain at 0 and will be imputed during
    preprocessing if a model is trained on Zeek-sourced data.
    """
    try:
        duration = _zeek_float(row.get("duration", "-"))
        if duration < 0:
            duration = 0.0

        orig_pkts = _zeek_int(row.get("orig_pkts", "-"))
        resp_pkts = _zeek_int(row.get("resp_pkts", "-"))
        orig_bytes = _zeek_int(row.get("orig_bytes", "-"))
        resp_bytes = _zeek_int(row.get("resp_bytes", "-"))
        proto_str = row.get("proto", "")
        proto_int = _proto_to_int(proto_str)

        safe_dur = max(duration, 1e-6)
        total_bytes = orig_bytes + resp_bytes
        total_pkts = orig_pkts + resp_pkts

        f = FlowFeatures(
            src_ip=row.get("id.orig_h", ""),
            src_port=_zeek_int(row.get("id.orig_p", "0")),
            dst_ip=row.get("id.resp_h", ""),
            dst_port=_zeek_int(row.get("id.resp_p", "0")),
            protocol=proto_int,
            timestamp=_zeek_float(row.get("ts", "0")),
            duration=duration,
            fwd_packet_count=orig_pkts,
            bwd_packet_count=resp_pkts,
            total_fwd_bytes=orig_bytes,
            total_bwd_bytes=resp_bytes,
            # Rough mean packet sizes (no per-packet data in conn.log)
            fwd_packet_len_mean=orig_bytes / orig_pkts if orig_pkts > 0 else 0.0,
            bwd_packet_len_mean=resp_bytes / resp_pkts if resp_pkts > 0 else 0.0,
            # Rates
            flow_bytes_per_sec=total_bytes / safe_dur,
            flow_packets_per_sec=total_pkts / safe_dur,
            fwd_packets_per_sec=orig_pkts / safe_dur,
            bwd_packets_per_sec=resp_pkts / safe_dur,
            label="unknown",
        )
        f.encode_label()
        return f

    except (KeyError, ValueError, ZeroDivisionError) as exc:
        logger.debug("zeek_parser.row_error", error=str(exc))
        return None


class ZeekConnLogParser:
    """Parse Zeek conn.log files (batch and streaming modes)."""

    # Zeek log header lines start with '#'
    COMMENT_CHAR = "#"

    def parse_file(self, path: Path) -> Generator[FlowFeatures, None, None]:
        """
        Parse a complete conn.log file.
        Yields one FlowFeatures per valid row.
        """
        path = Path(path)
        if not path.exists():
            logger.error("zeek_parser.file_not_found", path=str(path))
            return

        fields: list[str] = []
        row_count = 0
        skip_count = 0

        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith(self.COMMENT_CHAR):
                    # Extract field names from the '#fields' header line
                    if line.startswith("#fields"):
                        fields = line.split("\t")[1:]  # skip '#fields' prefix
                    continue
                if not fields:
                    logger.warning("zeek_parser.no_fields_header", path=str(path))
                    continue

                values = line.split("\t")
                if len(values) != len(fields):
                    skip_count += 1
                    continue

                row = dict(zip(fields, values))
                feat = _row_to_features(row)
                if feat:
                    row_count += 1
                    yield feat
                else:
                    skip_count += 1

        logger.info(
            "zeek_parser.file_done",
            path=str(path),
            rows_parsed=row_count,
            rows_skipped=skip_count,
        )

    def watch_directory(
        self,
        directory: Path,
        on_flow: Callable[[FlowFeatures], None],
        poll_interval: float = 1.0,
    ) -> None:
        """
        Watch a directory for new Zeek conn.log entries.
        Tails the most recently modified conn.log, emitting flows
        line-by-line as Zeek writes them.

        Runs synchronously — call from a dedicated thread.
        """
        directory = Path(directory)
        logger.info("zeek_parser.watch_start", directory=str(directory))

        seen_files: set[Path] = set()
        file_positions: dict[Path, int] = {}

        while True:
            conn_logs = sorted(
                directory.glob("conn*.log"), key=lambda p: p.stat().st_mtime
            )
            if not conn_logs:
                time.sleep(poll_interval)
                continue

            for log_path in conn_logs:
                if log_path not in seen_files:
                    seen_files.add(log_path)
                    file_positions[log_path] = 0
                    logger.info("zeek_parser.new_log", path=str(log_path))

            # Read new lines from the current (most recently modified) log
            current = conn_logs[-1]
            pos = file_positions.get(current, 0)

            try:
                with current.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    new_lines = fh.readlines()
                    file_positions[current] = fh.tell()

                fields: list[str] = []
                for line in new_lines:
                    line = line.rstrip("\n")
                    if not line or line.startswith(self.COMMENT_CHAR):
                        if line.startswith("#fields"):
                            fields = line.split("\t")[1:]
                        continue
                    if not fields:
                        continue
                    values = line.split("\t")
                    if len(values) != len(fields):
                        continue
                    row = dict(zip(fields, values))
                    feat = _row_to_features(row)
                    if feat:
                        on_flow(feat)

            except OSError as exc:
                logger.warning(
                    "zeek_parser.read_error", path=str(current), error=str(exc)
                )

            time.sleep(poll_interval)
