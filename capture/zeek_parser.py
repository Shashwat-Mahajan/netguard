"""
capture/zeek_parser.py
Parses Zeek logs: conn.log -> FlowFeatures, notice.log -> ZeekNotice.

Both Zeek output styles are supported and detected line by line:
  * JSON lines (LogAscii::use_json=T, the default in the Zeek 9 Docker image)
  * Classic tab-separated files with a '#fields' header

Zeek conn.log fields used:
  ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p, proto,
  duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts

orig = the connection initiator, which matches our forward direction.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.logger import get_logger
from processor.feature_schema import FlowFeatures

logger = get_logger(__name__)

# Zeek uses '-' (and '(empty)') for null in tab-separated logs.
_ZEEK_NULL = {"-", "(empty)", ""}
COMMENT_CHAR = "#"


def _is_null(val: Any) -> bool:
    return val is None or (isinstance(val, str) and val in _ZEEK_NULL)


def _zeek_float(val: Any, default: float = 0.0) -> float:
    return default if _is_null(val) else float(val)


def _zeek_int(val: Any, default: int = 0) -> int:
    return default if _is_null(val) else int(float(val))


def _proto_to_int(proto: Any) -> int:
    return {"tcp": 6, "udp": 17, "icmp": 1, "icmp6": 58}.get(
        str(proto or "").lower(), 0
    )


def _parse_line(
    line: str, fields: list[str]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Turn one log line into a row dict. Returns (row or None, fields)."""
    line = line.rstrip("\r\n")
    if not line:
        return None, fields
    if line.startswith("{"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return None, fields
        return (row if isinstance(row, dict) else None), fields
    if line.startswith(COMMENT_CHAR):
        if line.startswith("#fields"):
            fields = line.split("\t")[1:]
        return None, fields
    if not fields:
        return None, fields
    values = line.split("\t")
    if len(values) != len(fields):
        return None, fields
    return dict(zip(fields, values, strict=True)), fields


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    fields: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            row, fields = _parse_line(line, fields)
            if row is not None:
                yield row


def _read_new_lines(path: Path, pos: int) -> tuple[list[str], int]:
    """Read complete new lines after byte offset pos (ignores a half-written line)."""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        if pos > fh.tell():  # file was replaced/rotated
            pos = 0
        fh.seek(pos)
        data = fh.read()
    end = data.rfind(b"\n")
    if end == -1:
        return [], pos
    chunk = data[: end + 1]
    return chunk.decode("utf-8", errors="replace").splitlines(), pos + len(chunk)


def _wait(stop: threading.Event | None, seconds: float) -> None:
    if stop is None:
        time.sleep(seconds)
    else:
        stop.wait(seconds)


# --------------------------------------------------------------------------
# conn.log -> FlowFeatures
# --------------------------------------------------------------------------
def _row_to_features(row: dict[str, Any]) -> FlowFeatures | None:
    """
    Convert one Zeek conn.log row to a FlowFeatures object.

    conn.log has connection-level totals only, so per-packet features (IAT,
    packet length spread) stay at 0.
    """
    try:
        duration = max(_zeek_float(row.get("duration")), 0.0)
        orig_pkts = _zeek_int(row.get("orig_pkts"))
        resp_pkts = _zeek_int(row.get("resp_pkts"))
        orig_bytes = _zeek_int(row.get("orig_bytes"))
        resp_bytes = _zeek_int(row.get("resp_bytes"))

        safe_dur = max(duration, 1e-6)
        total_bytes = orig_bytes + resp_bytes
        total_pkts = orig_pkts + resp_pkts

        f = FlowFeatures(
            src_ip=str(row.get("id.orig_h") or ""),
            src_port=_zeek_int(row.get("id.orig_p")),
            dst_ip=str(row.get("id.resp_h") or ""),
            dst_port=_zeek_int(row.get("id.resp_p")),
            protocol=_proto_to_int(row.get("proto")),
            timestamp=_zeek_float(row.get("ts")),
            duration=duration,
            fwd_packet_count=orig_pkts,
            bwd_packet_count=resp_pkts,
            total_fwd_bytes=orig_bytes,
            total_bwd_bytes=resp_bytes,
            fwd_packet_len_mean=orig_bytes / orig_pkts if orig_pkts > 0 else 0.0,
            bwd_packet_len_mean=resp_bytes / resp_pkts if resp_pkts > 0 else 0.0,
            flow_bytes_per_sec=total_bytes / safe_dur,
            flow_packets_per_sec=total_pkts / safe_dur,
            fwd_packets_per_sec=orig_pkts / safe_dur,
            bwd_packets_per_sec=resp_pkts / safe_dur,
            label="unknown",
        )
        f.encode_label()
        return f
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        logger.debug("zeek_parser.row_error", error=str(exc))
        return None


class ZeekConnLogParser:
    """Parse Zeek conn.log files (batch and tail modes)."""

    def parse_file(self, path: Path) -> Iterator[FlowFeatures]:
        path = Path(path)
        if not path.exists():
            logger.error("zeek_parser.file_not_found", path=str(path))
            return

        parsed = 0
        skipped = 0
        for row in _iter_rows(path):
            feat = _row_to_features(row)
            if feat:
                parsed += 1
                yield feat
            else:
                skipped += 1
        logger.info(
            "zeek_parser.file_done",
            path=str(path),
            rows_parsed=parsed,
            rows_skipped=skipped,
        )

    def watch_directory(
        self,
        directory: Path,
        on_flow: Callable[[FlowFeatures], None],
        poll_interval: float = 1.0,
        stop: threading.Event | None = None,
    ) -> None:
        """
        Tail the most recently modified conn*.log in a directory and call
        on_flow for each new flow. Runs until `stop` is set (or forever if
        stop is None), so call it from a dedicated thread.
        """
        directory = Path(directory)
        logger.info("zeek_parser.watch_start", directory=str(directory))

        positions: dict[Path, int] = {}
        fields_by_file: dict[Path, list[str]] = {}

        while stop is None or not stop.is_set():
            try:
                logs = sorted(
                    directory.glob("conn*.log"), key=lambda p: p.stat().st_mtime
                )
            except OSError:
                logs = []

            if logs:
                current = logs[-1]
                if current not in positions:
                    positions[current] = 0
                    fields_by_file[current] = []
                    logger.info("zeek_parser.new_log", path=str(current))
                try:
                    lines, positions[current] = _read_new_lines(
                        current, positions[current]
                    )
                except OSError as exc:
                    logger.warning(
                        "zeek_parser.read_error", path=str(current), error=str(exc)
                    )
                    lines = []

                fields = fields_by_file[current]
                for line in lines:
                    row, fields = _parse_line(line, fields)
                    if row is None:
                        continue
                    feat = _row_to_features(row)
                    if feat:
                        on_flow(feat)
                fields_by_file[current] = fields

            _wait(stop, poll_interval)


# --------------------------------------------------------------------------
# notice.log -> ZeekNotice
# --------------------------------------------------------------------------
_NOTICE_SEVERITY = {
    "SSH::Password_Guessing": "high",
    "SSH::Login_By_Password_Guesser": "high",
    "FTP::Bruteforcing": "high",
    "HTTP::SQL_Injection_Attacker": "high",
    "HTTP::SQL_Injection_Victim": "medium",
    "Scan::Port_Scan": "medium",
    "Scan::Address_Scan": "medium",
    "SSL::Certificate_Expired": "medium",
    "SSL::Certificate_Expires_Soon": "low",
    "SSL::Invalid_Server_Cert": "low",
    "Traceroute::Detected": "low",
}


def notice_severity(note: str, msg: str = "") -> str:
    """
    Map a Zeek notice type to high/medium/low. Unknown types are medium.

    Invalid_Server_Cert is low by default: it also fires for every
    certificate when Zeek has no CA store. A self-signed certificate is
    more interesting, so that one is medium.
    """
    if note == "SSL::Invalid_Server_Cert":
        m = msg.lower()
        if "self" in m and "signed" in m:
            return "medium"
    return _NOTICE_SEVERITY.get(note, "medium")


@dataclass(frozen=True)
class ZeekNotice:
    timestamp: float
    note: str
    msg: str
    sub: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: str
    uid: str
    severity: str

    @property
    def signature(self) -> str:
        return f"Zeek {self.note}"


def _row_to_notice(row: dict[str, Any]) -> ZeekNotice | None:
    note = str(row.get("note") or "")
    if not note:
        return None
    try:
        msg = str(row.get("msg") or "")
        # Connection fields are absent for some notices (e.g. scans); then
        # Zeek only gives src / dst / p.
        dst_port_raw = row.get("id.resp_p")
        if _is_null(dst_port_raw):
            dst_port_raw = row.get("p")
        return ZeekNotice(
            timestamp=_zeek_float(row.get("ts")),
            note=note,
            msg=msg,
            sub=str(row.get("sub") or ""),
            src_ip=str(row.get("id.orig_h") or row.get("src") or ""),
            src_port=_zeek_int(row.get("id.orig_p")),
            dst_ip=str(row.get("id.resp_h") or row.get("dst") or ""),
            dst_port=_zeek_int(dst_port_raw),
            proto=str(row.get("proto") or ""),
            uid=str(row.get("uid") or ""),
            severity=notice_severity(note, msg),
        )
    except (ValueError, TypeError) as exc:
        logger.debug("zeek_parser.notice_error", error=str(exc))
        return None


class ZeekNoticeParser:
    """Parse a Zeek notice.log file."""

    def parse_file(self, path: Path) -> Iterator[ZeekNotice]:
        path = Path(path)
        if not path.exists():
            logger.error("zeek_parser.file_not_found", path=str(path))
            return
        for row in _iter_rows(path):
            notice = _row_to_notice(row)
            if notice:
                yield notice
