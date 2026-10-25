import json
import threading
from pathlib import Path

import pytest

from capture.zeek_parser import (
    ZeekConnLogParser,
    ZeekNoticeParser,
    notice_severity,
)

SSL_CONN = {
    "ts": 1792955671.996531,
    "uid": "CxVKf14HVEUeZkR5Rk",
    "id.orig_h": "192.168.2.5",
    "id.orig_p": 64393,
    "id.resp_h": "57.155.101.212",
    "id.resp_p": 443,
    "proto": "tcp",
    "service": "ssl",
    "duration": 0.4615650177001953,
    "orig_bytes": 1911,
    "resp_bytes": 4687,
    "conn_state": "SF",
    "orig_pkts": 12,
    "resp_pkts": 12,
}

# Half-open IPv6 flow: no duration, no byte counts, no service.
S0_CONN = {
    "ts": 1792955690.809223,
    "uid": "CV0kw71PRpdI84SK9f",
    "id.orig_h": "fe80::9f2b:42cb:870e:ee69",
    "id.orig_p": 64410,
    "id.resp_h": "2606:4700:10::ac42:93f3",
    "id.resp_p": 443,
    "proto": "tcp",
    "conn_state": "S0",
    "orig_pkts": 1,
    "resp_pkts": 0,
}

NOTICE = {
    "ts": 1792955672.195328,
    "uid": "CxVKf14HVEUeZkR5Rk",
    "id.orig_h": "192.168.2.5",
    "id.orig_p": 64393,
    "id.resp_h": "57.155.101.212",
    "id.resp_p": 443,
    "proto": "tcp",
    "note": "SSL::Invalid_Server_Cert",
    "msg": "SSL certificate validation failed with (unable to get local issuer certificate)",
    "sub": "CN=settings-win.data.microsoft.com",
    "src": "192.168.2.5",
    "dst": "57.155.101.212",
    "p": 443,
}


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_json_conn_log_parses(tmp_path):
    log = tmp_path / "conn.log"
    _write(log, [SSL_CONN, S0_CONN])
    flows = list(ZeekConnLogParser().parse_file(log))
    assert len(flows) == 2

    f = flows[0]
    assert f.src_ip == "192.168.2.5"
    assert f.dst_port == 443
    assert f.protocol == 6
    assert f.fwd_packet_count == 12
    assert f.total_bwd_bytes == 4687
    assert f.duration == pytest.approx(0.4615650177)

    s0 = flows[1]
    assert s0.duration == 0.0
    assert s0.fwd_packet_count == 1
    assert s0.bwd_packet_count == 0
    assert s0.total_fwd_bytes == 0


def test_tsv_conn_log_parses(tmp_path):
    log = tmp_path / "conn.log"
    header = (
        "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\t"
        "proto\tduration\torig_bytes\tresp_bytes\torig_pkts\tresp_pkts"
    )
    row = "1792955658.0\tC1\t10.0.0.1\t5000\t10.0.0.2\t80\ttcp\t-\t-\t-\t1\t0"
    log.write_text("#separator \\x09\n" + header + "\n" + row + "\n", encoding="utf-8")
    flows = list(ZeekConnLogParser().parse_file(log))
    assert len(flows) == 1
    assert flows[0].dst_port == 80
    assert flows[0].protocol == 6
    assert flows[0].duration == 0.0


def test_missing_file_yields_nothing(tmp_path):
    assert list(ZeekConnLogParser().parse_file(tmp_path / "nope.log")) == []


def test_bad_lines_are_skipped(tmp_path):
    log = tmp_path / "conn.log"
    log.write_text(
        "this is not json\n{broken json\n" + json.dumps(SSL_CONN) + "\n",
        encoding="utf-8",
    )
    assert len(list(ZeekConnLogParser().parse_file(log))) == 1


def test_notice_parses(tmp_path):
    log = tmp_path / "notice.log"
    _write(log, [NOTICE])
    notices = list(ZeekNoticeParser().parse_file(log))
    assert len(notices) == 1
    n = notices[0]
    assert n.note == "SSL::Invalid_Server_Cert"
    assert n.src_ip == "192.168.2.5"
    assert n.dst_port == 443
    assert n.severity == "low"
    assert n.signature == "Zeek SSL::Invalid_Server_Cert"


def test_notice_severity_map():
    assert notice_severity("SSH::Password_Guessing") == "high"
    assert notice_severity("Some::Unknown_Notice") == "medium"
    assert (
        notice_severity(
            "SSL::Invalid_Server_Cert",
            "validation failed with (self signed certificate)",
        )
        == "medium"
    )


def test_notice_without_conn_fields(tmp_path):
    log = tmp_path / "notice.log"
    _write(
        log,
        [
            {
                "ts": 1.0,
                "note": "Scan::Port_Scan",
                "msg": "1.2.3.4 scanned at least 50 unique ports",
                "src": "1.2.3.4",
                "dst": "5.6.7.8",
                "p": 22,
            }
        ],
    )
    n = next(iter(ZeekNoticeParser().parse_file(log)))
    assert n.src_ip == "1.2.3.4"
    assert n.dst_ip == "5.6.7.8"
    assert n.dst_port == 22
    assert n.severity == "medium"


def test_watch_directory_emits_and_stops(tmp_path):
    _write(tmp_path / "conn.log", [SSL_CONN])
    stop = threading.Event()
    flows = []

    def on_flow(f):
        flows.append(f)
        stop.set()

    t = threading.Thread(
        target=ZeekConnLogParser().watch_directory,
        args=(tmp_path, on_flow, 0.01, stop),
        daemon=True,
    )
    t.start()
    t.join(timeout=5)
    stop.set()
    assert not t.is_alive()
    assert len(flows) == 1
