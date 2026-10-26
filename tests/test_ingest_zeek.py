import json
from types import SimpleNamespace

from streaming.zeek import ingest_notices, notice_to_alert, parse_notice_file


def _notice(**kw):
    base = dict(
        timestamp=1700000000.0,
        note="SSL::Invalid_Server_Cert",
        msg="SSL certificate validation failed with "
        "(unable to get local issuer certificate)",
        sub="CN=example.com",
        src_ip="10.0.0.5",
        dst_ip="20.1.2.3",
        severity="low",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class FakeWriter:
    def __init__(self):
        self.written = 0
        self.alerts = []

    def write(self, a):
        self.alerts.append(a)
        self.written += 1


def test_ssl_invalid_cert_stays_low_and_unknown():
    a = notice_to_alert(_notice())
    assert a.source == "zeek"
    assert a.rule == "zeek:SSL::Invalid_Server_Cert"
    assert a.severity == "low"
    assert (a.label, a.label_encoded) == ("Unknown", -1)
    assert "issuer certificate" in a.detail


def test_brute_force_maps_to_r2l_high():
    a = notice_to_alert(_notice(note="SSH::Password_Guessing", severity="high"))
    assert a.severity == "high"
    assert a.label == "Unknown"
    a = notice_to_alert(_notice(note="FTP::Bruteforcing", severity="high"))
    assert (a.label, a.label_encoded) == ("R2L", 3)


def test_scan_maps_to_probe():
    a = notice_to_alert(_notice(note="Scan::Port_Scan", severity="medium"))
    assert (a.label, a.label_encoded) == ("Probe", 2)
    assert a.severity == "medium"


def test_missing_conn_fields_and_bad_severity_are_safe():
    a = notice_to_alert(_notice(src_ip=None, dst_ip="", severity="weird"))
    assert a.src_ip == ""
    assert a.dst_ip == ""
    assert a.severity == "low"


def test_real_parser_roundtrip(tmp_path):
    p = tmp_path / "notice.log"
    row = {
        "ts": 1700000000.5,
        "uid": "C1",
        "id.orig_h": "10.0.0.5",
        "id.orig_p": 50000,
        "id.resp_h": "20.1.2.3",
        "id.resp_p": 443,
        "proto": "tcp",
        "note": "SSL::Invalid_Server_Cert",
        "msg": "SSL certificate validation failed with "
        "(unable to get local issuer certificate)",
        "sub": "CN=example.com",
    }
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    alerts = parse_notice_file(p)
    assert len(alerts) == 2
    assert all(a.source == "zeek" and a.severity == "low" for a in alerts)
    assert alerts[0].src_ip == "10.0.0.5"


def test_ingest_counts_and_restamp(tmp_path):
    p = tmp_path / "notice.log"
    row = {"ts": 1.0, "note": "SSL::Invalid_Server_Cert", "msg": "x"}
    p.write_text(json.dumps(row) + "\n")
    w = FakeWriter()
    written, found = ingest_notices(p, w, restamp=True)
    assert (written, found) == (1, 1)
    assert w.alerts[0].timestamp > 1_000_000_000


def test_missing_file_writes_nothing(tmp_path):
    w = FakeWriter()
    assert ingest_notices(tmp_path / "nope.log", w) == (0, 0)
    assert w.written == 0
