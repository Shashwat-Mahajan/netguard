import json

from streaming.suricata import parse_eve_line

ALERT = {
    "timestamp": "2026-10-25T07:37:33.123456+0000",
    "event_type": "alert",
    "src_ip": "10.9.9.9",
    "dest_ip": "10.0.0.9",
    "alert": {
        "signature_id": 9000001,
        "signature": "NETGUARD-TEST SYN flood",
        "category": "Attempted Denial of Service",
        "severity": 1,
    },
}


def test_alert_is_parsed():
    a = parse_eve_line(json.dumps(ALERT))
    assert a.source == "suricata" and a.rule == "suricata:9000001"
    assert a.label == "DoS" and a.severity == "high" and a.src_ip == "10.9.9.9"


def test_non_alert_and_garbage_are_skipped():
    assert parse_eve_line(json.dumps({"event_type": "flow"})) is None
    assert parse_eve_line("not json") is None
    assert parse_eve_line(json.dumps({"event_type": "alert"})) is None


def test_unknown_category_gets_unknown_label():
    e = json.loads(json.dumps(ALERT))
    e["alert"].update(category="Misc activity", signature="odd thing")
    assert parse_eve_line(json.dumps(e)).label == "Unknown"
