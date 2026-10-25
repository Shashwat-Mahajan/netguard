from types import SimpleNamespace

from capture.rules import RuleAlert
from streaming.thehive import TheHiveClient


def _client(calls, code=201, **kw):
    def post(url, payload, headers, timeout):
        calls.append((url, payload, headers))
        return code

    return TheHiveClient("http://hive:9001", "KEY", http_post=post, **kw)


def _rule(sev="high", hit=False):
    return RuleAlert(
        "port_scan",
        "Probe",
        2,
        sev,
        "10.8.8.8",
        "10.0.0.9",
        50,
        "50 ports",
        1000.0,
        intel_hit=hit,
    )


def test_disabled_without_url_or_key():
    c = TheHiveClient("", "")
    assert not c.enabled and c.send_rule_alert(_rule()) is False


def test_high_rule_alert_is_sent_with_auth():
    calls = []
    c = _client(calls)
    assert c.send_rule_alert(_rule()) is True
    url, payload, headers = calls[0]
    assert url.endswith("/api/v1/alert") and headers["Authorization"] == "Bearer KEY"
    assert payload["severity"] == 3 and "port_scan" in payload["tags"]


def test_below_threshold_skipped_unless_intel_hit():
    calls = []
    c = _client(calls)
    assert c.send_rule_alert(_rule("medium")) is False and calls == []
    assert c.send_rule_alert(_rule("medium", hit=True)) is True


def test_duplicates_sent_once():
    calls = []
    c = _client(calls)
    a = _rule()
    assert c.send_rule_alert(a) and not c.send_rule_alert(a)
    assert len(calls) == 1


def test_unreachable_thehive_never_raises():
    def boom(*a):
        raise RuntimeError("down")

    c = TheHiveClient("http://hive:9001", "KEY", http_post=boom)
    assert c.send_rule_alert(_rule()) is False and c.failed == 1


def test_ml_alert_only_for_confirmed_threats():
    calls = []
    c = _client(calls)
    base = dict(
        flow_id="f1",
        label="DoS",
        confidence=0.95,
        src_ip="1.1.1.1",
        src_port=1,
        dst_ip="2.2.2.2",
        dst_port=80,
        is_anomaly=False,
        timestamp=1000.0,
        top_features={"syn_flag_count": 2.0},
        report="",
    )
    threat = SimpleNamespace(is_threat=lambda: True, **base)
    benign = SimpleNamespace(is_threat=lambda: False, **base)
    assert c.send_ml_alert(benign) is False
    assert (
        c.send_ml_alert(threat) is True
        and "syn_flag_count" in calls[0][1]["description"]
    )
