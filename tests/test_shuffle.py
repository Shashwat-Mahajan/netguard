from types import SimpleNamespace

from streaming.shuffle import SoarClient


def _a(sev="high", aid="1", hit=False):
    return SimpleNamespace(
        alert_id=aid, source="signature", rule="syn_flood", label="DoS",
        severity=sev, src_ip="1.1.1.1", dst_ip="2.2.2.2", flow_count=5,
        detail="d", timestamp=1.0, intel="", intel_hit=hit,
    )  # fmt: skip


def test_disabled_without_url():
    c = SoarClient(http_post=lambda *a: 200)
    assert not c.enabled and c.send_rule_alert(_a()) is False


def test_high_alert_is_posted_once():
    calls = []
    c = SoarClient("http://x", http_post=lambda u, p, t: calls.append(p) or 200)
    assert c.send_rule_alert(_a()) is True
    assert c.send_rule_alert(_a()) is False
    assert len(calls) == 1 and calls[0]["rule"] == "syn_flood"


def test_low_severity_skipped_unless_intel_hit():
    c = SoarClient("http://x", http_post=lambda *a: 200)
    assert c.send_rule_alert(_a("low", "2")) is False
    assert c.send_rule_alert(_a("low", "3", hit=True)) is True


def test_failure_never_raises_and_allows_retry():
    def boom(*a):
        raise RuntimeError("down")

    c = SoarClient("http://x", http_post=boom)
    assert c.send_rule_alert(_a()) is False and c.failed == 1
    c._post = lambda *a: 200
    assert c.send_rule_alert(_a()) is True
