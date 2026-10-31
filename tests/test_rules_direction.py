"""tests/test_rules_direction.py — reversed (server-reply) flows are not scans."""

from __future__ import annotations

from types import SimpleNamespace

from capture.rules import SignatureEngine


def _flow(ts, sport, dport, src="10.0.0.9", dst="10.9.9.9"):
    return SimpleNamespace(
        timestamp=ts,
        src_ip=src,
        dst_ip=dst,
        src_port=sport,
        dst_port=dport,
        syn_flag_count=1,
        ack_flag_count=1,
        bwd_packet_count=1,
    )


def test_reversed_reply_flows_do_not_look_like_a_scan():
    e = SignatureEngine()
    alerts = [
        a for p in range(2000, 2200) for a in e.observe(_flow(100 + p * 0.001, 22, p))
    ]
    assert [a.rule for a in alerts] == []


def test_real_scan_from_ephemeral_ports_still_fires():
    e = SignatureEngine()
    alerts = [
        a
        for p in range(1, 101)
        for a in e.observe(
            _flow(100 + p * 0.01, 50000, p, src="10.9.9.9", dst="10.0.0.9")
        )
    ]
    assert [a.rule for a in alerts] == ["port_scan"]
