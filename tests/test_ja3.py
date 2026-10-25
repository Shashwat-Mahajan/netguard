import json

from streaming.ja3 import Ja3Blocklist, ja3_alerts_from_eve, parse_blocklist

H = "a" * 32
CSV = f"# header\n{H},2020-01-01,2020-02-01,Emotet C2\nnotahash,x\n"


def _bl(tmp_path):
    return Ja3Blocklist("u", tmp_path, fetch=lambda u: CSV)


def test_parse_ignores_comments_and_bad_rows():
    assert parse_blocklist(CSV) == {H: "Emotet C2"}


def test_tls_event_with_listed_ja3_alerts_once(tmp_path):
    ev = {
        "event_type": "tls",
        "timestamp": "2026-10-25T07:00:00+0000",
        "src_ip": "10.0.0.5",
        "dest_ip": "45.1.1.1",
        "tls": {"ja3": {"hash": H}},
    }
    p = tmp_path / "eve.json"
    p.write_text(
        "\n".join(
            [json.dumps(ev)] * 2
            + [json.dumps({**ev, "tls": {"ja3": {"hash": "b" * 32}}})]
        )
    )
    out = ja3_alerts_from_eve(p, _bl(tmp_path))
    assert len(out) == 1 and out[0].source == "ja3" and "Emotet" in out[0].detail


def test_offline_uses_cache(tmp_path):
    _bl(tmp_path).size()

    def boom(u):
        raise RuntimeError("offline")

    assert Ja3Blocklist("u", tmp_path, refresh_hours=0.00001, fetch=boom).size() == 1
