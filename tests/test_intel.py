import json
from types import SimpleNamespace

from capture.pipeline import FlowPipeline
from capture.rules import SignatureEngine
from streaming.intel import ThreatIntel

C2_IP = "45.33.32.156"
FEODO = [{"ip_address": C2_IP, "malware": "Emotet", "port": 443, "status": "online"}]


def _ti(tmp_path, calls=None, **kw):
    def get(url, headers=None):
        if calls is not None:
            calls.append(url)
        if "feodo" in url:
            return FEODO
        raise RuntimeError("unexpected")

    return ThreatIntel("http://feodo/x", tmp_path, http_get=get, **kw)


def test_private_ips_are_never_looked_up(tmp_path):
    calls = []
    ti = _ti(tmp_path, calls, abuseipdb_key="k")
    assert ti.enrich_ip("10.0.0.5") == {} and ti.enrich_ip("192.168.1.1") == {}
    assert calls == []


def test_feodo_hit(tmp_path):
    ti = _ti(tmp_path)
    info, hit = ti.enrich_alert(C2_IP, "10.0.0.1")
    assert hit and info[C2_IP]["feodo"]["malware"] == "Emotet"


def test_feodo_uses_disk_cache_when_download_fails(tmp_path):
    _ti(tmp_path).feodo_size()  # writes cache file

    def boom(url, headers=None):
        raise RuntimeError("offline")

    ti = ThreatIntel("http://feodo/x", tmp_path, refresh_hours=0.0001, http_get=boom)
    assert ti.feodo_size() == 1


def test_no_keys_means_no_api_calls(tmp_path):
    calls = []
    _ti(tmp_path, calls).enrich_ip("45.33.32.157")
    assert all("abuseipdb" not in c and "otx" not in c for c in calls)


def test_abuseipdb_score_counts_as_hit(tmp_path):
    def get(url, headers=None):
        if "feodo" in url:
            return []
        return {
            "data": {"abuseConfidenceScore": 90, "totalReports": 5, "countryCode": "XX"}
        }

    ti = ThreatIntel("http://feodo/x", tmp_path, abuseipdb_key="k", http_get=get)
    _, hit = ti.enrich_alert("198.51.100.0")  # documentation range is not global
    assert hit is False
    _, hit = ti.enrich_alert("8.8.4.4")
    assert hit is True


def test_pipeline_attaches_intel_to_alert(tmp_path):
    class NoModels:
        def predict(self, f):
            return None

    class W:
        def __init__(self):
            self.rows = []

        def write(self, a):
            self.rows.append(a)

    def make_flow(i):
        return SimpleNamespace(
            timestamp=100 + i * 0.001,
            src_ip=C2_IP,
            dst_ip="10.0.0.9",
            dst_port=80,
            syn_flag_count=1,
            ack_flag_count=0,
            bwd_packet_count=0,
        )

    w = W()
    p = FlowPipeline(
        NoModels(),
        rules=SignatureEngine(synflood_flows=10),
        alert_writer=w,
        intel=_ti(tmp_path),
    )
    for i in range(20):
        p.handle_flow(make_flow(i))
    assert w.rows[0].intel_hit is True
    assert json.loads(w.rows[0].intel)[C2_IP]["feodo"]["malware"] == "Emotet"
