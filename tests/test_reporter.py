from models.incident_reporter import IncidentReporter
from streaming.consumer import PredictionResult

FLOW = {
    "src_ip": "10.9.9.9",
    "dst_ip": "10.0.0.9",
    "src_port": 1,
    "dst_port": 80,
    "protocol": 6,
    "duration": 0.001,
    "fwd_packet_count": 1000,
    "bwd_packet_count": 0,
}
FEATS = {"syn_flag_count": 2.1, "flow_packets_per_sec": 1.4}


def _ok_post(url, payload, headers, timeout):
    return {"choices": [{"message": {"content": " LLM says hello "}}]}


def _boom(url, payload, headers, timeout):
    raise RuntimeError("down")


def test_template_backend_needs_no_llm():
    r = IncidentReporter("template").generate(FLOW, "DoS", 0.9, FEATS)
    assert r["backend"] == "template" and "DoS" in r["report"]


def test_ollama_used_when_up():
    r = IncidentReporter("auto", http_get=lambda u, t: True, http_post=_ok_post)
    out = r.generate(FLOW, "DoS", 0.9, FEATS)
    assert out["backend"] == "ollama" and out["report"] == "LLM says hello"


def test_auto_falls_back_to_nim_then_template():
    r = IncidentReporter(
        "auto", nim_key="k", http_get=lambda u, t: False, http_post=_ok_post
    )
    assert r.generate(FLOW, "DoS", 0.9, FEATS)["backend"] == "nim"
    r2 = IncidentReporter("auto", http_get=lambda u, t: False, http_post=_ok_post)
    assert r2.generate(FLOW, "DoS", 0.9, FEATS)["backend"] == "template"


def test_failing_backend_degrades_to_template():
    r = IncidentReporter("ollama", http_get=lambda u, t: True, http_post=_boom)
    assert r.generate(FLOW, "DoS", 0.9, FEATS)["backend"] == "template"


def test_prediction_result_new_fields_default():
    p = PredictionResult(label="Benign")
    assert p.top_features == {} and p.report == ""
