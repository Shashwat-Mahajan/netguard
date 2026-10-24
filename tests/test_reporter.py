from models.incident_reporter import IncidentReporter
from streaming.consumer import PredictionResult
from models.incident_reporter import build_prompt, ungrounded_numbers

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


def test_prompt_contains_real_feature_values():
    p = build_prompt(
        FLOW, "DoS", 0.9, {"syn_flag_count": 2.1}, {"syn_flag_count": 1000.0}
    )
    assert "value=1000" in p


def test_invented_numbers_are_detected():
    allowed = [1000.0, 90.0, 0.9]
    assert ungrounded_numbers("1000 SYN packets, 90% confidence", allowed) == []
    assert ungrounded_numbers("a delay of 382 ms from 10.0.0.5", allowed) == ["382"]


def test_report_with_invented_number_gets_a_note():
    def post(url, payload, headers, timeout):
        return {"choices": [{"message": {"content": "Latency was 382 ms."}}]}

    r = IncidentReporter("auto", http_get=lambda u, t: True, http_post=post)
    out = r.generate(FLOW, "DoS", 0.9, FEATS, {"syn_flag_count": 1000.0})
    assert (
        out["ungrounded"] == ["382"] and "not found in the alert data" in out["report"]
    )
