from fastapi.testclient import TestClient

from api.main import app
from api.routers.alerts import unify_alerts


def test_unify_orders_newest_first_and_tags_sources():
    ml = [
        {
            "flow_id": "f1",
            "timestamp": 100,
            "label": "DoS",
            "confidence": 0.9,
            "src_ip": "1.1.1.1",
            "dst_ip": "2.2.2.2",
        }
    ]
    sig = [
        {
            "alert_id": "a1",
            "timestamp": 200,
            "rule": "port_scan",
            "label": "Probe",
            "severity": "high",
            "src_ip": "3.3.3.3",
            "dst_ip": "2.2.2.2",
            "detail": "x",
        }
    ]
    out = unify_alerts(ml, sig, 10)
    assert [a["source"] for a in out] == ["signature", "ml"]
    assert out[0]["rule"] == "port_scan" and out[1]["rule"] == "classifier"


def test_unify_respects_limit():
    sig = [{"alert_id": str(i), "timestamp": i} for i in range(10)]
    assert len(unify_alerts([], sig, 3)) == 3


def test_alert_endpoints_require_auth():
    c = TestClient(app)
    assert c.get("/api/v1/rule-alerts").status_code == 401
    assert c.get("/api/v1/alerts/all").status_code == 401
