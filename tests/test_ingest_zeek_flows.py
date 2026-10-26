import json
from types import SimpleNamespace

from capture.pipeline import FlowPipeline
from streaming.zeek import ingest_flows


def _write_conn(path, n=2):
    rows = []
    for i in range(n):
        rows.append(
            {
                "ts": 1700000000.0 + i,
                "uid": f"C{i}",
                "id.orig_h": "10.0.0.5",
                "id.orig_p": 50000 + i,
                "id.resp_h": "20.1.2.3",
                "id.resp_p": 443,
                "proto": "tcp",
                "service": "ssl",
                "duration": 1.5,
                "orig_bytes": 500,
                "resp_bytes": 3000,
                "conn_state": "SF",
                "orig_pkts": 8,
                "resp_pkts": 9,
                "orig_ip_bytes": 900,
                "resp_ip_bytes": 3500,
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class CollectingPipeline:
    def __init__(self, fail_first=False):
        self.flows = []
        self.fail_first = fail_first

    def handle_flow(self, flow):
        if self.fail_first and not self.flows and not hasattr(self, "tried"):
            self.tried = True
            raise RuntimeError("boom")
        self.flows.append(flow)


class NoModelEngine:
    def predict(self, flow):
        return None


def test_flows_are_sent_to_pipeline(tmp_path):
    p = tmp_path / "conn.log"
    _write_conn(p, 2)
    pipe = CollectingPipeline()
    assert ingest_flows(p, pipe) == 2
    assert len(pipe.flows) == 2


def test_missing_conn_log_sends_nothing(tmp_path):
    pipe = CollectingPipeline()
    assert ingest_flows(tmp_path / "nope.log", pipe) == 0
    assert pipe.flows == []


def test_one_failing_flow_does_not_stop_the_rest(tmp_path):
    p = tmp_path / "conn.log"
    _write_conn(p, 3)
    pipe = CollectingPipeline(fail_first=True)
    assert ingest_flows(p, pipe) == 2
    assert len(pipe.flows) == 2


def test_real_flow_pipeline_accepts_zeek_flows(tmp_path):
    p = tmp_path / "conn.log"
    _write_conn(p, 2)
    pipe = FlowPipeline(NoModelEngine(), restamp=True)
    assert ingest_flows(p, pipe) == 2
    assert pipe.summary["flows_seen"] == 2
    assert pipe.summary["dropped"] == 2
    assert isinstance(SimpleNamespace(), object)
