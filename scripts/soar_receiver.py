"""python scripts/soar_receiver.py [port]
Tiny SOAR-style playbook: receives a NetGuard alert (JSON POST), adds a
priority from threat intel and creates a TheHive alert. Local lab use only."""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming.thehive import TheHiveClient  # noqa: E402

CLIENT = TheHiveClient.from_settings()


def handle(alert: dict) -> dict:
    hit = bool(alert.get("intel_hit"))
    priority = "critical" if hit else alert.get("severity", "low")
    ok = False
    if CLIENT.enabled:
        a = type("A", (), alert)()
        a.intel_hit = hit
        ok = CLIENT.send_rule_alert(a)
    return {"priority": priority, "thehive_created": ok}


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            result = handle(json.loads(self.rfile.read(n)))
            body, code = json.dumps(result).encode(), 200
        except Exception as exc:
            body, code = json.dumps({"error": str(exc)}).encode(), 400
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"SOAR receiver on http://127.0.0.1:{port}")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
