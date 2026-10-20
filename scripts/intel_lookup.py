"""python scripts/intel_lookup.py [ip ...]   (no args: shows a few IPs from the Feodo list)"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming.intel import ThreatIntel  # noqa: E402

ti = ThreatIntel.from_settings()
print("Feodo entries loaded:", ti.feodo_size())
ips = sys.argv[1:] or list((ti._feodo or {}).keys())[:3] or ["8.8.8.8"]
for ip in ips:
    print(
        ip,
        json.dumps(ti.enrich_ip(ip)),
        "HIT" if ti.is_hit(ti.enrich_ip(ip)) else "clean",
    )
