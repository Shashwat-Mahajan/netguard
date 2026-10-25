"""python scripts/ingest_suricata.py [data/suricata/eve.json] [--restamp]
Loads Suricata alerts into netguard.alerts. --restamp sets them to now (for replayed pcaps).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming.alert_writer import ch_alert_writer  # noqa: E402
from streaming.suricata import parse_eve_file  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = args[0] if args else "data/suricata/eve.json"
alerts = parse_eve_file(path)
if "--restamp" in sys.argv:
    for a in alerts:
        a.timestamp = time.time()
if not ch_alert_writer.start():
    sys.exit("ClickHouse unreachable")
for a in alerts:
    ch_alert_writer.write(a)
print(
    f"ingested {ch_alert_writer.written} of {len(alerts)} Suricata alerts from {path}"
)
