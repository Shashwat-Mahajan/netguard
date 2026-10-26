"""python scripts/ingest_zeek.py [data/zeek/notice.log] [--restamp]
Loads Zeek notices into netguard.alerts (source='zeek').
--restamp sets them to now (for replayed pcaps)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming.alert_writer import ch_alert_writer  # noqa: E402
from streaming.zeek import ingest_notices  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = args[0] if args else "data/zeek/notice.log"
if not Path(path).is_file():
    sys.exit(f"{path} not found (run scripts\\run_zeek.bat first)")
if not ch_alert_writer.start():
    sys.exit("ClickHouse unreachable")
written, found = ingest_notices(path, ch_alert_writer, "--restamp" in sys.argv)
print(f"ingested {written} of {found} alerts from {path}")
