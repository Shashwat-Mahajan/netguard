"""python scripts/compare_detectors.py [minutes=60]
Which detector flagged which source IP: NetGuard rules, Suricata, ML classifier."""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming.clickhouse_reader import ch_reader  # noqa: E402

mins = int(sys.argv[1]) if len(sys.argv) > 1 else 60
seen: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

for src, source, rule in ch_reader._query(
    "SELECT src_ip, source, rule FROM netguard.alerts "
    "WHERE timestamp >= now() - INTERVAL %(m)s MINUTE",
    {"m": mins},
):
    seen[src][source].add(rule)
for (src,) in ch_reader._query(
    "SELECT DISTINCT src_ip FROM netguard.flows "
    "WHERE is_threat = 1 AND timestamp >= now() - INTERVAL %(m)s MINUTE",
    {"m": mins},
):
    seen[src]["ml"].add("classifier")

print(f"{'source IP':<18}{'netguard rules':<28}{'suricata':<24}{'ML'}")
for ip, d in sorted(seen.items()):
    print(
        f"{ip:<18}{','.join(sorted(d['signature'])) or '-':<28}"
        f"{','.join(sorted(d['suricata'])) or '-':<24}{'yes' if d['ml'] else '-'}"
    )
