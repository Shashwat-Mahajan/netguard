"""python scripts/ingest_zeek.py [data/zeek/notice.log] [--restamp] [--flows] [--no-notices]
Loads Zeek notices into netguard.alerts (source='zeek').
--flows       also run conn.log (same folder) through the ML models + rules.
--no-notices  skip the notices (use with --flows to avoid duplicate alerts).
--restamp     sets everything to now (for replayed pcaps)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402
from streaming.alert_writer import ch_alert_writer  # noqa: E402
from streaming.zeek import ingest_flows, ingest_notices  # noqa: E402


def run_flows(conn_path: Path, restamp: bool) -> int:
    from capture.pipeline import FlowPipeline
    from capture.rules import SignatureEngine
    from streaming.clickhouse_writer import ch_writer
    from streaming.consumer import InferenceEngine
    from streaming.intel import ThreatIntel
    from streaming.thehive import TheHiveClient

    engine = InferenceEngine()
    engine.load()
    if not engine.stats["models_loaded"]["xgboost"]:
        print(
            "ERROR: models not found in data/processed. Run scripts/train_model.py first."
        )
        return 1
    ch_writer.start()
    if not ch_writer.stats["connected"]:
        print("WARNING: ClickHouse unreachable. Flows will be scored but NOT saved.")
    rules = SignatureEngine.from_settings() if settings.rules_enabled else None
    intel = ThreatIntel.from_settings() if settings.intel_enabled else None
    pipeline = FlowPipeline(
        engine,
        writer=ch_writer,
        restamp=restamp,
        rules=rules,
        alert_writer=ch_alert_writer,
        intel=intel,
        thehive=TheHiveClient.from_settings(),
    )
    try:
        n = ingest_flows(conn_path, pipeline)
    finally:
        ch_writer.stop()  # flushes the last batch
    print(f"flows sent to the pipeline: {n} from {conn_path}")
    print("Summary:", pipeline.summary)
    return 0


def main() -> int:
    flags = set(sys.argv[1:])
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0] if args else "data/zeek/notice.log")
    restamp = "--restamp" in flags

    if not ch_alert_writer.start():
        print("ClickHouse unreachable")
        return 1

    if "--no-notices" not in flags:
        if not path.is_file():
            print(f"{path} not found (run scripts\\run_zeek.bat first)")
            return 1
        written, found = ingest_notices(path, ch_alert_writer, restamp)
        print(f"ingested {written} of {found} alerts from {path}")

    if "--flows" in flags:
        conn = path.with_name("conn.log")
        if not conn.is_file():
            print(f"{conn} not found (run scripts\\run_zeek.bat first)")
            return 1
        return run_flows(conn, restamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
