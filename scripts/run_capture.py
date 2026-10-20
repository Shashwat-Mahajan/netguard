"""
scripts/run_capture.py
Standalone capture process: packets (live or from a pcap) -> rules + inference -> ClickHouse.
The API reads the results back from ClickHouse.

  python scripts/run_capture.py --list-interfaces
  python scripts/run_capture.py --pcap data/test.pcap
  python scripts/run_capture.py --interface "Ethernet"      (live: needs admin/root)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.pipeline import FlowPipeline, replay_pcap  # noqa: E402
from config.logger import configure_logging, get_logger  # noqa: E402
from config.settings import settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="NetGuard capture process")
    ap.add_argument("--pcap", help="replay this pcap file instead of sniffing")
    ap.add_argument("--interface", default=settings.capture_interface)
    ap.add_argument("--max-packets", type=int, default=0)
    ap.add_argument(
        "--keep-timestamps",
        action="store_true",
        help="pcap replay: keep original packet times (default: stamp as now)",
    )
    ap.add_argument(
        "--publish",
        action="store_true",
        help="also publish flows to Kafka (needs KAFKA_BOOTSTRAP_SERVERS)",
    )
    ap.add_argument("--list-interfaces", action="store_true")
    args = ap.parse_args()

    if args.list_interfaces:
        from scapy.all import get_if_list  # type: ignore

        print("\n".join(get_if_list()))
        return 0

    configure_logging()
    log = get_logger("run_capture")

    from streaming.clickhouse_writer import ch_writer
    from streaming.consumer import InferenceEngine

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

    from capture.rules import SignatureEngine
    from streaming.alert_writer import ch_alert_writer
    from streaming.intel import ThreatIntel

    rules = SignatureEngine.from_settings() if settings.rules_enabled else None
    if rules is not None and not ch_alert_writer.start():
        print("WARNING: could not store rule alerts (ClickHouse unreachable).")

    intel = ThreatIntel.from_settings() if settings.intel_enabled else None

    producer = None
    if args.publish:
        from streaming.producer import FlowProducer

        producer = FlowProducer()
        producer.start()

    pipeline = FlowPipeline(
        engine,
        writer=ch_writer,
        producer=producer,
        restamp=bool(args.pcap) and not args.keep_timestamps,
        rules=rules,
        alert_writer=ch_alert_writer,
        intel=intel,
    )

    try:
        if args.pcap:
            n = replay_pcap(args.pcap, pipeline, args.max_packets)
            print(f"Replayed {n} packets from {args.pcap}")
        else:
            from capture.sniffer import PacketCapture

            cap = PacketCapture(
                interface=args.interface,
                on_flow=pipeline.handle_flow,
                max_packets=args.max_packets,
            )
            cap.start()
            print(
                f"Capturing on {args.interface}. Ctrl+C to stop "
                f"(flows finish after {settings.capture_flow_timeout}s idle)."
            )
            try:
                while cap.is_running():
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            log.info("capture.stopping", packets=cap.packet_count)
            cap.stop()
    finally:
        ch_writer.stop()  # flushes the last batch
        if producer is not None:
            producer.stop()

    print("Summary:", pipeline.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
