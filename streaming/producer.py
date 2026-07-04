"""
streaming/producer.py
Kafka/Redpanda producer — sends completed flows to the traffic topic.

Flow:
  PacketCapture → FlowFeatures → producer.send() → Redpanda topic
  ZeekParser    → FlowFeatures → producer.send() → Redpanda topic

Each message is a JSON-serialised FlowFeatures object.
The consumer (FastAPI) reads from this topic and runs ML inference.

Why Redpanda instead of raw Kafka?
  - Redpanda is Kafka-API compatible — same client code works with both
  - Single binary, no ZooKeeper, starts in seconds
  - Free community edition, no limits at dev/research scale
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Callable

from config.logger import get_logger
from config.settings import settings
from processor.feature_schema import FlowFeatures

log = get_logger(__name__)


def _flow_to_dict(f: FlowFeatures) -> dict:
    """Serialise FlowFeatures to a JSON-safe dict."""
    return {
        "flow_id":            f.flow_id,
        "src_ip":             f.src_ip,
        "dst_ip":             f.dst_ip,
        "src_port":           f.src_port,
        "dst_port":           f.dst_port,
        "protocol":           f.protocol,
        "timestamp":          f.timestamp,
        "duration":           f.duration,
        "fwd_packet_count":   f.fwd_packet_count,
        "bwd_packet_count":   f.bwd_packet_count,
        "total_fwd_bytes":    f.total_fwd_bytes,
        "total_bwd_bytes":    f.total_bwd_bytes,
        "flow_bytes_per_sec": f.flow_bytes_per_sec,
        "flow_packets_per_sec": f.flow_packets_per_sec,
        "syn_flag_count":     f.syn_flag_count,
        "fin_flag_count":     f.fin_flag_count,
        "rst_flag_count":     f.rst_flag_count,
        "psh_flag_count":     f.psh_flag_count,
        "label":              f.label,
        "label_encoded":      f.label_encoded,
        # Full ML feature vector
        "features":           f.to_vector().tolist(),
        "ingested_at":        time.time(),
    }


class FlowProducer:
    """
    Sends FlowFeatures objects to a Redpanda/Kafka topic.

    Falls back to a no-op mode if Kafka is not configured
    (settings.kafka_enabled == False) so Phase 1/2 still work
    without Redpanda running.

    Usage:
        producer = FlowProducer()
        producer.start()
        producer.send(flow_features)
        producer.stop()
    """

    def __init__(self) -> None:
        self._producer = None
        self._topic = settings.kafka_topic_traffic
        self._enabled = settings.kafka_enabled
        self._sent = 0
        self._failed = 0

    def start(self) -> None:
        if not self._enabled:
            log.info("producer.disabled", reason="KAFKA_BOOTSTRAP_SERVERS not set")
            return

        try:
            from kafka import KafkaProducer  # type: ignore
            self._producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",              # wait for all replicas
                retries=3,
                max_block_ms=5000,       # 5s timeout if broker unavailable
                compression_type="lz4",  # compress messages
            )
            log.info(
                "producer.started",
                bootstrap=settings.kafka_bootstrap_servers,
                topic=self._topic,
            )
            print(f"  ✓  Producer connected → {settings.kafka_bootstrap_servers}")
        except Exception as exc:
            log.error("producer.start_failed", error=str(exc))
            self._producer = None

    def send(self, flow: FlowFeatures) -> bool:
        """
        Send one flow to the Kafka topic.
        Returns True if sent, False if dropped (no connection / disabled).
        """
        if not self._enabled or self._producer is None:
            return False

        try:
            payload = _flow_to_dict(flow)
            # Use flow_id as the message key — ensures ordering per flow
            self._producer.send(
                self._topic,
                key=flow.flow_id,
                value=payload,
            )
            self._sent += 1
            if self._sent % 100 == 0:
                log.info("producer.throughput", sent=self._sent, failed=self._failed)
            return True
        except Exception as exc:
            self._failed += 1
            log.warning("producer.send_failed", error=str(exc), flow_id=flow.flow_id)
            return False

    def flush(self) -> None:
        """Block until all buffered messages are sent."""
        if self._producer:
            self._producer.flush(timeout=10)

    def stop(self) -> None:
        if self._producer:
            self.flush()
            self._producer.close()
            log.info("producer.stopped", total_sent=self._sent, total_failed=self._failed)

    @property
    def stats(self) -> dict:
        return {"sent": self._sent, "failed": self._failed, "enabled": self._enabled}
