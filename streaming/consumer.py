"""
streaming/consumer.py
Kafka/Redpanda consumer — reads flows and runs ML inference.

Pipeline per message:
  Redpanda topic → deserialise → scale features →
  XGBoost classify → IsoForest anomaly check → SHAP explanation (alerts) →
  merge result → publish to alerts topic

The consumer runs in its own thread so the FastAPI server
stays responsive while inference happens in the background.

Heavy model imports (xgboost, sklearn) are deferred to
InferenceEngine.load() so that importing this module — and therefore
the FastAPI app — works even when ML dependencies are unavailable.
"""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Callable

from config.logger import get_logger
from config.settings import settings
from processor.feature_schema import FlowFeatures

if TYPE_CHECKING:
    from models.anomaly_detector import AnomalyDetector
    from models.explainer import SHAPExplainer
    from models.xgboost_classifier import IntrusionClassifier
    from processor.preprocessor import DataPreprocessor

log = get_logger(__name__)


class PredictionResult:
    """Result of classifying one flow."""

    __slots__ = [
        "flow_id",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "timestamp",
        "duration",
        "fwd_packet_count",
        "bwd_packet_count",
        "total_fwd_bytes",
        "total_bwd_bytes",
        "flow_bytes_per_sec",
        "flow_packets_per_sec",
        "syn_flag_count",
        "fin_flag_count",
        "rst_flag_count",
        "psh_flag_count",
        "label",
        "label_encoded",
        "confidence",
        "is_anomaly",
        "anomaly_score",
        "class_probs",
        "inference_ms",
        "top_features",
        "report",
        "report_backend",
        "feature_values",
    ]

    def __init__(self, **kwargs):
        self.top_features = {}
        self.report = ""
        self.report_backend = ""
        self.feature_values = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}

    def is_threat(self, confidence_threshold: float = 0.7) -> bool:
        """Confirmed threat: the classifier says attack with enough confidence."""
        return self.label_encoded != 0 and self.confidence >= confidence_threshold

    def is_suspicious(self, confidence_threshold: float = 0.7) -> bool:
        """Not a confirmed threat, but the Isolation Forest flagged it as unusual."""
        return bool(self.is_anomaly) and not self.is_threat(confidence_threshold)

    def severity(self) -> str:
        if self.is_threat():
            return "high"
        if self.is_suspicious():
            return "low"
        return "none"


class InferenceEngine:
    """
    Loads models once at startup, classifies flows at inference time.
    Thread-safe — can be called from multiple consumer threads.
    """

    def __init__(self) -> None:
        self._clf: IntrusionClassifier | None = None
        self._det: AnomalyDetector | None = None
        self._pp: DataPreprocessor | None = None
        self._explainer: SHAPExplainer | None = None
        self._explainer_failed = False
        self._shap_lock = threading.Lock()
        self._lock = threading.Lock()
        self._total = 0
        self._threats = 0

    def load(self) -> None:
        """Load all models. Call once at startup."""
        from models.anomaly_detector import AnomalyDetector
        from models.xgboost_classifier import IntrusionClassifier
        from processor.preprocessor import DataPreprocessor

        log.info("inference.loading")

        self._pp = DataPreprocessor()
        scaler_path = settings.data_processed_dir / "scaler.joblib"
        if scaler_path.exists():
            self._pp.load_scaler(scaler_path)
            log.info("inference.scaler_loaded")
        else:
            log.warning("inference.no_scaler", hint="Run Phase 1 ingestion first")

        self._clf = IntrusionClassifier()
        model_path = settings.data_processed_dir / "xgboost_model.joblib"
        if model_path.exists():
            self._clf.load(model_path)
            log.info("inference.xgboost_loaded")
        else:
            log.warning("inference.no_xgboost", hint="Run Phase 2 training first")

        self._det = AnomalyDetector()
        det_path = settings.data_processed_dir / "isolation_forest.joblib"
        if det_path.exists():
            self._det.load(det_path)
            log.info("inference.isoforest_loaded")
        else:
            log.warning("inference.no_isoforest")

        log.info("inference.ready")

    def _explain(self, vec, label_int: int, top_n: int = 8) -> dict:
        """Top SHAP features for the predicted class. Never raises."""
        with self._shap_lock:
            try:
                if self._explainer is None:
                    if self._explainer_failed:
                        return {}
                    from models.explainer import SHAPExplainer

                    ex = SHAPExplainer(self._clf)
                    ex.build()
                    self._explainer = ex
                out = self._explainer.explain_single(vec, label_int, top_n=top_n)
                return out["top_features"]
            except Exception as exc:
                self._explainer_failed = True
                log.warning("inference.shap_unavailable", error=str(exc))
                return {}

    def predict(
        self, flow: FlowFeatures, explain: bool | None = None
    ) -> PredictionResult | None:
        """
        Run full inference on one flow. Returns None if models are not loaded.
        explain=None: SHAP only for threats/suspicious flows (cheap for bulk capture)
        explain=True: always; explain=False: never.
        """
        if self._clf is None or self._pp is None:
            return None

        t0 = time.perf_counter()

        try:
            # Scale features
            vec = self._pp.transform_features(flow)

            # XGBoost classification
            label_name, label_int, confidence, class_probs = self._clf.predict_single(
                vec[0]
            )

            # Isolation Forest anomaly check
            is_anomaly, anomaly_score = False, 0.0
            if self._det and self._det.is_fitted():
                is_anomaly, anomaly_score = self._det.predict_single(vec[0])

            inference_ms = (time.perf_counter() - t0) * 1000

            result = PredictionResult(
                flow_id=flow.flow_id,
                src_ip=flow.src_ip,
                dst_ip=flow.dst_ip,
                src_port=flow.src_port,
                dst_port=flow.dst_port,
                protocol=flow.protocol,
                timestamp=flow.timestamp,
                duration=flow.duration,
                fwd_packet_count=flow.fwd_packet_count,
                bwd_packet_count=flow.bwd_packet_count,
                total_fwd_bytes=flow.total_fwd_bytes,
                total_bwd_bytes=flow.total_bwd_bytes,
                flow_bytes_per_sec=flow.flow_bytes_per_sec,
                flow_packets_per_sec=flow.flow_packets_per_sec,
                syn_flag_count=flow.syn_flag_count,
                fin_flag_count=flow.fin_flag_count,
                rst_flag_count=flow.rst_flag_count,
                psh_flag_count=flow.psh_flag_count,
                label=label_name,
                label_encoded=label_int,
                confidence=round(confidence, 4),
                is_anomaly=is_anomaly,
                anomaly_score=round(anomaly_score, 4),
                class_probs=class_probs,
                inference_ms=round(inference_ms, 2),
            )

            do_explain = (
                explain
                if explain is not None
                else (result.is_threat() or result.is_suspicious())
            )
            if do_explain:
                result.top_features = self._explain(vec[0], label_int)
                result.feature_values = {
                    k: float(getattr(flow, k, 0) or 0) for k in result.top_features
                }

            with self._lock:
                self._total += 1
                if result.is_threat():
                    self._threats += 1

            if result.is_threat():
                log.warning(
                    "inference.threat_detected",
                    flow_id=flow.flow_id,
                    label=label_name,
                    confidence=confidence,
                    src=f"{flow.src_ip}:{flow.src_port}",
                    dst=f"{flow.dst_ip}:{flow.dst_port}",
                )
            elif result.is_suspicious():
                log.debug(
                    "inference.suspicious_flow",
                    flow_id=flow.flow_id,
                    anomaly_score=anomaly_score,
                )

            return result

        except Exception as exc:
            log.error("inference.predict_error", error=str(exc), flow_id=flow.flow_id)
            return None

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_flows": self._total,
                "threats": self._threats,
                "benign": self._total - self._threats,
                "threat_rate": round(self._threats / max(self._total, 1), 4),
                "models_loaded": {
                    "xgboost": self._clf is not None and self._clf.is_fitted(),
                    "isolation_forest": self._det is not None and self._det.is_fitted(),
                    "scaler": self._pp is not None,
                },
            }


class FlowConsumer:
    """
    Kafka consumer loop — reads from traffic topic, runs inference,
    publishes results to alerts topic and calls on_result callback.

    Usage:
        engine = InferenceEngine()
        engine.load()

        consumer = FlowConsumer(engine, on_result=my_handler)
        consumer.start()   # runs in background thread
        ...
        consumer.stop()
    """

    def __init__(
        self,
        engine: InferenceEngine,
        on_result: Callable[[PredictionResult], None] | None = None,
    ) -> None:
        self._engine = engine
        self._on_result = on_result or (lambda r: None)
        self._running = False
        self._thread: threading.Thread | None = None
        self._consumer = None

    def start(self) -> None:
        if not settings.kafka_enabled:
            log.info("consumer.disabled", reason="KAFKA_BOOTSTRAP_SERVERS not set")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._consume_loop,
            daemon=True,
            name="flow-consumer",
        )
        self._thread.start()
        log.info("consumer.started", topic=settings.kafka_topic_traffic)

    def stop(self) -> None:
        self._running = False
        if self._consumer:
            self._consumer.close()
        log.info("consumer.stopped")

    def _consume_loop(self) -> None:
        try:
            from kafka import KafkaConsumer  # type: ignore

            self._consumer = KafkaConsumer(
                settings.kafka_topic_traffic,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                group_id="netguard-inference",
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=1000,
            )
            log.info("consumer.connected")

            while self._running:
                for msg in self._consumer:
                    if not self._running:
                        break
                    self._handle_message(msg.value)

        except Exception as exc:
            log.error("consumer.loop_error", error=str(exc))
            self._running = False

    def _handle_message(self, payload: dict) -> None:
        try:
            # Reconstruct FlowFeatures from the message
            flow = FlowFeatures(
                flow_id=payload.get("flow_id", ""),
                src_ip=payload.get("src_ip", ""),
                dst_ip=payload.get("dst_ip", ""),
                src_port=payload.get("src_port", 0),
                dst_port=payload.get("dst_port", 0),
                protocol=payload.get("protocol", 0),
                timestamp=payload.get("timestamp", 0.0),
                duration=payload.get("duration", 0.0),
                fwd_packet_count=payload.get("fwd_packet_count", 0),
                bwd_packet_count=payload.get("bwd_packet_count", 0),
                total_fwd_bytes=payload.get("total_fwd_bytes", 0),
                total_bwd_bytes=payload.get("total_bwd_bytes", 0),
                flow_bytes_per_sec=payload.get("flow_bytes_per_sec", 0.0),
                flow_packets_per_sec=payload.get("flow_packets_per_sec", 0.0),
                syn_flag_count=payload.get("syn_flag_count", 0),
                fin_flag_count=payload.get("fin_flag_count", 0),
                rst_flag_count=payload.get("rst_flag_count", 0),
                psh_flag_count=payload.get("psh_flag_count", 0),
                label=payload.get("label", "unknown"),
            )

            # Run inference
            result = self._engine.predict(flow)
            if result:
                self._on_result(result)

        except Exception as exc:
            log.error("consumer.handle_error", error=str(exc))
