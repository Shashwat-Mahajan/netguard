"""
config/logger.py
Structured JSON logging using structlog.
Every log line is a JSON object — easy to ingest into Grafana Loki / ELK.
"""

from __future__ import annotations

import logging
import logging.config
import sys

import structlog

from config.settings import settings


def configure_logging() -> None:
    """Call once at application startup."""

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_production:
        # JSON output for log aggregators
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable coloured output for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)

    # Silence noisy third-party loggers
    for name in ("scapy.runtime", "urllib3", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
