"""
config/settings.py
Central configuration — loaded once at startup via pydantic-settings.
All values can be overridden by environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    secret_key: str = Field(default="insecure-dev-key", min_length=16)

    # ── Capture ────────────────────────────────────────────────────
    capture_interface: str = "eth0"
    capture_flow_timeout: int = Field(default=120, ge=10, le=3600)
    capture_max_packets: int = Field(default=0, ge=0)  # 0 = unlimited

    # ── Data paths ─────────────────────────────────────────────────
    data_raw_dir: Path = Path("./data/raw")
    data_processed_dir: Path = Path("./data/processed")
    log_dir: Path = Path("./logs")
    zeek_log_dir: Path = Path("/opt/zeek/logs/current")

    # ── API ────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_workers: int = Field(default=4, ge=1, le=32)

    # ── Kafka / Redpanda (Phase 3) ─────────────────────────────────
    kafka_bootstrap_servers: str = ""
    kafka_topic_traffic: str = "netguard.traffic"
    kafka_topic_alerts: str = "netguard.alerts"

    @field_validator("data_raw_dir", "data_processed_dir", "log_dir", mode="after")
    @classmethod
    def ensure_dirs_exist(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def kafka_enabled(self) -> bool:
        return bool(self.kafka_bootstrap_servers)


# ── Singleton ──────────────────────────────────────────────────────────────────
settings = Settings()
