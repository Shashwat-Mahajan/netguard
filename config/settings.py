"""
config/settings.py
Central configuration — loaded once at startup via pydantic-settings.
All values can be overridden by environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
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

    # ── CORS ───────────────────────────────────────────────────────
    # Comma-separated list (plain string avoids pydantic-settings'
    # JSON-parsing requirement for list env vars).
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

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

    # ── ClickHouse ─────────────────────────────────────────────────
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_db: str = "netguard"
    clickhouse_user: str = "netguard"
    clickhouse_password: str = ""

    # ── Auth ───────────────────────────────────────────────────────
    auth_db_path: Path = Path("./data/auth.db")
    jwt_expire_minutes: int = Field(default=60, ge=5, le=1440)
    # Optional: creates the first admin at startup if it doesn't exist yet.
    admin_username: str = ""
    admin_password: str = ""

    @field_validator("data_raw_dir", "data_processed_dir", "log_dir", mode="after")
    @classmethod
    def ensure_dirs_exist(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def _reject_dev_key_in_prod(self) -> "Settings":
        if self.app_env == "production" and self.secret_key == "insecure-dev-key":
            raise ValueError("SECRET_KEY must be set explicitly in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def kafka_enabled(self) -> bool:
        return bool(self.kafka_bootstrap_servers)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# ── Singleton ──────────────────────────────────────────────────────────────────
settings = Settings()
