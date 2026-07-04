from __future__ import annotations
import time
from pathlib import Path
from fastapi import APIRouter
from config.settings import settings

router = APIRouter(tags=["status"])

def _dir_stats(directory: Path) -> dict:
    if not directory.exists():
        return {"exists": False, "files": 0, "size_mb": 0}
    files = list(directory.rglob("*"))
    size = sum(f.stat().st_size for f in files if f.is_file())
    return {
        "exists": True,
        "files": len([f for f in files if f.is_file()]),
        "size_mb": round(size / 1_048_576, 2),
    }

@router.get("/status")
async def pipeline_status():
    processed = settings.data_processed_dir
    return {
        "timestamp": time.time(),
        "environment": settings.app_env,
        "datasets": {
            "cicids2017": {
                "raw": _dir_stats(settings.data_raw_dir / "cicids2017"),
                "cached": (processed / "cicids2017_combined.parquet").exists(),
            },
            "nslkdd": {
                "raw": _dir_stats(settings.data_raw_dir / "nslkdd"),
                "cached": (processed / "nslkdd_train.parquet").exists(),
            },
        },
        "models": {
            "xgboost":         (processed / "xgboost_model.joblib").exists(),
            "isolation_forest":(processed / "isolation_forest.joblib").exists(),
            "scaler":          (processed / "scaler.joblib").exists(),
        },
        "pipeline": {
            "scaler_fitted":    (processed / "scaler.joblib").exists(),
            "kafka_enabled":    settings.kafka_enabled,
            "capture_interface":settings.capture_interface,
        },
    }
