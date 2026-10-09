from __future__ import annotations
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest, REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from api.routers import status as status_router
from api.routers import predict as predict_router
from config.logger import configure_logging, get_logger
from config.settings import settings

log = get_logger(__name__)

def _metric(cls, name, desc, labels=None):
    try:
        return cls(name, desc, labels) if labels else cls(name, desc)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

FLOWS_TOTAL  = _metric(Counter, "netguard_flows_total",       "Total flows", ["source"])
ACTIVE_FLOWS = _metric(Gauge,   "netguard_active_flows",      "Active flows")
API_REQUESTS = _metric(Counter, "netguard_api_requests_total","API requests", ["method","path","status"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("netguard.startup", env=settings.app_env, version="1.0.0-phase3")

    try:
        from api.routers.predict import get_engine
        get_engine()
        log.info("models.preloaded")
    except Exception as e:
        log.warning("models.preload_failed", error=str(e))

    ch_writer = None
    try:
        from streaming.clickhouse_writer import ch_writer as _ch_writer
        _ch_writer.start()  # logs its own success/failure
        ch_writer = _ch_writer
    except Exception as e:
        log.warning("clickhouse.writer.failed", error=str(e))

    yield

    if ch_writer is not None:
        ch_writer.stop()
    log.info("netguard.shutdown")

app = FastAPI(
    title="NetGuard API",
    description="AI-powered Network Intrusion Detection System",
    version="1.0.0-phase3",
    docs_url="/docs"  if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware,
    allow_origins=settings.cors_origin_list, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        if API_REQUESTS:
            API_REQUESTS.labels(
                method=request.method,
                path=request.url.path,
                status=str(response.status_code),
            ).inc()
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        return response

app.add_middleware(MetricsMiddleware)
app.include_router(status_router.router,  prefix="/api/v1")
app.include_router(predict_router.router, prefix="/api/v1")

@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/ready", tags=["ops"])
async def ready():
    checks = {
        "data_processed_dir": settings.data_processed_dir.exists(),
        "xgboost_model":      (settings.data_processed_dir / "xgboost_model.joblib").exists(),
        "scaler":             (settings.data_processed_dir / "scaler.joblib").exists(),
    }
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail={"checks": checks})
    return {"status": "ready", "checks": checks}

@app.get("/metrics", tags=["ops"])
async def metrics():
    return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
