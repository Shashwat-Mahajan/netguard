"""
api/routers/auth.py
POST /api/v1/auth/login           — username + password -> JWT   (public)
GET  /api/v1/auth/me              — who am I                      (viewer)
POST /api/v1/auth/users           — create user                   (admin)
GET  /api/v1/auth/users           — list users                    (admin)
POST /api/v1/auth/api-keys        — create API key, shown once    (admin)
GET  /api/v1/auth/api-keys        — list keys (never the secret)  (admin)
DELETE /api/v1/auth/api-keys/{id} — revoke key                    (admin)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api import auth as auth_mod
from api.auth import Principal, require_admin, require_viewer
from config.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=72)  # bcrypt limit


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=12, max_length=72)
    role: str = "viewer"


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    role: str = "viewer"


@router.post("/login")
def login(req: LoginRequest):
    principal = auth_mod.auth_store.authenticate(req.username, req.password)
    if principal is None:
        log.warning("auth.login_failed", username=req.username)
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token, expires = auth_mod.create_token(principal)
    log.info("auth.login", username=principal.name, role=principal.role)
    return {"access_token": token, "token_type": "bearer", "expires_in": expires}


@router.get("/me")
def me(principal: Principal = Depends(require_viewer)):
    return {"name": principal.name, "role": principal.role, "kind": principal.kind}


@router.post("/users", status_code=201)
def create_user(body: UserCreate, admin: Principal = Depends(require_admin)):
    try:
        user = auth_mod.auth_store.create_user(body.username, body.password, body.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log.info("auth.user_created", username=body.username, role=body.role, by=admin.name)
    return user


@router.get("/users")
def list_users(_: Principal = Depends(require_admin)):
    return {"users": auth_mod.auth_store.list_users()}


@router.post("/api-keys", status_code=201)
def create_api_key(body: ApiKeyCreate, admin: Principal = Depends(require_admin)):
    try:
        record, raw = auth_mod.auth_store.create_api_key(body.name, body.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log.info("auth.api_key_created", name=body.name, role=body.role, by=admin.name)
    return {
        **record,
        "api_key": raw,
        "note": "Store this now. It cannot be shown again.",
    }


@router.get("/api-keys")
def list_api_keys(_: Principal = Depends(require_admin)):
    return {"api_keys": auth_mod.auth_store.list_api_keys()}


@router.delete("/api-keys/{key_id}")
def revoke_api_key(key_id: str, admin: Principal = Depends(require_admin)):
    if not auth_mod.auth_store.revoke_api_key(key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    log.info("auth.api_key_revoked", id=key_id, by=admin.name)
    return {"revoked": key_id}
