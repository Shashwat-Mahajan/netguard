"""
api/auth.py
Authentication and authorisation: JWT for people, API keys for machines.

Storage is a small SQLite file (single-row updates such as revoking a key
don't suit ClickHouse). Passwords use bcrypt; API keys are random, shown once,
and only their SHA-256 hash is stored.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import bcrypt
import jwt
from fastapi import Depends, HTTPException, WebSocketException
from starlette.requests import HTTPConnection

from config.logger import get_logger
from config.settings import settings

log = get_logger(__name__)

ROLES = ("admin", "viewer")
MIN_PASSWORD_LEN = 12
_ALGO = "HS256"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL,
    disabled      INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    prefix     TEXT NOT NULL,
    key_hash   TEXT NOT NULL UNIQUE,
    role       TEXT NOT NULL,
    revoked    INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
"""


@dataclass
class Principal:
    name: str
    role: str
    kind: str  # "user" | "api_key"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False


_DUMMY_HASH = hash_password("dummy-password-for-timing")


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class AuthStore:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── users ─────────────────────────────────────────────────
    def create_user(self, username: str, password: str, role: str) -> dict:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if len(password) < MIN_PASSWORD_LEN:
            raise ValueError(f"password must be at least {MIN_PASSWORD_LEN} characters")
        pw_hash = hash_password(password)
        try:
            with self._lock:
                self._db.execute(
                    "INSERT INTO users (username, password_hash, role, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (username, pw_hash, role, time.time()),
                )
                self._db.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("username already exists") from exc
        return {"username": username, "role": role}

    def get_user(self, username: str) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

    def authenticate(self, username: str, password: str) -> Principal | None:
        row = self.get_user(username)
        # Always run one bcrypt check so response time doesn't reveal valid usernames.
        ok = verify_password(password, row["password_hash"] if row else _DUMMY_HASH)
        if row is None or not ok or row["disabled"]:
            return None
        return Principal(name=row["username"], role=row["role"], kind="user")

    def list_users(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT username, role, disabled, created_at FROM users ORDER BY username"
            ).fetchall()
        return [dict(r) for r in rows]

    def bootstrap_admin(self) -> None:
        if not (settings.admin_username and settings.admin_password):
            return
        if self.get_user(settings.admin_username):
            return
        try:
            self.create_user(settings.admin_username, settings.admin_password, "admin")
            log.info("auth.admin_created", username=settings.admin_username)
        except ValueError as exc:
            log.warning("auth.admin_bootstrap_failed", error=str(exc))

    # ── API keys ──────────────────────────────────────────────
    def create_api_key(self, name: str, role: str) -> tuple[dict, str]:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        raw = "ng_" + secrets.token_urlsafe(32)
        key_id = str(uuid.uuid4())
        with self._lock:
            self._db.execute(
                "INSERT INTO api_keys (id, name, prefix, key_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key_id, name, raw[:8], _hash_key(raw), role, time.time()),
            )
            self._db.commit()
        return {"id": key_id, "name": name, "role": role, "prefix": raw[:8]}, raw

    def verify_api_key(self, raw: str) -> Principal | None:
        with self._lock:
            row = self._db.execute(
                "SELECT name, role, revoked FROM api_keys WHERE key_hash = ?",
                (_hash_key(raw),),
            ).fetchone()
        if row is None or row["revoked"]:
            return None
        return Principal(name=f"key:{row['name']}", role=row["role"], kind="api_key")

    def list_api_keys(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, name, prefix, role, revoked, created_at "
                "FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke_api_key(self, key_id: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,)
            )
            self._db.commit()
            return cur.rowcount > 0


auth_store = AuthStore(settings.auth_db_path)


# ── JWT ───────────────────────────────────────────────────────────────────────


def create_token(principal: Principal) -> tuple[str, int]:
    expires = settings.jwt_expire_minutes * 60
    token = jwt.encode(
        {"sub": principal.name, "exp": int(time.time()) + expires},
        settings.secret_key,
        algorithm=_ALGO,
    )
    return token, expires


def _principal_from_token(token: str) -> Principal | None:
    try:
        claims = jwt.decode(token, settings.secret_key, algorithms=[_ALGO])
    except jwt.PyJWTError:
        return None
    # Role comes from the database, so disabling a user or changing a role
    # takes effect immediately instead of when the token expires.
    row = auth_store.get_user(claims.get("sub", ""))
    if row is None or row["disabled"]:
        return None
    return Principal(name=row["username"], role=row["role"], kind="user")


# ── FastAPI dependencies (HTTPConnection works for HTTP *and* WebSocket) ───────


def _unauthorized(is_ws: bool = False) -> Exception:
    if is_ws:
        # Closes the socket with policy-violation code 1008
        return WebSocketException(code=1008, reason="Not authenticated")
    return HTTPException(
        status_code=401,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_principal(conn: HTTPConnection) -> Principal:
    is_ws = conn.scope.get("type") == "websocket"

    key = conn.headers.get("x-api-key") or (
        conn.query_params.get("api_key") if is_ws else None
    )
    if key:
        principal = auth_store.verify_api_key(key)
        if principal:
            return principal
        raise _unauthorized(is_ws)

    authz = conn.headers.get("authorization", "")
    token = authz[7:] if authz.lower().startswith("bearer ") else None
    # Browsers can't set headers on WebSockets, so allow ?token= there only.
    if token is None and is_ws:
        token = conn.query_params.get("token")
    if token:
        principal = _principal_from_token(token)
        if principal:
            return principal
    raise _unauthorized(is_ws)


def require_viewer(principal: Principal = Depends(get_principal)) -> Principal:
    return principal  # viewer and admin both pass


def require_admin(principal: Principal = Depends(get_principal)) -> Principal:
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return principal
