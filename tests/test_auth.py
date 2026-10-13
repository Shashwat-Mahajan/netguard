"""tests/test_auth.py — auth rules, using a throw-away SQLite file per test."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import auth as auth_mod
from api.auth import AuthStore, hash_password, verify_password
from api.main import app

PW = "correct-horse-battery"


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = AuthStore(tmp_path / "auth.db")
    monkeypatch.setattr(auth_mod, "auth_store", s)
    yield s
    s.close()


@pytest.fixture
def client(store):
    return TestClient(app)  # no `with`, so lifespan (ClickHouse etc.) is not run


def _login(client, username, password=PW):
    r = client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    return r


def _bearer(client, username):
    return {
        "Authorization": f"Bearer {_login(client, username).json()['access_token']}"
    }


def test_password_hash_roundtrip():
    h = hash_password("abc")
    assert h != "abc"
    assert verify_password("abc", h)
    assert not verify_password("abd", h)


def test_short_password_rejected(store):
    with pytest.raises(ValueError):
        store.create_user("bob", "short", "viewer")


def test_unauthenticated_requests_get_401(client):
    assert client.get("/api/v1/flows").status_code == 401
    assert client.get("/api/v1/alerts").status_code == 401
    assert client.get("/api/v1/status").status_code == 401


def test_health_stays_open(client):
    assert client.get("/health").status_code == 200


def test_login_wrong_password(client, store):
    store.create_user("alice", PW, "viewer")
    assert _login(client, "alice", "wrong-password-here").status_code == 401
    assert _login(client, "nobody").status_code == 401


def test_viewer_can_read_but_not_admin_actions(client, store):
    store.create_user("alice", PW, "viewer")
    h = _bearer(client, "alice")
    assert client.get("/api/v1/auth/me", headers=h).json()["role"] == "viewer"
    assert (
        client.post(
            "/api/v1/auth/api-keys", headers=h, json={"name": "x", "role": "viewer"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/predict",
            headers=h,
            json={
                "src_ip": "1.1.1.1",
                "dst_ip": "2.2.2.2",
                "src_port": 1,
                "dst_port": 2,
            },
        ).status_code
        == 403
    )


def test_api_key_lifecycle(client, store):
    store.create_user("root", PW, "admin")
    h = _bearer(client, "root")

    created = client.post(
        "/api/v1/auth/api-keys", headers=h, json={"name": "siem", "role": "viewer"}
    )
    assert created.status_code == 201
    body = created.json()
    key = body["api_key"]

    me = client.get("/api/v1/auth/me", headers={"X-API-Key": key})
    assert me.status_code == 200 and me.json()["role"] == "viewer"

    listing = client.get("/api/v1/auth/api-keys", headers=h).json()["api_keys"]
    assert all("api_key" not in k and "key_hash" not in k for k in listing)

    assert (
        client.delete(f"/api/v1/auth/api-keys/{body['id']}", headers=h).status_code
        == 200
    )
    assert client.get("/api/v1/auth/me", headers={"X-API-Key": key}).status_code == 401


def test_disabled_user_token_stops_working(client, store):
    store.create_user("alice", PW, "viewer")
    h = _bearer(client, "alice")
    store._db.execute("UPDATE users SET disabled = 1 WHERE username = 'alice'")
    store._db.commit()
    assert client.get("/api/v1/auth/me", headers=h).status_code == 401


def test_websocket_requires_credentials(client, store):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/v1/stream"):
            pass
