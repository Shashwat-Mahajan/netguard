"""api/limiter.py — shared slowapi limiter (in-memory, per process)."""

from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings

# config_filename=None makes slowapi read its own copy of .env with the
# Windows default encoding, which breaks on non-ASCII bytes. Our settings
# already handle .env (as UTF-8), so point slowapi at nothing.
limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.rate_limit_enabled,
    config_filename="config/slowapi.env",
)
