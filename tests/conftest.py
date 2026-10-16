import pytest

from api.limiter import limiter


@pytest.fixture(autouse=True)
def _limiter_off():
    """Other tests log in many times; keep limits off unless a test opts in."""
    limiter.enabled = False
    limiter.reset()
    yield
    limiter.enabled = False
    limiter.reset()
