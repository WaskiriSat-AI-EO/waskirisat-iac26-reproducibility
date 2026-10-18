"""CDSE OAuth2 token management.

Obtains short-lived access tokens from the Copernicus Data Space
Ecosystem identity service using the Resource Owner Password
Credentials grant.  Credentials are read from environment variables
``CDSE_USER`` and ``CDSE_PASSWORD`` (typically via a ``.env`` file).
"""

import logging
import os
import time
from typing import Callable, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
_CLIENT_ID = "cdse-public"
_TIMEOUT_S = 30


def get_access_token(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Optional[str]:
    """Obtain a CDSE OAuth2 access token.

    Authenticates against the Copernicus Data Space Ecosystem identity
    server using the password grant.  If *username* or *password* are
    omitted, they are read from the ``CDSE_USER`` and ``CDSE_PASSWORD``
    environment variables (loaded from ``.env`` if present).

    Args:
        username: CDSE account username.  Defaults to ``$CDSE_USER``.
        password: CDSE account password.  Defaults to ``$CDSE_PASSWORD``.

    Returns:
        Bearer access token string, or ``None`` on any failure.
    """
    if username is None:
        username = os.getenv("CDSE_USER")
    if password is None:
        password = os.getenv("CDSE_PASSWORD")

    if not username or not password:
        logger.error(
            "CDSE credentials not provided.  Set CDSE_USER and "
            "CDSE_PASSWORD environment variables or pass them explicitly."
        )
        return None

    payload = {
        "client_id": _CLIENT_ID,
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    try:
        response = requests.post(_TOKEN_URL, data=payload, timeout=_TIMEOUT_S)
        response.raise_for_status()
        token = response.json()["access_token"]
        logger.info("CDSE authentication successful.")
        return token
    except requests.HTTPError as exc:
        logger.error(
            "CDSE authentication HTTP error %s: %s", exc.response.status_code, exc
        )
    except Exception as exc:
        logger.error("CDSE authentication failed: %s", exc)

    return None


class TokenProvider:
    """Reuse one CDSE access token until shortly before it expires.

    The identity server rate-limits password logins: one login per granule
    was fine at ~2 min per granule, but at ~5 s per granule (node downloads,
    several roots in parallel) it answered HTTP 400/403.  This logs in once,
    reuses the token for ``expires_in − margin_s`` seconds (CDSE: 600 s), and
    backs off between failed logins.

    Args:
        margin_s: Seconds before expiry at which a new token is fetched.
        backoff_s: Waits between failed login attempts; one attempt more
            than its length is made before giving up.
        clock, sleep: Injectable for tests.
    """

    def __init__(
        self,
        margin_s: float = 120.0,
        backoff_s: Sequence[float] = (30.0, 60.0, 120.0, 240.0),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._margin_s = margin_s
        self._backoff_s = tuple(backoff_s)
        self._clock = clock
        self._sleep = sleep
        self._token: Optional[str] = None
        self._valid_until = 0.0
        self.logins = 0

    def invalidate(self) -> None:
        """Force a new login on the next :meth:`get` (e.g. after an HTTP 401)."""
        self._token = None

    def get(self) -> Optional[str]:
        """Return a valid token, logging in only when needed."""
        if self._token and self._clock() < self._valid_until:
            return self._token
        waits = list(self._backoff_s)
        while True:
            token, expires_in = _login()
            self.logins += 1
            if token:
                self._token = token
                self._valid_until = self._clock() + max(
                    expires_in - self._margin_s, 30.0
                )
                return token
            if not waits:
                return None
            wait = waits.pop(0)
            logger.warning("CDSE login failed; retrying in %.0f s.", wait)
            self._sleep(wait)


def _login() -> Tuple[Optional[str], float]:
    """One password-grant login: ``(token, expires_in_s)`` or ``(None, 0)``."""
    username = os.getenv("CDSE_USER")
    password = os.getenv("CDSE_PASSWORD")
    if not username or not password:
        logger.error("CDSE credentials not provided (CDSE_USER / CDSE_PASSWORD).")
        return None, 0.0
    payload = {
        "client_id": _CLIENT_ID,
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        response = requests.post(_TOKEN_URL, data=payload, timeout=_TIMEOUT_S)
        response.raise_for_status()
        body = response.json()
        logger.info("CDSE authentication successful.")
        return body["access_token"], float(body.get("expires_in", 600))
    except requests.HTTPError as exc:
        logger.error(
            "CDSE authentication HTTP error %s: %s", exc.response.status_code, exc
        )
    except Exception as exc:
        logger.error("CDSE authentication failed: %s", exc)
    return None, 0.0
