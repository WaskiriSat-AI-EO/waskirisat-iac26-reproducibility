"""CDSE OAuth2 token management.

Obtains short-lived access tokens from the Copernicus Data Space
Ecosystem identity service using the Resource Owner Password
Credentials grant.  Credentials are read from environment variables
``CDSE_USER`` and ``CDSE_PASSWORD`` (typically via a ``.env`` file).
"""

import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
_CLIENT_ID = "cdse-public"
_TIMEOUT_S  = 30


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
        "client_id":  _CLIENT_ID,
        "username":   username,
        "password":   password,
        "grant_type": "password",
    }

    try:
        response = requests.post(_TOKEN_URL, data=payload, timeout=_TIMEOUT_S)
        response.raise_for_status()
        token = response.json()["access_token"]
        logger.info("CDSE authentication successful.")
        return token
    except requests.HTTPError as exc:
        logger.error("CDSE authentication HTTP error %s: %s", exc.response.status_code, exc)
    except Exception as exc:
        logger.error("CDSE authentication failed: %s", exc)

    return None
