"""API authentication via API keys.

Simple but effective: API keys stored in config or environment variable.
Supports multiple keys for different clients.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# Header name for API key
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthManager:
    """Manages API key authentication."""

    def __init__(self, api_keys: list[str] | None = None):
        """Initialize with a list of valid API keys.

        Keys can be provided directly or via AMEM_API_KEYS env var
        (comma-separated).
        """
        self._enabled = True
        if api_keys:
            self._keys = set(api_keys)
        else:
            env_keys = os.environ.get("AMEM_API_KEYS", "")
            if env_keys:
                self._keys = set(k.strip() for k in env_keys.split(",") if k.strip())
            else:
                # No keys configured → auth disabled
                self._keys = set()
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and len(self._keys) > 0

    def validate(self, key: Optional[str]) -> bool:
        """Validate an API key."""
        if not self.enabled:
            return True  # Auth disabled
        if not key:
            return False
        return key in self._keys

    def require(self, key: Optional[str]):
        """Validate or raise HTTPException."""
        if not self.validate(key):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    @staticmethod
    def generate_key() -> str:
        """Generate a secure random API key."""
        return f"amem_{secrets.token_urlsafe(32)}"


def get_auth_dependency(auth_manager: AuthManager):
    """Create a FastAPI dependency for API key auth."""
    async def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)):
        auth_manager.require(api_key)
    return verify_api_key
