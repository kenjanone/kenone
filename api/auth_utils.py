"""
auth_utils.py — Shared JWT + password helpers for PlusOne backend.

Uses:
  - python-jose  for JWT signing/verification
  - passlib      for bcrypt password hashing

Required env var:
  SECRET_KEY    — a long random string, e.g. `openssl rand -hex 32`
                  Never commit this to git.

Token lifetime:   60 minutes (configurable via ACCESS_TOKEN_EXPIRE_MINUTES)
Algorithm:        HS256
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-use-a-long-random-string")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "43200"))

# ── Password hashing ──────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return bcrypt hash of the plain-text password."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    return _pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> str:
    """
    Sign and return a JWT containing `data`.
    The token expires in `expires_minutes` minutes (default: ACCESS_TOKEN_EXPIRE_MINUTES).
    """
    payload = data.copy()
    expire  = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload["exp"] = expire
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT.
    Returns the payload dict if valid, or None if expired / tampered.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
