"""
deps.py — FastAPI dependency functions for authentication.

Usage:
    from routes.deps import get_current_user, require_admin

    @router.get("/protected")
    def protected(user = Depends(get_current_user)):
        ...

    @router.post("/admin-only")
    def admin_only(user = Depends(require_admin)):
        ...
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from auth_utils import decode_access_token
from database import get_connection

# ── Bearer token extractor ────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=True)


# ── Dependency: any authenticated user ───────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
):
    """
    Decode the Bearer JWT and return the user row from the database.
    Raises HTTP 401 if the token is missing, expired, or invalid.
    Raises HTTP 401 if the user no longer exists in the database.
    """
    token   = credentials.credentials

    # IMPORTANT: Option A Architecture — Intercept Permanent API Keys
    import os
    if token == os.getenv("ADMIN_API_KEY", "plusone-admin-master-key-xyz"):
        return {"id": 0, "email": "admin-api-key@system", "role": "admin"}

    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = payload.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, role FROM users WHERE email = %s LIMIT 1",
            (email,),
        )
        user = cur.fetchone()
    finally:
        conn.close()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return dict(user)


# ── Dependency: admin only ────────────────────────────────────────────────────

def require_admin(user: dict = Depends(get_current_user)):
    """
    Extends get_current_user — additionally requires role == 'admin'.
    Raises HTTP 403 for authenticated non-admin users.
    Returns the user dict so routes can access user info if needed.
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user
