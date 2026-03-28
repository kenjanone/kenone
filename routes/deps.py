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
import os

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
    # --- [NEW] ADMIN API KEY BYPASS -------------------------------------------
    # We check if the request provides the permanent master key in the headers.
    # If it matches our extremely secure environment variable, we grant it INSTANT
    # admin privileges, entirely bypassing the 30-day JWT decoding.
    # This guarantees that a 4-hour batch scrape running overnight will NEVER expire!
    master_key = os.getenv("ADMIN_API_KEY")
    token = credentials.credentials
    
    if master_key and token == master_key:
        return {"id": 0, "email": "admin@api-key.system", "role": "admin"}
    # -------------------------------------------------------------------------

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
