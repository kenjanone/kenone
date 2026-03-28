"""
routes/auth.py — JWT authentication for PlusOne.

Endpoints:
  POST /api/auth/register              — self-signup (open, 7-day trial)
  POST /api/auth/login                 — email + password → JWT token
  POST /api/auth/payment               — user submits payment reference (authenticated)
  POST /api/auth/create-user           — admin creates a new account
  GET  /api/auth/me                    — current user info (requires token)
  GET  /api/auth/users                 — list all users (admin only)
  DELETE /api/auth/users/{user_id}     — delete a user (admin only)
  PUT  /api/auth/users/{user_id}/role  — change role (admin only)
  PUT  /api/auth/users/{user_id}/activate — activate/set plan (admin only)
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Optional
import os

from auth_utils import hash_password, verify_password, create_access_token  # root-level, not api.auth_utils
from routes.deps import get_current_user, require_admin
from database import get_connection

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    phone: Optional[str] = None

class PaymentRequest(BaseModel):
    payment_ref: str
    payment_amount: str
    payment_method: str          # "MTN MoMo" | "Airtel Money" | "Bank Transfer" | etc.
    plan: str                    # "basic" | "pro" | "elite"

class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: Optional[str] = "user"
    phone: Optional[str] = None

class ChangeRoleRequest(BaseModel):
    role: str

class ActivateRequest(BaseModel):
    plan: str                    # "basic" | "pro" | "elite" | "admin"
    months: Optional[int] = 1   # how many months to activate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user_row_to_dict(row: dict) -> dict:
    """Convert a DB user row to the standard API user dict."""
    def iso(v):
        return v.isoformat() if v else None
    return {
        "id":                    row["id"],
        "email":                 row["email"],
        "role":                  row["role"],
        "plan":                  row.get("plan", "trial"),
        "phone":                 row.get("phone"),
        "is_active":             row.get("is_active", False),
        "trial_expires_at":      iso(row.get("trial_expires_at")),
        "subscription_expires_at": iso(row.get("subscription_expires_at")),
        "payment_ref":           row.get("payment_ref"),
        "payment_method":        row.get("payment_method"),
        "payment_amount":        row.get("payment_amount"),
        "created_at":            iso(row.get("created_at")),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/auth/register", status_code=201)
def register(payload: RegisterRequest):
    """
    Public self-signup. Creates account with a 7-day free trial.
    No admin involvement needed.
    """
    email = payload.email.lower().strip()
    if not email:
        raise HTTPException(status_code=422, detail="Email is required.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")

    hashed = hash_password(payload.password)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO users (email, password_hash, role, phone, plan, is_active,
                               trial_expires_at)
            VALUES (%s, %s, 'user', %s, 'trial', false,
                    NOW() + INTERVAL '7 days')
            RETURNING id, email, role, phone, plan, is_active,
                      trial_expires_at, subscription_expires_at,
                      payment_ref, payment_method, payment_amount, created_at
            """,
            (email, hashed, payload.phone),
        )
        new_user = cur.fetchone()
        conn.commit()
    except Exception as exc:
        conn.rollback()
        if "unique" in str(exc).lower():
            raise HTTPException(status_code=409, detail=f"An account with email '{email}' already exists.")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")
    finally:
        conn.close()

    token = create_access_token({"sub": new_user["email"], "role": new_user["role"]})
    return {
        "success": True,
        "message": "Account created! Your 7-day free trial is now active. Submit a payment reference to subscribe after the trial.",
        "token": token,
        "token_type": "bearer",
        "user": _user_row_to_dict(new_user),
    }


@router.post("/api/auth/login")
def login(payload: LoginRequest):
    """Authenticate with email + password. Returns a signed JWT token."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, email, password_hash, role, phone, plan, is_active,
                      trial_expires_at, subscription_expires_at,
                      payment_ref, payment_method, payment_amount, created_at
               FROM users WHERE email = %s LIMIT 1""",
            (payload.email.lower().strip(),),
        )
        user = cur.fetchone()
    finally:
        conn.close()

    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token({"sub": user["email"], "role": user["role"]})
    
    # [ADMIN BYPASS FIX] If the user is an admin, securely push the Permanent Master Key to them
    # so their browser extension can bypass 30-day JWT timeouts on huge scraping uploads.
    api_key = os.getenv("ADMIN_API_KEY") if user["role"] == "admin" else None
    
    return {
        "success": True,
        "token": token,
        "token_type": "bearer",
        "api_key": api_key,
        "user": _user_row_to_dict(user),
    }


@router.post("/api/auth/payment")
def submit_payment(
    payload: PaymentRequest,
    user: dict = Depends(get_current_user),
):
    """
    User submits their payment reference for admin review.
    Saves the reference — admin will verify and activate the plan.
    """
    if payload.plan not in ("basic", "pro", "elite"):
        raise HTTPException(status_code=422, detail="plan must be 'basic', 'pro', or 'elite'.")

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """UPDATE users
               SET payment_ref = %s, payment_amount = %s, payment_method = %s
               WHERE id = %s
               RETURNING id, email, role, phone, plan, is_active,
                         trial_expires_at, subscription_expires_at,
                         payment_ref, payment_method, payment_amount, created_at""",
            (payload.payment_ref, payload.payment_amount, payload.payment_method, user["id"]),
        )
        updated = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to save payment reference.")
    finally:
        conn.close()

    return {
        "success": True,
        "message": "Payment reference received! An admin will verify and activate your plan within 24 hours.",
        "user": _user_row_to_dict(updated),
    }


@router.post("/api/auth/create-user", status_code=201)
def create_user(
    payload: CreateUserRequest,
    _admin: dict = Depends(require_admin),
):
    """Admin-only: create a new user or admin account directly."""
    if payload.role not in ("user", "admin"):
        raise HTTPException(status_code=422, detail="role must be 'user' or 'admin'.")

    email = payload.email.lower().strip()
    if not email:
        raise HTTPException(status_code=422, detail="Email is required.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")

    # Admins get is_active=true and admin plan; users get trial
    plan = "admin" if payload.role == "admin" else "trial"
    is_active = payload.role == "admin"

    hashed = hash_password(payload.password)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO users (email, password_hash, role, phone, plan, is_active,
                                  trial_expires_at)
               VALUES (%s, %s, %s, %s, %s, %s,
                       NOW() + INTERVAL '7 days')
               RETURNING id, email, role, phone, plan, is_active,
                         trial_expires_at, subscription_expires_at,
                         payment_ref, payment_method, payment_amount, created_at""",
            (email, hashed, payload.role, payload.phone, plan, is_active),
        )
        new_user = cur.fetchone()
        conn.commit()
    except Exception as exc:
        conn.rollback()
        if "unique" in str(exc).lower():
            raise HTTPException(status_code=409, detail=f"A user with email '{email}' already exists.")
        raise HTTPException(status_code=500, detail="Failed to create user.")
    finally:
        conn.close()

    return {"success": True, "user": _user_row_to_dict(new_user)}


@router.get("/api/auth/me")
def me(user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's full profile."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, email, role, phone, plan, is_active,
                      trial_expires_at, subscription_expires_at,
                      payment_ref, payment_method, payment_amount, created_at
               FROM users WHERE id = %s LIMIT 1""",
            (user["id"],),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="User not found.")
    return _user_row_to_dict(row)


@router.get("/api/auth/users")
def list_users(_admin: dict = Depends(require_admin)):
    """Admin only — list all registered users with subscription status."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, email, role, phone, plan, is_active,
                      trial_expires_at, subscription_expires_at,
                      payment_ref, payment_method, payment_amount, created_at
               FROM users ORDER BY id"""
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return {"users": [_user_row_to_dict(r) for r in rows]}


@router.put("/api/auth/users/{user_id}/activate")
def activate_user(
    user_id: int,
    payload: ActivateRequest,
    _admin: dict = Depends(require_admin),
):
    """Admin only — set a user's plan and activate their subscription."""
    if payload.plan not in ("basic", "pro", "elite", "admin"):
        raise HTTPException(status_code=422, detail="plan must be 'basic', 'pro', 'elite', or 'admin'.")
    if payload.months < 1 or payload.months > 12:
        raise HTTPException(status_code=422, detail="months must be between 1 and 12.")

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """UPDATE users
               SET plan = %s, is_active = true,
                   subscription_expires_at = NOW() + (%s * INTERVAL '30 days')
               WHERE id = %s
               RETURNING id, email, role, phone, plan, is_active,
                         trial_expires_at, subscription_expires_at,
                         payment_ref, payment_method, payment_amount, created_at""",
            (payload.plan, payload.months, user_id),
        )
        updated = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to activate subscription.")
    finally:
        conn.close()

    if updated is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    return {"success": True, "user": _user_row_to_dict(updated)}


@router.put("/api/auth/users/{user_id}/role")
def change_role(
    user_id: int,
    payload: ChangeRoleRequest,
    _admin: dict = Depends(require_admin),
):
    """Admin only — change a user's role between 'user' and 'admin'."""
    if payload.role not in ("user", "admin"):
        raise HTTPException(status_code=422, detail="role must be 'user' or 'admin'.")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET role = %s WHERE id = %s RETURNING id, email, role",
            (payload.role, user_id),
        )
        updated = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to update role.")
    finally:
        conn.close()

    if updated is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    return {"success": True, "user": dict(updated)}


@router.delete("/api/auth/users/{user_id}", status_code=200)
def delete_user(
    user_id: int,
    admin: dict = Depends(require_admin),
):
    """Admin only — delete a user account. Cannot delete yourself."""
    if admin["id"] == user_id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id = %s RETURNING id", (user_id,))
        deleted = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete user.")
    finally:
        conn.close()

    if deleted is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    return {"success": True, "deleted_id": user_id}
