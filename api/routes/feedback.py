"""
User Feedback API
=================
POST /api/feedback                  — Submit feedback
GET  /api/feedback                  — List all feedback (most recent first)
POST /api/feedback/{id}/reply       — Admin reply to feedback
PATCH /api/feedback/{id}            — Mark feedback as reviewed (admin)
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from database import get_connection
import logging

router = APIRouter()
log = logging.getLogger(__name__)


class FeedbackIn(BaseModel):
    name: Optional[str] = None          # optional display name / anonymous
    email: Optional[str] = None         # optional
    category: str = "general"           # 'prediction', 'feature', 'bug', 'general'
    message: str
    rating: Optional[int] = None        # 1-5 star rating


class ReplyIn(BaseModel):
    reply_text: str
    admin_email: Optional[str] = None


# ── POST /api/feedback ─────────────────────────────────────────────────────────
@router.post("")
def submit_feedback(body: FeedbackIn):
    """Submit user feedback. Creates the table if it doesn't exist yet."""
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    if body.rating is not None and not (1 <= body.rating <= 5):
        raise HTTPException(status_code=422, detail="Rating must be 1–5.")

    conn = get_connection()
    cur = conn.cursor()
    try:
        # Auto-create table on first use (safe for Railway / Supabase)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id          SERIAL PRIMARY KEY,
                name        TEXT,
                email       TEXT,
                category    TEXT DEFAULT 'general',
                message     TEXT NOT NULL,
                rating      SMALLINT CHECK (rating BETWEEN 1 AND 5),
                reviewed    BOOLEAN DEFAULT FALSE,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            INSERT INTO user_feedback (name, email, category, message, rating)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, created_at
        """, (
            body.name or "Anonymous",
            body.email,
            body.category,
            body.message.strip(),
            body.rating,
        ))
        row = cur.fetchone()
        conn.commit()
        return {"success": True, "id": row["id"], "created_at": row["created_at"].isoformat()}
    except Exception as e:
        conn.rollback()
        log.error("feedback insert failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ── GET /api/feedback ──────────────────────────────────────────────────────────
@router.get("")
def list_feedback(
    category: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    reviewed: Optional[bool] = Query(None),
):
    """List feedback, newest first. Useful for an admin view."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        where_clauses = []
        params = []
        if category:
            where_clauses.append("category = %s")
            params.append(category)
        if reviewed is not None:
            where_clauses.append("reviewed = %s")
            params.append(reviewed)
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        params.append(limit)
        cur.execute(f"""
            SELECT id, name, email, category, message, rating, reviewed,
                   reply_text, replied_at, replied_by, created_at
            FROM user_feedback
            {where}
            ORDER BY created_at DESC
            LIMIT %s
        """, params)
        rows = cur.fetchall()
        return {"count": len(rows), "feedback": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ── PATCH /api/feedback/{id} ───────────────────────────────────────────────────
@router.patch("/{feedback_id}")
def mark_reviewed(feedback_id: int):
    """Mark a feedback item as reviewed."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE user_feedback SET reviewed = TRUE WHERE id = %s RETURNING id",
            (feedback_id,)
        )
        row = cur.fetchone()
        conn.commit()
        if not row:
            raise HTTPException(status_code=404, detail="Feedback not found.")
        return {"success": True, "id": feedback_id}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ── POST /api/feedback/{id}/reply ──────────────────────────────────────────────
@router.post("/{feedback_id}/reply")
def reply_to_feedback(feedback_id: int, body: ReplyIn):
    """Admin replies to a feedback message. Stores reply text and timestamp."""
    if not body.reply_text.strip():
        raise HTTPException(status_code=422, detail="Reply cannot be empty.")
    conn = get_connection()
    cur = conn.cursor()
    try:
        for col, coldef in [
            ("reply_text",  "TEXT"),
            ("replied_at",  "TIMESTAMPTZ"),
            ("replied_by",  "TEXT"),
        ]:
            cur.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE user_feedback ADD COLUMN IF NOT EXISTS {col} {coldef};
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$;
            """)
        cur.execute(
            """
            UPDATE user_feedback
               SET reply_text = %s,
                   replied_at  = NOW(),
                   replied_by  = %s,
                   reviewed    = TRUE
             WHERE id = %s
            RETURNING id
            """,
            (body.reply_text.strip(), body.admin_email or "admin", feedback_id)
        )
        row = cur.fetchone()
        conn.commit()
        if not row:
            raise HTTPException(status_code=404, detail="Feedback not found.")
        return {"success": True, "id": feedback_id}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        log.error("feedback reply failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
