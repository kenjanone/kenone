"""
Admin Settings API
==================
Endpoints:
  GET  /api/settings          → list all app settings
  GET  /api/settings/{key}    → get one setting by key
  PUT  /api/settings/{key}    → create or update a setting

Settings stored in the `app_settings` table (key/value pairs).

Known keys:
  consensus_interval_hours   int   How often the auto-consensus job runs (default: 6)
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from database import get_connection
from routes.deps import require_admin

log    = logging.getLogger(__name__)
router = APIRouter()


class SettingUpdate(BaseModel):
    value:       str
    description: Optional[str] = None


@router.get("")
def get_all_settings():
    """Return all app settings."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT key, value, description, updated_at FROM app_settings ORDER BY key")
        rows = cur.fetchall()
        return {"settings": [dict(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


@router.get("/{key}")
def get_setting(key: str):
    """Return a single setting by key."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT key, value, description, updated_at FROM app_settings WHERE key = %s", (key,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Setting '{key}' not found.")
        return dict(row)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


@router.put("/{key}")
def upsert_setting(key: str, body: SettingUpdate, _admin: dict = Depends(require_admin)):
    """Create or update a setting. Returns the updated row."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO app_settings (key, value, description, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (key) DO UPDATE
              SET value       = EXCLUDED.value,
                  description = COALESCE(EXCLUDED.description, app_settings.description),
                  updated_at  = NOW()
            RETURNING key, value, description, updated_at
        """, (key, body.value, body.description))
        row = cur.fetchone()
        conn.commit()
        log.info("Setting updated: %s = %s", key, body.value)
        return {"updated": True, "setting": dict(row)}
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()
