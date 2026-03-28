from fastapi import APIRouter, HTTPException
from database import get_connection

router = APIRouter()

@router.get("")
def list_leagues():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM leagues ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return rows

@router.get("/{league_id}")
def get_league(league_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM leagues WHERE id = %s", (league_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="League not found")
    return row

@router.post("")
def create_league(name: str, country: str = None, fbref_id: int = None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO leagues (name, country, fbref_id) VALUES (%s, %s, %s) RETURNING *",
        (name, country, fbref_id)
    )
    row = cur.fetchone()
    conn.commit()
    conn.close()
    return row

@router.delete("/{league_id}")
def delete_league(league_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM leagues WHERE id = %s RETURNING id", (league_id,))
    row = cur.fetchone()
    conn.commit()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="League not found")
    return {"deleted": league_id}
