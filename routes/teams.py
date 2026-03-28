from fastapi import APIRouter, HTTPException
from typing import Optional
from database import get_connection

router = APIRouter()

@router.get("")
def list_teams(league_id: Optional[int] = None):
    conn = get_connection()
    cur = conn.cursor()
    if league_id:
        cur.execute("SELECT t.*, l.name AS league FROM teams t JOIN leagues l ON l.id=t.league_id WHERE t.league_id=%s ORDER BY t.name", (league_id,))
    else:
        cur.execute("SELECT t.*, l.name AS league FROM teams t JOIN leagues l ON l.id=t.league_id ORDER BY l.name, t.name")
    rows = cur.fetchall()
    conn.close()
    return rows

@router.get("/{team_id}")
def get_team(team_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT t.*, l.name AS league FROM teams t JOIN leagues l ON l.id=t.league_id WHERE t.id=%s", (team_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Team not found")
    return row

@router.get("/{team_id}/head-to-head/{opponent_id}")
def head_to_head(team_id: int, opponent_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT m.match_date, m.gameweek, s.name AS season, l.name AS league,
               ht.name AS home_team, ht.logo_url AS home_logo, m.home_score, m.away_score, at.name AS away_team, at.logo_url AS away_logo,
               m.venue, m.score_raw
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        JOIN leagues l ON l.id = m.league_id
        JOIN seasons s ON s.id = m.season_id
        WHERE (m.home_team_id=%s AND m.away_team_id=%s)
           OR (m.home_team_id=%s AND m.away_team_id=%s)
        ORDER BY m.match_date DESC
    """, (team_id, opponent_id, opponent_id, team_id))
    rows = cur.fetchall()
    conn.close()
    return rows

@router.delete("/{team_id}")
def delete_team(team_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM teams WHERE id=%s RETURNING id", (team_id,))
    row = cur.fetchone()
    conn.commit()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"deleted": team_id}
