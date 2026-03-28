from fastapi import APIRouter, HTTPException
from typing import Optional
from database import get_connection

router = APIRouter()

@router.get("")
def list_matches(
    league_id: Optional[int] = None,
    season_id: Optional[int] = None,
    team: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    has_score: Optional[str] = None,  # "true" = results only, "false" = fixtures only
    limit: int = 50,
    offset: int = 0
):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT m.id, m.match_date, m.gameweek, m.start_time, m.score_raw,
               m.home_score, m.away_score, m.attendance, m.venue, m.referee, m.round,
               ht.name AS home_team, at.name AS away_team,
               ht.logo_url AS home_logo, at.logo_url AS away_logo,
               l.name AS league, s.name AS season
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        JOIN leagues l ON l.id = m.league_id
        JOIN seasons s ON s.id = m.season_id
        WHERE 1=1
    """
    params = []
    if league_id:
        query += " AND m.league_id = %s"; params.append(league_id)
    if season_id:
        query += " AND m.season_id = %s"; params.append(season_id)
    if team:
        query += " AND (ht.name ILIKE %s OR at.name ILIKE %s)"
        params += [f"%{team}%", f"%{team}%"]
    if date_from:
        query += " AND m.match_date >= %s"; params.append(date_from)
    if date_to:
        query += " AND m.match_date <= %s"; params.append(date_to)
    if has_score == "true":
        query += " AND m.home_score IS NOT NULL"
        query += " ORDER BY m.match_date DESC LIMIT %s OFFSET %s"
    elif has_score == "false":
        # Only show genuinely upcoming fixtures: no score AND date is today or future.
        # Without the date guard, past matches that were never synced with scores
        # appear as upcoming fixtures (e.g. Aug 2024 games with home_score IS NULL).
        query += " AND m.home_score IS NULL AND m.match_date >= CURRENT_DATE"
        query += " ORDER BY m.match_date ASC LIMIT %s OFFSET %s"
    else:
        query += " ORDER BY m.match_date DESC LIMIT %s OFFSET %s"
    params += [limit, offset]
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows

@router.get("/{match_id}")
def get_match(match_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT m.*, ht.name AS home_team, ht.logo_url AS home_logo,
               at.name AS away_team, at.logo_url AS away_logo,
               l.name AS league, s.name AS season
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        JOIN leagues l ON l.id = m.league_id
        JOIN seasons s ON s.id = m.season_id
        WHERE m.id = %s
    """, (match_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Match not found")
    return row

@router.put("/{match_id}")
def update_match_result(match_id: int, home_score: int, away_score: int, score_raw: str = None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE matches SET home_score=%s, away_score=%s, score_raw=%s
        WHERE id=%s RETURNING *
    """, (home_score, away_score, score_raw, match_id))
    row = cur.fetchone()
    conn.commit()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Match not found")
    return row

@router.delete("/{match_id}")
def delete_match(match_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM matches WHERE id=%s RETURNING id", (match_id,))
    row = cur.fetchone()
    conn.commit()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Match not found")
    return {"deleted": match_id}
