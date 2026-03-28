from fastapi import APIRouter
from typing import Optional
from database import get_connection

router = APIRouter()

@router.get("")
def get_squad_stats(
    team_id: Optional[int] = None,
    league_id: Optional[int] = None,
    season_id: Optional[int] = None,
    split: Optional[str] = None  # 'for' or 'against'
):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT ts.*, t.name AS team, t.logo_url AS logo_url, l.name AS league, s.name AS season
        FROM team_squad_stats ts
        JOIN teams t ON t.id = ts.team_id
        JOIN leagues l ON l.id = ts.league_id
        JOIN seasons s ON s.id = ts.season_id
        WHERE 1=1
    """
    params = []
    if team_id:
        query += " AND ts.team_id = %s"; params.append(team_id)
    if league_id:
        query += " AND ts.league_id = %s"; params.append(league_id)
    if season_id:
        query += " AND ts.season_id = %s"; params.append(season_id)
    if split:
        query += " AND ts.split = %s"; params.append(split)
    query += " ORDER BY t.name, ts.split"
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows
