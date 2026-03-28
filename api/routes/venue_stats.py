from fastapi import APIRouter
from typing import Optional
from database import get_connection

router = APIRouter()

@router.get("")
def get_venue_stats(
    team_id: Optional[int] = None,
    league_id: Optional[int] = None,
    season_id: Optional[int] = None,
    venue: Optional[str] = None  # 'home' or 'away'
):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT tv.*, t.name AS team, t.logo_url, l.name AS league, s.name AS season
        FROM team_venue_stats tv
        JOIN teams t ON t.id = tv.team_id
        JOIN leagues l ON l.id = tv.league_id
        JOIN seasons s ON s.id = tv.season_id
        WHERE 1=1
    """
    params = []
    if team_id:
        query += " AND tv.team_id = %s"; params.append(team_id)
    if league_id:
        query += " AND tv.league_id = %s"; params.append(league_id)
    if season_id:
        query += " AND tv.season_id = %s"; params.append(season_id)
    if venue:
        query += " AND tv.venue = %s"; params.append(venue)
    query += " ORDER BY t.name, tv.venue"
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows
