from fastapi import APIRouter
from typing import Optional
from database import get_connection

router = APIRouter()

@router.get("")
def get_players(
    season_id: Optional[int] = None,
    team_id: Optional[int] = None,
    league_id: Optional[int] = None,
    position: Optional[str] = None,
    min_goals: Optional[int] = None,
    search: Optional[str] = None,
    sort_by: str = "goals",
    limit: int = 50,
    offset: int = 0
):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT ps.id, ps.player_name, ps.nationality, ps.position,
               ps.age, ps.games, ps.games_starts, ps.minutes, ps.minutes_90s,
               ps.goals, ps.assists, ps.standard_stats,
               t.name AS team, t.logo_url AS logo_url, l.name AS league, s.name AS season
        FROM player_stats ps
        LEFT JOIN teams t ON t.id = ps.team_id
        LEFT JOIN leagues l ON l.id = t.league_id
        JOIN seasons s ON s.id = ps.season_id
        WHERE 1=1
    """
    params = []
    if season_id:
        query += " AND ps.season_id = %s"; params.append(season_id)
    if team_id:
        query += " AND ps.team_id = %s"; params.append(team_id)
    if league_id:
        query += " AND t.league_id = %s"; params.append(league_id)
    if position:
        query += " AND ps.position ILIKE %s"; params.append(f"%{position}%")
    if min_goals:
        query += " AND ps.goals >= %s"; params.append(min_goals)
    if search:
        query += " AND ps.player_name ILIKE %s"; params.append(f"%{search}%")

    allowed_sort = ["goals", "assists", "games", "minutes", "player_name"]
    sort_col = sort_by if sort_by in allowed_sort else "goals"
    query += f" ORDER BY ps.{sort_col} DESC LIMIT %s OFFSET %s"
    params += [limit, offset]

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows

@router.get("/top-scorers")
def top_scorers(season_id: Optional[int] = None, league_id: Optional[int] = None, limit: int = 20):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT ps.player_name, ps.nationality, ps.position, ps.goals, ps.assists,
               ps.games, ps.minutes_90s, t.name AS team, t.logo_url AS logo_url, l.name AS league, s.name AS season
        FROM player_stats ps
        LEFT JOIN teams t ON t.id = ps.team_id
        LEFT JOIN leagues l ON l.id = t.league_id
        JOIN seasons s ON s.id = ps.season_id
        WHERE ps.goals IS NOT NULL
    """
    params = []
    if season_id:
        query += " AND ps.season_id = %s"; params.append(season_id)
    if league_id:
        query += " AND t.league_id = %s"; params.append(league_id)
    query += " ORDER BY ps.goals DESC LIMIT %s"
    params.append(limit)
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows
