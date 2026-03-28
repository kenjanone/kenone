from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from database import get_connection
from routes.deps import require_admin
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

class SyncEnrichmentPayload(BaseModel):
    league: str
    odds: Optional[List[dict]] = []
    injuries: Optional[List[dict]] = []
    clubelo: Optional[List[dict]] = []

def safe_num(val):
    if val is None or val == "": return None
    try: return float(val)
    except: return None

def _get_league(cur, name: str):
    cur.execute("SELECT id FROM leagues WHERE name ILIKE %s LIMIT 1", (name.strip(),))
    res = cur.fetchone()
    if res: return res["id"]
    cur.execute("INSERT INTO leagues (name) VALUES (%s) RETURNING id", (name.title().strip(),))
    return cur.fetchone()["id"]

def _get_team(cur, name: str, league_id: int, team_cache: dict):
    # In-memory deduplication during massive sync batches
    cache_key = (name.lower().strip(), league_id)
    if cache_key in team_cache:
        return team_cache[cache_key]

    # Primitive mapping, assumes exact or subset matching will be handled by data normalizers later
    cur.execute("SELECT id FROM teams WHERE name ILIKE %s AND league_id = %s LIMIT 1", (name.strip(), league_id))
    res = cur.fetchone()
    if res:
        team_cache[cache_key] = res["id"]
        return res["id"]
        
    cur.execute("INSERT INTO teams (name, league_id) VALUES (%s, %s) RETURNING id", (name.strip(), league_id))
    new_id = cur.fetchone()["id"]
    team_cache[cache_key] = new_id
    return new_id

def _find_match(cur, home_team_id, away_team_id, match_date: str):
    if not match_date:
        # Fallback if no date is found
        cur.execute("""
            SELECT id FROM matches 
            WHERE home_team_id = %s AND away_team_id = %s 
            ORDER BY match_date DESC LIMIT 1
        """, (home_team_id, away_team_id))
    else:
        # Odds files/FBref dates can differ by 1-2 days due to timezone disparities.
        # We search within a generous 3-day window of the declared match date to perfectly align historical matches.
        cur.execute("""
            SELECT id FROM matches 
            WHERE home_team_id = %s AND away_team_id = %s 
              AND match_date >= %s::date - INTERVAL '3 days' 
              AND match_date <= %s::date + INTERVAL '3 days'
            LIMIT 1
        """, (home_team_id, away_team_id, match_date, match_date))
        
    res = cur.fetchone()
    return res["id"] if res else None

def _parse_odds_date(d_str):
    if not d_str: return None
    try:
        if "/" in d_str:
            parts = d_str.split("/")
            if len(parts[2]) == 2: parts[2] = "20" + parts[2]
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
        return d_str
    except:
        return None

@router.post("/enrichment")
def sync_enrichment(payload: SyncEnrichmentPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        default_league_id = _get_league(cur, payload.league)
        team_cache = {}
        
        # Helper to determine the correct league dynamically per row.
        # This prevents team-corruption when batch syncing multi-league CSV files (like fixtures.csv)
        def _resolve_league(row_obj):
            row_league = row_obj.get("League") or row_obj.get("league")
            if row_league and row_league != "Unknown League" and row_league != "Global Enrichment Sync":
                return _get_league(cur, row_league)
            return default_league_id
            
        odds_count = 0
        injuries_count = 0
        clubelo_count = 0
        
        # ── 1. Match Odds ───────────────────────────────────────────────────────────
        if payload.odds:
            for row in payload.odds:
                home_name = row.get("HomeTeam") or row.get("Home")
                away_name = row.get("AwayTeam") or row.get("Away")
                if not home_name or not away_name: continue
                
                l_id = _resolve_league(row)
                h_id = _get_team(cur, home_name, l_id, team_cache)
                a_id = _get_team(cur, away_name, l_id, team_cache)
                m_date = _parse_odds_date(row.get("Date"))
                m_id = _find_match(cur, h_id, a_id, m_date)
                
                if m_id:
                    h_odds = safe_num(row.get("B365H"))
                    d_odds = safe_num(row.get("B365D"))
                    a_odds = safe_num(row.get("B365A"))
                    cur.execute("""
                        INSERT INTO match_odds (match_id, b365_home_win, b365_draw, b365_away_win, raw_data)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (match_id) DO UPDATE SET
                            b365_home_win=EXCLUDED.b365_home_win,
                            b365_draw=EXCLUDED.b365_draw,
                            b365_away_win=EXCLUDED.b365_away_win,
                            raw_data=EXCLUDED.raw_data,
                            scraped_at=NOW()
                    """, (m_id, h_odds, d_odds, a_odds, json.dumps(row)))
                    odds_count += 1

        # ── 2. Player Injuries ──────────────────────────────────────────────────────
        if payload.injuries:
            for row in payload.injuries:
                club = row.get("Club")
                player = row.get("Player")
                if not club or not player: continue
                
                l_id = _resolve_league(row)
                t_id = _get_team(cur, club, l_id, team_cache)
                cur.execute("""
                    INSERT INTO player_injuries (team_id, player_name, injury_type, return_date, raw_data)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (team_id, player_name) DO UPDATE SET
                        injury_type=EXCLUDED.injury_type,
                        return_date=EXCLUDED.return_date,
                        raw_data=EXCLUDED.raw_data,
                        scraped_at=NOW()
                """, (t_id, player, row.get("Injury"), row.get("Return_Date"), json.dumps(row)))
                injuries_count += 1

        # ── 3. Team ClubElo ─────────────────────────────────────────────────────────
        if payload.clubelo:
            for row in payload.clubelo:
                raw_data = row.get("raw", row)
                home = raw_data.get("Home") or raw_data.get("HomeTeam") or row.get("home")
                away = raw_data.get("Away") or raw_data.get("AwayTeam") or row.get("away")
                target_date = raw_data.get("Date") or row.get("date")
                
                l_id = _resolve_league(row)
                
                # We may not have exact Elo numbers from the Fixtures endpoint, but we save the predicted Probs!
                for tm, elo_val in [(home, raw_data.get("Home_Elo")), (away, raw_data.get("Away_Elo"))]:
                    if not tm or not target_date: continue
                    t_id = _get_team(cur, tm, l_id, team_cache)
                    cur.execute("""
                        INSERT INTO team_clubelo (team_id, elo_date, elo, raw_data)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (team_id, elo_date) DO UPDATE SET
                            elo=EXCLUDED.elo,
                            raw_data=EXCLUDED.raw_data,
                            scraped_at=NOW()
                    """, (t_id, target_date, safe_num(elo_val), json.dumps(row)))
                    clubelo_count += 1
                
        conn.commit()
        return {
            "success": True, 
            "inserted": {
                "match_odds": odds_count,
                "player_injuries": injuries_count,
                "team_clubelo": clubelo_count
            }
        }
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Enrichment Sync Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
