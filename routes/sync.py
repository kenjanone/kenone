import json
import re
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Any
from database import get_connection
from routes.deps import require_admin

def safe_num(val):
    """Safely convert FBref values to a number, returning None for non-numeric."""
    if val is None:
        return None
    s = str(val).strip().replace(",", "").replace("%", "").replace("N/A", "").replace("nan", "")
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


def safe_text(val):
    """Extract plain text from a value that may be a dict/link object or plain string."""
    if val is None:
        return ""
    if isinstance(val, dict):
        return str(val.get("text", val.get("name", ""))).strip()
    s = str(val).strip()
    if s.startswith("{") and ("'text'" in s or '"text"' in s):
        try:
            import ast
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                return str(d.get("text", d.get("name", ""))).strip()
        except Exception:
            pass
    return s


def trunc(val, max_len: int):
    """Cap a string to max_len to respect VARCHAR column limits."""
    if val is None:
        return None
    s = str(val).strip()
    return s[:max_len] if s else None


def safe_age_int(val):
    """Convert FBref age/birth_year to an integer."""
    if val is None:
        return None
    s = safe_text(val) if isinstance(val, dict) else str(val).strip()
    s = s.replace(",", "").strip()
    s = s.split("-")[0].split(".")[0].strip()
    try:
        return int(s) if s else None
    except (ValueError, TypeError):
        return None


router = APIRouter()


from routes.prediction_log import do_evaluate_predictions

def _auto_evaluate_predictions(conn) -> int:
    """
    Grade all unevaluated prediction_log rows whose match is now complete.
    Called automatically after each sync so performance metrics stay current.
    Returns the number of rows updated.
    """
    try:
        updated = do_evaluate_predictions(conn)
        conn.commit()
        return updated
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        return 0

class TableData(BaseModel):
    headers: List[str] = []
    rows: List[List[Any]] = []
    rowCount: Optional[int] = None

class SyncPayload(BaseModel):
    league: str
    season: str
    tables: Optional[List[TableData]] = None
    fixtures: Optional[List[dict]] = None
    stats: Optional[List[dict]] = None
    player_stats: Optional[List[dict]] = None
    playerStats: Optional[List[dict]] = None
    team_logos: Optional[dict] = None

def tables_to_fixtures(tables):
    result = []
    for table in tables:
        headers = [h.strip().lower() for h in table.headers]
        for row in table.rows:
            if len(row) < 3:
                continue
            r = dict(zip(headers, row))
            home = safe_text(r.get("home_team", r.get("home", r.get("home team", ""))))
            away = safe_text(r.get("away_team", r.get("away", r.get("away team", ""))))
            if not home or not away or home.lower() in ("home", "home_team", ""):
                continue
            result.append({
                "home_team":  home,
                "away_team":  away,
                "date":       safe_text(r.get("date", r.get("dates", ""))),
                "start_time": safe_text(r.get("start_time", r.get("time", ""))),
                "score":      trunc(safe_text(r.get("score", "")), 30),
                "gameweek":   safe_text(r.get("gameweek", r.get("wk", r.get("round", "")))),
                "dayofweek":  safe_text(r.get("dayofweek", r.get("day", ""))),
                "venue":      safe_text(r.get("venue", "")),
                "attendance": safe_num(r.get("attendance", None)),
                "referee":    safe_text(r.get("referee", "")),
                "round":      trunc(safe_text(r.get("round", r.get("gameweek", ""))), 100),
            })
    return result


def tables_to_squad_stats(tables):
    result = []
    for table in tables:
        headers = [h.strip().lower() for h in table.headers]
        for row in table.rows:
            if len(row) < 2:
                continue
            r = dict(zip(headers, row))
            team = safe_text(r.get("squad", r.get("team", "")))
            if not team or team.lower() in ("squad", "team", ""):
                continue
            extra = {k: v for k, v in r.items() if k not in ("squad", "team")}
            result.append({
                "team": team,
                "players_used": r.get("# pl", r.get("players used", r.get("players_used", None))),
                "avg_age": r.get("age", r.get("avg age", None)),
                "possession": r.get("poss", r.get("possession", None)),
                "games": r.get("mp", r.get("games", None)),
                "games_starts": r.get("starts", r.get("games_starts", None)),
                "minutes": r.get("min", r.get("minutes", None)),
                "minutes_90s": r.get("90s", r.get("minutes_90s", None)),
                "goals": r.get("gls", r.get("goals", None)),
                "assists": r.get("ast", r.get("assists", None)),
                "standard_stats": extra,
            })
    return result


def tables_to_player_stats(tables):
    result = []
    for table in tables:
        headers = [h.strip().lower() for h in table.headers]
        for row in table.rows:
            if len(row) < 2:
                continue
            r = dict(zip(headers, row))
            name = safe_text(r.get("player", ""))
            if not name or name.lower() in ("player", ""):
                continue
            extra = {k: v for k, v in r.items() if k not in ("player",)}
            raw_nat = safe_text(r.get("nationality", r.get("nation", "")) or "").strip()
            nationality = raw_nat.split()[-1] if raw_nat else ""
            result.append({
                "player":        name,
                "nationality":   trunc(nationality, 10),
                "position":      trunc(safe_text(r.get("position", r.get("pos", ""))), 20),
                "team":          safe_text(r.get("team", r.get("squad", ""))),
                "age":           safe_age_int(r.get("age", None)),
                "birth_year":    safe_age_int(r.get("birth_year", r.get("born", None))),
                "games":         safe_num(r.get("games", r.get("mp", None))),
                "games_starts":  safe_num(r.get("games_starts", r.get("starts", None))),
                "minutes":       safe_num(r.get("minutes", r.get("min", None))),
                "minutes_90s":   safe_num(r.get("minutes_90s", r.get("90s", None))),
                "goals":         safe_num(r.get("goals", r.get("gls", None))),
                "assists":       safe_num(r.get("assists", r.get("ast", None))),
                "standard_stats": extra,
            })
    return result


def tables_to_home_away_stats(tables):
    """Parse FBref home/away split table (Table 2 on stats pages).
    Returns two dicts per team: one for 'home' and one for 'away'.
    FBref column pattern: home_games, home_wins, home_ties, home_losses,
    home_goals_for, home_goals_against, home_goal_diff, home_points,
    and same with 'away_' prefix.
    """
    result = []
    for table in tables:
        headers = [h.strip().lower() for h in table.headers]
        for row in table.rows:
            r = dict(zip(headers, row))
            team = safe_text(r.get("team", r.get("squad", "")))
            if not team or team.lower() in ("team", "squad", ""):
                continue
            # Home row
            result.append({
                "team": team,
                "venue": "home",
                "games":          safe_num(r.get("home_games",         r.get("home_mp", None))),
                "wins":           safe_num(r.get("home_wins",          r.get("home_w",  None))),
                "draws":          safe_num(r.get("home_ties",          r.get("home_d",  None))),
                "losses":         safe_num(r.get("home_losses",        r.get("home_l",  None))),
                "goals_for":      safe_num(r.get("home_goals_for",     r.get("home_gf", None))),
                "goals_against":  safe_num(r.get("home_goals_against", r.get("home_ga", None))),
                "goal_diff":      safe_num(r.get("home_goal_diff",     r.get("home_gd", None))),
                "points":         safe_num(r.get("home_points",        r.get("home_pts",None))),
            })
            # Away row
            result.append({
                "team": team,
                "venue": "away",
                "games":          safe_num(r.get("away_games",         r.get("away_mp", None))),
                "wins":           safe_num(r.get("away_wins",          r.get("away_w",  None))),
                "draws":          safe_num(r.get("away_ties",          r.get("away_d",  None))),
                "losses":         safe_num(r.get("away_losses",        r.get("away_l",  None))),
                "goals_for":      safe_num(r.get("away_goals_for",     r.get("away_gf", None))),
                "goals_against":  safe_num(r.get("away_goals_against", r.get("away_ga", None))),
                "goal_diff":      safe_num(r.get("away_goal_diff",     r.get("away_gd", None))),
                "points":         safe_num(r.get("away_points",        r.get("away_pts",None))),
            })
    return result


def detect_table_type(table):
    headers_lower = [h.strip().lower() for h in table.headers]
    headers_set = set(headers_lower)

    # -- Home/Away split table (Table 2 on FBref stats pages) ---
    if ("home_games" in headers_set or "home_wins" in headers_set) and ("rank" in headers_set or "rk" in headers_set) and ("team" in headers_set or "squad" in headers_set):
        return "standings_home_away"

    # -- Standings ---
    has_rank  = "rank" in headers_set or "rk" in headers_set
    has_pts   = "points" in headers_set or "pts" in headers_set
    has_squad = "team" in headers_set or "squad" in headers_set
    has_wins  = "wins" in headers_set or "w" in headers_set
    if has_rank and has_pts and has_squad and has_wins:
        return "standings"

    # -- Fixtures ---
    has_home_team = "home_team" in headers_set or "home" in headers_set
    has_date      = "date" in headers_set
    has_score     = "score" in headers_set
    if has_home_team or (has_date and has_score):
        return "fixtures"

    # -- Player stats ---
    if "player" in headers_set:
        return "player_stats"

    return "squad_stats"


def get_or_create(cur, table, unique_cols, extra_cols={}):
    where = " AND ".join(f"{k}=%s" for k in unique_cols)
    cur.execute(f"SELECT id FROM {table} WHERE {where}", list(unique_cols.values()))
    row = cur.fetchone()
    if row:
        return row["id"]
    all_cols = {**unique_cols, **extra_cols}
    cols = ", ".join(all_cols.keys())
    placeholders = ", ".join(["%s"] * len(all_cols))
    cur.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) RETURNING id", list(all_cols.values()))
    return cur.fetchone()["id"]


def get_or_create_league(cur, name):
    clean = name.strip()
    if clean.isdigit():
        raise ValueError(f"Invalid league name '{clean}'")
    cur.execute("SELECT id FROM leagues WHERE name ILIKE %s LIMIT 1", (clean,))
    row = cur.fetchone()
    if row:
        return row["id"]
    normalized = clean.title()
    cur.execute("INSERT INTO leagues (name) VALUES (%s) RETURNING id", (normalized,))
    return cur.fetchone()["id"]


def get_or_create_season(cur, name):
    clean = name.strip()
    cur.execute("SELECT id FROM seasons WHERE name ILIKE %s LIMIT 1", (clean,))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur.execute("INSERT INTO seasons (name) VALUES (%s) RETURNING id", (clean,))
    return cur.fetchone()["id"]


def get_or_create_team(cur, name, league_id):
    clean = safe_text(name) or name.strip()
    cur.execute("SELECT id FROM teams WHERE name ILIKE %s AND league_id = %s LIMIT 1", (clean, league_id))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur.execute("INSERT INTO teams (name, league_id) VALUES (%s, %s) RETURNING id", (clean, league_id))
    return cur.fetchone()["id"]


def parse_score(score_raw):
    if not score_raw or str(score_raw).strip() in ("", "nan", "None"):
        return None, None
    # Strip any parenthetical suffix like "(aet)", "(5-3 pen)", then split on dash variant
    cleaned = re.sub(r"\s*\(.*?\)", "", str(score_raw)).strip()
    for sep in ["–", "-", "—"]:
        if sep in cleaned:
            parts = cleaned.split(sep)
            try:
                return int(parts[0].strip()), int(parts[1].strip())
            except (ValueError, IndexError):
                return None, None
    return None, None


def parse_date(raw):
    if not raw or str(raw).strip() in ("", "nan", "None"):
        return None
    s = str(raw).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s[:10]


@router.get("/status")
def sync_status():
    """Return per-league sync history from scrape_log + live row counts from DB."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Per-league: last sync time and total rows inserted
        cur.execute("""
            SELECT
                l.name            AS league,
                s.name            AS season,
                sl.page_type,
                SUM(sl.rows_inserted) AS rows,
                MAX(sl.scraped_at)    AS last_sync
            FROM scrape_log sl
            JOIN leagues  l ON l.id = sl.league_id
            JOIN seasons  s ON s.id = sl.season_id
            GROUP BY l.name, s.name, sl.page_type
            ORDER BY l.name, sl.page_type
        """)
        log_rows = cur.fetchall()

        # Live counts per league from key tables
        cur.execute("""
            SELECT l.name AS league,
                COUNT(DISTINCT m.id)   AS fixtures,
                COUNT(DISTINCT tvs.id) AS home_away_rows,
                COUNT(DISTINCT st.id)  AS standings_rows
            FROM leagues l
            LEFT JOIN matches          m   ON m.league_id   = l.id
            LEFT JOIN team_venue_stats tvs ON tvs.league_id = l.id
            LEFT JOIN league_standings st  ON st.league_id  = l.id
            GROUP BY l.name
            ORDER BY l.name
        """)
        live_rows = cur.fetchall()

        # Build structured response
        by_league = {}
        for r in log_rows:
            lg = r["league"]
            if lg not in by_league:
                by_league[lg] = {"league": lg, "season": r["season"], "log": [], "live": {}}
            by_league[lg]["log"].append({
                "type": r["page_type"],
                "rows": r["rows"],
                "last_sync": r["last_sync"].isoformat() if r["last_sync"] else None
            })

        for r in live_rows:
            lg = r["league"]
            if lg not in by_league:
                by_league[lg] = {"league": lg, "season": None, "log": [], "live": {}}
            by_league[lg]["live"] = {
                "fixtures": r["fixtures"],
                "home_away_rows": r["home_away_rows"],
                "standings_rows": r["standings_rows"]
            }

        return {"success": True, "leagues": list(by_league.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/all")
def sync_all(payload: SyncPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        league_id = get_or_create_league(cur, payload.league)
        season_id = get_or_create_season(cur, payload.season)
        fixtures_list = payload.fixtures or []
        stats_list = payload.stats or []
        players_list = payload.playerStats or payload.player_stats or []
        standings_list = []
        ha_split_list  = []  # Home/Away split table (Table 2 on FBref stats pages)
        if payload.tables:
            for t in payload.tables:
                ttype = detect_table_type(t)
                if ttype == "standings":
                    standings_list.extend(tables_to_standings([t]))
                elif ttype == "fixtures":
                    fixtures_list.extend(tables_to_fixtures([t]))
                elif ttype == "player_stats":
                    players_list.extend(tables_to_player_stats([t]))
                elif ttype == "standings_home_away":
                    ha_split_list.extend(tables_to_home_away_stats([t]))
                else:
                    stats_list.extend(tables_to_squad_stats([t]))
        fx  = _insert_fixtures(cur, league_id, season_id, payload.league, fixtures_list)
        st  = _insert_squad_stats(cur, league_id, season_id, stats_list)
        pl  = _insert_player_stats(cur, season_id, payload.league, players_list)
        sd  = _insert_standings(cur, league_id, season_id, standings_list)
        ha  = _insert_home_away_stats(cur, league_id, season_id, ha_split_list)
        logos = 0
        if payload.team_logos:
            logos = _update_team_logos(cur, league_id, payload.team_logos)
        conn.commit()
        log_scrape(cur, league_id, season_id, "sync_all", fx + st + pl + sd + ha, 0)
        conn.commit()
        # Auto-evaluate predictions whose matches just received scores
        evaluated = _auto_evaluate_predictions(conn)
        return {"success": True, "fixtures_inserted": fx, "stats_inserted": st, "players_inserted": pl, "standings_inserted": sd, "home_away_inserted": ha, "logos_updated": logos, "predictions_evaluated": evaluated}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/fixtures")
def sync_fixtures(payload: SyncPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        league_id = get_or_create_league(cur, payload.league)
        season_id = get_or_create_season(cur, payload.season)
        rows = payload.fixtures or []
        if payload.tables:
            rows.extend(tables_to_fixtures(payload.tables))
        inserted = _insert_fixtures(cur, league_id, season_id, payload.league, rows)
        conn.commit()
        # Auto-evaluate predictions whose matches just received scores
        evaluated = _auto_evaluate_predictions(conn)
        return {"success": True, "matches_inserted": inserted, "predictions_evaluated": evaluated}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/stats")
def sync_stats(payload: SyncPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        league_id = get_or_create_league(cur, payload.league)
        season_id = get_or_create_season(cur, payload.season)
        rows = payload.stats or []
        if payload.tables:
            rows.extend(tables_to_squad_stats(payload.tables))
        inserted = _insert_squad_stats(cur, league_id, season_id, rows)
        conn.commit()
        return {"success": True, "stats_inserted": inserted}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/player-stats")
def sync_player_stats(payload: SyncPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        season_id = get_or_create_season(cur, payload.season)
        rows = payload.player_stats or payload.playerStats or []
        if payload.tables:
            rows.extend(tables_to_player_stats(payload.tables))
        inserted = _insert_player_stats(cur, season_id, payload.league, rows)
        conn.commit()
        return {"success": True, "players_inserted": inserted}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


def tables_to_standings(tables):
    result = []
    for table in tables:
        headers = [h.strip().lower() for h in table.headers]
        for row in table.rows:
            if len(row) < 3:
                continue
            r = dict(zip(headers, row))
            team = safe_text(r.get("squad", r.get("team", "")))
            if not team or team.lower() in ("squad", "team", ""):
                continue
            result.append({
                "rank":          safe_num(r.get("rk", r.get("rank", r.get("pos", r.get("#", None))))),
                "team":          team,
                "games":         safe_num(r.get("mp", r.get("games", r.get("pld", None)))),
                "wins":          safe_num(r.get("w", r.get("wins", None))),
                "ties":          safe_num(r.get("d", r.get("draws", r.get("ties", None)))),
                "losses":        safe_num(r.get("l", r.get("losses", None))),
                "goals_for":     safe_num(r.get("gf", r.get("goals_for", None))),
                "goals_against": safe_num(r.get("ga", r.get("goals_against", None))),
                "goal_diff":     safe_num(r.get("gd", r.get("goal_diff", None))),
                "points":        safe_num(r.get("pts", r.get("points", r.get("pt", None)))),
                "points_avg":    safe_num(r.get("pts/g", r.get("pts_avg", r.get("points_avg", None)))),
            })
    return result


def _insert_fixtures(cur, league_id, season_id, league_name, fixtures):
    count = 0
    for f in fixtures:
        home = str(f.get("home_team", "")).strip()
        away = str(f.get("away_team", "")).strip()
        if not home or not away:
            continue
        home_id = get_or_create_team(cur, home, league_id)
        away_id = get_or_create_team(cur, away, league_id)
        home_score, away_score = parse_score(f.get("score"))
        match_date = parse_date(f.get("date"))
        cur.execute("""
            INSERT INTO matches (league_id, season_id, home_team_id, away_team_id,
                gameweek, dayofweek, match_date, start_time, home_score, away_score, score_raw,
                attendance, venue, referee, round)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE SET
                -- Fix corrupt rows that had league_id/season_id=None from old imports
                league_id=COALESCE(matches.league_id, EXCLUDED.league_id),
                season_id=COALESCE(matches.season_id, EXCLUDED.season_id),
                home_score=EXCLUDED.home_score,
                away_score=EXCLUDED.away_score,
                score_raw=EXCLUDED.score_raw,
                attendance=EXCLUDED.attendance,
                venue=EXCLUDED.venue,
                referee=EXCLUDED.referee,
                is_played=EXCLUDED.home_score IS NOT NULL,
                updated_at=NOW()
        """, (
            league_id, season_id, home_id, away_id,
            safe_num(f.get("gameweek")),    safe_num(f.get("dayofweek")),
            match_date,                     safe_text(f.get("start_time", "")) or None,
            home_score, away_score,         safe_text(f.get("score", "")),
            safe_num(f.get("attendance")),  safe_text(f.get("venue", "")),
            safe_text(f.get("referee", "")), safe_text(f.get("round", ""))
        ))
        count += 1
    return count


def _insert_squad_stats(cur, league_id, season_id, stats_rows):
    count = 0
    for row in stats_rows:
        team_raw = safe_text(row.get("team", ""))
        if not team_raw:
            continue
        split = "against" if team_raw.startswith("vs ") else "for"
        team_name = team_raw[3:].strip() if split == "against" else team_raw
        team_id = get_or_create_team(cur, team_name, league_id)
        cur.execute("""
            INSERT INTO team_squad_stats
                (team_id, league_id, season_id, split, players_used, avg_age, possession,
                 games, games_starts, minutes, minutes_90s, goals, assists,
                 standard_stats, goalkeeping, shooting, playing_time, misc_stats)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (team_id, season_id, split) DO UPDATE SET
                goals=EXCLUDED.goals, assists=EXCLUDED.assists,
                standard_stats=EXCLUDED.standard_stats,
                scraped_at=NOW()
        """, (
            team_id, league_id, season_id, split,
            safe_num(row.get("players_used")), safe_num(row.get("avg_age")), safe_num(row.get("possession")),
            safe_num(row.get("games")), safe_num(row.get("games_starts")), safe_num(row.get("minutes")), safe_num(row.get("minutes_90s")),
            safe_num(row.get("goals")), safe_num(row.get("assists")),
            json.dumps(row.get("standard_stats") or {}),
            json.dumps(row.get("goalkeeping") or {}),
            json.dumps(row.get("shooting") or {}),
            json.dumps(row.get("playing_time") or {}),
            json.dumps(row.get("misc_stats") or {}),
        ))
        count += 1
    return count


def _insert_home_away_stats(cur, league_id, season_id, rows):
    """Insert/update home and away venue stats per team into team_venue_stats."""
    count = 0
    for row in rows:
        team_name = row.get("team", "")
        if not team_name:
            continue
        team_id = get_or_create_team(cur, team_name, league_id)
        cur.execute("""
            INSERT INTO team_venue_stats
                (team_id, league_id, season_id, venue,
                 games, wins, draws, losses, goals_for, goals_against, goal_diff, points)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (team_id, season_id, venue) DO UPDATE SET
                league_id     = EXCLUDED.league_id,
                games         = EXCLUDED.games,
                wins          = EXCLUDED.wins,
                draws         = EXCLUDED.draws,
                losses        = EXCLUDED.losses,
                goals_for     = EXCLUDED.goals_for,
                goals_against = EXCLUDED.goals_against,
                goal_diff     = EXCLUDED.goal_diff,
                points        = EXCLUDED.points,
                updated_at    = NOW()
        """, (
            team_id, league_id, season_id, row.get("venue"),
            safe_num(row.get("games")),
            safe_num(row.get("wins")),
            safe_num(row.get("draws")),
            safe_num(row.get("losses")),
            safe_num(row.get("goals_for")),
            safe_num(row.get("goals_against")),
            safe_num(row.get("goal_diff")),
            safe_num(row.get("points")),
        ))
        count += 1
    return count


def _insert_player_stats(cur, season_id, league_name, players):
    count = 0
    for p in players:
        name = str(p.get("player", "")).strip()
        if not name or name.lower() in ("player", ""):
            continue
        team_name = str(p.get("team", "")).strip()
        team_id = None
        if team_name:
            cur.execute("SELECT id FROM leagues WHERE name ILIKE %s LIMIT 1", (f"%{league_name}%",))
            lg = cur.fetchone()
            if lg:
                team_id = get_or_create_team(cur, team_name, lg["id"])
        cur.execute("""
            INSERT INTO player_stats
                (player_name, nationality, position, team_id, season_id,
                 age, birth_year, games, games_starts, minutes, minutes_90s,
                 goals, assists, standard_stats)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (player_name, team_id, season_id) DO UPDATE SET
                goals=EXCLUDED.goals, assists=EXCLUDED.assists,
                standard_stats=EXCLUDED.standard_stats,
                scraped_at=NOW()
        """, (
            name,
            safe_text(p.get("nationality", "")),
            safe_text(p.get("position", "")),
            team_id, season_id,
            safe_age_int(p.get("age")),
            safe_age_int(p.get("birth_year")),
            safe_num(p.get("games")),
            safe_num(p.get("games_starts")),
            safe_num(p.get("minutes")),
            safe_num(p.get("minutes_90s")),
            safe_num(p.get("goals")),
            safe_num(p.get("assists")),
            json.dumps(p.get("standard_stats") or {})
        ))
        count += 1
    return count


def _insert_standings(cur, league_id, season_id, rows):
    count = 0
    for row in rows:
        team_name = row.get("team", "")
        if not team_name:
            continue
        team_id = get_or_create_team(cur, team_name, league_id)
        cur.execute("""
            INSERT INTO league_standings
                (league_id, season_id, team_id, rank, games, wins, ties, losses,
                 goals_for, goals_against, goal_diff, points, points_avg)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (team_id, league_id, season_id) DO UPDATE SET
                rank          = EXCLUDED.rank,
                games         = EXCLUDED.games,
                wins          = EXCLUDED.wins,
                ties          = EXCLUDED.ties,
                losses        = EXCLUDED.losses,
                goals_for     = EXCLUDED.goals_for,
                goals_against = EXCLUDED.goals_against,
                goal_diff     = EXCLUDED.goal_diff,
                points        = EXCLUDED.points,
                points_avg    = EXCLUDED.points_avg,
                scraped_at    = NOW()
        """, (
            league_id, season_id, team_id,
            row.get("rank"), row.get("games"),
            row.get("wins"), row.get("ties"), row.get("losses"),
            row.get("goals_for"), row.get("goals_against"), row.get("goal_diff"),
            row.get("points"), row.get("points_avg")
        ))
        count += 1
    return count


def _update_standings_home_away(cur, league_id, season_id, rows):
    for r in rows:
        team_name = r["team"]
        split_json = json.dumps(r["split"])
        cur.execute("""
            UPDATE league_standings ls
            SET    home_away_split = %s
            FROM   teams t
            WHERE  t.id = ls.team_id
              AND  ls.league_id = %s
              AND  ls.season_id = %s
              AND  LOWER(t.name) LIKE LOWER(%s)
        """, (split_json, league_id, season_id, f"%{team_name}%"))

def _update_team_logos(cur, league_id, team_logos):
    """Update team logos from the scraper mapping."""
    count = 0
    for team_name, logo_url in team_logos.items():
        if not team_name or not logo_url:
            continue
        try:
            # We use an ILIKE match on name and league_id to find the team
            cur.execute("""
                UPDATE teams
                SET logo_url = %s
                WHERE league_id = %s AND name ILIKE %s
                  AND (logo_url IS NULL OR logo_url != %s)
            """, (logo_url, league_id, f"%{team_name}%", logo_url))
            if cur.rowcount > 0:
                count += cur.rowcount
        except Exception:
            pass
    return count

def log_scrape(cur, league_id, season_id, page_type, inserted, updated):
    try:
        cur.execute("""
            INSERT INTO scrape_log (league_id, season_id, page_type, rows_inserted, rows_updated)
            VALUES (%s, %s, %s, %s, %s)
        """, (league_id, season_id, page_type, inserted, updated))
    except Exception:
        pass
