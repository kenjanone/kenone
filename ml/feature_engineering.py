"""
Feature Engineering for PlusOne Prediction Engine
=====================================================
Extracts 80+ features per team from ALL database tables:
  league_standings, team_squad_stats (standard/shooting/goalkeeping/
  playing_time/misc — for & against splits), player_stats,
  matches (current season form + previous season performance), H2H.

Also captures Scoring/Conceding Patterns (streakiness, blanks, blowouts).
"""

import math

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _f(val, default=0.0):
    """Safely cast to float."""
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _j(jsonb_dict, key, default=0.0):
    """Pull a float value from a JSONB dict stored in Supabase."""
    if not isinstance(jsonb_dict, dict):
        return default
    return _f(jsonb_dict.get(key, default), default)


def _safe_div(num, den, default=0.0):
    try:
        return num / den if den and den != 0 else default
    except Exception:
        return default


# ─── League averages (normalization baseline) ─────────────────────────────────

def get_league_averages(cur, league_id, season_id):
    """
    Compute per-game averages across the whole league for normalisation.
    Also computes venue-specific averages (home/away) from team_venue_stats.
    Returns dict with avg_gf_pg, avg_ga_pg, avg_pts_avg, n_teams,
    plus avg_home_gf_pg, avg_home_ga_pg, avg_away_gf_pg, avg_away_ga_pg.
    """
    # Overall standings averages
    cur.execute("""
        SELECT
            AVG(goals_for::float / NULLIF(games, 0))     AS avg_gf_pg,
            AVG(goals_against::float / NULLIF(games, 0)) AS avg_ga_pg,
            AVG(points_avg)                              AS avg_pts_avg,
            COUNT(*)                                     AS n_teams
        FROM league_standings
        WHERE league_id = %s AND season_id = %s
    """, (league_id, season_id))
    row = cur.fetchone()
    if not row or not row["n_teams"]:
        return {
            "avg_gf_pg": 1.35, "avg_ga_pg": 1.35, "avg_pts_avg": 1.35, "n_teams": 20,
            "avg_home_gf_pg": 1.55, "avg_home_ga_pg": 1.15,
            "avg_away_gf_pg": 1.15, "avg_away_ga_pg": 1.55,
        }

    # Venue-specific averages from team_venue_stats
    cur.execute("""
        SELECT venue,
            AVG(goals_for::float / NULLIF(games, 0))     AS avg_gf_pg,
            AVG(goals_against::float / NULLIF(games, 0)) AS avg_ga_pg
        FROM team_venue_stats
        WHERE league_id = %s AND season_id = %s AND games > 0
        GROUP BY venue
    """, (league_id, season_id))
    venue_avgs = {r["venue"]: r for r in cur.fetchall()}
    h = venue_avgs.get("home", {})
    a = venue_avgs.get("away", {})

    avg_gf = _f(row["avg_gf_pg"], 1.35)
    return {
        "avg_gf_pg":       avg_gf,
        "avg_ga_pg":       _f(row["avg_ga_pg"],   avg_gf),
        "avg_pts_avg":     _f(row["avg_pts_avg"], 1.35),
        "n_teams":         int(row["n_teams"] or 20),
        # Venue: home teams score more at home, away teams score less away
        "avg_home_gf_pg":  _f(h.get("avg_gf_pg"), avg_gf * 1.15),
        "avg_home_ga_pg":  _f(h.get("avg_ga_pg"), avg_gf * 0.85),
        "avg_away_gf_pg":  _f(a.get("avg_gf_pg"), avg_gf * 0.85),
        "avg_away_ga_pg":  _f(a.get("avg_ga_pg"), avg_gf * 1.15),
    }


# ─── League style (base rates per league) ─────────────────────────────────────

def compute_league_style(cur, league_id):
    """
    Compute the characteristic style of a league from its historical results.
    These tell the model what "normal" looks like in each competition:
      - Serie A is defensive (low goals, high draw rate)
      - Bundesliga is open (high goals, lower draw rate)
      - Some leagues have higher home advantage than others

    Returns dict with 6 features keyed by name.
    Falls back to global football averages if no data is available.
    Uses ALL completed matches for the league regardless of season (more data
    = more stable base rates; draws from the same matches table).
    """
    DEFAULTS = {
        "league_home_win_rate":      0.44,
        "league_draw_rate":          0.25,
        "league_away_win_rate":      0.31,
        "league_goals_pg":           2.70,
        "league_btts_rate":          0.50,
        "league_home_advantage_score": 1.42,  # HW% / AW%
    }
    try:
        cur.execute("""
            SELECT
                COUNT(*)                                                 AS n,
                SUM(CASE WHEN home_score  > away_score THEN 1 ELSE 0 END) AS hw,
                SUM(CASE WHEN home_score  = away_score THEN 1 ELSE 0 END) AS d,
                SUM(CASE WHEN home_score  < away_score THEN 1 ELSE 0 END) AS aw,
                AVG(home_score::float + away_score::float)                AS avg_goals,
                SUM(CASE WHEN home_score > 0 AND away_score > 0 THEN 1 ELSE 0 END)
                                                                         AS btts
            FROM matches
            WHERE league_id = %s
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
        """, (league_id,))
        row = cur.fetchone()
        if not row or not row["n"] or int(row["n"]) < 10:
            return DEFAULTS

        n    = int(row["n"])
        hw   = float(row["hw"] or 0)
        d    = float(row["d"]  or 0)
        aw   = float(row["aw"] or 0)
        goals = _f(row["avg_goals"], 2.70)
        btts  = float(row["btts"] or 0)

        hw_rate = hw / n
        aw_rate = aw / n
        return {
            "league_home_win_rate":      round(hw_rate,              4),
            "league_draw_rate":          round(d / n,                4),
            "league_away_win_rate":      round(aw_rate,              4),
            "league_goals_pg":           round(goals,                4),
            "league_btts_rate":          round(btts / n,             4),
            "league_home_advantage_score": round(
                _safe_div(hw_rate, aw_rate, DEFAULTS["league_home_advantage_score"]), 4
            ),
        }
    except Exception:
        return DEFAULTS


def compute_form(cur, team_id, venue=None, n=5):
    """
    Last n completed matches for a team (optionally filtered to home/away).
    Returns dict: form_score (0-1), goals_scored_avg, goals_conceded_avg,
                  win_streak (consecutive wins), results (list of 'W'/'D'/'L').
    venue: None=all, 'home'=home only, 'away'=away only
    """
    venue_clause = ""
    params = [team_id, team_id]
    if venue == "home":
        venue_clause = " AND home_team_id = %s"
        params = [team_id, team_id, team_id]
    elif venue == "away":
        venue_clause = " AND away_team_id = %s"
        params = [team_id, team_id, team_id]

    cur.execute(f"""
        SELECT home_team_id, away_team_id, home_score, away_score
        FROM matches
        WHERE (home_team_id = %s OR away_team_id = %s)
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          {venue_clause}
        ORDER BY match_date DESC
        LIMIT %s
    """, params + [n])
    rows = cur.fetchall()

    if not rows:
        return {"form_score": 0.5, "goals_scored_avg": 1.35, "goals_conceded_avg": 1.35,
                "win_streak": 0, "results": []}

    pts_total = 0
    gf_total = 0
    ga_total = 0
    results = []
    streak = 0
    streak_active = True

    for r in rows:
        is_home = (r["home_team_id"] == team_id)
        gf = _f(r["home_score"] if is_home else r["away_score"])
        ga = _f(r["away_score"] if is_home else r["home_score"])
        gf_total += gf
        ga_total += ga
        if gf > ga:
            pts_total += 3
            results.append("W")
            if streak_active:
                streak += 1
        elif gf == ga:
            pts_total += 1
            results.append("D")
            streak_active = False
        else:
            results.append("L")
            streak_active = False

    n_actual = len(rows)
    max_pts = n_actual * 3
    return {
        "form_score":        _safe_div(pts_total, max_pts, 0.5),
        "goals_scored_avg":  _safe_div(gf_total, n_actual, 1.2),
        "goals_conceded_avg":_safe_div(ga_total, n_actual, 1.2),
        "win_streak":        streak,
        "results":           results,
    }


# ─── Head-to-head ─────────────────────────────────────────────────────────────

def compute_h2h(cur, home_team_id, away_team_id, n=10):
    """
    Last n head-to-head matches between the two teams (both directions).
    Returns: home_wins, draws, away_wins, home_goals_avg, away_goals_avg,
             last_5 (list of 'H'/'D'/'A' from perspective of home_team_id)
    """
    cur.execute("""
        SELECT home_team_id, away_team_id, home_score, away_score
        FROM matches
        WHERE ((home_team_id = %s AND away_team_id = %s)
            OR (home_team_id = %s AND away_team_id = %s))
          AND home_score IS NOT NULL
        ORDER BY match_date DESC
        LIMIT %s
    """, (home_team_id, away_team_id, away_team_id, home_team_id, n))
    rows = cur.fetchall()

    h_wins = d = a_wins = 0
    hg_tot = ag_tot = 0
    last_5 = []

    for r in rows:
        if r["home_team_id"] == home_team_id:
            hg, ag = _f(r["home_score"]), _f(r["away_score"])
        else:
            hg, ag = _f(r["away_score"]), _f(r["home_score"])
        hg_tot += hg
        ag_tot += ag
        if hg > ag:
            h_wins += 1
            last_5.append("H")
        elif hg == ag:
            d += 1
            last_5.append("D")
        else:
            a_wins += 1
            last_5.append("A")

    total = h_wins + d + a_wins or 1
    return {
        "h2h_home_wins":    h_wins,
        "h2h_draws":        d,
        "h2h_away_wins":    a_wins,
        "h2h_home_win_pct": _safe_div(h_wins, total),
        "h2h_away_win_pct": _safe_div(a_wins, total),
        "h2h_home_gf_avg":  _safe_div(hg_tot, total),
        "h2h_away_gf_avg":  _safe_div(ag_tot, total),
        "h2h_last_5":       last_5[:5],
    }


# ─── Squad stats block ────────────────────────────────────────────────────────

def _get_squad_stat(cur, team_id, season_id, split):
    """Fetch the team_squad_stats row for a given split ('for' or 'against')."""
    cur.execute("""
        SELECT players_used, avg_age, possession, games, minutes_90s,
               goals, assists, standard_stats, goalkeeping, shooting,
               playing_time, misc_stats
        FROM team_squad_stats
        WHERE team_id = %s AND season_id = %s AND split = %s
        LIMIT 1
    """, (team_id, season_id, split))
    return cur.fetchone()


# ─── Player stats features ────────────────────────────────────────────────────

def build_player_features(cur, team_id, season_id):
    """
    Aggregate player_stats for a team into team-level features:
    - Top scorer goals (star player quality)
    - Goal concentration (% of team goals from top scorer — high = more predictable/vulnerable)
    - Number of goal scorers (squad depth)
    - Top assister assists count
    - Average goals per player (distributed attack)
    - Avg age of scoring players (youth vs experience in attack)
    - Non-penalty top scorer goals per90 (pure quality)
    Return a flat dict of player-derived features.
    """
    cur.execute("""
        SELECT player_name, goals, assists, minutes, minutes_90s,
               age, standard_stats
        FROM player_stats
        WHERE team_id = %s AND season_id = %s
          AND goals IS NOT NULL
        ORDER BY goals DESC
    """, (team_id, season_id))
    players = cur.fetchall()

    feats = {}
    if not players:
        for k in ["top_scorer_goals", "goal_concentration", "n_scorers",
                  "squad_depth_scorers", "top_assister_assists", "avg_goals_per_player",
                  "scorer_avg_age", "top_scorer_goals_per90", "team_player_assists_pg",
                  "attack_dependency"]:
            feats[k] = 0.0
        return feats

    total_goals      = sum(_f(p["goals"]) for p in players)
    total_assists    = sum(_f(p["assists"]) for p in players if p["assists"])
    scorers          = [p for p in players if _f(p["goals"]) > 0]
    top             = players[0]
    top_goals        = _f(top["goals"])
    top_m90          = _f(top["minutes_90s"]) or 1

    feats["top_scorer_goals"]     = top_goals
    feats["goal_concentration"]   = _safe_div(top_goals, total_goals, 0.0)   # high → star-dependent
    feats["n_scorers"]            = _f(len(scorers))                         # squad depth
    feats["squad_depth_scorers"]  = _safe_div(len(scorers), len(players), 0.5)
    feats["top_assister_assists"] = _f(players[0]["assists"]) if players else 0.0
    feats["avg_goals_per_player"] = _safe_div(total_goals, len(players), 0.0)
    feats["scorer_avg_age"]       = (_safe_div(sum(_f(p["age"]) for p in scorers),
                                                len(scorers), 26.0) if scorers else 26.0)
    ss = top["standard_stats"] if isinstance(top["standard_stats"], dict) else {}
    feats["top_scorer_goals_per90"] = _j(ss, "goals_per90", _safe_div(top_goals, top_m90))

    cur.execute("""
        SELECT COUNT(DISTINCT m.id) AS n_games
        FROM matches m
        JOIN team_squad_stats ts ON ts.team_id = %s
        WHERE (m.home_team_id = %s OR m.away_team_id = %s)
          AND m.home_score IS NOT NULL AND ts.season_id = %s
        LIMIT 1
    """, (team_id, team_id, team_id, season_id))
    gm_row = cur.fetchone()
    n_games = _f(gm_row["n_games"]) if gm_row and gm_row["n_games"] else 1
    feats["team_player_assists_pg"] = _safe_div(total_assists, n_games)
    # Attack dependency risk: 1 player with >40% goals is a single point of failure
    feats["attack_dependency"] = 1.0 if feats["goal_concentration"] > 0.40 else 0.0

    return feats


# ─── Previous season features ─────────────────────────────────────────────────

def build_prev_season_features(cur, team_id, league_id):
    """
    Retrieve the team's performance from the previous season using the
    SCORES & FIXTURES / standings data already in the database.

    Previous season data tells us:
    - Was the team promoted? (different calibre)
    - Historical consistency (not just current form)
    - Whether they had a strong/weak last season

    Returns a flat dict of prev_* features (zero-filled if no prior data).
    """
    ZERO = {
        "prev_rank_norm": 0.5, "prev_points_avg": 0.0,
        "prev_wins_pct": 0.0,  "prev_goals_for_pg": 0.0,
        "prev_goals_ag_pg": 0.0, "prev_goal_diff_pg": 0.0,
        "prev_season_found": 0.0,
    }

    cur.execute("""
        SELECT ls.season_id, s.name AS season_name,
               ls.rank, ls.games, ls.wins, ls.ties, ls.losses,
               ls.goals_for, ls.goals_against, ls.goal_diff, ls.points_avg,
               COUNT(*) OVER (PARTITION BY ls.league_id) AS league_team_count
        FROM league_standings ls
        JOIN seasons s ON s.id = ls.season_id
        WHERE ls.team_id = %s AND ls.league_id = %s
        ORDER BY s.name DESC
        OFFSET 1 LIMIT 1
    """, (team_id, league_id))
    prev = cur.fetchone()

    if not prev:
        return ZERO

    g = _f(prev["games"]) or 1
    n = _f(prev["league_team_count"]) or 20
    return {
        "prev_rank_norm":    1.0 - _safe_div(_f(prev["rank"]) - 1, n - 1),
        "prev_points_avg":   _f(prev["points_avg"]),
        "prev_wins_pct":     _safe_div(_f(prev["wins"]),          g),
        "prev_goals_for_pg": _safe_div(_f(prev["goals_for"]),    g),
        "prev_goals_ag_pg":  _safe_div(_f(prev["goals_against"]), g),
        "prev_goal_diff_pg": _safe_div(_f(prev["goal_diff"]),     g),
        "prev_season_found": 1.0,
    }


# ─── Previous season match-level form ─────────────────────────────────────────

def build_prev_season_form(cur, team_id):
    """
    From matches table: compute win/draw/loss/gf/ga rates for the team
    in all matches from the PREVIOUS season (not the most recent season_id).
    Uses SCORES & FIXTURES data already populated.
    """
    cur.execute("""
        SELECT season_id
        FROM (
            SELECT DISTINCT m.season_id, s.name
            FROM matches m
            JOIN seasons s ON s.id = m.season_id
            WHERE (m.home_team_id = %s OR m.away_team_id = %s)
              AND m.home_score IS NOT NULL
            ORDER BY s.name DESC
            OFFSET 1 LIMIT 1
        ) sub
    """, (team_id, team_id))
    row = cur.fetchone()
    if not row:
        return {"prev_form_score": 0.5, "prev_gf_pg": 1.35, "prev_ga_pg": 1.35,
                "prev_match_wins_pct": 0.0}

    prev_sid = row["season_id"]
    cur.execute("""
        SELECT home_team_id, away_team_id, home_score, away_score
        FROM matches
        WHERE (home_team_id = %s OR away_team_id = %s)
          AND season_id = %s
          AND home_score IS NOT NULL
    """, (team_id, team_id, prev_sid))
    matches = cur.fetchall()
    if not matches:
        return {"prev_form_score": 0.5, "prev_gf_pg": 1.35, "prev_ga_pg": 1.35,
                "prev_match_wins_pct": 0.0}

    wins = pts = gf_t = ga_t = 0
    for m in matches:
        is_home = m["home_team_id"] == team_id
        gf  = _f(m["home_score"] if is_home else m["away_score"])
        ga  = _f(m["away_score"] if is_home else m["home_score"])
        gf_t += gf; ga_t += ga
        if gf > ga:  wins += 1; pts += 3
        elif gf == ga: pts += 1

    n = len(matches)
    return {
        "prev_form_score":      _safe_div(pts, n * 3, 0.5),
        "prev_gf_pg":          _safe_div(gf_t, n),
        "prev_ga_pg":          _safe_div(ga_t, n),
        "prev_match_wins_pct": _safe_div(wins, n),
    }


# ─── Scoring and Conceding Patterns ───────────────────────────────────────────

def build_scoring_patterns(cur, team_id, season_id):
    """
    Analyzes the match-by-match variance to detect patterns:
    - consistency (variance of goals scored/conceded vs the mean)
    - blanks (rate of scoring 0 goals)
    - blowouts (rate of scoring 3+ goals)
    - leaky (rate of conceding 3+ goals)
    - clean_sheets (rate of conceding 0 goals)
    """
    cur.execute("""
        SELECT home_score, away_score, home_team_id
        FROM matches
        WHERE (home_team_id = %s OR away_team_id = %s)
          AND season_id = %s
          AND home_score IS NOT NULL
        ORDER BY match_date ASC
    """, (team_id, team_id, season_id))
    matches = cur.fetchall()

    if not matches:
        return {
            "goals_scored_variance": 0.0, "goals_conceded_variance": 0.0,
            "blank_rate": 0.0, "blowout_rate": 0.0,
            "defensive_collapse_rate": 0.0, "clean_sheet_rate": 0.0
        }

    n = len(matches)
    scored = []
    conceded = []
    blanks = blowouts = collapses = clean_sheets = 0

    for m in matches:
        is_home = (m["home_team_id"] == team_id)
        gf = _f(m["home_score"] if is_home else m["away_score"])
        ga = _f(m["away_score"] if is_home else m["home_score"])

        scored.append(gf)
        conceded.append(ga)

        if gf == 0: blanks += 1
        if gf >= 3: blowouts += 1
        if ga == 0: clean_sheets += 1
        if ga >= 3: collapses += 1

    mean_gf = sum(scored) / n
    mean_ga = sum(conceded) / n

    # Variance: indicates streakiness / unpredictability
    var_gf = sum((x - mean_gf) ** 2 for x in scored) / n if n > 1 else 0.0
    var_ga = sum((x - mean_ga) ** 2 for x in conceded) / n if n > 1 else 0.0

    return {
        "goals_scored_variance":   _f(var_gf),
        "goals_conceded_variance": _f(var_ga),
        "blank_rate":              _safe_div(blanks, n),
        "blowout_rate":            _safe_div(blowouts, n),
        "defensive_collapse_rate": _safe_div(collapses, n),
        "clean_sheet_rate":        _safe_div(clean_sheets, n),
    }


# ─── Main team feature builder ────────────────────────────────────────────────

def build_venue_stats(cur, team_id, season_id):
    """
    Fetch per-venue (home/away) goal records from team_venue_stats table.
    Returns dict with:
      home_gf_pg, home_ga_pg, home_win_rate, home_games,
      away_gf_pg, away_ga_pg, away_win_rate, away_games
    All floats, zero-safe.
    """
    cur.execute("""
        SELECT venue, games, wins, goals_for, goals_against
        FROM team_venue_stats
        WHERE team_id = %s AND season_id = %s AND games > 0
    """, (team_id, season_id))
    rows = {r["venue"]: r for r in cur.fetchall()}

    def _venue(v, fallback_gf, fallback_ga):
        r = rows.get(v)
        if not r:
            return {f"{v}_gf_pg": fallback_gf, f"{v}_ga_pg": fallback_ga,
                    f"{v}_win_rate": 0.33, f"{v}_games": 0}
        g = _f(r["games"]) or 1
        return {
            f"{v}_gf_pg":   _safe_div(_f(r["goals_for"]),    g, fallback_gf),
            f"{v}_ga_pg":   _safe_div(_f(r["goals_against"]), g, fallback_ga),
            f"{v}_win_rate": _safe_div(_f(r["wins"]),         g, 0.33),
            f"{v}_games":    _f(r["games"]),
        }

    result = {}
    result.update(_venue("home", 1.55, 1.15))   # home teams score more
    result.update(_venue("away", 1.15, 1.55))   # away teams score less
    return result


def build_team_features(cur, team_id, league_id, season_id, league_avgs=None):
    """
    Build features for a single team from league standings, squad stats,
    recent form, player stats, and previous season performance.
    Returns a flat dict with all features (floats, NaN-safe).
    """
    feats = {}

    # ── 1. League standings ───────────────────────────────────────────────
    cur.execute("""
        SELECT rank, games, wins, ties, losses,
               goals_for, goals_against, goal_diff, points, points_avg,
               home_away_split
        FROM league_standings
        WHERE team_id = %s AND league_id = %s AND season_id = %s
        LIMIT 1
    """, (team_id, league_id, season_id))
    st = cur.fetchone()

    n_teams = (league_avgs or {}).get("n_teams", 20)
    avg_gf  = (league_avgs or {}).get("avg_gf_pg", 1.3)
    avg_ga  = (league_avgs or {}).get("avg_ga_pg", 1.3)

    if st:
        g = _f(st["games"]) or 1
        gf_pg  = _safe_div(_f(st["goals_for"]),    g)
        ga_pg  = _safe_div(_f(st["goals_against"]), g)
        gd_pg  = _safe_div(_f(st["goal_diff"]),     g)
        feats["rank_norm"]       = 1.0 - _safe_div(_f(st["rank"]) - 1, n_teams - 1)  # 1=top, 0=bottom
        feats["points_avg"]      = _f(st["points_avg"])
        feats["wins_pct"]        = _safe_div(_f(st["wins"]),   g)
        feats["draws_pct"]       = _safe_div(_f(st["ties"]),   g)
        feats["losses_pct"]      = _safe_div(_f(st["losses"]), g)
        feats["goals_for_pg"]    = gf_pg
        feats["goals_against_pg"]= ga_pg
        feats["goal_diff_pg"]    = gd_pg
        feats["attack_strength"] = _safe_div(gf_pg, avg_gf, 1.0)   # >1 = above avg attack
        feats["defence_strength"]= _safe_div(avg_ga, ga_pg, 1.0)   # >1 = above avg defence

        ha = st["home_away_split"] if isinstance(st["home_away_split"], dict) else {}
        feats["home_wins_pct"]      = _j(ha, "home_win_pct", feats["wins_pct"])
        feats["home_goals_for_pg"]  = _j(ha, "home_gf_pg",   gf_pg)
        feats["home_goals_ag_pg"]   = _j(ha, "home_ga_pg",   ga_pg)
        feats["home_points_avg"]    = _j(ha, "home_pts_avg",  feats["points_avg"])
        feats["away_wins_pct"]      = _j(ha, "away_win_pct",  feats["wins_pct"])
        feats["away_goals_for_pg"]  = _j(ha, "away_gf_pg",    gf_pg)
        feats["away_goals_ag_pg"]   = _j(ha, "away_ga_pg",    ga_pg)
        feats["away_points_avg"]    = _j(ha, "away_pts_avg",  feats["points_avg"])
    else:
        for key in ["rank_norm","points_avg","wins_pct","draws_pct","losses_pct",
                    "goals_for_pg","goals_against_pg","goal_diff_pg",
                    "attack_strength","defence_strength",
                    "home_wins_pct","home_goals_for_pg","home_goals_ag_pg","home_points_avg",
                    "away_wins_pct","away_goals_for_pg","away_goals_ag_pg","away_points_avg"]:
            feats[key] = 0.5 if "pct" in key or key in ("rank_norm",) else 0.0

    # ── 2. Squad stats — FOR split ──────────────────────────────────────
    sq_for = _get_squad_stat(cur, team_id, season_id, "for")
    if sq_for:
        feats["possession"]       = _f(sq_for["possession"], 50.0)
        feats["avg_age"]          = _f(sq_for["avg_age"], 26.0)
        feats["players_used"]     = _f(sq_for["players_used"], 20)
        m90 = _f(sq_for["minutes_90s"]) or 1

        ss = sq_for["standard_stats"] or {}
        feats["goals_per90"]             = _j(ss, "goals_per90",             0.0)
        feats["assists_per90"]           = _j(ss, "assists_per90",           0.0)
        feats["goals_assists_per90"]     = _j(ss, "goals_assists_per90",     0.0)
        feats["goals_pens_per90"]        = _j(ss, "goals_pens_per90",        0.0)
        feats["goals_assists_pens_per90"]= _j(ss, "goals_assists_pens_per90",0.0)
        feats["cards_yellow_per90"]      = _safe_div(_j(ss, "cards_yellow", 0.0), m90)
        feats["cards_red_per90"]         = _safe_div(_j(ss, "cards_red",    0.0), m90)
        pens_att = _j(ss, "pens_att", 0.0)
        feats["pen_conversion"]          = _safe_div(_j(ss, "pens_made", 0.0), pens_att, 0.75)

        sh = sq_for["shooting"] or {}
        feats["shots_per90"]                = _j(sh, "shots_per90",                0.0)
        feats["shots_on_target_per90"]      = _j(sh, "shots_on_target_per90",      0.0)
        feats["shots_on_target_pct"]        = _j(sh, "shots_on_target_pct",        0.0)
        feats["goals_per_shot"]             = _j(sh, "goals_per_shot",             0.0)
        feats["goals_per_shot_on_target"]   = _j(sh, "goals_per_shot_on_target",   0.0)

        pt = sq_for["playing_time"] or {}
        feats["points_per_game_pt"]  = _j(pt, "points_per_game",    0.0)
        feats["plus_minus_per90"]    = _j(pt, "plus_minus_per90",   0.0)
        on_gf = _j(pt, "on_goals_for",     0.0)
        on_ga = _j(pt, "on_goals_against", 0.0)
        feats["on_field_goal_ratio"] = _safe_div(on_gf, on_gf + on_ga, 0.5)
        feats["squad_completeness"]  = _safe_div(_j(pt, "games_complete", 0.0),
                                                  _f(sq_for["games"], 1))

        ms = sq_for["misc_stats"] or {}
        feats["fouls_per90"]          = _safe_div(_j(ms, "fouls",         0.0), m90)
        feats["fouled_per90"]         = _safe_div(_j(ms, "fouled",        0.0), m90)
        feats["offsides_per90"]       = _safe_div(_j(ms, "offsides",      0.0), m90)
        feats["crosses_per90"]        = _safe_div(_j(ms, "crosses",       0.0), m90)
        feats["interceptions_per90"]  = _safe_div(_j(ms, "interceptions", 0.0), m90)
        feats["tackles_won_per90"]    = _safe_div(_j(ms, "tackles_won",   0.0), m90)
        pens_con = _j(ms, "pens_conceded", 0.0) or 1
        feats["pen_area_ratio"]       = _safe_div(_j(ms, "pens_won", 0.0), pens_con, 1.0)
        feats["own_goals"]            = _j(ms, "own_goals", 0.0)
        feats["discipline_score"]     = (feats["cards_yellow_per90"] * 1 +
                                          feats["cards_red_per90"]   * 3)
    else:
        for key in ["possession","avg_age","players_used","goals_per90","assists_per90",
                    "goals_assists_per90","goals_pens_per90","goals_assists_pens_per90",
                    "cards_yellow_per90","cards_red_per90","pen_conversion",
                    "shots_per90","shots_on_target_per90","shots_on_target_pct",
                    "goals_per_shot","goals_per_shot_on_target","points_per_game_pt",
                    "plus_minus_per90","on_field_goal_ratio","squad_completeness",
                    "fouls_per90","fouled_per90","offsides_per90","crosses_per90",
                    "interceptions_per90","tackles_won_per90","pen_area_ratio",
                    "own_goals","discipline_score"]:
            feats[key] = 0.0

    # ── 3. Squad stats — AGAINST split (defensive stats) ─────────────────
    sq_ag = _get_squad_stat(cur, team_id, season_id, "against")
    if sq_ag:
        gk = sq_ag["goalkeeping"] or {}
        feats["gk_goals_ag_per90"]     = _j(gk, "gk_goals_against_per90",  1.3)
        feats["gk_shots_faced_per90"]  = _j(gk, "gk_shots_on_target_against", 4.0)
        feats["gk_save_pct"]           = _j(gk, "gk_save_pct",             65.0)
        feats["gk_clean_sheets_pct"]   = _j(gk, "gk_clean_sheets_pct",     25.0)
        gk_g = _j(gk, "gk_games", 1.0) or 1
        feats["gk_win_rate"]           = _safe_div(_j(gk, "gk_wins", 0.0), gk_g)
        pen_att_gk = _j(gk, "gk_pens_att", 0.0) or 1
        feats["gk_pen_save_pct"]       = _safe_div(_j(gk, "gk_pens_saved", 0.0), pen_att_gk)

        sh_ag = sq_ag["shooting"] or {}
        feats["opp_shots_per90"]           = _j(sh_ag, "shots_per90",          0.0)
        feats["opp_shots_on_tgt_per90"]    = _j(sh_ag, "shots_on_target_per90",0.0)
        feats["opp_goals_per_shot"]        = _j(sh_ag, "goals_per_shot",        0.0)
    else:
        for key in ["gk_goals_ag_per90","gk_shots_faced_per90","gk_save_pct",
                    "gk_clean_sheets_pct","gk_win_rate","gk_pen_save_pct",
                    "opp_shots_per90","opp_shots_on_tgt_per90","opp_goals_per_shot"]:
            feats[key] = 0.0

    # ── 4. Recent form (combined xG-style stats) ────────────────────────
    form_all  = compute_form(cur, team_id, venue=None,   n=5)
    form_home = compute_form(cur, team_id, venue="home", n=5)
    form_away = compute_form(cur, team_id, venue="away", n=5)

    feats["form_score"]          = form_all["form_score"]
    feats["form_gf_avg"]         = form_all["goals_scored_avg"]
    feats["form_ga_avg"]         = form_all["goals_conceded_avg"]
    feats["win_streak"]          = _f(form_all["win_streak"])
    feats["home_form_score"]     = form_home["form_score"]
    feats["away_form_score"]     = form_away["form_score"]

    # Derived: expected goals estimate from current-season shooting
    feats["xg_estimate"] = (
        feats.get("shots_on_target_per90", 0.0) * feats.get("goals_per_shot_on_target", 0.0)
        if feats.get("shots_on_target_per90")
        else feats.get("goals_per90", 1.2)
    )

    # ── 5. Player standard stats ──────────────────────────────────────
    player_feats = build_player_features(cur, team_id, season_id)
    feats.update(player_feats)

    prev_st   = build_prev_season_features(cur, team_id, league_id)
    prev_form = build_prev_season_form(cur, team_id)
    feats.update(prev_st)
    feats.update(prev_form)

    # ── 7. Match-by-match scoring patterns (streakiness, blanks, blowouts) ──
    patterns = build_scoring_patterns(cur, team_id, season_id)
    feats.update(patterns)

    # ── 8. Venue stats (home/away splits from team_venue_stats) ────────────
    venue = build_venue_stats(cur, team_id, season_id)
    feats.update(venue)   # adds home_gf_pg, home_ga_pg, away_gf_pg, away_ga_pg, etc.

    return feats


# ─── Match-level feature vector ───────────────────────────────────────────────

def build_match_features(cur, home_team_id, away_team_id, league_id, season_id):
    """
    Combine home + away team features + per-feature differentials + league-style
    context into a single flat feature vector ready for the ML model.

    Also returns raw home/away feature dicts for the rich prediction output.
    """
    avgs = get_league_averages(cur, league_id, season_id)
    style = compute_league_style(cur, league_id)   # ← league-level base rates

    home_feats = build_team_features(cur, home_team_id, league_id, season_id, avgs)
    away_feats = build_team_features(cur, away_team_id, league_id, season_id, avgs)
    h2h        = compute_h2h(cur, home_team_id, away_team_id)

    vector = {}

    for k, v in home_feats.items():
        vector[f"home_{k}"] = v
    for k, v in away_feats.items():
        vector[f"away_{k}"] = v
        # Differential = home - away (positive = home advantage)
        if k in home_feats:
            vector[f"diff_{k}"] = home_feats[k] - v

    # H2H features
    for k, v in h2h.items():
        if k != "h2h_last_5":
            vector[k] = _f(v)

    # ── League-style context (6 features) ──────────────────────────────────
    # These anchor the prediction to the specific league's characteristics:
    # the model learns that 44% home wins is "normal" for Serie A but slightly
    # above par for the Bundesliga. Without these, cross-league comparisons
    # produce miscalibrated probabilities.
    vector.update(style)

    # ── League-relative team strength (4 features) ─────────────────────────
    # How does each team's attack/defence compare to THIS league's average?
    # Useful for teams playing in multiple competitions — their raw stats are
    # normalised to the competition they're currently playing in.
    lg_goals = style["league_goals_pg"] or 2.70
    half_goals = lg_goals / 2            # rough expected team goals/game
    vector["home_attack_rel_league"] = _safe_div(
        home_feats.get("goals_for_pg", 0.0), half_goals, 1.0
    )
    vector["away_attack_rel_league"] = _safe_div(
        away_feats.get("goals_for_pg", 0.0), half_goals, 1.0
    )
    vector["home_defence_rel_league"] = _safe_div(
        half_goals, home_feats.get("goals_against_pg", half_goals) or half_goals, 1.0
    )
    vector["away_defence_rel_league"] = _safe_div(
        half_goals, away_feats.get("goals_against_pg", half_goals) or half_goals, 1.0
    )

    # Match-level constant: home advantage exists in all football data
    vector["home_advantage"] = 1.0

    # Sort for consistent column ordering
    feature_names = sorted(vector.keys())
    feature_values = [vector[k] for k in feature_names]

    return feature_values, feature_names, home_feats, away_feats, h2h


# ─── Training dataset builder ─────────────────────────────────────────────────

def build_training_dataset(cur, skip_errors=True):
    """
    Fetch all completed matches from DB, build feature vectors + labels.
    Returns: X (list of lists), y (list of ints), match_ids (list)
    Labels: 0=Home Win, 1=Draw, 2=Away Win
    """
    cur.execute("""
        SELECT m.id, m.home_team_id, m.away_team_id,
               m.league_id, m.season_id,
               m.home_score, m.away_score
        FROM matches m
        WHERE m.home_score IS NOT NULL
          AND m.away_score IS NOT NULL
          AND m.league_id IS NOT NULL
          AND m.season_id IS NOT NULL
        ORDER BY m.match_date DESC
    """)
    matches = cur.fetchall()

    X, y, match_ids = [], [], []
    errors = 0

    for m in matches:
        try:
            fv, _, _, _, _ = build_match_features(
                cur,
                m["home_team_id"],
                m["away_team_id"],
                m["league_id"],
                m["season_id"],
            )
            hs, as_ = _f(m["home_score"]), _f(m["away_score"])
            if hs > as_:
                label = 0
            elif hs == as_:
                label = 1
            else:
                label = 2

            X.append(fv)
            y.append(label)
            match_ids.append(m["id"])
        except Exception:
            errors += 1
            if not skip_errors:
                raise
            continue

    return X, y, match_ids, errors
