import logging
import math
from datetime import datetime

log = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _f(val, default=0.0):
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default

def _parse_market_value(mv_str: str) -> float:
    """
    Parses strings like '€50.00m', '€125k', '- ' into a float representing Millions (€).
    """
    if not mv_str or not isinstance(mv_str, str):
        return 0.0
    s = mv_str.lower().replace("€", "").replace(" ", "").strip()
    if s in ["", "-", "unknown"]:
        return 0.0
    
    multiplier = 1.0
    if s.endswith("m"):
        s = s[:-1]
        multiplier = 1.0
    elif s.endswith("k"):
        s = s[:-1]
        multiplier = 0.001
        
    try:
        return float(s) * multiplier
    except ValueError:
        return 0.0

# ─── Main Extraction Function ─────────────────────────────────────────────────

def build_enrichment_features(cur, home_team_id: int, away_team_id: int, match_date: str = None):
    """
    Queries the database's enrichment tables (`team_clubelo`, `player_injuries`, `match_odds`)
    to build a 1D feature array exclusively focused on the enrichment domain.
    
    If data is missing for a table, defaults are returned cleanly to prevent Pipeline crashes.
    """
    feats = {}
    
    # Optional match date context. If None, we just take the most recent data (for future predictions)
    # If predicting historically, we use match_date to filter.
    date_filter = ""
    date_params = []
    if match_date:
        date_filter = " AND scraped_at <= %s::timestamp + INTERVAL '1 day' "
        date_params = [match_date]
        
    # ── 1. Match Odds ───────────────────────────────────────────────────────────
    # We find the most recent odds row where these two teams played.
    cur.execute(f"""
        SELECT mo.b365_home_win, mo.b365_draw, mo.b365_away_win, mo.raw_data
        FROM match_odds mo
        JOIN matches m ON m.id = mo.match_id
        WHERE m.home_team_id = %s AND m.away_team_id = %s
        ORDER BY mo.scraped_at DESC LIMIT 1
    """, (home_team_id, away_team_id))
    odds_row = cur.fetchone()
    
    if odds_row and odds_row["b365_home_win"]:
        hw = _f(odds_row["b365_home_win"])
        dw = _f(odds_row["b365_draw"])
        aw = _f(odds_row["b365_away_win"])
        
        # Convert odds to implied probabilities (1x2)
        margin_1x2 = (1.0/hw + 1.0/dw + 1.0/aw) if (hw and dw and aw) else 1.0
        feats["odds_home_prob"] = (1.0/hw) / margin_1x2 if hw else 0.35
        feats["odds_draw_prob"] = (1.0/dw) / margin_1x2 if dw else 0.30
        feats["odds_away_prob"] = (1.0/aw) / margin_1x2 if aw else 0.35
        
        # Parse additional JSON raw_data features (Over/Under, Asian Handicap, etc.)
        raw_odds = odds_row.get("raw_data") or {}
        
        # Over/Under 2.5
        o25 = _f(raw_odds.get("B365>2.5", raw_odds.get("Max>2.5", 0)))
        u25 = _f(raw_odds.get("B365<2.5", raw_odds.get("Max<2.5", 0)))
        margin_ou = (1.0/o25 + 1.0/u25) if (o25 > 0 and u25 > 0) else 1.0
        feats["odds_over_25_prob"]  = (1.0/o25) / margin_ou if o25 > 0 else 0.50
        feats["odds_under_25_prob"] = (1.0/u25) / margin_ou if u25 > 0 else 0.50
        
        # Asian Handicap (e.g. -1.5, +0.5)
        ah_size = _f(raw_odds.get("AHh", 0.0))
        ahh     = _f(raw_odds.get("B365AHH", 0))
        aha     = _f(raw_odds.get("B365AHA", 0))
        margin_ah = (1.0/ahh + 1.0/aha) if (ahh > 0 and aha > 0) else 1.0
        
        feats["odds_asian_hdcp_size"] = ah_size
        feats["odds_ah_home_prob"]    = (1.0/ahh) / margin_ah if ahh > 0 else 0.50
        feats["odds_ah_away_prob"]    = (1.0/aha) / margin_ah if aha > 0 else 0.50
        
    else:
        # Defaults if no match odds row
        feats["odds_home_prob"] = 0.35
        feats["odds_draw_prob"] = 0.30
        feats["odds_away_prob"] = 0.35
        feats["odds_over_25_prob"] = 0.50
        feats["odds_under_25_prob"] = 0.50
        feats["odds_asian_hdcp_size"] = 0.0
        feats["odds_ah_home_prob"] = 0.50
        feats["odds_ah_away_prob"] = 0.50

    # ── 2. Team ClubElo ─────────────────────────────────────────────────────────
    # We pull the most recent Elo values for both home and away
    cur.execute(f"""
        SELECT elo, raw_data FROM team_clubelo
        WHERE team_id = %s {date_filter}
        ORDER BY elo_date DESC LIMIT 1
    """, [home_team_id] + date_params)
    home_elo_row = cur.fetchone()

    cur.execute(f"""
        SELECT elo, raw_data FROM team_clubelo
        WHERE team_id = %s {date_filter}
        ORDER BY elo_date DESC LIMIT 1
    """, [away_team_id] + date_params)
    away_elo_row = cur.fetchone()
    
    h_elo = _f(home_elo_row["elo"]) if home_elo_row and home_elo_row["elo"] else 1500.0
    a_elo = _f(away_elo_row["elo"]) if away_elo_row and away_elo_row["elo"] else 1500.0
    
    feats["clubelo_home_rating"] = h_elo
    feats["clubelo_away_rating"] = a_elo
    feats["clubelo_gap"]         = h_elo - a_elo
    
    # Try to unpack exact match probabilities from the raw_data JSONB if available
    h_elo_prob = 0.35
    d_elo_prob = 0.30
    a_elo_prob = 0.35
    
    # The JSON structure places the probabilities inside `{"raw": {"GD=0": 0.25, "GD=1": 0.15}}`
    raw_json = (home_elo_row or {}).get("raw_data") or (away_elo_row or {}).get("raw_data") or {}
    raw_inner = raw_json.get("raw", {})
    
    if raw_inner and "GD=0" in raw_inner:
        d_elo_prob = _f(raw_inner.get("GD=0", 0.30))
        
        # Sum GD>0 for home win
        hw_sum = 0.0
        for i in range(1, 10):
            hw_sum += _f(raw_inner.get(f"GD={i}", 0.0))
        h_elo_prob = hw_sum if hw_sum > 0 else 0.35
        
        # Sum GD<0 for away win
        aw_sum = 0.0
        for i in range(-9, 0):
            aw_sum += _f(raw_inner.get(f"GD={i}", 0.0))
        a_elo_prob = aw_sum if aw_sum > 0 else 0.35
        
        # Normalize
        total = h_elo_prob + d_elo_prob + a_elo_prob
        if total > 0:
            h_elo_prob, d_elo_prob, a_elo_prob = h_elo_prob/total, d_elo_prob/total, a_elo_prob/total
            
    feats["clubelo_home_prob"] = h_elo_prob
    feats["clubelo_draw_prob"] = d_elo_prob
    feats["clubelo_away_prob"] = a_elo_prob

    # ── 3. Player Injuries (Transfermarkt) ──────────────────────────────────────
    # We find all active injuries for a team.
    # To determine impact, we sum the Market_Value of injured players.
    
    def _compute_injury_impact(team_id):
        # We look for the most recent scrape for this team
        cur.execute("""
            SELECT MAX(scraped_at) as last_scrape 
            FROM player_injuries WHERE team_id = %s
        """, (team_id,))
        last = cur.fetchone()
        if not last or not last["last_scrape"]:
            return {"count": 0, "total_value_m": 0.0}
            
        latest_time = last["last_scrape"]
        # Pull all injuries scraped around that identical batch window (within 24h)
        cur.execute("""
            SELECT raw_data 
            FROM player_injuries 
            WHERE team_id = %s 
              AND scraped_at >= %s::timestamp - INTERVAL '1 day'
        """, (team_id, latest_time))
        injuries = cur.fetchall()
        
        val_sum = 0.0
        count = 0
        for inj in injuries:
            raw = inj.get("raw_data", {})
            val_sum += _parse_market_value(raw.get("Market_Value", ""))
            count += 1
            
        return {"count": count, "total_value_m": val_sum}

    h_inj = _compute_injury_impact(home_team_id)
    a_inj = _compute_injury_impact(away_team_id)
    
    feats["home_injured_players"] = float(h_inj["count"])
    feats["home_injured_value_m"] = h_inj["total_value_m"]
    feats["away_injured_players"] = float(a_inj["count"])
    feats["away_injured_value_m"] = a_inj["total_value_m"]
    feats["injury_value_gap_m"]   = h_inj["total_value_m"] - a_inj["total_value_m"]

    return feats
