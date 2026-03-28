"""
Betting Markets API Routes
===========================
GET  /api/markets              — Full market sheet for a fixture (needs DC model trained)
POST /api/markets/value        — Find value bets given bookmaker odds
GET  /api/markets/arb          — Scan for arb across multiple bookmakers
GET  /api/markets/dc/status    — DC model status
POST /api/markets/dc/train     — Train (or retrain) DC model
GET  /api/markets/dc/leaderboard — Elo leaderboard
"""


from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Optional
import logging
import threading
import time

from database import get_connection
from ml.dc_engine  import (predict_dc_match, train_dc_model,
                            dc_status, get_dc_predictor)
from ml.markets    import MarketPricer, ValueDetector, ArbitrageScanner

router = APIRouter()
log = logging.getLogger(__name__)

# ─── Training state (persistent thread, survives navigation) ──────────────────
_dc_training_state: dict = {}

def _run_dc_training():
    global _dc_training_state, _upcoming_cache
    _dc_training_state = {"status": "running", "started_at": time.time()}
    try:
        result = train_dc_model()
        _dc_training_state = {"status": "done", "result": result}
        # Bust the upcoming cache so Free Picks reflects the new model immediately
        _upcoming_cache = {"data": None, "expires_at": 0.0}
        if result.get("trained"):
            log.info("DC retrain complete: %d matches", result.get("n_matches", 0))
        else:
            log.error("DC retrain failed: %s", result.get("error"))
    except Exception as e:
        _dc_training_state = {"status": "error", "error": str(e)}
        log.error("DC training exception: %s", e)


# ─── Helper: log DC prediction to prediction_log ──────────────────────────────

def _log_prediction_bg(match_id: int, home_team: str, away_team: str,
                       league: str, match_date, predicted_outcome: str,
                       confidence: str, confidence_score: float,
                       home_win_prob: float, draw_prob: float,
                       away_win_prob: float):
    try:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO prediction_log
                    (match_id, home_team, away_team, league, match_date,
                     predicted, confidence, confidence_score,
                     home_win_prob, draw_prob, away_win_prob)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                int(match_id),
                str(home_team), str(away_team), str(league), match_date,
                str(predicted_outcome), str(confidence),
                float(confidence_score),
                float(home_win_prob), float(draw_prob), float(away_win_prob),
            ))
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.exception("prediction_log insert failed")



# ─── 15-min cache for /upcoming (makes Free Picks near-instant after first hit) 
_upcoming_cache: dict = {"data": None, "expires_at": 0.0}
_UPCOMING_TTL = 15 * 60   # seconds


# ─── GET /api/markets/upcoming ────────────────────────────────────────────────

def _best_engine_bet_recommendations(engine_name: str, probs: dict, xg_h: float, xg_a: float) -> list:
    """Derive precise bet recommendations explicitly from the most historically accurate champion engine."""
    hw = probs.get("home_win", 0.33)
    dr = probs.get("draw", 0.33)
    aw = probs.get("away_win", 0.34)
    lead = max(hw, dr, aw)
    if lead == hw:
        outcome, outcome_label = "home_win", "Home Win"
    elif lead == aw:
        outcome, outcome_label = "away_win", "Away Win"
    else:
        outcome, outcome_label = "draw", "Draw"

    bets = []
    display_name = "XGBoost ML" if engine_name == "ml" else engine_name.title()
    
    if lead >= 0.55:
        bets.append({"bet": f"✅ Strong Pick — {outcome_label} (Powered by {display_name})", "prob": round(lead * 100), "tier": "high"})
    elif lead >= 0.47:
        bets.append({"bet": f"💡 Value Bet — {outcome_label} (Powered by {display_name})", "prob": round(lead * 100), "tier": "medium"})
    if xg_h >= 1.1 and xg_a >= 1.0:
        bets.append({"bet": "⚽ Both Teams to Score (recommended)", "prob": None, "tier": "btts"})
    if dr >= 0.33 and outcome != "draw":
        bets.append({"bet": f"⚠️ Draw value — {round(dr * 100)}% probability", "prob": round(dr * 100), "tier": "draw_value"})
    return bets


@router.get("/upcoming")
def upcoming_dc_predictions(
    league_id: Optional[int] = Query(None, description="Filter by league ID"),
    limit: int               = Query(30,   description="Max fixtures"),
):
    """
    Bulk Consensus predictions for all upcoming fixtures.
    Results are cached for 15 minutes to keep the Free Picks page near-instant.
    """
    global _upcoming_cache

    from ml.consensus_engine import upcoming_consensus_fast

    # Return from cache if still fresh
    cache_key = (league_id, limit)
    now = time.time()
    if (
        _upcoming_cache.get("data") is not None
        and _upcoming_cache.get("expires_at", 0) > now
        and _upcoming_cache.get("key") == cache_key
    ):
        log.debug("Returning /upcoming from cache (%.0fs remaining)",
                  _upcoming_cache["expires_at"] - now)
        return _upcoming_cache["data"]

    # ── Bulk Generate Consensus Pipeline ──
    raw_results = upcoming_consensus_fast(league_id, limit)
    
    results = []
    from .predictions import _log_prediction_to_db as _log_db

    for pred in raw_results:
        try:
            fx = pred["match"]
            fx_id = pred["fixture_id"]
            consensus = pred["consensus"]
            engines   = pred["engines"]

            hw   = consensus.get("home_win", 0.33)
            dr   = consensus.get("draw",     0.33)
            aw   = consensus.get("away_win", 0.34)
            lead = max(hw, dr, aw)
            outcome = consensus.get("predicted_outcome", "Home Win")
            confidence = consensus.get("confidence", "Medium")

            markets = pred.get("markets", {})
            xg_h = float(markets.get("home_xg", 0))
            xg_a = float(markets.get("away_xg", 0))
            pred_h = max(0, round(xg_h))
            pred_a = max(0, round(xg_a))
            weight_map = pred.get("weights_used", {})

            # ── Pluck the Champion Engine dynamically ──
            valid_engines = {k: v for k, v in weight_map.items() if k in ["dc", "ml", "legacy", "enrichment"]}
            champion_engine_name = "ml" # safe fallback
            if valid_engines:
                champion_engine_name = max(valid_engines, key=valid_engines.get)
            
            champion_probs = engines.get(champion_engine_name, {"home_win": hw, "draw": dr, "away_win": aw})

            result_entry = {
                "predicted_outcome": outcome,
                "confidence":        confidence,
                "confidence_score":  round(lead, 4),
                "probabilities": {
                    "home_win": round(hw, 4),
                    "draw":     round(dr, 4),
                    "away_win": round(aw, 4),
                },
                "expected_goals": {
                    "home_xg":       round(xg_h, 2),
                    "away_xg":       round(xg_a, 2),
                    "predicted_score": f"{pred_h}-{pred_a}",
                },
                "match": {
                    "match_id":      fx_id,
                    "home_team":     fx["home_team"],
                    "away_team":     fx["away_team"],
                    "home_team_id":  fx["home_team_id"],
                    "away_team_id":  fx["away_team_id"],
                    "home_logo":     fx.get("home_logo"),
                    "away_logo":     fx.get("away_logo"),
                    "league":        fx.get("league", ""),
                    "league_id":     fx.get("league_id", 0),
                    "date":          pred["match_date"],
                    "gameweek":      pred["gameweek"],
                    "season":        fx.get("season", ""),
                },
                # MAP consensus engines payload to the frontend's expected breakdown
                "model_breakdown": engines,
                "weights": weight_map,
                "bet_recommendations": _best_engine_bet_recommendations(champion_engine_name, champion_probs, xg_h, xg_a),
                "engine": "consensus",
            }
            results.append(result_entry)
            
            dc_out  = engines.get("dc", {}).get("predicted_outcome")
            ml_out  = engines.get("ml", {}).get("predicted_outcome")
            leg_out = engines.get("legacy", {}).get("predicted_outcome")
            enr_out = engines.get("enrichment", {}).get("predicted_outcome")
            
            # Auto-log handled internally background thread
            threading.Thread(
                target=_log_db,
                args=(result_entry, int(fx_id), dc_out, ml_out, enr_out, leg_out, f"{pred_h}-{pred_a}"),
                daemon=True,
            ).start()
        except Exception as exc:
            log.debug("Formatting skip fixture %s vs %s: %s", pred.get("match", {}).get("home_team"), pred.get("match", {}).get("away_team"), exc)
            continue

    response = {"count": len(results), "predictions": results, "engine": "consensus"}
    _upcoming_cache = {"data": response, "expires_at": time.time() + _UPCOMING_TTL, "key": cache_key}
    log.info("Built /upcoming cache: %d consensus predictions, TTL %ds", len(results), _UPCOMING_TTL)
    return response



# ─── GET /api/markets/match-preview ──────────────────────────────────────────

@router.get("/match-preview")
def match_preview(
    home_team_id: int  = Query(..., description="Home team ID"),
    away_team_id: int  = Query(..., description="Away team ID"),
):
    """
    Rich pre-match data for the Free Picks card expand view.
    Returns: last-5 results per team (with scores), season stats,
             H2H last-8 with scores, next-3 upcoming per team.
    """
    conn = get_connection()
    cur  = conn.cursor()

    try:
        # ── Helper: last N played results for one team ──────────────────
        def recent_results(tid: int, n: int = 5):
            cur.execute("""
                SELECT
                    m.match_date, m.gameweek,
                    CASE WHEN m.home_team_id = %(t)s THEN at2.name ELSE ht2.name END AS opponent,
                    CASE WHEN m.home_team_id = %(t)s THEN 'H' ELSE 'A'           END AS venue,
                    CASE WHEN m.home_team_id = %(t)s THEN m.home_score ELSE m.away_score END AS ts,
                    CASE WHEN m.home_team_id = %(t)s THEN m.away_score ELSE m.home_score END AS os,
                    l.name AS league
                FROM matches m
                JOIN teams  ht2 ON ht2.id = m.home_team_id
                JOIN teams  at2 ON at2.id = m.away_team_id
                JOIN leagues l  ON l.id   = m.league_id
                WHERE (m.home_team_id = %(t)s OR m.away_team_id = %(t)s)
                  AND m.home_score IS NOT NULL
                ORDER BY m.match_date DESC
                LIMIT %(n)s
            """, {"t": tid, "n": n})
            out = []
            for r in cur.fetchall():
                ts = r["ts"] or 0;  os = r["os"] or 0
                out.append({
                    "date":     str(r["match_date"]) if r["match_date"] else None,
                    "gameweek": r["gameweek"],
                    "opponent": r["opponent"],
                    "venue":    r["venue"],
                    "score":    f"{ts}-{os}",
                    "result":   "W" if ts > os else ("L" if ts < os else "D"),
                    "league":   r["league"],
                })
            return out

        # ── Helper: season aggregate stats ─────────────────────────────
        def season_stats(tid: int):
            cur.execute("""
                SELECT
                    COUNT(*) AS played,
                    SUM(CASE WHEN (home_team_id=%(t)s AND home_score>away_score)
                              OR  (away_team_id=%(t)s AND away_score>home_score) THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN home_score=away_score THEN 1 ELSE 0 END) AS draws,
                    SUM(CASE WHEN (home_team_id=%(t)s AND home_score<away_score)
                              OR  (away_team_id=%(t)s AND away_score<home_score) THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN home_team_id=%(t)s THEN home_score ELSE away_score END) AS gf,
                    SUM(CASE WHEN home_team_id=%(t)s THEN away_score ELSE home_score END) AS ga,
                    SUM(CASE WHEN (home_team_id=%(t)s AND away_score=0)
                              OR  (away_team_id=%(t)s AND home_score=0) THEN 1 ELSE 0 END) AS cs
                FROM matches
                WHERE (home_team_id=%(t)s OR away_team_id=%(t)s)
                  AND home_score IS NOT NULL
            """, {"t": tid})
            r = cur.fetchone()
            if not r or not r["played"]: return {}
            p = r["played"]
            return {
                "played":             p,
                "wins":               r["wins"]  or 0,
                "draws":              r["draws"] or 0,
                "losses":             r["losses"] or 0,
                "goals_for":          r["gf"] or 0,
                "goals_against":      r["ga"] or 0,
                "avg_goals_for":      round((r["gf"] or 0) / p, 2),
                "avg_goals_against":  round((r["ga"] or 0) / p, 2),
                "clean_sheets":       r["cs"] or 0,
                "clean_sheet_pct":    round((r["cs"] or 0) / p * 100),
            }

        # ── Helper: next N upcoming for one team (excluding the H2H match) ──
        def upcoming_fixtures(tid: int, excl_opp: int, n: int = 3):
            cur.execute("""
                SELECT
                    m.match_date, m.gameweek,
                    CASE WHEN m.home_team_id=%(t)s THEN at3.name ELSE ht3.name END AS opponent,
                    CASE WHEN m.home_team_id=%(t)s THEN 'H' ELSE 'A'           END AS venue,
                    l.name AS league
                FROM matches m
                JOIN teams  ht3 ON ht3.id = m.home_team_id
                JOIN teams  at3 ON at3.id = m.away_team_id
                JOIN leagues l  ON l.id   = m.league_id
                WHERE (m.home_team_id=%(t)s OR m.away_team_id=%(t)s)
                  AND m.home_score IS NULL
                  AND m.match_date >= CURRENT_DATE
                  AND NOT ((m.home_team_id=%(t)s AND m.away_team_id=%(e)s)
                        OR (m.away_team_id=%(t)s AND m.home_team_id=%(e)s))
                ORDER BY m.match_date ASC
                LIMIT %(n)s
            """, {"t": tid, "e": excl_opp, "n": n})
            return [
                {"date":     str(r["match_date"]) if r["match_date"] else None,
                 "gameweek": r["gameweek"],
                 "opponent": r["opponent"],
                 "venue":    r["venue"],
                 "league":   r["league"]}
                for r in cur.fetchall()
            ]

        # ── H2H ────────────────────────────────────────────────────────
        cur.execute("""
            SELECT
                m.match_date, m.gameweek,
                ht4.name AS home_team,  at4.name AS away_team,
                m.home_team_id,         m.away_team_id,
                m.home_score,           m.away_score,
                l.name AS league
            FROM matches m
            JOIN teams  ht4 ON ht4.id = m.home_team_id
            JOIN teams  at4 ON at4.id = m.away_team_id
            JOIN leagues l  ON l.id   = m.league_id
            WHERE ((m.home_team_id=%(h)s AND m.away_team_id=%(a)s)
                OR (m.home_team_id=%(a)s AND m.away_team_id=%(h)s))
              AND m.home_score IS NOT NULL
            ORDER BY m.match_date DESC
            LIMIT 8
        """, {"h": home_team_id, "a": away_team_id})
        h2h_rows = cur.fetchall()

        hw = dr = aw = 0
        h2h_matches = []
        for r in h2h_rows:
            hs = r["home_score"] or 0
            as_ = r["away_score"] or 0
            if hs > as_:
                result_code = "H"
                if r["home_team_id"] == home_team_id: hw += 1
                else: aw += 1
            elif hs < as_:
                result_code = "A"
                if r["away_team_id"] == home_team_id: hw += 1
                else: aw += 1
            else:
                result_code = "D"
                dr += 1
            h2h_matches.append({
                "date":      str(r["match_date"]) if r["match_date"] else None,
                "gameweek":  r["gameweek"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "score":     f"{hs}-{as_}",
                "result":    result_code,
                "league":    r["league"],
            })

        return {
            "home": {
                "team_id":  home_team_id,
                "form":     recent_results(home_team_id),
                "stats":    season_stats(home_team_id),
                "upcoming": upcoming_fixtures(home_team_id, away_team_id),
            },
            "away": {
                "team_id":  away_team_id,
                "form":     recent_results(away_team_id),
                "stats":    season_stats(away_team_id),
                "upcoming": upcoming_fixtures(away_team_id, home_team_id),
            },
            "h2h": {
                "summary": {"home_wins": hw, "draws": dr, "away_wins": aw, "total": hw + dr + aw},
                "matches":  h2h_matches,
            },
        }
    finally:
        conn.close()


@router.get("")
def get_markets(
    home_team_id: int = Query(..., description="Home team ID"),
    away_team_id: int = Query(..., description="Away team ID"),
    n_sim: int        = Query(100_000, ge=10_000, le=500_000,
                              description="Monte Carlo simulations"),
):
    """
    Full betting market sheet for a fixture.
    Expected goals come from the DC ensemble model.
    Includes: 1X2, Asian Handicap, O/U, BTTS, Correct Score,
              Double Chance, Draw No Bet, Win to Nil, Team Goals.
    """
    dc = get_dc_predictor()
    if dc is None or not dc.fitted:
        raise HTTPException(
            status_code=503,
            detail="DC model not trained. POST /api/markets/dc/train first.",
        )

    result = predict_dc_match(home_team_id, away_team_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    lam_h = result.get("exp_home_goals", 1.35)
    lam_a = result.get("exp_away_goals", 1.10)

    pricer = MarketPricer(lam_h, lam_a, n_sim=n_sim)
    sheet  = pricer.full_sheet()

    return {
        "fixture": {
            "home_team":    result["home_team"],
            "away_team":    result["away_team"],
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
        },
        "ensemble_prediction": {
            "calibrated":   result["calibrated"],
            "prediction":   result["prediction"],
            "confidence":   result["confidence"],
            "models":       result["models"],
        },
        "markets": sheet,
        "model_info": result.get("model_info", {}),
    }


# ─── POST /api/markets/value ──────────────────────────────────────────────────

class ValueRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    market_odds: dict        # {"home_win": 2.45, "draw": 3.40, "away_win": 2.95}
    min_edge_pct: float = 3.0


@router.post("/value")
def get_value_bets(req: ValueRequest):
    """
    Find value bets by comparing DC model probabilities vs supplied bookmaker odds.
    Returns all outcomes where model edge > min_edge_pct%.
    """
    dc = get_dc_predictor()
    if dc is None or not dc.fitted:
        raise HTTPException(status_code=503,
                            detail="DC model not trained. POST /api/markets/dc/train first.")

    result = predict_dc_match(req.home_team_id, req.away_team_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    cal = result["calibrated"]
    model_probs = {
        "home_win": cal["home_win"],
        "draw":     cal["draw"],
        "away_win": cal["away_win"],
    }

    detector = ValueDetector(min_edge=req.min_edge_pct / 100)
    bets = detector.scan(model_probs, req.market_odds, market_name="1x2")

    return {
        "fixture": {
            "home_team": result["home_team"],
            "away_team": result["away_team"],
        },
        "model_probs":  model_probs,
        "value_bets":   bets,
        "n_value_bets": len(bets),
    }


# ─── GET /api/markets/arb ─────────────────────────────────────────────────────

class ArbRequest(BaseModel):
    books_odds: dict    # {"home_win": {"Pinnacle": 2.45, "Bet365": 2.40}, ...}


@router.post("/arb")
def scan_arbitrage(req: ArbRequest):
    """
    Scan for arbitrage opportunities across multiple bookmakers.
    Pass {outcome: {bookmaker: decimal_odds}} and we'll find risk-free arbs.
    """
    result = ArbitrageScanner.find_arb(req.books_odds)
    return result


# ─── GET /api/markets/dc/status ──────────────────────────────────────────────

@router.get("/dc/status")
def get_dc_status():
    """Current DC model status."""
    st = dc_status()
    # If the background thread is currently training, tell the frontend so polling continues
    if _dc_training_state.get("status") == "running":
        st["training_status"] = "running"
    elif "status" in _dc_training_state:
        st["training_status"] = _dc_training_state["status"]
    return st


# ─── POST /api/markets/dc/train ──────────────────────────────────────────────

@router.post("/dc/train")
def train_dc():
    """
    Train (or retrain) the DC + Elo + xG ensemble from DB data.
    Runs asynchronously in a daemon thread so it survives frontend disconnects.
    """
    global _dc_training_state
    if _dc_training_state.get("status") == "running":
        return {"message": "DC model training is already running."}

    thread = threading.Thread(target=_run_dc_training, daemon=True)
    thread.start()
    return {"message": "DC model training started in background thread."}


# ─── GET /api/markets/dc/predict ─────────────────────────────────────────────

@router.get("/dc/predict")
def dc_predict(
    home_team_id: int = Query(...),
    away_team_id: int = Query(...),
):
    """
    Full DC ensemble prediction for a fixture.
    Returns per-model breakdown (Dixon-Coles, Elo, xG, Ensemble, Calibrated),
    Monte Carlo markets, and confidence score.
    """
    dc = get_dc_predictor()
    if dc is None or not dc.fitted:
        raise HTTPException(status_code=503,
                            detail="DC model not trained. POST /api/markets/dc/train first.")
    result = predict_dc_match(home_team_id, away_team_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ─── GET /api/markets/dc/leaderboard ─────────────────────────────────────────

@router.get("/dc/leaderboard")
def elo_leaderboard():
    """Elo rating leaderboard across all teams in the DC model."""
    dc = get_dc_predictor()
    if dc is None or not dc.fitted:
        raise HTTPException(status_code=503,
                            detail="DC model not trained. POST /api/markets/dc/train first.")
    return {"leaderboard": dc.elo_leaderboard()}
