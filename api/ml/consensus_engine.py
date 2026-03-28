"""
Dynamic Consensus Engine
========================
Aggregates three independent prediction engines into one synthesised output:
  1. DC Engine      — Dixon-Coles Poisson ensemble (dc_engine.py)
  2. ML Engine      — XGBoost + RandomForest ensemble (prediction_engine.py)
  3. Legacy Engine  — Heuristic rule-based predictor (predictions.py helpers)

Dynamic Weighting:
  Before blending, queries prediction_log for each engine's 30-day trailing
  accuracy.  Engines that have proved more accurate recently receive a higher
  weight in the blend.  Falls back to fixed defaults if < MIN_GRADED_ROWS rows
  have been graded.

Default weights (when insufficient history):
  DC 45% | ML 35% | Legacy 20%

Usage:
  from ml.consensus_engine import run_consensus
  result = run_consensus(home_team_id, away_team_id, league_id, season_id)
"""

import logging
import math
from typing import Optional

from database import get_connection

log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "dc":         0.45,
    "ml":         0.30,
    "enrichment": 0.00,
    "legacy":     0.25,
}

# Minimum graded rows required before we trust historical weights
MIN_GRADED_ROWS = 10

# Trailing window for accuracy calculation (days)
ACCURACY_WINDOW_DAYS = 30

OUTCOME_LABELS = ["Home Win", "Draw", "Away Win"]


# ─── Legacy heuristic helpers (self-contained, no circular imports) ───────────

def _safe_div(a, b, default=0.0):
    try:
        return float(a) / float(b) if b else default
    except Exception:
        return default


def _run_legacy_engine(cur, home_team_id: int, away_team_id: int) -> dict:
    """
    Lightweight heuristic prediction from league_standings + team_squad_stats.
    Returns {home_win, draw, away_win, predicted_outcome}.
    Falls back to equal probs on any error.
    """
    import numpy as np  # lazy import
    FALLBACK = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35,
                "predicted_outcome": "Home Win"}
    try:
        cur.execute("""
            SELECT ls.wins, ls.ties, ls.losses, ls.games,
                   ls.goals_for, ls.goals_against, ls.points
            FROM   league_standings ls
            WHERE  ls.team_id = %s
            ORDER  BY ls.season_id DESC LIMIT 1
        """, (home_team_id,))
        h_st = cur.fetchone()

        cur.execute("""
            SELECT ls.wins, ls.ties, ls.losses, ls.games,
                   ls.goals_for, ls.goals_against, ls.points
            FROM   league_standings ls
            WHERE  ls.team_id = %s
            ORDER  BY ls.season_id DESC LIMIT 1
        """, (away_team_id,))
        a_st = cur.fetchone()

        cur.execute("""
            SELECT ts.goals, ts.games, ts.possession
            FROM   team_squad_stats ts
            WHERE  ts.team_id = %s AND ts.split = 'for'
            ORDER  BY ts.season_id DESC LIMIT 1
        """, (home_team_id,))
        h_sq = cur.fetchone()

        cur.execute("""
            SELECT ts.goals, ts.games, ts.possession
            FROM   team_squad_stats ts
            WHERE  ts.team_id = %s AND ts.split = 'for'
            ORDER  BY ts.season_id DESC LIMIT 1
        """, (away_team_id,))
        a_sq = cur.fetchone()

        h_gpg = _safe_div(h_sq["goals"] if h_sq else 0,
                           h_sq["games"] if h_sq else 1, 1.2)
        a_gpg = _safe_div(a_sq["goals"] if a_sq else 0,
                           a_sq["games"] if a_sq else 1, 1.0)
        h_wr  = _safe_div(h_st["wins"] if h_st else 0,
                           h_st["games"] if h_st else 1, 0.40)
        a_wr  = _safe_div(a_st["wins"] if a_st else 0,
                           a_st["games"] if a_st else 1, 0.35)

        h_str = 0.6 * h_gpg + 0.4 * h_wr
        a_str = 0.6 * a_gpg + 0.4 * a_wr
        total = h_str + a_str + 0.001

        r_h = max(0.1, h_str / total + 0.06)   # home advantage bump
        r_a = max(0.1, a_str / total - 0.03)
        r_d = max(0.1, 1 - r_h - r_a)
        s   = r_h + r_a + r_d
        home_p = round(r_h / s, 4)
        away_p = round(r_a / s, 4)
        draw_p = round(1 - home_p - away_p, 4)

        probs = [home_p, draw_p, away_p]
        idx   = int(np.argmax(probs))
        outcome = OUTCOME_LABELS[idx]

        return {"home_win": home_p, "draw": draw_p, "away_win": away_p,
                "predicted_outcome": outcome}
    except Exception as exc:
        log.warning("Legacy engine error: %s", exc)
        return FALLBACK


# ─── Dynamic weight computation ───────────────────────────────────────────────

def _fetch_dynamic_weights(cur) -> dict:
    """
    Query prediction_log for each engine's trailing accuracy over the last
    ACCURACY_WINDOW_DAYS days.

    Returns normalised weight dict: {"dc": w1, "ml": w2, "legacy": w3}.
    Falls back to DEFAULT_WEIGHTS if not enough graded rows exist.
    """
    try:
        cur.execute(f"""
            SELECT
                COUNT(*)                                                 AS total,
                SUM(CASE WHEN dc_correct         THEN 1 ELSE 0 END)::float AS dc_correct,
                SUM(CASE WHEN ml_correct         THEN 1 ELSE 0 END)::float AS ml_correct,
                SUM(CASE WHEN enrichment_correct THEN 1 ELSE 0 END)::float AS enrichment_correct,
                SUM(CASE WHEN legacy_correct     THEN 1 ELSE 0 END)::float AS legacy_correct
            FROM prediction_log
            WHERE correct IS NOT NULL
              AND evaluated_at >= NOW() - INTERVAL '{ACCURACY_WINDOW_DAYS} days'
              AND dc_predicted_outcome         IS NOT NULL
              AND ml_predicted_outcome         IS NOT NULL
              AND enrichment_predicted_outcome IS NOT NULL
              AND legacy_predicted_outcome     IS NOT NULL
        """)
        row = cur.fetchone()
        if not row:
            return DEFAULT_WEIGHTS.copy()

        total = int(row["total"] or 0)
        if total < MIN_GRADED_ROWS:
            log.info("Consensus: only %d graded rows → using default weights", total)
            return DEFAULT_WEIGHTS.copy()

        dc_acc     = float(row["dc_correct"]         or 0) / total
        ml_acc     = float(row["ml_correct"]         or 0) / total
        enr_acc    = 0.0
        legacy_acc = float(row["legacy_correct"]     or 0) / total

        # Accuracy directly becomes the unnormalised weight
        weight_sum = dc_acc + ml_acc + enr_acc + legacy_acc
        if weight_sum < 0.01:
            return DEFAULT_WEIGHTS.copy()

        weights = {
            "dc":         round(dc_acc     / weight_sum, 4),
            "ml":         round(ml_acc     / weight_sum, 4),
            "enrichment": round(enr_acc    / weight_sum, 4),
            "legacy":     round(legacy_acc / weight_sum, 4),
        }
        log.info(
            "Consensus: dynamic weights from %d rows — DC=%.1f%% ML=%.1f%% ENR=%.1f%% Legacy=%.1f%%",
            total, weights["dc"]*100, weights["ml"]*100, weights["enrichment"]*100, weights["legacy"]*100,
        )
        return weights

    except Exception as exc:
        log.warning("Consensus: dynamic weight query failed (%s) → using defaults", exc)
        return DEFAULT_WEIGHTS.copy()


# ─── Probability blending ─────────────────────────────────────────────────────

def _blend(dc: dict, ml: dict, enrichment: dict, legacy: dict, weights: dict) -> dict:
    """Weighted linear blend of four probability distributions."""
    w_dc     = weights.get("dc", 0.0)
    w_ml     = weights.get("ml", 0.0)
    w_enr    = weights.get("enrichment", 0.0)
    w_legacy = weights.get("legacy", 0.0)

    hw = (w_dc * dc["home_win"] + w_ml * ml["home_win"] + w_enr * enrichment["home_win"] + w_legacy * legacy["home_win"])
    dr = (w_dc * dc["draw"]     + w_ml * ml["draw"]     + w_enr * enrichment["draw"]     + w_legacy * legacy["draw"])
    aw = (w_dc * dc["away_win"] + w_ml * ml["away_win"] + w_enr * enrichment["away_win"] + w_legacy * legacy["away_win"])

    total = hw + dr + aw or 1.0
    hw, dr, aw = hw / total, dr / total, aw / total

    return {
        "home_win": round(float(hw), 4),
        "draw":     round(float(dr), 4),
        "away_win": round(float(aw), 4),
    }


def _confidence_from_entropy(probs: dict) -> tuple:
    """Shannon-entropy-based confidence score (0–100) and label."""
    import numpy as np  # lazy import
    p = np.array([probs["home_win"], probs["draw"], probs["away_win"]])
    p = np.clip(p, 1e-10, 1.0)
    entropy     = float(-np.sum(p * np.log(p)))
    max_entropy = float(-np.log(1.0 / 3.0))
    score = round((1.0 - entropy / max_entropy) * 100, 1)
    if score >= 30:
        label = "High"
    elif score >= 15:
        label = "Medium"
    else:
        label = "Low"
    return score, label


def _agreement_level(dc_out: str, ml_out: str, enr_out: str, legacy_out: str) -> str:
    outcomes = [dc_out, ml_out, enr_out, legacy_out]
    unique = len(set(outcomes))
    if unique == 1:
        return "full"
    if unique == 2:
        # Check if 3 engines agree (majority) vs 2-2 tie
        from collections import Counter
        counts = Counter(outcomes).values()
        if max(counts) >= 3:
            return "majority"
    return "split"


# ─── Main public function ─────────────────────────────────────────────────────

def run_consensus(
    home_team_id: int,
    away_team_id: int,
    league_id:    int,
    season_id:    int,
) -> dict:
    """
    Run all three engines, blend results with dynamic weights, return a
    comprehensive consensus prediction.

    Return schema:
      consensus:   {home_win, draw, away_win, predicted_outcome,
                    confidence, confidence_score}
      engines:     {dc: {...}, ml: {...}, legacy: {...}}
      weights_used:{dc, ml, legacy, source}
      agreement:   "full" | "majority" | "split"
      markets:     {btts_yes, over_2_5, ...}
    """
    conn = get_connection()
    cur  = conn.cursor()

    errors = []

    try:
        # ── 1. Compute dynamic weights ─────────────────────────────────────────
        weights = _fetch_dynamic_weights(cur)
        weight_source = (
            "dynamic_historical"
            if weights != DEFAULT_WEIGHTS
            else "default_fallback"
        )

        # ── 2. Run DC engine ──────────────────────────────────────────────────
        try:
            from ml.dc_engine import predict_dc_match  # lazy import
            import numpy as np
            dc_raw = predict_dc_match(home_team_id, away_team_id)
            if "error" in dc_raw:
                raise RuntimeError(dc_raw["error"])
            dc_probs = {
                "home_win": float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("home_win", 0.35)),
                "draw":     float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("draw",     0.30)),
                "away_win": float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("away_win", 0.35)),
            }
            # Normalise
            s = sum(dc_probs.values()) or 1
            dc_probs = {k: round(v / s, 4) for k, v in dc_probs.items()}
            dc_outcome_idx = int(np.argmax([dc_probs["home_win"], dc_probs["draw"], dc_probs["away_win"]]))
            dc_outcome = OUTCOME_LABELS[dc_outcome_idx]
        except Exception as exc:
            log.warning("Consensus: DC engine error: %s", exc)
            errors.append(f"dc: {exc}")
            dc_probs   = DEFAULT_WEIGHTS.copy()   # crude fallback
            dc_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            dc_outcome = "Home Win"
            weights["dc"] = 0.0   # zero-weight broken engine

        # ── 3. Run ML engine ──────────────────────────────────────────────────
        try:
            import ml.prediction_engine as ml_engine  # lazy import
            ml_raw = ml_engine.predict_match(home_team_id, away_team_id,
                                             league_id, season_id)
            if "error" in ml_raw:
                raise RuntimeError(ml_raw["error"])
            ml_probs_inner = ml_raw.get("probabilities", {})
            ml_probs = {
                "home_win": float(ml_probs_inner.get("home_win", 0.35)),
                "draw":     float(ml_probs_inner.get("draw",     0.30)),
                "away_win": float(ml_probs_inner.get("away_win", 0.35)),
            }
            s = sum(ml_probs.values()) or 1
            ml_probs = {k: round(v / s, 4) for k, v in ml_probs.items()}
            ml_outcome = ml_raw.get("predicted_outcome", "Home Win")
            ml_match_info = ml_raw.get("match", {})
            home_name = ml_match_info.get("home_team", f"Team {home_team_id}")
            away_name = ml_match_info.get("away_team", f"Team {away_team_id}")
            league_name = ml_match_info.get("league", "")
            season_name = ml_match_info.get("season", "")
            expected_goals = ml_raw.get("expected_goals", {})
        except Exception as exc:
            log.warning("Consensus: ML engine error: %s", exc)
            errors.append(f"ml: {exc}")
            ml_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            ml_outcome = "Home Win"
            home_name  = f"Team {home_team_id}"
            away_name  = f"Team {away_team_id}"
            league_name = ""
            season_name = ""
            expected_goals = {}
            weights["ml"] = 0.0

        # ── 4. Run Enrichment engine ──────────────────────────────────────────
        try:
            predict_enrichment = lambda *args: {"error": "Enrichment disabled by user"}  # lazy import
            
            # Find exact match date if it exists
            cur.execute("""
                SELECT match_date FROM matches 
                WHERE home_team_id = %s AND away_team_id = %s AND season_id = %s LIMIT 1
            """, (home_team_id, away_team_id, season_id))
            mr = cur.fetchone()
            m_date = str(mr["match_date"]) if mr and mr["match_date"] else None
            
            enr_raw = predict_enrichment(home_team_id, away_team_id, m_date)
            if "error" in enr_raw:
                raise RuntimeError(enr_raw["error"])
                
            enr_probs = {
                "home_win": float(enr_raw.get("home_win", 0.35)),
                "draw":     float(enr_raw.get("draw",     0.30)),
                "away_win": float(enr_raw.get("away_win", 0.35)),
            }
            s = sum(enr_probs.values()) or 1
            enr_probs = {k: round(v / s, 4) for k, v in enr_probs.items()}
            enr_outcome = enr_raw.get("predicted_outcome", "Home Win")
            enr_features = enr_raw.get("_features", {})
        except Exception as exc:
            log.warning("Consensus: Enrichment engine error: %s", exc)
            errors.append(f"enrichment: {exc}")
            enr_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            enr_outcome = "Home Win"
            enr_features = {}
            weights["enrichment"] = 0.0

        # ── 5. Run Legacy engine ──────────────────────────────────────────────
        try:
            legacy_raw     = _run_legacy_engine(cur, home_team_id, away_team_id)
            legacy_probs   = {k: v for k, v in legacy_raw.items()
                              if k in ("home_win", "draw", "away_win")}
            legacy_outcome = legacy_raw.get("predicted_outcome", "Home Win")
        except Exception as exc:
            log.warning("Consensus: Legacy engine error: %s", exc)
            errors.append(f"legacy: {exc}")
            legacy_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            legacy_outcome = "Home Win"
            weights["legacy"] = 0.0

        # ── 6. Re-normalise weights if any engine failed ─────────────────────
        w_sum = sum(weights.values())
        if w_sum < 0.01:
            weights = DEFAULT_WEIGHTS.copy()
            w_sum = 1.0
        weights = {k: round(v / w_sum, 4) for k, v in weights.items()}

        # ── 7. Blend ──────────────────────────────────────────────────────────
        blended = _blend(dc_probs, ml_probs, enr_probs, legacy_probs, weights)

        # ── 8. Consensus outcome + confidence ─────────────────────────────────
        import numpy as np  # lazy import (already imported above in DC block)
        idx = int(np.argmax([blended["home_win"], blended["draw"], blended["away_win"]]))
        consensus_outcome   = OUTCOME_LABELS[idx]
        confidence_score, confidence_label = _confidence_from_entropy(blended)
        agreement = _agreement_level(dc_outcome, ml_outcome, enr_outcome, legacy_outcome)

        # ── 8. Simple market proxies from blended probs + xG ─────────────────
        home_xg = float(expected_goals.get("home_xg", 1.35))
        away_xg = float(expected_goals.get("away_xg", 1.10))
        btts_yes  = round(1 - math.exp(-home_xg) - math.exp(-away_xg)
                          + math.exp(-(home_xg + away_xg)), 4)
        btts_yes  = round(max(0.0, min(btts_yes, 1.0)), 4)
        over_2_5  = round(1 - sum(
            (math.exp(-(home_xg + away_xg)) * ((home_xg + away_xg) ** k) / math.factorial(k))
            for k in range(3)
        ), 4)
        over_2_5  = round(max(0.0, min(over_2_5, 1.0)), 4)

        return {
            "match": {
                "home_team":    home_name,
                "away_team":    away_name,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "league":       league_name,
                "season":       season_name,
            },
            "consensus": {
                "home_win":        blended["home_win"],
                "draw":            blended["draw"],
                "away_win":        blended["away_win"],
                "predicted_outcome": consensus_outcome,
                "confidence":      confidence_label,
                "confidence_score": confidence_score,
            },
            "engines": {
                "dc": {
                    "home_win":          dc_probs["home_win"],
                    "draw":              dc_probs["draw"],
                    "away_win":          dc_probs["away_win"],
                    "predicted_outcome": dc_outcome,
                },
                "ml": {
                    "home_win":          ml_probs["home_win"],
                    "draw":              ml_probs["draw"],
                    "away_win":          ml_probs["away_win"],
                    "predicted_outcome": ml_outcome,
                },
                "enrichment": {
                    "home_win":          enr_probs["home_win"],
                    "draw":              enr_probs["draw"],
                    "away_win":          enr_probs["away_win"],
                    "predicted_outcome": enr_outcome,
                    "diagnostics":       enr_features,
                },
                "legacy": {
                    "home_win":          legacy_probs["home_win"],
                    "draw":              legacy_probs["draw"],
                    "away_win":          legacy_probs["away_win"],
                    "predicted_outcome": legacy_outcome,
                },
            },
            "weights_used": {**weights, "source": weight_source},
            "agreement":    agreement,
            "markets": {
                "btts_yes":  btts_yes,
                "btts_no":   round(1 - btts_yes, 4),
                "over_2_5":  over_2_5,
                "under_2_5": round(1 - over_2_5, 4),
                "home_xg":   round(home_xg, 2),
                "away_xg":   round(away_xg, 2),
            },
            # Individual engine picks — stored in prediction_log for grading
            "_engine_picks": {
                "dc":         dc_outcome,
                "ml":         ml_outcome,
                "enrichment": enr_outcome,
                "legacy":     legacy_outcome,
            },
            **({"_errors": errors} if errors else {}),
        }

    finally:
        conn.close()

def upcoming_consensus_fast(league_id: int = None, limit: int = 50) -> list:
    """
    Bulk-predicts upcoming matches for the Consensus Engine at sub-second speeds.
    Bypasses the 30-second loop limitation by instantiating the DB DataCache exactly ONCE
    and sharing it across all native engines for the entire fixture set.
    """
    conn = get_connection()
    cur  = conn.cursor()
    errors = []
    
    try:
        # Compute dynamic weights once
        weights = _fetch_dynamic_weights(cur)
        weight_source = "dynamic_historical" if weights != DEFAULT_WEIGHTS else "default_fallback"

        # Bulk load fixtures
        query = """
            SELECT m.id, m.home_team_id, m.away_team_id,
                   m.league_id, m.season_id, m.match_date, m.gameweek,
                   ht.name AS home_name, at.name AS away_name,
                   ht.logo_url AS home_logo, at.logo_url AS away_logo
            FROM matches m
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams at ON at.id = m.away_team_id
            WHERE m.home_score IS NULL AND m.match_date >= CURRENT_DATE
        """
        params = []
        if league_id:
            query += " AND m.league_id = %s"
            params.append(league_id)
        query += " ORDER BY m.match_date ASC LIMIT %s"
        params.append(limit)
        cur.execute(query, params)
        fixtures = [dict(r) for r in cur.fetchall()]

        if not fixtures:
            return []

        # Bulk load DB state exactly once
        from ml.batch_features import DataCache, _build_match_features
        import ml.prediction_engine as ml_engine
        
        ml_cache = DataCache(cur)
        ml_model = ml_engine._get_engine()
        
        # We need the DC engine in memory too
        from ml.dc_engine import predict_dc_match
        
        # We need Enrichment features lazily
        predict_enrichment = lambda *args: {"error": "Enrichment disabled by user"}
        
        results = []
        import numpy as np

        for fx in fixtures:
            match_payload = {
                "home_team":    fx["home_name"],
                "away_team":    fx["away_name"],
                "home_team_id": fx["home_team_id"],
                "away_team_id": fx["away_team_id"],
                "home_logo":    fx["home_logo"],
                "away_logo":    fx["away_logo"],
                "league":       "",
                "season":       "",
            }
            
            # --- DC 
            try:
                dc_raw = predict_dc_match(fx["home_team_id"], fx["away_team_id"])
                if "error" in dc_raw: raise RuntimeError(dc_raw["error"])
                dc_probs = {
                    "home_win": float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("home_win", 0.35)),
                    "draw":     float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("draw",     0.30)),
                    "away_win": float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("away_win", 0.35)),
                }
                s = sum(dc_probs.values()) or 1
                dc_probs = {k: round(v / s, 4) for k, v in dc_probs.items()}
                dc_outcome = OUTCOME_LABELS[int(np.argmax([dc_probs["home_win"], dc_probs["draw"], dc_probs["away_win"]]))]
            except Exception as e:
                dc_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
                dc_outcome = "Home Win"
                weights["dc"] = 0.0
                
            # --- ML Engine (Instant Cache Bypass)
            try:
                if not ml_model or not ml_model.is_trained:
                    raise RuntimeError("ML Model not trained.")
                fv, feat_names, home_feats, away_feats, h2h = _build_match_features(
                    ml_cache, fx["home_team_id"], fx["away_team_id"], fx["league_id"], fx["season_id"],
                    match_id=fx["id"], match_date=fx["match_date"]
                )
                raw_proba = ml_model.predict_proba(fv)
                calibrated = ml_engine._apply_calibration({
                    "home_win": raw_proba["home_win"],
                    "draw":     raw_proba["draw"],
                    "away_win": raw_proba["away_win"]
                })
                # Re-normalize just in case
                sm = sum(calibrated.values()) or 1
                ml_probs = {k: round(v/sm, 4) for k, v in calibrated.items()}
                ml_outcome = OUTCOME_LABELS[int(np.argmax([ml_probs["home_win"], ml_probs["draw"], ml_probs["away_win"]]))]
                
                home_xg = ml_engine._compute_venue_xg(home_feats, away_feats, "home")
                away_xg = ml_engine._compute_venue_xg(away_feats, home_feats, "away")
            except Exception as e:
                ml_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
                ml_outcome = "Home Win"
                home_xg, away_xg = 1.35, 1.10
                weights["ml"] = 0.0

            # --- Enrichment
            try:
                enr_raw = predict_enrichment(fx["home_team_id"], fx["away_team_id"], str(fx["match_date"]) if fx["match_date"] else None)
                if "error" in enr_raw and "all defaults" not in str(enr_raw.get("error", "")):
                    raise RuntimeError(enr_raw["error"])
                
                # If it's the "all defaults" explicit abstention loop:
                if "error" in enr_raw and "all defaults" in str(enr_raw.get("error")):
                    enr_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
                    enr_outcome = "—"
                    enr_features = enr_raw.get("_features", {})
                    weights["enrichment"] = 0.0
                else:
                    enr_probs = {
                        "home_win": float(enr_raw.get("home_win", 0.35)),
                        "draw":     float(enr_raw.get("draw",     0.30)),
                        "away_win": float(enr_raw.get("away_win", 0.35)),
                    }
                    s = sum(enr_probs.values()) or 1
                    enr_probs = {k: round(v / s, 4) for k, v in enr_probs.items()}
                    enr_outcome = enr_raw.get("predicted_outcome", "Home Win")
                    enr_features = enr_raw.get("_features", {})
            except Exception as e:
                enr_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
                enr_outcome = "Home Win"
                enr_features = {}
                weights["enrichment"] = 0.0
                
            # --- Legacy
            try:
                legacy_raw = _run_legacy_engine(cur, fx["home_team_id"], fx["away_team_id"])
                legacy_probs = {k: v for k, v in legacy_raw.items() if k in ("home_win", "draw", "away_win")}
                legacy_outcome = legacy_raw.get("predicted_outcome", "Home Win")
            except Exception as e:
                legacy_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
                legacy_outcome = "Home Win"
                weights["legacy"] = 0.0
                
            # Re-normalize weights for this specific match loop
            match_weights = weights.copy()
            w_sum = sum(match_weights.values())
            if w_sum < 0.01:
                match_weights = DEFAULT_WEIGHTS.copy()
                w_sum = 1.0
            match_weights = {k: round(v / w_sum, 4) for k, v in match_weights.items()}

            blended = _blend(dc_probs, ml_probs, enr_probs, legacy_probs, match_weights)
            idx = int(np.argmax([blended["home_win"], blended["draw"], blended["away_win"]]))
            consensus_outcome = OUTCOME_LABELS[idx]
            confidence_score, confidence_label = _confidence_from_entropy(blended)
            agreement = _agreement_level(dc_outcome, ml_outcome, enr_outcome, legacy_outcome)

            import math
            btts_yes  = round(1 - math.exp(-home_xg) - math.exp(-away_xg) + math.exp(-(home_xg + away_xg)), 4)
            btts_yes  = round(max(0.0, min(btts_yes, 1.0)), 4)
            over_2_5  = round(1 - sum((math.exp(-(home_xg + away_xg)) * ((home_xg + away_xg) ** k) / math.factorial(k)) for k in range(3)), 4)
            over_2_5  = round(max(0.0, min(over_2_5, 1.0)), 4)
            
            results.append({
                "fixture_id": fx["id"],
                "match_date": str(fx["match_date"]) if fx["match_date"] else None,
                "gameweek":   fx["gameweek"],
                "match":      match_payload,
                "consensus": {
                    "home_win":          blended["home_win"],
                    "draw":              blended["draw"],
                    "away_win":          blended["away_win"],
                    "predicted_outcome": consensus_outcome,
                    "confidence":        confidence_label,
                    "confidence_score":  confidence_score,
                },
                "engines": {
                    "dc": {"home_win": dc_probs["home_win"], "draw": dc_probs["draw"], "away_win": dc_probs["away_win"], "predicted_outcome": dc_outcome},
                    "ml": {"home_win": ml_probs["home_win"], "draw": ml_probs["draw"], "away_win": ml_probs["away_win"], "predicted_outcome": ml_outcome},
                    "enrichment": {"home_win": enr_probs["home_win"], "draw": enr_probs["draw"], "away_win": enr_probs["away_win"], "predicted_outcome": enr_outcome, "diagnostics": enr_features},
                    "legacy": {"home_win": legacy_probs["home_win"], "draw": legacy_probs["draw"], "away_win": legacy_probs["away_win"], "predicted_outcome": legacy_outcome},
                },
                "weights_used": {**match_weights, "source": weight_source},
                "agreement": agreement,
                "markets": {
                    "btts_yes":  btts_yes,
                    "btts_no":   round(1 - btts_yes, 4),
                    "over_2_5":  over_2_5,
                    "under_2_5": round(1 - over_2_5, 4),
                    "home_xg":   round(home_xg, 2),
                    "away_xg":   round(away_xg, 2),
                },
                "_engine_picks": {
                    "dc":         dc_outcome,
                    "ml":         ml_outcome,
                    "enrichment": enr_outcome,
                    "legacy":     legacy_outcome,
                }
            })
            
        return results

    finally:
        conn.close()
