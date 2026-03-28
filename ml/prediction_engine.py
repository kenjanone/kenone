"""
Prediction Engine for PlusOne
==================================
Singleton-style engine that:
  1. Trains on ALL historical matches from the database
  2. Predicts match outcomes with rich, detailed output
  3. Persists trained model to disk (survives API restarts)
  4. Auto-loads saved model on startup if present
  5. Applies feedback calibration from prediction_log outcomes
     (call POST /api/predictions/recalibrate after evaluating predictions)
"""

import os
import time
from database import get_connection
from ml.feature_engineering import (
    build_match_features,
    build_training_dataset,
    compute_h2h,
)
from ml.ml_models import EnsemblePredictor
from ml.model_store import save_to_db, load_from_db
from ml.batch_features import build_training_dataset_fast, DataCache, _build_team_features, _build_match_features
from ml.feedback_calibrator import get_calibrator, reset_calibrator

# ─── Singleton state ──────────────────────────────────────────────────────────

_engine: EnsemblePredictor = None
_meta = {
    "trained_at": None,
    "n_samples": 0,
    "train_accuracy": None,
    "cv_accuracy": None,
    "errors": 0,
    "feature_names": [],
    "calibrator_samples": 0,
    "calibrator_accuracy": None,
}

OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}


def _get_engine() -> EnsemblePredictor:
    global _engine
    if _engine is None:
        loaded = load_from_db()
        if loaded is None:
            loaded = EnsemblePredictor.load()
        if loaded and loaded.is_trained:
            _engine = loaded
            _meta["n_samples"]      = loaded.n_samples
            _meta["train_accuracy"] = loaded.train_accuracy
            _meta["cv_accuracy"]    = loaded.cv_accuracy
            _meta["feature_names"]  = loaded.feature_names_ or []
    return _engine


def _apply_calibration(probs: dict) -> dict:
    """
    Apply feedback calibrator to raw model probabilities.
    Falls through unchanged if calibrator not fitted yet.
    Always re-applies a 2% floor after calibration so no outcome
    can ever display as 0% regardless of what isotonic regression learned.
    """
    try:
        cal = get_calibrator()
        if cal.is_fitted:
            calibrated = cal.apply(probs)
            _meta["calibrator_samples"]  = cal.n_samples
            _meta["calibrator_accuracy"] = cal.post_accuracy
            # Re-apply floor after calibration (calibrator may squeeze probs to 0)
            MIN_PROB = 0.02
            hw = max(calibrated["home_win"], MIN_PROB)
            dr = max(calibrated["draw"],     MIN_PROB)
            aw = max(calibrated["away_win"], MIN_PROB)
            total = hw + dr + aw
            return {
                "home_win": round(hw / total, 4),
                "draw":     round(dr / total, 4),
                "away_win": round(aw / total, 4),
            }
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Calibration apply failed: %s", exc)
    return probs


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model():
    """
    Pull all completed matches from the database, build 70+ feature vectors,
    train the XGBoost+RandomForest ensemble, and save to disk.
    Returns a status dict.
    """
    global _engine, _meta
    conn = get_connection()
    cur  = conn.cursor()
    try:
        t0 = time.time()
        X, y, match_ids, match_dates, errors = build_training_dataset_fast(cur, skip_errors=True)
        conn.close()

        if len(X) < 20:
            return {
                "success": False,
                "error": f"Not enough training data. Found {len(X)} completed matches — need at least 20.",
                "matches_found": len(X),
            }

        model = EnsemblePredictor()

        conn3 = get_connection()
        cur3  = conn3.cursor()
        feat_names = []
        try:
            cur3.execute("""
                SELECT m.id, m.home_team_id, m.away_team_id, m.league_id, m.season_id
                FROM matches m
                WHERE m.home_score IS NOT NULL
                  AND m.league_id IS NOT NULL AND m.season_id IS NOT NULL
                LIMIT 1
            """)
            sample = cur3.fetchone()
            if sample:
                from ml.batch_features import DataCache, _build_match_features as _batch_build
                cache = DataCache(cur3)
                _, feat_names, _, _, _ = _batch_build(
                    cache,
                    sample["home_team_id"],
                    sample["away_team_id"],
                    sample["league_id"],
                    sample["season_id"],
                )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Could not extract feature names during training: %s", exc)
        finally:
            conn3.close()

        model.train(X, y, feature_names=feat_names or None, match_dates=match_dates)
        model.save()
        save_to_db(model)

        _engine = model
        _meta.update({
            "trained_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_samples":     model.n_samples,
            "train_accuracy":model.train_accuracy,
            "cv_accuracy":   model.cv_accuracy,
            "errors":        errors,
            "feature_names": model.feature_names_ or [],
        })

        elapsed = round(time.time() - t0, 1)
        return {
            "success": True,
            "matches_trained":  model.n_samples,
            "train_accuracy":   round(model.train_accuracy or 0, 4),
            "cv_accuracy":      round(model.cv_accuracy    or 0, 4),
            "errors_skipped":   errors,
            "elapsed_seconds":  elapsed,
            "n_features":       len(model.feature_names_ or []),
        }
    except Exception:
        conn.close()
        raise


# ─── Recalibration ────────────────────────────────────────────────────────────

def recalibrate_from_log() -> dict:
    """
    Fit the feedback calibrator from evaluated prediction_log rows.
    Call this after running POST /api/prediction-log/evaluate.

    The calibrator learns which probability ranges the model is systematically
    wrong about and adjusts future outputs accordingly.

    Returns a summary dict with pre/post accuracy and improvement.
    """
    from ml.feedback_calibrator import FeedbackCalibrator
    cal = FeedbackCalibrator()
    result = cal.fit_from_db()

    if result.get("success"):
        cal.save()
        reset_calibrator()   # force singleton reload next prediction
        _meta["calibrator_samples"]  = cal.n_samples
        _meta["calibrator_accuracy"] = cal.post_accuracy
        import logging
        logging.getLogger(__name__).info(
            "Recalibration complete: %d samples, %.1f%% → %.1f%%",
            cal.n_samples,
            result["pre_accuracy_pct"],
            result["post_accuracy_pct"],
        )

    return result


# ─── Prediction ───────────────────────────────────────────────────────────────

import math

def _poisson_prob(lam: float, k: int) -> float:
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def _derive_markets(h_xg: float, a_xg: float) -> tuple:
    p_h0 = _poisson_prob(h_xg, 0)
    p_a0 = _poisson_prob(a_xg, 0)
    btts_no = p_h0 + p_a0 - (p_h0 * p_a0)
    btts_yes = 1.0 - btts_no
    
    u25 = 0.0
    for i in range(3):
        for j in range(3 - i):
            u25 += _poisson_prob(h_xg, i) * _poisson_prob(a_xg, j)
            
    return round(btts_yes, 4), round(1.0 - u25, 4)


def _compute_venue_xg(attacker_feats: dict, defender_feats: dict, venue: str) -> float:
    if venue == "home":
        attack_rate  = attacker_feats.get("home_gf_pg", 0.0)
        defence_rate = defender_feats.get("away_ga_pg", 0.0)
    else:
        attack_rate  = attacker_feats.get("away_gf_pg", 0.0)
        defence_rate = defender_feats.get("home_ga_pg", 0.0)

    season_gf = attacker_feats.get("goals_for_pg", 0.0)
    form_gf   = attacker_feats.get("form_gf_avg",  0.0)

    if attack_rate < 0.1:
        attack_rate = season_gf if season_gf > 0.1 else (form_gf if form_gf > 0.1 else 1.35)
    if defence_rate < 0.1:
        defence_rate = defender_feats.get("goals_against_pg", 1.35)

    if form_gf > 0.1:
        attack_rate = 0.65 * attack_rate + 0.35 * form_gf

    LEAGUE_AVG_GA = 1.35
    defence_factor = defence_rate / LEAGUE_AVG_GA if defence_rate > 0 else 1.0
    xg = attack_rate * defence_factor
    return round(max(0.5, min(xg, 5.0)), 2)


def _key_factors(home_feats: dict, away_feats: dict,
                 home_name: str, away_name: str) -> list:
    factors = []

    def pct_diff(h, a, label, higher_better=True):
        if h == 0 and a == 0:
            return
        diff = h - a
        if abs(diff) < 0.01:
            return
        better = home_name if (diff > 0) == higher_better else away_name
        worse  = away_name if better == home_name else home_name
        pct    = abs(round(diff / max(abs(a), abs(h), 0.001) * 100, 1))
        factors.append(f"{better} {pct}% better {label} than {worse}")

    pct_diff(home_feats.get("goals_per90",           0), away_feats.get("goals_per90",           0), "goals/90")
    pct_diff(home_feats.get("shots_on_target_per90", 0), away_feats.get("shots_on_target_per90", 0), "shots on target/90")
    pct_diff(home_feats.get("goals_per_shot",        0), away_feats.get("goals_per_shot",        0), "shot conversion rate")
    pct_diff(home_feats.get("gk_save_pct",           0), away_feats.get("gk_save_pct",           0), "GK save%")
    pct_diff(home_feats.get("gk_clean_sheets_pct",   0), away_feats.get("gk_clean_sheets_pct",   0), "clean sheet rate")

    hf = home_feats.get("form_score", 0.5)
    af = away_feats.get("form_score", 0.5)
    if abs(hf - af) > 0.1:
        better = home_name if hf > af else away_name
        hpts = round(hf * 15); apts = round(af * 15)
        factors.append(f"{better} better recent form: {hpts}/15 vs {apts}/15 pts (last 5)")

    hp = home_feats.get("prev_rank_norm", 0.5)
    ap = away_feats.get("prev_rank_norm", 0.5)
    if abs(hp - ap) > 0.15:
        better = home_name if hp > ap else away_name
        factors.append(f"{better} finished significantly higher last season")

    hga = home_feats.get("goals_against_pg", 0)
    aga = away_feats.get("goals_against_pg", 0)
    if aga > hga + 0.3:
        factors.append(f"{away_name} concede {round(aga, 1)} goals/game — vulnerable defence")
    elif hga > aga + 0.3:
        factors.append(f"{home_name} concede {round(hga, 1)} goals/game — defensive concern")

    if home_feats.get("attack_dependency") == 1.0:
        factors.append(f"{home_name} attack heavily reliant on one player (>40% of goals)")
    if away_feats.get("attack_dependency") == 1.0:
        factors.append(f"{away_name} attack heavily reliant on one player (>40% of goals)")

    h_blank = home_feats.get("blank_rate", 0)
    a_blank = away_feats.get("blank_rate", 0)
    if h_blank > 0.35: factors.append(f"{home_name} struggles to score (blanked in {int(h_blank*100)}% of games)")
    if a_blank > 0.35: factors.append(f"{away_name} struggles to score (blanked in {int(a_blank*100)}% of games)")

    h_blowout = home_feats.get("blowout_rate", 0)
    a_blowout = away_feats.get("blowout_rate", 0)
    if h_blowout > 0.25: factors.append(f"{home_name} high explosive potential (3+ goals in {int(h_blowout*100)}% of games)")
    if a_blowout > 0.25: factors.append(f"{away_name} high explosive potential (3+ goals in {int(a_blowout*100)}% of games)")

    h_collapse = home_feats.get("defensive_collapse_rate", 0)
    a_collapse = away_feats.get("defensive_collapse_rate", 0)
    if h_collapse > 0.25: factors.append(f"{home_name} prone to defensive collapse (conceded 3+ in {int(h_collapse*100)}% of games)")
    if a_collapse > 0.25: factors.append(f"{away_name} prone to defensive collapse (conceded 3+ in {int(a_collapse*100)}% of games)")

    pct_diff(home_feats.get("squad_depth_scorers", 0), away_feats.get("squad_depth_scorers", 0), "squad scoring depth")

    hp_pos = home_feats.get("possession", 50)
    ap_pos = away_feats.get("possession", 50)
    if abs(hp_pos - ap_pos) > 5:
        dom = home_name if hp_pos > ap_pos else away_name
        factors.append(f"{dom} dominates possession ({round(max(hp_pos,ap_pos),1)}% avg)")

    return factors[:6]


def _outcome_from_probs(probs: dict) -> tuple:
    """Return (predicted_outcome_str, confidence_label, confidence_score)."""
    hw = probs["home_win"]
    d  = probs["draw"]
    aw = probs["away_win"]
    vals = [hw, d, aw]
    labels = ["Home Win", "Draw", "Away Win"]
    idx = int(max(range(3), key=lambda i: vals[i]))
    predicted = labels[idx]

    score = max(vals)
    label = "High" if score >= 0.55 else ("Medium" if score >= 0.42 else "Low")
    return predicted, label, score


def predict_match(home_team_id: int, away_team_id: int,
                  league_id: int, season_id: int) -> dict:
    """
    Full rich prediction for one match with feedback calibration applied.
    Uses the same batch_features path as training to guarantee feature-count consistency.
    """
    engine = _get_engine()
    if engine is None or not engine.is_trained:
        return {"error": "Model not trained. POST /api/predictions/train first."}

    conn = get_connection()
    cur  = conn.cursor()
    try:
        # ── 1. Resolve names (always from DB, never fall back to "Team N") ──────
        cur.execute("SELECT id, name FROM teams WHERE id IN (%s, %s)",
                    (home_team_id, away_team_id))
        name_map = {r["id"]: r["name"] for r in cur.fetchall()}
        home_name = name_map.get(home_team_id, f"Team {home_team_id}")
        away_name = name_map.get(away_team_id, f"Team {away_team_id}")

        cur.execute("SELECT name FROM leagues WHERE id = %s", (league_id,))
        lg_row = cur.fetchone()
        league_name = lg_row["name"] if lg_row else ""

        cur.execute("SELECT name FROM seasons WHERE id = %s", (season_id,))
        ss_row = cur.fetchone()
        season_name = ss_row["name"] if ss_row else ""

        cur.execute(
            "SELECT id, match_date FROM matches WHERE home_team_id = %s AND away_team_id = %s"
            " AND season_id = %s LIMIT 1",
            (home_team_id, away_team_id, season_id),
        )
        match_row = cur.fetchone()
        match_id_or_none = match_row["id"] if match_row else None
        match_date_val   = match_row["match_date"] if match_row else None

        # ── 2. Build features via BATCH path (identical to training) ────────────
        # This guarantees the feature vector has the exact same shape / ordering
        # as the vectors used when the model's StandardScaler was fitted.
        from ml.batch_features import DataCache, _build_match_features as _batch_build
        cache = DataCache(cur)          # bulk-loads all tables (no more DB calls needed)
        conn.close()                    # cache holds everything in memory
        conn = None

        fv, feat_names, home_feats, away_feats, h2h = _batch_build(
            cache, home_team_id, away_team_id, league_id, season_id,
            match_id=match_id_or_none, match_date=match_date_val
        )

        raw_proba = engine.predict_proba(fv)
        top_feats = engine.get_top_features(fv, n=6)

        # Apply feedback calibration
        proba_dict = {
            "home_win": raw_proba["home_win"],
            "draw":     raw_proba["draw"],
            "away_win": raw_proba["away_win"],
        }
        calibrated = _apply_calibration(proba_dict)

        # Re-derive outcome + confidence from calibrated probs
        predicted_outcome, confidence, confidence_score = _outcome_from_probs(calibrated)

        home_xg = _compute_venue_xg(home_feats, away_feats, "home")
        away_xg = _compute_venue_xg(away_feats, home_feats, "away")
        factors = _key_factors(home_feats, away_feats, home_name, away_name)

        def cmp(feats, venue_key):
            return {
                "attack":    round(feats.get("attack_strength",   1.0), 3),
                "defence":   round(feats.get("defence_strength",  1.0), 3),
                "form":      round(feats.get(f"{venue_key}_form_score", feats.get("form_score", 0.5)), 3),
                "possession":round(feats.get("possession", 50.0), 1),
                "goals_pg":  round(feats.get("goals_for_pg", 0.0), 2),
                "concede_pg":round(feats.get("goals_against_pg", 0.0), 2),
                "gk_save_pct":round(feats.get("gk_save_pct", 0.0), 1),
                "shots_ot_pg":round(feats.get("shots_on_target_per90", 0.0), 2),
                "top_scorer_goals": int(feats.get("top_scorer_goals", 0)),
                "prev_rank_norm":   round(feats.get("prev_rank_norm", 0.5), 3),
                "prev_form":        round(feats.get("prev_form_score", 0.5), 3),
                "blank_rate":       round(feats.get("blank_rate", 0.0), 3),
                "clean_sheet_rate": round(feats.get("clean_sheet_rate", 0.0), 3),
            }

        btts_prob, o25_prob = _derive_markets(home_xg, away_xg)

        cal = get_calibrator()
        return {
            "match": {
                "match_id":   match_id_or_none,
                "home_team":  home_name,
                "away_team":  away_name,
                "league":     league_name,
                "season":     season_name,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
            },
            "probabilities": calibrated,
            "raw_probabilities": {
                "home_win": raw_proba["home_win"],
                "draw":     raw_proba["draw"],
                "away_win": raw_proba["away_win"],
            },
            "predicted_outcome":  predicted_outcome,
            "confidence":         confidence,
            "confidence_score":   confidence_score,
            "expected_goals": {
                "home_xg": home_xg,
                "away_xg": away_xg,
                "predicted_score": f"{round(home_xg)}-{round(away_xg)}",
            },
            "key_factors":    factors,
            "team_comparison": {
                "home": cmp(home_feats, "home"),
                "away": cmp(away_feats, "away"),
            },
            "markets": {
                "btts_yes": btts_prob,
                "over_2_5": o25_prob,
            },
            "h2h": {
                "home_wins":  h2h["h2h_home_wins"],
                "draws":      h2h["h2h_draws"],
                "away_wins":  h2h["h2h_away_wins"],
                "home_win_pct": round(h2h["h2h_home_win_pct"], 3),
                "away_win_pct": round(h2h["h2h_away_win_pct"], 3),
                "last_5":     h2h["h2h_last_5"],
            },
            "top_features": top_feats,
            "model_info": {
                "n_trained_on":          _meta.get("n_samples", 0),
                "cv_accuracy":           _meta.get("cv_accuracy"),
                "trained_at":            _meta.get("trained_at"),
                "calibrated":            cal.is_fitted,
                "calibrator_samples":    _meta.get("calibrator_samples", 0),
                "calibrator_accuracy":   _meta.get("calibrator_accuracy"),
            },
        }
    finally:
        if conn:
            try: conn.close()
            except: pass



# ─── Upcoming fixtures ────────────────────────────────────────────────────────

def predict_upcoming(league_id: int = None, limit: int = 50) -> list:
    engine = _get_engine()
    if engine is None or not engine.is_trained:
        return []

    conn = get_connection()
    cur  = conn.cursor()
    try:
        query = """
            SELECT m.id, m.home_team_id, m.away_team_id,
                   m.league_id, m.season_id, m.match_date, m.gameweek
            FROM matches m
            WHERE m.home_score IS NULL AND m.match_date >= CURRENT_DATE
        """
        params = []
        if league_id:
            query += " AND m.league_id = %s"; params.append(league_id)
        query += " ORDER BY m.match_date ASC LIMIT %s"; params.append(limit)
        cur.execute(query, params)
        fixtures = cur.fetchall()
        conn.close()

        results = []
        for fx in fixtures:
            try:
                pred = predict_match(fx["home_team_id"], fx["away_team_id"],
                                     fx["league_id"], fx["season_id"])
                pred["fixture_id"] = fx["id"]
                pred["match_date"] = str(fx["match_date"]) if fx["match_date"] else None
                pred["gameweek"]   = fx["gameweek"]
                results.append(pred)
            except Exception:
                continue
        return results
    except Exception:
        conn.close()
        return []


def predict_upcoming_fast(league_id: int = None, limit: int = 50) -> list:
    from ml.feedback_calibrator import recalibrate_with_feedback
    try:
        recalibrate_with_feedback()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Pre-prediction recalibration failed: %s", exc)

    eng = _get_engine()
    if eng is None or not eng.is_trained:
        return []

    conn = get_connection()
    cur  = conn.cursor()
    try:
        query = """
            SELECT m.id, m.home_team_id, m.away_team_id,
                   m.league_id, m.season_id, m.match_date, m.gameweek,
                   ht.name AS home_name, at.name AS away_name,
                   ht.logo_url AS home_logo, at.logo_url AS away_logo,
                   l.name AS league_name, s.name AS season_name
            FROM matches m
            JOIN teams   ht ON ht.id = m.home_team_id
            JOIN teams   at ON at.id = m.away_team_id
            JOIN leagues l  ON l.id  = m.league_id
            JOIN seasons s  ON s.id  = m.season_id
            WHERE m.home_score IS NULL AND m.match_date >= CURRENT_DATE
        """
        params = []
        if league_id:
            query += " AND m.league_id = %s"; params.append(league_id)
        query += " ORDER BY m.match_date ASC LIMIT %s"; params.append(limit)
        cur.execute(query, params)
        fixtures = [dict(r) for r in cur.fetchall()]

        if not fixtures:
            conn.close()
            return []

        cache = DataCache(cur)
        conn.close()

        results = []
        for fx in fixtures:
            try:
                htid = fx["home_team_id"]
                atid = fx["away_team_id"]
                lid  = fx["league_id"]
                sid  = fx["season_id"]

                fv, feat_names, home_feats, away_feats, h2h = _build_match_features(
                    cache, htid, atid, lid, sid,
                    match_id=fx["id"], match_date=fx["match_date"]
                )
                raw_proba = eng.predict_proba(fv)

                # Apply feedback calibration
                proba_dict = {
                    "home_win": raw_proba["home_win"],
                    "draw":     raw_proba["draw"],
                    "away_win": raw_proba["away_win"],
                }
                calibrated = _apply_calibration(proba_dict)
                predicted_outcome, confidence, confidence_score = _outcome_from_probs(calibrated)

                home_xg = _compute_venue_xg(home_feats, away_feats, "home")
                away_xg = _compute_venue_xg(away_feats, home_feats, "away")
                btts_prob, o25_prob = _derive_markets(home_xg, away_xg)
                factors = _key_factors(home_feats, away_feats,
                                       fx["home_name"], fx["away_name"])

                results.append({
                    "fixture_id":   fx["id"],
                    "match_date":   str(fx["match_date"]) if fx["match_date"] else None,
                    "gameweek":     fx["gameweek"],
                    "match": {
                        "match_id":     fx["id"],
                        "home_team":    fx["home_name"],
                        "away_team":    fx["away_name"],
                        "home_logo":    fx["home_logo"],
                        "away_logo":    fx["away_logo"],
                        "league":       fx["league_name"],
                        "season":       fx["season_name"],
                        "home_team_id": htid,
                        "away_team_id": atid,
                    },
                    "probabilities":     calibrated,
                    "raw_probabilities": {
                        "home_win": raw_proba["home_win"],
                        "draw":     raw_proba["draw"],
                        "away_win": raw_proba["away_win"],
                    },
                    "predicted_outcome":  predicted_outcome,
                    "confidence":         confidence,
                    "confidence_score":   confidence_score,
                    "expected_goals": {
                        "home_xg":         home_xg,
                        "away_xg":         away_xg,
                        "predicted_score": f"{round(home_xg)}-{round(away_xg)}",
                    },
                    "markets": {
                        "btts_yes": btts_prob,
                        "over_2_5": o25_prob,
                    },
                    "key_factors": factors,
                    "model_info": {
                        "n_trained_on":        _meta.get("n_samples", 0),
                        "cv_accuracy":         _meta.get("cv_accuracy"),
                        "trained_at":          _meta.get("trained_at"),
                        "calibrated":          get_calibrator().is_fitted,
                        "calibrator_samples":  _meta.get("calibrator_samples", 0),
                    },
                })
            except Exception:
                continue

        return results

    except Exception:
        try: conn.close()
        except Exception: pass
        return predict_upcoming(league_id=league_id, limit=limit)


# ─── Status ───────────────────────────────────────────────────────────────────

def get_status() -> dict:
    engine = _get_engine()
    cal    = get_calibrator()
    return {
        "model_trained":         engine is not None and engine.is_trained,
        "trained_at":            _meta.get("trained_at"),
        "n_samples":             _meta.get("n_samples", 0),
        "train_accuracy":        _meta.get("train_accuracy"),
        "cv_accuracy":           _meta.get("cv_accuracy"),
        "n_features":            len(_meta.get("feature_names", [])),
        "feature_names":         _meta.get("feature_names", [])[:20],
        "calibrated":            cal.is_fitted,
        "calibrator_samples":    cal.n_samples,
        "calibrator_accuracy":   cal.post_accuracy,
    }


# Auto-load on import
_get_engine()
