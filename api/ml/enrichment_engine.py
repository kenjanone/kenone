"""
Enrichment Machine Learning Engine
===================================
A standalone predictive engine strictly trained on Enrichment Data:
- ClubElo Goal Expectancy & Ratings
- Transfermarkt Injury Squad Decimation (Market Values)
- Historical Betting Odds Movement

This isolated engine guarantees that if a scraper drops missing data, the core DC/ML engines survive untouched, allowing Consensus to dynamically discard this engine's output.
"""

import os
import time
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from database import get_connection

from ml.enrichment_features import build_enrichment_features

log = logging.getLogger(__name__)

# ─── Singleton state ──────────────────────────────────────────────────────────

_engine = None

class EnrichmentPredictor:
    def __init__(self):
        # We use a blend of RF (low variance) and XGB (high accuracy)
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, 
                                 eval_metric='mlogloss', random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names_ = []
        self.n_samples = 0
        
    def train(self, X_df: pd.DataFrame, y_series: pd.Series):
        self.feature_names_ = list(X_df.columns)
        self.n_samples = len(X_df)
        
        X_scaled = self.scaler.fit_transform(X_df)
        self.rf.fit(X_scaled, y_series)
        self.xgb.fit(X_scaled, y_series)
        
        self.is_trained = True
        
    def predict_proba(self, feature_dict: dict) -> dict:
        if not self.is_trained:
            return {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            
        row = []
        for col in self.feature_names_:
            row.append(feature_dict.get(col, 0.0))
            
        X_arr = np.array([row])
        X_scaled = self.scaler.transform(X_arr)
        
        rf_p = self.rf.predict_proba(X_scaled)[0]
        xgb_p = self.xgb.predict_proba(X_scaled)[0]
        
        # Blend 50/50
        hw = (rf_p[0] + xgb_p[0]) / 2.0
        dr = (rf_p[1] + xgb_p[1]) / 2.0
        aw = (rf_p[2] + xgb_p[2]) / 2.0
        
        return {"home_win": round(hw, 4), "draw": round(dr, 4), "away_win": round(aw, 4)}

    def save(self, filepath="enrichment_model.pkl"):
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            log.error(f"Failed to save enrichment model: {e}")

    @staticmethod
    def load(filepath="enrichment_model.pkl"):
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                log.error(f"Failed to load enrichment model: {e}")
        return None

def _get_enrichment_engine() -> EnrichmentPredictor:
    global _engine
    if _engine is None:
        loaded = EnrichmentPredictor.load()
        if loaded and loaded.is_trained:
            _engine = loaded
    return _engine

# ─── Training Orchestrator ───────────────────────────────────────────────────

def train_enrichment_model() -> dict:
    """
    Extracts all historical matches, pulls their enrichment states at that point in time,
    trains the models, and persists the singleton.
    """
    global _engine
    conn = get_connection()
    cur = conn.cursor()
    try:
        t0 = time.time()
        
        # Target only matches with valid scores
        cur.execute("""
            SELECT id, home_team_id, away_team_id, match_date, home_score, away_score
            FROM matches
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        """)
        matches = cur.fetchall()
        
        # --- BATCH PRE-FETCH ENRICHMENT DATA TO AVOID 50K+ QUERIES N+1 TIMEOUT ---
        log.info(f"Batch fetching enrichment data for {len(matches)} matches...")
        
        # 1. Preload latest Odds per match
        cur.execute("SELECT match_id, b365_home_win, b365_draw, b365_away_win, raw_data FROM match_odds ORDER BY scraped_at ASC")
        odds_dict = {}
        for row in cur.fetchall():
            odds_dict[row["match_id"]] = row
            
        # 2. Preload ClubElo
        cur.execute("SELECT team_id, elo_date, elo, raw_data FROM team_clubelo ORDER BY team_id, elo_date ASC")
        from collections import defaultdict
        import bisect
        elo_dict = defaultdict(list)
        for row in cur.fetchall():
            # store tuple (date_string, row)
            d_str = str(row["elo_date"])
            elo_dict[row["team_id"]].append((d_str, row))
            
        # Helper to get Elo at a point in time
        def get_elo(t_id, m_date):
            arr = elo_dict[t_id]
            if not arr: return None
            if not m_date: return arr[-1][1]
            idx = bisect.bisect_right([x[0] for x in arr], m_date)
            if idx == 0: return None
            return arr[idx-1][1]

        # 3. Preload player injuries (simplified grouping by team and scrape batch)
        # We group raw_data by team_id and scrape date (YYYY-MM-DD or ignoring time). We just use a flat list and bisect.
        cur.execute("SELECT team_id, scraped_at, raw_data FROM player_injuries ORDER BY team_id, scraped_at ASC")
        inj_dict = defaultdict(list)
        for row in cur.fetchall():
            inj_dict[row["team_id"]].append((str(row["scraped_at"]), row))
            
        def get_injuries(t_id, m_date):
            from ml.enrichment_features import _parse_market_value
            arr = inj_dict[t_id]
            if not arr: return {"count": 0, "total_value_m": 0.0}
            if not m_date: 
                latest = arr[-1][0]
            else:
                idx = bisect.bisect_right([x[0] for x in arr], m_date)
                if idx == 0: return {"count": 0, "total_value_m": 0.0}
                latest = arr[idx-1][0]
                
            # Grab all injuries happening on that exact same day string (ignoring hours if we matched the day prefix)
            day_prefix = latest[:10]
            val_sum = 0.0
            count = 0
            # Look backwards from the matched index for ALL players injured on that same date
            start_i = len(arr)-1 if not m_date else idx-1
            for i in range(start_i, -1, -1):
                if arr[i][0][:10] != day_prefix:
                    break
                raw = arr[i][1].get("raw_data", {})
                val_sum += _parse_market_value(raw.get("Market_Value", ""))
                count += 1
            return {"count": count, "total_value_m": val_sum}

        # --- PROCESS MATCHES IN MEMORY ---
        X_list = []
        y_list = []
        skipped = 0
        from ml.enrichment_features import _f
        
        for m in matches:
            if m["home_score"] > m["away_score"]: outcome = 0
            elif m["home_score"] == m["away_score"]: outcome = 1
            else: outcome = 2
                
            m_date = str(m["match_date"]) if m["match_date"] else None
            m_id = m["id"]
            h_id = m["home_team_id"]
            a_id = m["away_team_id"]
            
            feats = {}
            # 1. Odds
            o_row = odds_dict.get(m_id)
            if o_row and o_row["b365_home_win"]:
                hw, dw, aw = _f(o_row["b365_home_win"]), _f(o_row["b365_draw"]), _f(o_row["b365_away_win"])
                marg = (1.0/hw + 1.0/dw + 1.0/aw) if (hw and dw and aw) else 1.0
                feats["odds_home_prob"] = (1.0/hw)/marg if hw else 0.35
                feats["odds_draw_prob"] = (1.0/dw)/marg if dw else 0.30
                feats["odds_away_prob"] = (1.0/aw)/marg if aw else 0.35
                raw_o = o_row.get("raw_data") or {}
                o25, u25 = _f(raw_o.get("B365>2.5", 0)), _f(raw_o.get("B365<2.5", 0))
                m_ou = (1.0/o25 + 1.0/u25) if (o25 > 0 and u25 > 0) else 1.0
                feats["odds_over_25_prob"]  = (1.0/o25)/m_ou if o25 > 0 else 0.50
                feats["odds_under_25_prob"] = (1.0/u25)/m_ou if u25 > 0 else 0.50
                ahs, ahh, aha = _f(raw_o.get("AHh", 0)), _f(raw_o.get("B365AHH", 0)), _f(raw_o.get("B365AHA", 0))
                m_ah = (1.0/ahh + 1.0/aha) if (ahh > 0 and aha > 0) else 1.0
                feats["odds_asian_hdcp_size"] = ahs
                feats["odds_ah_home_prob"] = (1.0/ahh)/m_ah if ahh > 0 else 0.50
                feats["odds_ah_away_prob"] = (1.0/aha)/m_ah if aha > 0 else 0.50
            else:
                for k, v in [("odds_home_prob",0.35),("odds_draw_prob",0.3),("odds_away_prob",0.35),("odds_over_25_prob",0.5),("odds_under_25_prob",0.5),("odds_asian_hdcp_size",0.0),("odds_ah_home_prob",0.5),("odds_ah_away_prob",0.5)]:
                    feats[k] = v

            # 2. ClubElo
            h_elo_r, a_elo_r = get_elo(h_id, m_date), get_elo(a_id, m_date)
            h_elo = _f(h_elo_r["elo"]) if h_elo_r and h_elo_r["elo"] else 1500.0
            a_elo = _f(a_elo_r["elo"]) if a_elo_r and a_elo_r["elo"] else 1500.0
            feats["clubelo_home_rating"] = h_elo
            feats["clubelo_away_rating"] = a_elo
            feats["clubelo_gap"] = h_elo - a_elo
            
            hp, dp, ap = 0.35, 0.30, 0.35
            rj = (h_elo_r or {}).get("raw_data") or (a_elo_r or {}).get("raw_data") or {}
            ri = rj.get("raw", {})
            if ri and "GD=0" in ri:
                dp = _f(ri.get("GD=0", 0.3))
                hw_s = sum(_f(ri.get(f"GD={i}", 0)) for i in range(1, 10))
                aw_s = sum(_f(ri.get(f"GD={i}", 0)) for i in range(-9, 0))
                hp = hw_s if hw_s > 0 else 0.35
                ap = aw_s if aw_s > 0 else 0.35
                t = hp + dp + ap
                if t > 0: hp, dp, ap = hp/t, dp/t, ap/t
            feats["clubelo_home_prob"], feats["clubelo_draw_prob"], feats["clubelo_away_prob"] = hp, dp, ap

            # 3. Injuries
            h_inj, a_inj = get_injuries(h_id, m_date), get_injuries(a_id, m_date)
            feats["home_injured_players"] = float(h_inj["count"])
            feats["home_injured_value_m"] = h_inj["total_value_m"]
            feats["away_injured_players"] = float(a_inj["count"])
            feats["away_injured_value_m"] = a_inj["total_value_m"]
            feats["injury_value_gap_m"] = h_inj["total_value_m"] - a_inj["total_value_m"]
            
            # Skip unpopulated generic rows
            if feats["clubelo_home_rating"] == 1500.0 and feats["odds_home_prob"] == 0.35:
                skipped += 1
                continue
                
            X_list.append(feats)
            y_list.append(outcome)
                
        if len(X_list) < 20:
            return {"success": False, "error": f"Insufficient enriched training data ({len(X_list)} rows).", "trained": 0}
            
        X_df = pd.DataFrame(X_list)
        y_series = pd.Series(y_list)
        
        # Replace completely missing NaNs with 0
        X_df = X_df.fillna(0.0)
        
        model = EnrichmentPredictor()
        model.train(X_df, y_series)
        model.save()
        _engine = model
        
        elapsed = round(time.time() - t0, 1)
        
        return {
            "success": True,
            "trained_rows": len(X_list),
            "skipped_rows": skipped,
            "elapsed_seconds": elapsed,
            "n_features": len(model.feature_names_)
        }
    finally:
        conn.close()

# ─── Inference API ────────────────────────────────────────────────────────────

def predict_enrichment(home_team_id: int, away_team_id: int, match_date: str = None) -> dict:
    """
    Returns the enrichment model probabilities for a single upcoming match.
    """
    eng = _get_enrichment_engine()
    if eng is None or not eng.is_trained:
        return {"error": "Enrichment model not trained.", "home_win": 0.35, "draw": 0.30, "away_win": 0.35}
        
    conn = get_connection()
    cur = conn.cursor()
    try:
        feats = build_enrichment_features(cur, home_team_id, away_team_id, match_date)
        
        # Default detection: If this match has literally zero enrichment data 
        # (no odds, no clubelo), the model's output would be arbitrary garbage.
        # We explicitly abstain so the Consensus Engine zero-weights this engine.
        if feats.get("clubelo_home_rating") == 1500.0 and feats.get("odds_home_prob") == 0.35:
            
            return {
                "error": "No enrichment data available (all defaults).",
                "home_win": 0.35, "draw": 0.30, "away_win": 0.35,
                "predicted_outcome": "—",
                "_features": feats
            }

        probs = eng.predict_proba(feats)
        
        idx = np.argmax([probs["home_win"], probs["draw"], probs["away_win"]])
        outcome_str = {0: "Home Win", 1: "Draw", 2: "Away Win"}[int(idx)]
        
        probs["predicted_outcome"] = outcome_str
        # Attach the raw features so we can display them in the console if needed
        probs["_features"] = feats
        
        return probs
    finally:
        conn.close()
