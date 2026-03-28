"""
Dixon-Coles Professional Prediction Engine
==========================================
Adapts the Professional Football Prediction Engine v2.0 to use real
database data from the PlusOne backend instead of synthetic fixtures.

Components:
  1. Dixon-Coles Poisson Model  (MLE + time decay)
  2. Dynamic Elo Rating System  (margin-of-victory weighted)
  3. xG Proxy Model             (rolling xG averages per team)
  4. Monte Carlo Simulation     (50,000 match simulations)
  5. Ensemble Blending          (DC 45% + Elo 30% + xG 25%)
  6. Probability Calibration    (Isotonic regression)

Usage:
  from ml.dc_engine import get_dc_predictor, train_dc_model, predict_dc_match
  train_dc_model(cur)                                     # fit on DB data
  result = predict_dc_match(cur, home_team_id, away_team_id)
"""

import logging
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression

from database import get_connection
from ml.model_store import save_to_db, load_from_db

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

CFG = {
    "dc_time_decay_xi":     0.0018,
    "dc_max_goals":         10,
    "elo_k_base":           40,
    "elo_home_advantage":   60,
    "elo_start":            1500,
    "elo_mov_factor":       0.25,
    "mc_simulations":       50_000,
    "ensemble_weights":     [0.45, 0.30, 0.25],
    "min_kelly_edge":       0.03,
    # How many months of match history to use for DC training.
    # 9 months ≈ 1 full football season.  Override via app_settings key
    # 'dc_lookback_months' (integer, 1–120).  Increasing this gives more
    # data but may dilute the time-decay weighting.
    "dc_lookback_months":   9,
}

HOME_ADVANTAGE = 1.22
LEAGUE_AVG_GOALS = 1.35

# ─── Singleton state ──────────────────────────────────────────────────────────

_dc_predictor = None
_dc_meta: dict = {}


def _get_lookback_months() -> int:
    """
    Read dc_lookback_months from app_settings (allows live changes without
    restarting the server). Falls back to CFG default (9 months).
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(
            "SELECT value FROM app_settings WHERE key = 'dc_lookback_months' LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        if row and row["value"]:
            val = int(row["value"])
            return max(1, min(val, 120))   # clamp to 1–120 months
    except Exception:
        pass
    return CFG["dc_lookback_months"]


# ══════════════════════════════════════════════════════════════════════════════
# DIXON-COLES MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DixonColesModel:
    """Dixon & Coles (1997) bivariate Poisson model with time decay."""

    def __init__(self, xi: float = CFG["dc_time_decay_xi"]):
        self.xi = xi
        self.params: dict = {}
        self.teams: list = []
        self.fitted: bool = False

    def _time_weight(self, dates: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
        days_ago = (ref_date - dates).dt.days.values.astype(float)
        return np.exp(-self.xi * days_ago)

    @staticmethod
    def _tau(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
        if   hg == 0 and ag == 0: return 1 - lam_h * lam_a * rho
        elif hg == 0 and ag == 1: return 1 + lam_h * rho
        elif hg == 1 and ag == 0: return 1 + lam_a * rho
        elif hg == 1 and ag == 1: return 1 - rho
        return 1.0

    def _neg_log_likelihood(self, params_vec: np.ndarray,
                             fixtures: pd.DataFrame,
                             weights: np.ndarray) -> float:
        n = len(self.teams)
        alpha = params_vec[:n]
        beta  = params_vec[n:2*n]
        gamma = params_vec[2*n]
        rho   = params_vec[2*n + 1]
        team_idx = {t: i for i, t in enumerate(self.teams)}
        ll = 0.0
        for i, row in fixtures.iterrows():
            hi = team_idx.get(row["home_team"], -1)
            ai = team_idx.get(row["away_team"], -1)
            if hi < 0 or ai < 0:
                continue
            lam_h = np.exp(alpha[hi] - beta[ai] + gamma)
            lam_a = np.exp(alpha[ai] - beta[hi])
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            tau = self._tau(hg, ag, lam_h, lam_a, rho)
            if tau <= 0:
                continue
            ll_m = (poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a) +
                    math.log(max(tau, 1e-10)))
            ll += weights[i] * ll_m
        return -ll

    def fit(self, fixtures: pd.DataFrame):
        if len(fixtures) < 20:
            log.warning("DC: fewer than 20 fixtures — skipping fit.")
            return self
        ref_date = fixtures["date"].max()
        self.teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
        n = len(self.teams)
        weights = self._time_weight(fixtures["date"], ref_date)
        x0 = np.zeros(2 * n + 2)
        x0[2*n]     =  0.30
        x0[2*n + 1] = -0.10
        bounds = [(-3, 3)] * (2 * n) + [(0, 1)] + [(-0.5, 0.5)]
        try:
            result = minimize(
                self._neg_log_likelihood, x0,
                args=(fixtures, weights),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-7},
            )
            alpha = result.x[:n]
            beta  = result.x[n:2*n]
            gamma = result.x[2*n]
            rho   = result.x[2*n + 1]
        except Exception as e:
            log.error("DC fit failed: %s", e)
            return self
        self.params = {t: {"attack": alpha[i], "defence": beta[i]}
                       for i, t in enumerate(self.teams)}
        self.params["_home_adv"] = gamma
        self.params["_rho"]      = rho
        self.fitted = True
        return self

    def score_matrix(self, home: str, away: str,
                     max_goals: int = CFG["dc_max_goals"]):
        if not self.fitted or home not in self.params or away not in self.params:
            return None, LEAGUE_AVG_GOALS * HOME_ADVANTAGE, LEAGUE_AVG_GOALS
        hp = self.params[home]
        ap = self.params[away]
        gamma = self.params["_home_adv"]
        rho   = self.params["_rho"]
        lam_h = np.exp(hp["attack"] - ap["defence"] + gamma)
        lam_a = np.exp(ap["attack"] - hp["defence"])
        mat = np.zeros((max_goals + 1, max_goals + 1))
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                tau = self._tau(hg, ag, lam_h, lam_a, rho)
                mat[hg, ag] = poisson.pmf(hg, lam_h) * poisson.pmf(ag, lam_a) * max(tau, 0)
        s = mat.sum()
        if s > 0:
            mat /= s
        return mat, lam_h, lam_a

    def predict(self, home: str, away: str) -> dict:
        mat, lam_h, lam_a = self.score_matrix(home, away)
        if mat is None:
            lam_h = LEAGUE_AVG_GOALS * HOME_ADVANTAGE
            lam_a = LEAGUE_AVG_GOALS
            p_hw = p_d = p_aw = 1/3
        else:
            p_hw = float(np.sum(np.tril(mat, -1)))
            p_d  = float(np.sum(np.diag(mat)))
            p_aw = float(np.sum(np.triu(mat, 1)))
        return {"model": "dixon_coles",
                "home_win": p_hw, "draw": p_d, "away_win": p_aw,
                "exp_home_goals": lam_h, "exp_away_goals": lam_a}


# ══════════════════════════════════════════════════════════════════════════════
# ELO RATING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class EloSystem:
    def __init__(self):
        self.ratings: dict = {}

    def _expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1 + 10 ** ((rb - ra) / 400))

    def _mov_multiplier(self, goal_diff: int, elo_diff: float) -> float:
        gd = abs(goal_diff)
        if gd == 0:
            return 1.0
        return math.log(gd + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def update(self, home: str, away: str, hg: int, ag: int):
        K  = CFG["elo_k_base"]
        ha = CFG["elo_home_advantage"]
        r_h = self.ratings.get(home, CFG["elo_start"])
        r_a = self.ratings.get(away, CFG["elo_start"])
        e_h = self._expected(r_h + ha, r_a)
        gd = hg - ag
        s_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        mov = self._mov_multiplier(gd, (r_h + ha) - r_a)
        delta = K * mov * (s_h - e_h)
        self.ratings[home] = r_h + delta
        self.ratings[away] = r_a - delta

    def fit(self, fixtures: pd.DataFrame):
        for _, r in fixtures.sort_values("date").iterrows():
            self.update(r.home_team, r.away_team, r.home_goals, r.away_goals)
        return self

    def predict(self, home: str, away: str) -> dict:
        ha  = CFG["elo_home_advantage"]
        r_h = self.ratings.get(home, CFG["elo_start"])
        r_a = self.ratings.get(away, CFG["elo_start"])
        e_h = self._expected(r_h + ha, r_a)
        e_a = self._expected(r_a, r_h + ha)
        raw_diff = abs(e_h - e_a)
        p_draw = max(0.18, 0.32 - 0.55 * raw_diff)
        p_hw   = e_h * (1 - p_draw)
        p_aw   = e_a * (1 - p_draw)
        total  = p_hw + p_draw + p_aw
        return {"model": "elo",
                "home_win": p_hw / total, "draw": p_draw / total, "away_win": p_aw / total,
                "home_elo": round(r_h, 1), "away_elo": round(r_a, 1),
                "elo_diff": round((r_h + ha) - r_a, 1)}


# ══════════════════════════════════════════════════════════════════════════════
# xG MODEL
# ══════════════════════════════════════════════════════════════════════════════

class XGModel:
    """Rolling xG averages per team from DB xg columns (with goal fallback)."""

    def __init__(self, window: int = 10):
        self.window = window
        self.team_xg_att: dict = {}
        self.team_xg_def: dict = {}
        self.league_xg_avg: float = LEAGUE_AVG_GOALS

    def fit(self, fixtures: pd.DataFrame):
        teams = set(fixtures.home_team) | set(fixtures.away_team)
        for team in teams:
            hm = fixtures[fixtures.home_team == team].tail(self.window)
            am = fixtures[fixtures.away_team == team].tail(self.window)
            # Use home_xg/away_xg if available, else fall back to goals
            xg_for = (list(hm.get("home_xg", hm.home_goals)) +
                      list(am.get("away_xg", am.away_goals)))
            xg_ag  = (list(hm.get("away_xg", hm.away_goals)) +
                      list(am.get("home_xg", am.home_goals)))
            self.team_xg_att[team] = np.mean(xg_for) if xg_for else self.league_xg_avg
            self.team_xg_def[team] = np.mean(xg_ag)  if xg_ag  else self.league_xg_avg
        avg_att = np.mean(list(self.team_xg_att.values())) if self.team_xg_att else self.league_xg_avg
        avg_def = np.mean(list(self.team_xg_def.values())) if self.team_xg_def else self.league_xg_avg
        self.league_xg_avg = (avg_att + avg_def) / 2
        return self

    def predict(self, home: str, away: str) -> dict:
        av = self.league_xg_avg or LEAGUE_AVG_GOALS
        h_att = self.team_xg_att.get(home, av)
        h_def = self.team_xg_def.get(home, av)
        a_att = self.team_xg_att.get(away, av)
        a_def = self.team_xg_def.get(away, av)
        exp_h = (h_att / av) * (a_def / av) * av * HOME_ADVANTAGE
        exp_a = (a_att / av) * (h_def / av) * av
        max_g = CFG["dc_max_goals"]
        p_hw = p_d = p_aw = 0.0
        for hg in range(max_g + 1):
            for ag in range(max_g + 1):
                p = poisson.pmf(hg, exp_h) * poisson.pmf(ag, exp_a)
                if   hg > ag: p_hw += p
                elif hg == ag: p_d  += p
                else:          p_aw += p
        total = p_hw + p_d + p_aw or 1
        return {"model": "xg",
                "home_win": p_hw / total, "draw": p_d / total, "away_win": p_aw / total,
                "exp_home_xg": round(exp_h, 3), "exp_away_xg": round(exp_a, 3)}


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class MonteCarloEngine:
    def __init__(self, n: int = CFG["mc_simulations"]):
        self.n = n

    def simulate(self, exp_h: float, exp_a: float) -> dict:
        home_g  = np.random.poisson(exp_h, self.n)
        away_g  = np.random.poisson(exp_a, self.n)
        total_g = home_g + away_g
        outcomes = np.where(home_g > away_g, 2, np.where(home_g == away_g, 1, 0))
        from collections import Counter
        score_counts = Counter(zip(home_g.tolist(), away_g.tolist()))
        top_scores = [
            {"score": f"{hg}-{ag}", "probability": round(cnt / self.n, 4)}
            for (hg, ag), cnt in score_counts.most_common(5)
        ]
        return {
            "home_win": float((outcomes == 2).mean()),
            "draw":     float((outcomes == 1).mean()),
            "away_win": float((outcomes == 0).mean()),
            "over_1_5":  float((total_g > 1).mean()),
            "over_2_5":  float((total_g > 2).mean()),
            "over_3_5":  float((total_g > 3).mean()),
            "under_2_5": float((total_g <= 2).mean()),
            "btts_yes":  float(((home_g > 0) & (away_g > 0)).mean()),
            "btts_no":   float(((home_g == 0) | (away_g == 0)).mean()),
            "home_cs":   float((away_g == 0).mean()),
            "away_cs":   float((home_g == 0).mean()),
            "1x":        float((outcomes >= 1).mean()),
            "x2":        float((outcomes <= 1).mean()),
            "12":        float((outcomes != 1).mean()),
            "exp_home_goals": float(home_g.mean()),
            "exp_away_goals": float(away_g.mean()),
            "top_scorelines": top_scores,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

class ProbabilityCalibrator:
    def __init__(self):
        self.calibrators = {}
        self.fitted = False

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray):
        for cls in range(3):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_probs[:, cls], (labels == cls).astype(float))
            self.calibrators[cls] = iso
        self.fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return raw_probs
        cal = np.zeros_like(raw_probs)
        for cls in range(3):
            cal[:, cls] = self.calibrators[cls].predict(raw_probs[:, cls])
        row_sums = np.where(cal.sum(axis=1, keepdims=True) == 0, 1,
                            cal.sum(axis=1, keepdims=True))
        return cal / row_sums


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE BLENDER
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleBlender:
    def __init__(self, weights=None):
        w = weights or CFG["ensemble_weights"]
        self.weights = np.array(w) / sum(w)

    def blend(self, dc: dict, elo: dict, xg: dict) -> dict:
        p_hw = self.weights[0]*dc["home_win"] + self.weights[1]*elo["home_win"] + self.weights[2]*xg["home_win"]
        p_d  = self.weights[0]*dc["draw"]     + self.weights[1]*elo["draw"]     + self.weights[2]*xg["draw"]
        p_aw = self.weights[0]*dc["away_win"] + self.weights[1]*elo["away_win"] + self.weights[2]*xg["away_win"]
        total = p_hw + p_d + p_aw or 1
        return {"home_win": float(p_hw/total), "draw": float(p_d/total), "away_win": float(p_aw/total)}


# ══════════════════════════════════════════════════════════════════════════════
# FULL PREDICTOR (uses team names internally, mapped from IDs externally)
# ══════════════════════════════════════════════════════════════════════════════

class DCPredictor:
    """
    Complete ensemble predictor. Fitted from DB data.
    All internal operations use team name strings.
    External callers pass team_ids; we resolve via self.team_names dict.
    """

    def __init__(self):
        self.dc          = DixonColesModel()
        self.elo         = EloSystem()
        self.xg          = XGModel()
        self.mc          = MonteCarloEngine()
        self.blender     = EnsembleBlender()
        self.calibrator  = ProbabilityCalibrator()
        self.team_names: dict = {}   # id → name
        self.fitted: bool = False
        self.n_matches: int = 0

    # ── Load fixtures from DB ─────────────────────────────────────────────────
    @staticmethod
    def _load_fixtures(cur, months: int = None) -> pd.DataFrame:
        """Load completed matches within the configured lookback window."""
        if months is None:
            months = _get_lookback_months()
        log.info("DCPredictor: loading fixtures with %d-month lookback", months)
        cur.execute("""
            SELECT
                m.id,
                m.match_date                AS date,
                m.home_score                AS home_goals,
                m.away_score                AS away_goals,
                m.home_team_id,
                m.away_team_id,
                m.season_id,
                m.league_id,
                ht.name                     AS home_team,
                at.name                     AS away_team
            FROM matches m
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams at ON at.id = m.away_team_id
            WHERE m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
              AND m.match_date >= CURRENT_DATE - (%(months)s || ' months')::INTERVAL
            ORDER BY m.match_date DESC
            LIMIT 5000
        """, {"months": months})
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        df["date"]       = pd.to_datetime(df["date"])
        df["home_goals"] = df["home_goals"].astype(float)
        df["away_goals"] = df["away_goals"].astype(float)
        return df

    # ── Fit all models ────────────────────────────────────────────────────────
    def fit(self, cur, months: int = None):
        months   = months or _get_lookback_months()
        log.info("DCPredictor: loading fixtures from DB…")
        fixtures = self._load_fixtures(cur, months=months)
        if fixtures.empty or len(fixtures) < 20:
            log.warning("DCPredictor: not enough fixtures (%d). Skipping.", len(fixtures))
            return self

        # Build team_id → name map
        for _, r in fixtures.iterrows():
            self.team_names[r["home_team_id"]] = r["home_team"]
            self.team_names[r["away_team_id"]] = r["away_team"]

        self.n_matches = len(fixtures)
        log.info("DCPredictor: fitting on %d matches…", self.n_matches)

        self.dc.fit(fixtures)
        self.elo.fit(fixtures)
        self.xg.fit(fixtures)

        # Calibration pass
        raw, labels = [], []
        for _, r in fixtures.iterrows():
            try:
                dc_p  = self.dc.predict(r.home_team, r.away_team)
                elo_p = self.elo.predict(r.home_team, r.away_team)
                xg_p  = self.xg.predict(r.home_team, r.away_team)
                bl    = self.blender.blend(dc_p, elo_p, xg_p)
                raw.append([bl["away_win"], bl["draw"], bl["home_win"]])
                outcome = (2 if r.home_goals > r.away_goals else
                           1 if r.home_goals == r.away_goals else 0)
                labels.append(outcome)
            except Exception:
                pass
        if len(raw) > 30:
            self.calibrator.fit(np.array(raw), np.array(labels))

        self.fitted = True
        log.info("DCPredictor: fitted successfully.")
        return self

    # ── Single match prediction ───────────────────────────────────────────────
    def predict(self, home_team_id: int, away_team_id: int) -> dict:
        home = self.team_names.get(home_team_id)
        away = self.team_names.get(away_team_id)

        # ── Graceful fallback for unknown team IDs ──────────────────────────────
        # Teams that play in multiple leagues (e.g. "Chelsea eng" in UCL) may
        # have a different ID than their domestic entry. We use their name if we
        # can look it up from the DB, otherwise fall back to league-average params.
        missing = []
        if not home: missing.append(("home", home_team_id))
        if not away: missing.append(("away", away_team_id))

        if missing:
            try:
                conn = get_connection()
                cur  = conn.cursor()
                ids = [tid for _, tid in missing]
                placeholders = ", ".join(["%s"] * len(ids))
                cur.execute(f"SELECT id, name FROM teams WHERE id IN ({placeholders})", ids)
                rows = {r["id"]: r["name"] for r in cur.fetchall()}
                conn.close()
                for side, tid in missing:
                    resolved = rows.get(tid)
                    if side == "home":
                        home = resolved
                    else:
                        away = resolved
            except Exception:
                pass

        # If we still don't have names, inject league-average model params
        if not home:
            home = f"__avg_{home_team_id}__"
            if home not in self.dc.params and self.dc.fitted:
                avg_atk = float(np.mean([v["attack"]  for v in self.dc.params.values() if isinstance(v, dict) and "attack" in v] or [0.0]))
                avg_def = float(np.mean([v["defence"] for v in self.dc.params.values() if isinstance(v, dict) and "defence" in v] or [0.0]))
                self.dc.params[home] = {"attack": avg_atk, "defence": avg_def}
            self.elo.ratings.setdefault(home, CFG["elo_start"])
            self.xg.team_xg_att.setdefault(home, self.xg.league_xg_avg)
            self.xg.team_xg_def.setdefault(home, self.xg.league_xg_avg)
        if not away:
            away = f"__avg_{away_team_id}__"
            if away not in self.dc.params and self.dc.fitted:
                avg_atk = float(np.mean([v["attack"]  for v in self.dc.params.values() if isinstance(v, dict) and "attack" in v] or [0.0]))
                avg_def = float(np.mean([v["defence"] for v in self.dc.params.values() if isinstance(v, dict) and "defence" in v] or [0.0]))
                self.dc.params[away] = {"attack": avg_atk, "defence": avg_def}
            self.elo.ratings.setdefault(away, CFG["elo_start"])
            self.xg.team_xg_att.setdefault(away, self.xg.league_xg_avg)
            self.xg.team_xg_def.setdefault(away, self.xg.league_xg_avg)

        # Store resolved names for future calls
        self.team_names[home_team_id] = home
        self.team_names[away_team_id] = away
        # ────────────────────────────────────────────────────────────────────────

        dc_pred  = self.dc.predict(home, away)
        elo_pred = self.elo.predict(home, away)
        xg_pred  = self.xg.predict(home, away)
        blended  = self.blender.blend(dc_pred, elo_pred, xg_pred)

        raw_arr = np.array([[blended["away_win"], blended["draw"], blended["home_win"]]])
        cal_arr = self.calibrator.calibrate(raw_arr)[0]
        calibrated = {"away_win": float(cal_arr[0]), "draw": float(cal_arr[1]), "home_win": float(cal_arr[2])}

        exp_h = (dc_pred["exp_home_goals"] * 0.5 + xg_pred["exp_home_xg"] * 0.5)
        exp_a = (dc_pred["exp_away_goals"] * 0.5 + xg_pred["exp_away_xg"] * 0.5)
        mc_results = self.mc.simulate(exp_h, exp_a)

        probs = np.array([calibrated["home_win"], calibrated["draw"], calibrated["away_win"]])
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = float(-np.log(1 / 3))
        confidence_score = round((1 - entropy / max_entropy) * 100, 1)

        outcome_label = max(calibrated, key=calibrated.get).replace("_", " ").title()

        return {
            "home_team":  home,
            "away_team":  away,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "models": {
                "dixon_coles": {k: round(v, 4) for k, v in dc_pred.items() if isinstance(v, float)},
                "elo":         {k: round(v, 4) if isinstance(v, float) else v for k, v in elo_pred.items()},
                "xg":          {k: round(v, 4) for k, v in xg_pred.items() if isinstance(v, float)},
            },
            "blended":    {k: round(v, 4) for k, v in blended.items()},
            "calibrated": {k: round(v, 4) for k, v in calibrated.items()},
            "markets":    mc_results,
            "prediction": outcome_label,
            "confidence": confidence_score,
            "exp_home_goals": round(exp_h, 2),
            "exp_away_goals": round(exp_a, 2),
            "model_info": {
                "n_trained_on": _dc_meta.get("n_matches", 0),
                "trained_at":   _dc_meta.get("trained_at"),
            }
        }

    # ── Elo leaderboard ───────────────────────────────────────────────────────
    def elo_leaderboard(self) -> list:
        return sorted(
            [{"team": t, "elo": round(r, 1), "delta": round(r - CFG["elo_start"], 1)}
             for t, r in self.elo.ratings.items()],
            key=lambda x: -x["elo"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON API
# ══════════════════════════════════════════════════════════════════════════════

def get_dc_predictor() -> DCPredictor | None:
    """Return the in-memory singleton. None if not yet trained."""
    return _dc_predictor


def train_dc_model() -> dict:
    """
    Train (or retrain) the DC ensemble from DB data.
    After training, automatically saves the model to Supabase so it
    survives Railway/Render redeploys.
    """
    global _dc_predictor, _dc_meta
    import time
    t0 = time.time()
    conn = get_connection()
    cur  = conn.cursor()
    try:
        predictor = DCPredictor()
        predictor.fit(cur)
        if not predictor.fitted:
            return {"trained": False, "error": "Not enough data in DB (need ≥20 completed matches)."}
        elapsed = round(time.time() - t0, 1)
        _dc_predictor = predictor
        _dc_meta = {
            "n_matches":  predictor.n_matches,
            "trained_at": __import__("datetime").datetime.utcnow().isoformat(),
            "elapsed_s":  elapsed,
        }
        log.info("DC model trained: %d matches, %.1fs", predictor.n_matches, elapsed)
        # — Persist to Supabase so the model survives redeploys —
        saved = save_to_db(predictor, name="dc_model")
        if saved:
            log.info("DC model saved to Supabase ml_models table.")
        else:
            log.warning("DC model could not be saved to Supabase (will retrain on next cold start).")
        return {"trained": True, **_dc_meta}
    except Exception as e:
        log.error("DC training error: %s", e)
        return {"trained": False, "error": str(e)}
    finally:
        conn.close()


def predict_dc_match(home_team_id: int, away_team_id: int) -> dict:
    """Run DC prediction for a single match. Returns error dict if untrained."""
    eng = get_dc_predictor()
    if eng is None or not eng.fitted:
        return {"error": "DC model not trained yet. POST /api/dc/train first."}
    return eng.predict(home_team_id, away_team_id)


def dc_status() -> dict:
    eng = get_dc_predictor()
    return {
        "dc_model_trained": eng is not None and eng.fitted,
        "n_matches":        _dc_meta.get("n_matches", 0),
        "trained_at":       _dc_meta.get("trained_at"),
        "n_teams":          len(eng.team_names) if eng else 0,
    }


# Auto-load or auto-train on startup (non-blocking daemon thread)
def _auto_train():
    global _dc_predictor, _dc_meta
    # 1️⃣  Try loading the saved model from Supabase first (fast cold-start)
    try:
        loaded = load_from_db(name="dc_model")
        if loaded is not None and getattr(loaded, "fitted", False):
            _dc_predictor = loaded
            # Reconstruct minimal meta from the loaded model
            _dc_meta = {
                "n_matches":  getattr(loaded, "n_matches", 0),
                "trained_at": "(loaded from Supabase)",
                "elapsed_s":  0,
            }
            log.info("DC model loaded from Supabase (%d teams).", len(loaded.team_names))
            return
    except Exception as e:
        log.warning("DC Supabase load failed, will train from scratch: %s", e)

    # 2️⃣  No saved model found — train from DB data
    try:
        result = train_dc_model()
        if result.get("trained"):
            log.info("DC auto-train complete: %d matches", result.get("n_matches", 0))
        else:
            log.warning("DC auto-train failed: %s", result.get("error"))
    except Exception as e:
        log.warning("DC auto-train exception (will retry on next /train call): %s", e)

import threading
threading.Thread(target=_auto_train, daemon=True).start()
