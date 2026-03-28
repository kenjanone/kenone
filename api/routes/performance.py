"""
Prediction Performance API Routes
===================================
GET /api/performance             — Overall metrics (Brier, RPS, accuracy, ROI)
GET /api/performance/drift       — Rolling 20-match Brier score over time
GET /api/performance/calibration — Calibration bin table
GET /api/performance/per-league  — Per-league accuracy and Brier breakdown
GET /api/performance/confusion   — Confusion matrix
GET /api/performance/markets     — Market accuracy (BTTS, Over 2.5)
"""

import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from database import get_connection
from ml.metrics import MetricsEngine

router = APIRouter()
log    = logging.getLogger(__name__)


def _load_completed(cur, league: str = None) -> pd.DataFrame:
    query = """
        SELECT
            id, home_team, away_team, league, match_date,
            home_win_prob, draw_prob, away_win_prob,
            predicted, actual, correct,
            confidence, confidence_score,
            btts_yes, over_2_5, home_xg, away_xg,
            btts_correct, over_2_5_correct
        FROM prediction_log
        WHERE actual IS NOT NULL
          AND home_win_prob IS NOT NULL
          AND draw_prob     IS NOT NULL
          AND away_win_prob IS NOT NULL
    """
    params = []
    if league:
        query += " AND league = %s"
        params.append(league)
    query += " ORDER BY match_date ASC NULLS LAST, created_at ASC"

    try:
        cur.execute(query, params)
        rows = cur.fetchall()
    except Exception as e:
        log.warning("prediction_log query failed: %s", e)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df = df.rename(columns={
        "home_win_prob": "prob_home_win",
        "draw_prob":     "prob_draw",
        "away_win_prob": "prob_away_win",
        "actual":        "actual_outcome",
        "predicted":     "predicted_outcome",
    })

    outcome_map = {
        "Home Win": 2, "home_win": 2, "2": 2,
        "Draw":     1, "draw":     1, "1": 1,
        "Away Win": 0, "away_win": 0, "0": 0,
    }
    if "actual_outcome" in df.columns:
        df["actual_int"] = df["actual_outcome"].astype(str).map(outcome_map)
    else:
        return pd.DataFrame()

    if "predicted_outcome" in df.columns:
        df["predicted_int"] = df["predicted_outcome"].astype(str).map(outcome_map)

    df = df.dropna(subset=["actual_int", "prob_home_win", "prob_draw", "prob_away_win"])
    return df


def _load_all_predictions(cur) -> pd.DataFrame:
    try:
        cur.execute("""
            SELECT league,
                   COUNT(*) AS total_predictions,
                   SUM(CASE WHEN actual IS NOT NULL THEN 1 ELSE 0 END) AS evaluated
            FROM prediction_log
            WHERE league IS NOT NULL
            GROUP BY league
            ORDER BY total_predictions DESC
        """)
        rows = cur.fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    except Exception as e:
        log.warning("_load_all_predictions failed: %s", e)
        return pd.DataFrame()


def _build_matrices(df: pd.DataFrame):
    p_aw = df["prob_away_win"].astype(float).values
    p_d  = df["prob_draw"].astype(float).values
    p_hw = df["prob_home_win"].astype(float).values
    probs    = np.stack([p_aw, p_d, p_hw], axis=1)
    outcomes = df["actual_int"].astype(int).values
    return probs, outcomes


# ─── GET /api/performance ─────────────────────────────────────────────────────

@router.get("")
def get_overall_performance(league: str = None):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur, league)
    finally:
        conn.close()

    if df.empty:
        return {"message": "No completed predictions in prediction_log yet.", "n": 0}

    probs, outcomes = _build_matrices(df)
    summary = MetricsEngine.full_summary(probs, outcomes)

    roi_records = []
    for _, r in df.iterrows():
        pred = r.get("predicted_int")
        if pd.isna(pred):
            continue
        pred = int(pred)
        odds_cols = {2: "market_home_odds", 1: "market_draw_odds", 0: "market_away_odds"}
        col = odds_cols.get(pred)
        if col and col in r and pd.notna(r[col]):
            roi_records.append({
                "predicted_outcome": pred,
                "actual_outcome":    int(r["actual_int"]),
                "odds_taken":        float(r[col]),
            })

    roi = MetricsEngine.roi(roi_records) if roi_records else {}
    return {**summary, "roi": roi, "league": league or "All Leagues"}


# ─── GET /api/performance/drift ───────────────────────────────────────────────

@router.get("/drift")
def get_rolling_drift(window: int = 20):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur)
    finally:
        conn.close()

    if df.empty or len(df) < window:
        return {"message": f"Need at least {window} completed predictions. Have {len(df)}.", "n": len(df)}

    df = df.reset_index(drop=True)
    rows = []
    for i in range(window, len(df) + 1):
        chunk = df.iloc[i - window: i]
        probs, outcomes = _build_matrices(chunk)
        rows.append({
            "match_number":  i,
            "rolling_brier": round(MetricsEngine.brier_score(probs, outcomes), 4),
            "rolling_rps":   round(MetricsEngine.rps(probs, outcomes), 4),
            "rolling_acc":   round(MetricsEngine.accuracy(probs, outcomes) * 100, 2),
        })

    return {"window": window, "n_predictions": len(df), "latest": rows[-1] if rows else {}, "drift": rows}


# ─── GET /api/performance/calibration ─────────────────────────────────────────

@router.get("/calibration")
def get_calibration():
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur)
    finally:
        conn.close()

    if df.empty:
        return {"message": "No completed predictions yet.", "bins": []}

    probs, outcomes = _build_matrices(df)
    cal_df = MetricsEngine.calibration(probs, outcomes)
    well_calibrated = int((cal_df["well_calibrated"]).sum())
    return {
        "n_predictions":        len(df),
        "n_bins":               len(cal_df),
        "well_calibrated_bins": well_calibrated,
        "bins":                 cal_df.to_dict(orient="records"),
    }


# ─── GET /api/performance/per-league ─────────────────────────────────────────

@router.get("/per-league")
def get_per_league():
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df_completed = _load_completed(cur)
        df_all       = _load_all_predictions(cur)
    finally:
        conn.close()

    if df_all.empty:
        return {"message": "No predictions yet.", "leagues": []}

    evaluated_metrics = {}
    if not df_completed.empty and "league" in df_completed.columns:
        for league, grp in df_completed.groupby("league"):
            if len(grp) < 1:
                continue
            probs, outcomes = _build_matrices(grp)
            evaluated_metrics[league] = {
                "evaluated":  len(grp),
                "accuracy":   round(MetricsEngine.accuracy(probs, outcomes) * 100, 2),
                "brier":      round(MetricsEngine.brier_score(probs, outcomes), 4),
                "rps":        round(MetricsEngine.rps(probs, outcomes), 4),
            }

    rows = []
    for _, row in df_all.iterrows():
        league    = row["league"]
        total     = int(row["total_predictions"])
        evaluated = int(row["evaluated"])
        metrics   = evaluated_metrics.get(league, {})
        rows.append({
            "league":      league,
            "matches":     total,
            "evaluated":   evaluated,
            "accuracy":    metrics.get("accuracy"),
            "brier":       metrics.get("brier"),
            "rps":         metrics.get("rps"),
            "has_results": evaluated > 0,
        })

    rows.sort(key=lambda x: (not x["has_results"], x.get("brier") or 999))
    return {"leagues": rows}


# ─── GET /api/performance/confusion ──────────────────────────────────────────

@router.get("/confusion")
def get_confusion_matrix():
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur)
    finally:
        conn.close()

    if df.empty:
        return {"message": "No completed predictions yet."}

    probs, outcomes = _build_matrices(df)
    cm = MetricsEngine.confusion_matrix(probs, outcomes)
    return {"n_predictions": len(df), "matrix": cm.to_dict(), "labels": ["Away Win", "Draw", "Home Win"]}


# ─── GET /api/performance/markets ────────────────────────────────────────────

@router.get("/markets")
def get_market_accuracy():
    """
    Market accuracy: BTTS and Over 2.5 graded performance.
    Only rows where btts_correct or over_2_5_correct have been filled
    (via /api/prediction-log/evaluate) are included.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT
                league,
                COUNT(*) FILTER (WHERE btts_correct IS NOT NULL)         AS btts_total,
                SUM(CASE WHEN btts_correct      THEN 1 ELSE 0 END)       AS btts_correct,
                COUNT(*) FILTER (WHERE over_2_5_correct IS NOT NULL)     AS over25_total,
                SUM(CASE WHEN over_2_5_correct  THEN 1 ELSE 0 END)       AS over25_correct,
                AVG(home_xg)  FILTER (WHERE home_xg IS NOT NULL)         AS avg_home_xg,
                AVG(away_xg)  FILTER (WHERE away_xg IS NOT NULL)         AS avg_away_xg,
                AVG(btts_yes) FILTER (WHERE btts_yes IS NOT NULL)        AS avg_btts_prob,
                AVG(over_2_5) FILTER (WHERE over_2_5 IS NOT NULL)        AS avg_over25_prob
            FROM prediction_log
            GROUP BY league
            ORDER BY btts_total DESC NULLS LAST
        """)
        rows = cur.fetchall()

        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE btts_correct IS NOT NULL)     AS btts_total,
                SUM(CASE WHEN btts_correct      THEN 1 ELSE 0 END)   AS btts_correct,
                COUNT(*) FILTER (WHERE over_2_5_correct IS NOT NULL) AS over25_total,
                SUM(CASE WHEN over_2_5_correct  THEN 1 ELSE 0 END)   AS over25_correct,
                AVG(home_xg) FILTER (WHERE home_xg IS NOT NULL)      AS avg_home_xg,
                AVG(away_xg) FILTER (WHERE away_xg IS NOT NULL)      AS avg_away_xg
            FROM prediction_log
        """)
        totals = cur.fetchone()
    finally:
        conn.close()

    def _pct(correct, total):
        return round(int(correct or 0) / int(total) * 100, 1) if total and int(total) > 0 else None

    overall = {}
    if totals:
        bt = int(totals["btts_total"]   or 0)
        ot = int(totals["over25_total"] or 0)
        overall = {
            "btts": {
                "total":        bt,
                "correct":      int(totals["btts_correct"]  or 0),
                "accuracy_pct": _pct(totals["btts_correct"],  bt),
            },
            "over_2_5": {
                "total":        ot,
                "correct":      int(totals["over25_correct"] or 0),
                "accuracy_pct": _pct(totals["over25_correct"], ot),
            },
            "avg_home_xg": round(float(totals["avg_home_xg"] or 0), 2),
            "avg_away_xg": round(float(totals["avg_away_xg"] or 0), 2),
        }

    by_league = []
    for r in rows:
        bt = int(r["btts_total"]   or 0)
        ot = int(r["over25_total"] or 0)
        if bt == 0 and ot == 0:
            continue
        by_league.append({
            "league":          r["league"] or "Unknown",
            "btts_total":      bt,
            "btts_correct":    int(r["btts_correct"]  or 0),
            "btts_accuracy":   _pct(r["btts_correct"],  bt),
            "over25_total":    ot,
            "over25_correct":  int(r["over25_correct"] or 0),
            "over25_accuracy": _pct(r["over25_correct"], ot),
            "avg_home_xg":     round(float(r["avg_home_xg"]  or 0), 2),
            "avg_away_xg":     round(float(r["avg_away_xg"]  or 0), 2),
            "avg_btts_prob":   round(float(r["avg_btts_prob"] or 0), 3),
            "avg_over25_prob": round(float(r["avg_over25_prob"] or 0), 3),
        })

    return {"overall": overall, "by_league": by_league}
