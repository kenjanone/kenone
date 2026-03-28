"""
Prediction Log — Standalone Performance Tracker
================================================
Tracks every prediction made by the ML engine against the actual result
once the match is completed. Gives real-world accuracy, not just training
cross-validation.

Endpoints:
  POST /api/prediction-log/record     → save a prediction before a match
  POST /api/prediction-log/evaluate   → scan completed matches, mark correct/wrong
  GET  /api/prediction-log/accuracy   → real-world accuracy stats and trend
  GET  /api/prediction-log            → full log with pagination

SQL to run ONCE in Supabase SQL Editor:
----------------------------------------
  CREATE TABLE IF NOT EXISTS prediction_log (
      id               SERIAL PRIMARY KEY,
      match_id         INT,
      home_team        TEXT NOT NULL,
      away_team        TEXT NOT NULL,
      league           TEXT,
      match_date       DATE,
      predicted        TEXT NOT NULL,
      confidence       TEXT,
      confidence_score FLOAT,
      home_win_prob    FLOAT,
      draw_prob        FLOAT,
      away_win_prob    FLOAT,
      actual           TEXT,
      correct          BOOLEAN,
      correct_score    TEXT,
      evaluated_at     TIMESTAMPTZ,
      created_at       TIMESTAMPTZ DEFAULT NOW(),
      home_xg          FLOAT,
      away_xg          FLOAT,
      btts_yes         FLOAT,
      over_2_5         FLOAT,
      btts_correct     BOOLEAN,
      over_2_5_correct BOOLEAN,
      dc_predicted_outcome     TEXT,
      ml_predicted_outcome     TEXT,
      legacy_predicted_outcome TEXT,
      dc_correct               BOOLEAN,
      ml_correct               BOOLEAN,
      legacy_correct           BOOLEAN,
      enrichment_correct       BOOLEAN
  );

  -- If table already exists, run these:
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS home_xg          FLOAT;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS away_xg          FLOAT;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS btts_yes         FLOAT;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS over_2_5         FLOAT;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS btts_correct     BOOLEAN;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS over_2_5_correct BOOLEAN;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS dc_predicted_outcome     TEXT;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS ml_predicted_outcome     TEXT;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS legacy_predicted_outcome TEXT;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS dc_correct               BOOLEAN;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS ml_correct               BOOLEAN;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS legacy_correct           BOOLEAN;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS enrichment_correct       BOOLEAN;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS correct_score    TEXT;
  ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS evaluated_at     TIMESTAMPTZ;
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Optional
from database import get_connection
from routes.deps import require_admin

router = APIRouter()


# ─── Request models ───────────────────────────────────────────────────────────

class RecordRequest(BaseModel):
    match_id:         Optional[int]   = None
    home_team:        str
    away_team:        str
    league:           Optional[str]   = None
    match_date:       Optional[str]   = None
    predicted:        str
    confidence:       Optional[str]   = None
    confidence_score: Optional[float] = None
    home_win_prob:    Optional[float] = None
    draw_prob:        Optional[float] = None
    away_win_prob:    Optional[float] = None
    btts_yes:         Optional[float] = None
    over_2_5:         Optional[float] = None
    home_xg:          Optional[float] = None
    away_xg:          Optional[float] = None


# ─── Helper ───────────────────────────────────────────────────────────────────

def _outcome_label(home_score: int, away_score: int) -> str:
    if home_score > away_score: return "Home Win"
    if away_score > home_score: return "Away Win"
    return "Draw"


# ─── POST /record ─────────────────────────────────────────────────────────────

@router.post("/record")
def record_prediction(req: RecordRequest):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO prediction_log
                (match_id, home_team, away_team, league, match_date,
                 predicted, confidence, confidence_score,
                 home_win_prob, draw_prob, away_win_prob,
                 btts_yes, over_2_5, home_xg, away_xg)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            req.match_id, req.home_team, req.away_team,
            req.league, req.match_date,
            req.predicted, req.confidence, req.confidence_score,
            req.home_win_prob, req.draw_prob, req.away_win_prob,
            req.btts_yes, req.over_2_5, req.home_xg, req.away_xg,
        ))
        row = cur.fetchone()
        conn.commit()
        return {"saved": True, "log_id": row["id"]}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─── Helper for evaluation ────────────────────────────────────────────────────

def do_evaluate_predictions(conn) -> int:
    """
    Grade un-evaluated prediction_log rows whose match has now completed.
    First tries to link any missing match_ids.
    """
    cur = conn.cursor()
    try:
        # 1. Backfill missing match_ids by matching team names (case-insensitive)
        #    and a ±7 day window so minor name differences or missing match_date
        #    don't leave rows permanently un-linked.
        cur.execute("""
            UPDATE prediction_log pl
            SET match_id = m.id
            FROM matches m
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams at ON at.id = m.away_team_id
            WHERE pl.match_id IS NULL
              AND LOWER(pl.home_team) = LOWER(ht.name)
              AND LOWER(pl.away_team) = LOWER(at.name)
              AND m.match_date >= (COALESCE(pl.match_date, pl.created_at::DATE) - INTERVAL '7 days')
              AND m.match_date <= (COALESCE(pl.match_date, pl.created_at::DATE) + INTERVAL '7 days')
              AND NOT EXISTS (
                  SELECT 1 FROM prediction_log ex
                  WHERE ex.match_id = m.id AND ex.id != pl.id
              )
        """)

        # 1b. Fallback: ILIKE fuzzy match for rows still without a match_id
        #     (handles "Manchester City" vs "Manchester City FC" etc.)
        cur.execute("""
            UPDATE prediction_log pl
            SET match_id = (
                SELECT m.id FROM matches m
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE ht.name ILIKE '%' || pl.home_team || '%'
                  AND at.name ILIKE '%' || pl.away_team || '%'
                  AND m.match_date >= (COALESCE(pl.match_date, pl.created_at::DATE) - INTERVAL '7 days')
                  AND m.match_date <= (COALESCE(pl.match_date, pl.created_at::DATE) + INTERVAL '7 days')
                  AND NOT EXISTS (
                      SELECT 1 FROM prediction_log ex
                      WHERE ex.match_id = m.id AND ex.id != pl.id
                  )
                LIMIT 1
            )
            WHERE pl.match_id IS NULL
              AND EXISTS (
                SELECT 1 FROM matches m
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE ht.name ILIKE '%' || pl.home_team || '%'
                  AND at.name ILIKE '%' || pl.away_team || '%'
                  AND m.match_date >= (COALESCE(pl.match_date, pl.created_at::DATE) - INTERVAL '7 days')
                  AND m.match_date <= (COALESCE(pl.match_date, pl.created_at::DATE) + INTERVAL '7 days')
                  AND NOT EXISTS (
                      SELECT 1 FROM prediction_log ex
                      WHERE ex.match_id = m.id AND ex.id != pl.id
                  )
              )
        """)

        # 2. Grade all un-evaluated rows
        cur.execute("""
            SELECT pl.id, pl.predicted, pl.btts_yes, pl.over_2_5,
                   pl.dc_predicted_outcome, pl.ml_predicted_outcome,
                   pl.legacy_predicted_outcome, pl.enrichment_predicted_outcome,
                   m.home_score, m.away_score
            FROM prediction_log pl
            JOIN matches m ON m.id = pl.match_id
            WHERE pl.correct IS NULL
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
        """)
        rows = cur.fetchall()

        updated = 0
        for r in rows:
            home_score  = int(r["home_score"])
            away_score  = int(r["away_score"])
            actual      = _outcome_label(home_score, away_score)
            correct     = (actual == r["predicted"])
            correct_score = f"{home_score}-{away_score}"
            total_goals = home_score + away_score
            both_scored = home_score > 0 and away_score > 0

            # Market grading
            btts_correct     = None
            over_2_5_correct = None
            if r["btts_yes"] is not None:
                btts_correct = ((float(r["btts_yes"]) > 0.5) == both_scored)
            if r["over_2_5"] is not None:
                over_2_5_correct = ((float(r["over_2_5"]) > 0.5) == (total_goals > 2))

            # Per-engine outcome grading
            dc_correct         = (r["dc_predicted_outcome"]         == actual) if r["dc_predicted_outcome"]         else None
            ml_correct         = (r["ml_predicted_outcome"]         == actual) if r["ml_predicted_outcome"]         else None
            legacy_correct     = (r["legacy_predicted_outcome"]     == actual) if r["legacy_predicted_outcome"]     else None
            enrichment_correct = (r["enrichment_predicted_outcome"] == actual) if r["enrichment_predicted_outcome"] else None

            cur.execute("""
                UPDATE prediction_log
                SET actual            = %s,
                    correct           = %s,
                    correct_score     = %s,
                    evaluated_at      = NOW(),
                    btts_correct      = %s,
                    over_2_5_correct  = %s,
                    dc_correct        = %s,
                    ml_correct        = %s,
                    legacy_correct    = %s,
                    enrichment_correct= %s
                WHERE id = %s
            """, (actual, correct, correct_score,
                  btts_correct, over_2_5_correct,
                  dc_correct, ml_correct, legacy_correct, enrichment_correct,
                  r["id"]))
            updated += 1
        return updated
    finally:
        cur.close()


# ─── POST /evaluate ───────────────────────────────────────────────────────────

@router.post("/evaluate")
def evaluate_predictions(_admin: dict = Depends(require_admin)):
    """
    Grade all un-evaluated prediction_log rows whose match has now completed.
    Grades: outcome correct/wrong, BTTS correct/wrong, Over 2.5 correct/wrong.
    Safe to call repeatedly.
    """
    conn = get_connection()
    try:
        updated = do_evaluate_predictions(conn)
        conn.commit()
        return {
            "evaluated": updated,
            "message": f"{updated} prediction(s) graded against real results.",
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─── GET /accuracy ────────────────────────────────────────────────────────────

@router.get("/accuracy")
def real_world_accuracy(
    league: Optional[str] = Query(None),
    last_n: int           = Query(50),
):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        base   = " FROM prediction_log WHERE correct IS NOT NULL"
        params: list = []
        if league:
            base += " AND league = %s"; params.append(league)

        cur.execute("SELECT COUNT(*) AS total, SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct_count" + base, params)
        overall       = cur.fetchone()
        total         = int(overall["total"] or 0)
        correct_count = int(overall["correct_count"] or 0)
        overall_pct   = round(correct_count / total * 100, 1) if total else None

        # By predicted outcome
        cur.execute("""
            SELECT predicted, COUNT(*) AS total,
                   SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct_count
        """ + base + " GROUP BY predicted ORDER BY predicted", params)
        by_outcome = [
            {
                "predicted":    r["predicted"],
                "total":        int(r["total"]),
                "correct":      int(r["correct_count"] or 0),
                "accuracy_pct": round(int(r["correct_count"] or 0) / int(r["total"]) * 100, 1)
                                if int(r["total"]) > 0 else None,
            }
            for r in cur.fetchall()
        ]

        # By confidence tier
        cur.execute("""
            SELECT confidence, COUNT(*) AS total,
                   SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct_count
        """ + base + " AND confidence IS NOT NULL GROUP BY confidence ORDER BY confidence", params)
        by_confidence = [
            {
                "confidence":   r["confidence"],
                "total":        int(r["total"]),
                "correct":      int(r["correct_count"] or 0),
                "accuracy_pct": round(int(r["correct_count"] or 0) / int(r["total"]) * 100, 1)
                                if int(r["total"]) > 0 else None,
            }
            for r in cur.fetchall()
        ]

        # Market accuracy
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE btts_correct IS NOT NULL)            AS btts_total,
                SUM(CASE WHEN btts_correct     THEN 1 ELSE 0 END)           AS btts_correct,
                COUNT(*) FILTER (WHERE over_2_5_correct IS NOT NULL)        AS over25_total,
                SUM(CASE WHEN over_2_5_correct THEN 1 ELSE 0 END)           AS over25_correct
        """ + base, params)
        mkt = cur.fetchone()
        btts_total    = int(mkt["btts_total"]    or 0)
        over25_total  = int(mkt["over25_total"]  or 0)
        market_accuracy = {
            "btts": {
                "total":        btts_total,
                "correct":      int(mkt["btts_correct"]  or 0),
                "accuracy_pct": round(int(mkt["btts_correct"] or 0) / btts_total * 100, 1) if btts_total else None,
            },
            "over_2_5": {
                "total":        over25_total,
                "correct":      int(mkt["over25_correct"] or 0),
                "accuracy_pct": round(int(mkt["over25_correct"] or 0) / over25_total * 100, 1) if over25_total else None,
            },
        }

        # Rolling last N
        cur.execute("SELECT correct, evaluated_at" + base + f" ORDER BY evaluated_at DESC LIMIT {last_n}", params)
        recent     = cur.fetchall()
        recent_pct = round(sum(1 for r in recent if r["correct"]) / len(recent) * 100, 1) if recent else None

        return {
            "total_evaluated":          total,
            "correct":                  correct_count,
            "overall_accuracy":         overall_pct,
            f"last_{last_n}_accuracy":  recent_pct,
            "by_outcome":               by_outcome,
            "by_confidence":            by_confidence,
            "market_accuracy":          market_accuracy,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─── GET / (full log) ─────────────────────────────────────────────────────────

@router.get("")
def list_prediction_log(
    league:    Optional[str]  = Query(None),
    correct:   Optional[bool] = Query(None),
    evaluated: Optional[bool] = Query(None),
    limit:     int            = Query(50),
    offset:    int            = Query(0),
):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        query  = "SELECT * FROM prediction_log WHERE 1=1"
        params: list = []
        if league:
            query += " AND league = %s"; params.append(league)
        if correct is not None:
            query += " AND correct = %s"; params.append(correct)
        if evaluated is True:
            query += " AND correct IS NOT NULL"
        elif evaluated is False:
            query += " AND correct IS NULL"
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params += [limit, offset]
        cur.execute(query, params)
        rows = cur.fetchall()
        return {"count": len(rows), "rows": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
