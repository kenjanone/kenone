-- ============================================================
-- PlusOne Backend SQL Migrations
-- Run once in the Supabase SQL Editor
-- ============================================================

-- ─── 1. ml_models table (DC Model Supabase Persistence) ───────────────────────
-- Stores the serialised DCPredictor (joblib bytes) so it survives Railway redeploys.

CREATE TABLE IF NOT EXISTS ml_models (
    id             SERIAL PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE DEFAULT 'dc_model',
    model_bytes    BYTEA NOT NULL,
    n_samples      INT,
    train_accuracy FLOAT,
    cv_accuracy    FLOAT,
    n_features     INT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);


-- ─── 2. prediction_log UNIQUE constraint on match_id ──────────────────────────
-- Required for ON CONFLICT (match_id) DO NOTHING deduplication in /upcoming
-- Only run this if the constraint doesn't already exist:

ALTER TABLE prediction_log
  ADD CONSTRAINT prediction_log_match_id_unique UNIQUE (match_id);


-- ─── 3. prediction_log table (if not created yet) ─────────────────────────────
-- Run this block only if prediction_log doesn't exist in your database.

CREATE TABLE IF NOT EXISTS prediction_log (
    id               SERIAL PRIMARY KEY,
    match_id         INT UNIQUE,           -- references matches(id)
    home_team        TEXT NOT NULL,
    away_team        TEXT NOT NULL,
    league           TEXT,
    match_date       DATE,
    predicted        TEXT NOT NULL,        -- 'Home Win' / 'Draw' / 'Away Win'
    confidence       TEXT,                 -- 'High' / 'Medium' / 'Low'
    confidence_score FLOAT,
    home_win_prob    FLOAT,
    draw_prob        FLOAT,
    away_win_prob    FLOAT,
    actual           TEXT,                 -- filled in by evaluate after match
    correct          BOOLEAN,              -- filled in by evaluate
    evaluated_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS prediction_log_match_id_idx ON prediction_log(match_id);
CREATE INDEX IF NOT EXISTS prediction_log_correct_idx  ON prediction_log(correct);
CREATE INDEX IF NOT EXISTS prediction_log_date_idx     ON prediction_log(match_date);
 
 
 -- ─── 4. teams table logo_url column ─────────────────────────────────────────
 -- Add column for storing team logos scraped from FBref
 
 ALTER TABLE teams
   ADD COLUMN IF NOT EXISTS logo_url TEXT;


-- ─── 5. prediction_log — new columns for per-engine outcomes & predicted score ─
-- Run these if you already created the prediction_log table previously.
-- They are safe to run multiple times (IF NOT EXISTS).

ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS predicted_score           TEXT;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS dc_predicted_outcome      TEXT;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS ml_predicted_outcome      TEXT;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS legacy_predicted_outcome  TEXT;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS dc_correct                BOOLEAN;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS ml_correct                BOOLEAN;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS legacy_correct            BOOLEAN;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS correct_score             TEXT;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS evaluated_at              TIMESTAMPTZ;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS home_xg                  FLOAT;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS away_xg                  FLOAT;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS btts_yes                 FLOAT;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS over_2_5                 FLOAT;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS btts_correct             BOOLEAN;
ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS over_2_5_correct         BOOLEAN;


-- ─── 6. app_settings — seed DC lookback window ───────────────────────────────
-- Change the value here or via PUT /api/settings/dc_lookback_months to tune
-- how many months of history the DC engine uses.  Default = 9 (≈ 1 season).
INSERT INTO app_settings (key, value, description)
VALUES ('dc_lookback_months', '9', 'Number of months of match history used to train the Dixon-Coles DC engine. Default 9 ≈ 1 full season.')
ON CONFLICT (key) DO NOTHING;


-- ─── 7. app_settings — auto-consensus fixture window ─────────────────────────
-- Controls how many days AHEAD the auto-consensus job scans for upcoming
-- fixtures when auto-filling prediction_log.
-- The original 7-day window was too narrow — matches scheduled >1 week out
-- were never picked up.  Setting to 30 covers full monthly gameweek schedules.
-- Adjustable any time via the admin Settings panel.
INSERT INTO app_settings (key, value, description)
VALUES (
    'consensus_lookback_days',
    '30',
    'How many days ahead the auto-consensus job scans for upcoming fixtures. Increase if fixtures are scheduled further out. Default 30.'
)
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
