-- ============================================================
-- Backfill script for prediction_log
-- Run once in Supabase SQL Editor AFTER deploying the fix.
-- ============================================================

-- 1. Re-grade dc_correct / ml_correct / legacy_correct for rows
--    that already have actual results but NULL engine-correct columns.
--    This will not affect rows where the columns were already filled.
UPDATE prediction_log
SET
    dc_correct     = CASE WHEN dc_predicted_outcome     IS NOT NULL THEN (dc_predicted_outcome     = actual) ELSE NULL END,
    ml_correct     = CASE WHEN ml_predicted_outcome     IS NOT NULL THEN (ml_predicted_outcome     = actual) ELSE NULL END,
    legacy_correct = CASE WHEN legacy_predicted_outcome IS NOT NULL THEN (legacy_predicted_outcome = actual) ELSE NULL END
WHERE actual IS NOT NULL
  AND (dc_correct IS NULL OR ml_correct IS NULL OR legacy_correct IS NULL);

-- 2. Check how many rows still have NULL per-engine PREDICTED outcomes
--    (these are rows logged before the fix — they can't be backfilled
--    automatically since the data was never stored).
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE dc_predicted_outcome IS NULL)     AS missing_dc,
    COUNT(*) FILTER (WHERE ml_predicted_outcome IS NULL)     AS missing_ml,
    COUNT(*) FILTER (WHERE legacy_predicted_outcome IS NULL) AS missing_legacy,
    COUNT(*) FILTER (WHERE dc_predicted_outcome IS NOT NULL) AS has_dc,
    COUNT(*) FILTER (WHERE ml_predicted_outcome IS NOT NULL) AS has_ml
FROM prediction_log;

-- 3. Check engine accuracy from rows that DO have per-engine data
SELECT
    COUNT(*) FILTER (WHERE dc_correct = TRUE)      AS dc_correct_count,
    COUNT(*) FILTER (WHERE dc_correct IS NOT NULL) AS dc_total,
    COUNT(*) FILTER (WHERE ml_correct = TRUE)      AS ml_correct_count,
    COUNT(*) FILTER (WHERE ml_correct IS NOT NULL) AS ml_total,
    COUNT(*) FILTER (WHERE legacy_correct = TRUE)       AS legacy_correct_count,
    COUNT(*) FILTER (WHERE legacy_correct IS NOT NULL)  AS legacy_total
FROM prediction_log
WHERE actual IS NOT NULL;
