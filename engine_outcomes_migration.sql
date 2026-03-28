-- Adds tracking columns for individual consensus engine predictions to prediction_log
-- Matches the backend insertions in _log_prediction_to_db

ALTER TABLE prediction_log 
ADD COLUMN IF NOT EXISTS enrichment_predicted_outcome VARCHAR(50),
ADD COLUMN IF NOT EXISTS legacy_predicted_outcome VARCHAR(50),
ADD COLUMN IF NOT EXISTS dc_predicted_outcome VARCHAR(50),
ADD COLUMN IF NOT EXISTS ml_predicted_outcome VARCHAR(50),
ADD COLUMN IF NOT EXISTS enrichment_correct BOOLEAN;
