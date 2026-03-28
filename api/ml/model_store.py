"""
Model Store — Supabase Persistence
====================================
Saves and loads the trained EnsemblePredictor to/from a PostgreSQL
(Supabase) table using joblib serialisation and a BYTEA column.

Why not disk?
  Railway and Render reset the container filesystem on every deploy.
  Storing the model in Supabase means it survives redeploys automatically.

SQL to run ONCE in the Supabase SQL Editor:
--------------------------------------------
  CREATE TABLE IF NOT EXISTS ml_models (
      id           SERIAL PRIMARY KEY,
      name         TEXT NOT NULL UNIQUE DEFAULT 'prediction_model',
      model_bytes  BYTEA NOT NULL,
      n_samples    INT,
      train_accuracy FLOAT,
      cv_accuracy  FLOAT,
      n_features   INT,
      created_at   TIMESTAMPTZ DEFAULT NOW()
  );

Usage:
  from ml.model_store import save_to_db, load_from_db
  save_to_db(model)       # stores / updates the single row
  model = load_from_db()  # returns EnsemblePredictor or None
"""

import io
import logging

import joblib
import psycopg2
from database import get_connection

log = logging.getLogger(__name__)

_MODEL_NAME = "prediction_model"


def save_to_db(model, name: str = _MODEL_NAME) -> bool:
    """
    Serialise `model` with joblib and upsert into the ml_models table.
    Returns True on success, False on failure (so training still works
    even if DB persistence fails).
    """
    try:
        buf = io.BytesIO()
        joblib.dump(model, buf)
        model_bytes = buf.getvalue()

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO ml_models
                    (name, model_bytes, n_samples, train_accuracy, cv_accuracy, n_features)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE
                    SET model_bytes    = EXCLUDED.model_bytes,
                        n_samples      = EXCLUDED.n_samples,
                        train_accuracy = EXCLUDED.train_accuracy,
                        cv_accuracy    = EXCLUDED.cv_accuracy,
                        n_features     = EXCLUDED.n_features,
                        created_at     = NOW()
                """,
                (
                    name,
                    psycopg2.Binary(model_bytes),
                    getattr(model, "n_samples", None),
                    getattr(model, "train_accuracy", None),
                    getattr(model, "cv_accuracy", None),
                    len(getattr(model, "feature_names_", None) or []),
                ),
            )
            conn.commit()
            log.info("Model '%s' saved to Supabase (%d bytes).", name, len(model_bytes))
            return True
        finally:
            conn.close()
    except Exception as exc:
        log.warning("Could not save model to DB (falling back to disk only): %s", exc)
        return False


def load_from_db(name: str = _MODEL_NAME):
    """
    Load the serialised EnsemblePredictor from Supabase.
    Returns an EnsemblePredictor instance or None if not found / error.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT model_bytes FROM ml_models WHERE name = %s LIMIT 1",
                (name,),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            log.info("No model named '%s' found in DB.", name)
            return None

        buf = io.BytesIO(bytes(row["model_bytes"]))
        model = joblib.load(buf)
        log.info("Model '%s' loaded from Supabase.", name)
        return model

    except Exception as exc:
        log.warning("Could not load model from DB: %s", exc)
        return None
