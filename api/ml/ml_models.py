"""
ML Models for PlusOne Prediction Engine
============================================
EnsemblePredictor: XGBoost + CalibratedClassifier(RandomForest)
soft-vote ensemble producing calibrated probabilities for
Home Win (0) / Draw (1) / Away Win (2).
"""

import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.joblib")


class EnsemblePredictor:
    """
    Soft-vote ensemble of calibrated XGBoost + RandomForest classifiers.
    Outputs calibrated probabilities that sum to 1.0.
    """

    def __init__(self):
        self.feature_names_ = None
        self.scaler_ = StandardScaler()
        self.is_trained = False
        self.train_accuracy = None
        self.cv_accuracy = None
        self.n_samples = 0
        self.feature_importances_ = {}

        # XGBoost — handles imbalanced classes well, fast, interpretable.
        # NOTE: sample_weight is passed at fit() time so XGB treats all
        # classes (Home Win / Draw / Away Win) with equal importance.
        # We calibrate it isotonically so its raw logits become true probs.
        xgb_base = XGBClassifier(
            n_estimators=400,           # more trees → smoother decision surface
            max_depth=5,
            learning_rate=0.04,         # slightly lower lr to pair with more trees
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,         # prevents splits on tiny leaf nodes
            gamma=0.1,                  # minimum gain required to make a split
            reg_alpha=0.1,              # L1 regularization (feature sparsity)
            reg_lambda=1.5,             # L2 regularization (weight decay)
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_cal = CalibratedClassifierCV(xgb_base, cv=3, method="isotonic")

        # RandomForest — diverse learner, good at irregular boundaries.
        # class_weight='balanced' already handles the HW/Draw/AW imbalance.
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_cal = CalibratedClassifierCV(rf, cv=3, method="isotonic")

        # Soft vote: equal weight — both models are now calibrated equally.
        self.model = VotingClassifier(
            estimators=[("xgb", xgb_cal), ("rf", rf_cal)],
            voting="soft",
            weights=[1, 1],
        )

    def train(self, X, y, feature_names=None, cv_folds=5, match_dates=None):
        """
        Train ensemble on (X, y).
        X: list of feature vectors
        y: list of labels (0=Home Win, 1=Draw, 2=Away Win)
        feature_names: optional list of feature name strings
        match_dates: optional list of date objects/strings — used to apply
                     exponential recency decay so recent matches matter more.
                     Formula: weight *= exp(-days_ago / 365)
                     A match from 1yr ago gets 37% weight vs a recent match.
        """
        import datetime
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        # Replace inf/nan with column means
        col_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_means, inds[1])

        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.n_samples = len(y)

        # Scale features (helps RF, doesn't hurt XGB)
        X_scaled = self.scaler_.fit_transform(X)

        # Compute balanced sample weights so XGBoost and RF treat
        # Home Win / Draw / Away Win with equal importance during training.
        # Without this, XGBoost (no built-in class weighting for multi-class)
        # collapses to always predicting the majority class (Home Win ~44%).
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)

        # Apply exponential recency decay if match_dates provided.
        # Recent matches are 3x more influential than 3-year-old data.
        if match_dates is not None and len(match_dates) == len(y):
            today = datetime.date.today()
            for i, d in enumerate(match_dates):
                try:
                    if isinstance(d, str):
                        d = datetime.date.fromisoformat(str(d)[:10])
                    elif hasattr(d, "date"):
                        d = d.date()
                    days_ago = max((today - d).days, 0)
                    recency = float(np.exp(-days_ago / 365.0))
                    sample_weights[i] *= recency
                except Exception:
                    pass  # silently keep original weight if date is malformed

        # Cross-validation accuracy (with sample weights)
        # sklearn ≥1.4 replaced fit_params= with params=; try new API first,
        # fall back to old API so we work with any installed version.
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(
                self.model, X_scaled, y, cv=skf,
                scoring="accuracy",
                params={"sample_weight": sample_weights},
                n_jobs=-1,
            )
        except (TypeError, ValueError):
            # sklearn < 1.4 uses fit_params= and raises ValueError/TypeError if params is passed
            cv_scores = cross_val_score(
                self.model, X_scaled, y, cv=skf,
                scoring="accuracy",
                fit_params={"sample_weight": sample_weights},
                n_jobs=-1,
            )
        self.cv_accuracy = float(np.mean(cv_scores))

        # Final fit on all data (pass balanced weights so XGB sees all classes equally)
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        y_pred = self.model.predict(X_scaled)
        self.train_accuracy = float(accuracy_score(y, y_pred))
        self.is_trained = True

        # Feature importances from XGBoost sub-model
        try:
            xgb_model = self.model.estimators_[0]
            imps = xgb_model.feature_importances_
            self.feature_importances_ = {
                name: float(imp)
                for name, imp in zip(self.feature_names_, imps)
            }
        except Exception:
            self.feature_importances_ = {}

        return self

    def predict_proba(self, x):
        """
        Predict calibrated probabilities for a single feature vector x.
        Returns dict: {home_win, draw, away_win, predicted_outcome, confidence,
                       confidence_score, label_int}
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = np.array(x, dtype=float).reshape(1, -1)

        # Fix non-finite values
        col_means = np.nan_to_num(
            np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0), nan=0.0
        )
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_means, inds[1])

        X_scaled = self.scaler_.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]

        # Map to named class probabilities
        classes = self.model.classes_
        prob_map = {int(c): float(p) for c, p in zip(classes, proba)}
        hw = prob_map.get(0, 0.33)
        dr = prob_map.get(1, 0.33)
        aw = prob_map.get(2, 0.34)

        # ── Probability floor ──────────────────────────────────────────────────
        # Apply a 2% floor (down from 5%) — just enough to avoid true-zero
        # probabilities breaking log-loss, but small enough not to steal
        # significant probability from correctly-predicted minority classes
        # (a 5% floor on 3 classes consumes 9% of probability budget).
        MIN_PROB = 0.02
        hw = max(hw, MIN_PROB)
        dr = max(dr, MIN_PROB)
        aw = max(aw, MIN_PROB)
        # ──────────────────────────────────────────────────────────────────────

        # Normalize to sum exactly to 1
        total = hw + dr + aw
        hw, dr, aw = hw / total, dr / total, aw / total

        predicted_int = int(np.argmax([hw, dr, aw]))
        predicted_label = LABELS[predicted_int]
        confidence_score = max(hw, dr, aw)

        if confidence_score >= 0.55:
            confidence = "High"
        elif confidence_score >= 0.42:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "home_win":         round(float(hw), 4),
            "draw":             round(float(dr), 4),
            "away_win":         round(float(aw), 4),
            "predicted_outcome": predicted_label,
            "label_int":        predicted_int,
            "confidence":       confidence,
            "confidence_score": round(float(confidence_score), 4),
        }

    def get_top_features(self, x, n=6):
        """
        Return the top n features by importance that influenced this prediction.
        """
        if not self.feature_importances_ or not self.feature_names_:
            return []
        items = sorted(self.feature_importances_.items(),
                       key=lambda kv: kv[1], reverse=True)[:n]
        return [{"feature": k, "importance": round(v, 4)} for k, v in items]

    def save(self, path=None):
        path = path or MODEL_PATH
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path=None):
        path = path or MODEL_PATH
        if not os.path.exists(path):
            return None
        obj = joblib.load(path)
        if isinstance(obj, cls):
            return obj
        return None
