"""
Professional Prediction Performance Metrics
============================================
Pure-math metrics engine for evaluating probabilistic football predictions.
No database dependency — feed it probability arrays and outcome labels.

Metrics:
  - Accuracy         (% correct 1X2 predictions)
  - Brier Score      (gold standard: MSE of probabilities; lower = better)
  - RPS              (Ranked Probability Score; penalises confident errors)
  - Log Loss         (cross-entropy)
  - Calibration      (predicted prob vs actual frequency per bucket)
  - Confusion Matrix (where is the model going wrong?)
  - ROI              (flat-stake betting return)

Usage:
  from ml.metrics import MetricsEngine
  probs   = np.array([[0.5, 0.3, 0.2], ...])   # shape (N, 3): [p_aw, p_d, p_hw]
  outcomes = np.array([0, 2, 1, ...])            # 0=Away Win 1=Draw 2=Home Win
  print(MetricsEngine.brier_score(probs, outcomes))
"""

import numpy as np
import pandas as pd
from scipy.stats import binom

__all__ = ["MetricsEngine"]


class MetricsEngine:
    """All scoring metrics for probabilistic football predictions."""

    # ── Brier Score ───────────────────────────────────────────────────────────
    @staticmethod
    def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Brier Score = mean( Σ_k (p_k − o_k)² )
        Range 0 (perfect) → 2 (worst). Random baseline ≈ 0.667.
        probs: shape (N, 3)  [p_aw, p_d, p_hw]
        outcomes: shape (N,) [0=AW, 1=D, 2=HW]
        """
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=int)
        n_classes = probs.shape[1]
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(outcomes)), outcomes] = 1
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # ── Ranked Probability Score ──────────────────────────────────────────────
    @staticmethod
    def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
        """
        RPS = mean over matches of Σ_k (CDF_pred_k − CDF_actual_k)²
        Lower = better. Random baseline ≈ 0.333.
        Standard metric in sports forecasting competitions.
        """
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=int)
        scores = []
        for i in range(len(outcomes)):
            p = probs[i]
            o_vec = np.zeros(3); o_vec[outcomes[i]] = 1.0
            cum_p = np.cumsum(p)[:-1]
            cum_o = np.cumsum(o_vec)[:-1]
            scores.append(np.sum((cum_p - cum_o) ** 2) / 2)
        return float(np.mean(scores))

    # ── Log Loss ──────────────────────────────────────────────────────────────
    @staticmethod
    def log_loss(probs: np.ndarray, outcomes: np.ndarray,
                 eps: float = 1e-15) -> float:
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=int)
        clipped = np.clip(probs, eps, 1 - eps)
        return float(-np.mean(np.log(clipped[np.arange(len(outcomes)), outcomes])))

    # ── Accuracy ──────────────────────────────────────────────────────────────
    @staticmethod
    def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=int)
        preds = np.argmax(probs, axis=1)
        return float((preds == outcomes).mean())

    # ── Calibration ───────────────────────────────────────────────────────────
    @staticmethod
    def calibration(probs: np.ndarray, outcomes: np.ndarray,
                    n_bins: int = 10) -> pd.DataFrame:
        """
        For each probability bucket, compare predicted probability vs
        actual observed frequency. A perfect model lies on the diagonal.
        """
        probs    = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=int)
        flat_probs  = probs.flatten()
        flat_labels = np.zeros(len(probs) * 3)
        for i, o in enumerate(outcomes):
            flat_labels[i * 3 + o] = 1.0

        bins = np.linspace(0, 1, n_bins + 1)
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (flat_probs >= lo) & (flat_probs < hi)
            if mask.sum() < 5:
                continue
            mean_pred = float(flat_probs[mask].mean())
            mean_act  = float(flat_labels[mask].mean())
            rows.append({
                "bin_center":     round((lo + hi) / 2, 2),
                "predicted_prob": round(mean_pred, 4),
                "actual_freq":    round(mean_act, 4),
                "n_samples":      int(mask.sum()),
                "gap":            round(mean_pred - mean_act, 4),
                "well_calibrated": abs(mean_pred - mean_act) < 0.05,
            })
        return pd.DataFrame(rows)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    @staticmethod
    def confusion_matrix(probs: np.ndarray, outcomes: np.ndarray) -> pd.DataFrame:
        probs    = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=int)
        preds    = np.argmax(probs, axis=1)
        labels   = ["Away Win", "Draw", "Home Win"]
        mat = np.zeros((3, 3), dtype=int)
        for pred, actual in zip(preds, outcomes):
            mat[actual, pred] += 1
        return pd.DataFrame(
            mat,
            index   =[f"Actual {l}" for l in labels],
            columns =[f"Pred {l}"   for l in labels],
        )

    # ── ROI ───────────────────────────────────────────────────────────────────
    @staticmethod
    def roi(records: list, stake: float = 1.0) -> dict:
        """
        records: list of dicts with keys:
          predicted_outcome (int 0/1/2),
          actual_outcome    (int 0/1/2),
          odds_taken        (float, decimal)
        """
        total_staked = total_return = wins = 0.0
        for r in records:
            if r.get("odds_taken") is None:
                continue
            total_staked += stake
            if r["predicted_outcome"] == r["actual_outcome"]:
                total_return += stake * r["odds_taken"]
                wins += 1
        if total_staked == 0:
            return {"roi_pct": 0, "profit": 0, "win_rate": 0, "bets": 0}
        profit = total_return - total_staked
        return {
            "roi_pct":       round(profit / total_staked * 100, 2),
            "profit":        round(profit, 2),
            "win_rate":      round(wins / len(records) * 100, 2) if records else 0,
            "bets":          len(records),
            "total_staked":  round(total_staked, 2),
            "total_return":  round(total_return, 2),
        }

    # ── Statistical Significance ──────────────────────────────────────────────
    @staticmethod
    def significance_test(probs: np.ndarray, outcomes: np.ndarray,
                          baseline: float = 0.45) -> dict:
        """
        Binomial test: is observed accuracy significantly better than baseline?
        p-value < 0.05 = statistically significant skill.
        """
        probs    = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=int)
        preds    = np.argmax(probs, axis=1)
        n = len(outcomes)
        k = int((preds == outcomes).sum())
        if n == 0:
            return {}
        p_val = float(binom.sf(k - 1, n, baseline))
        return {
            "n_predictions":         n,
            "n_correct":             k,
            "observed_accuracy_pct": round(k / n * 100, 2),
            "baseline_accuracy_pct": round(baseline * 100, 2),
            "p_value":               round(p_val, 4),
            "significant":           p_val < 0.05,
        }

    # ── Full summary ──────────────────────────────────────────────────────────
    @staticmethod
    def full_summary(probs: np.ndarray, outcomes: np.ndarray) -> dict:
        """Convenience: run all scalar metrics at once."""
        probs    = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=int)
        return {
            "n":           len(outcomes),
            "accuracy":    round(MetricsEngine.accuracy(probs, outcomes), 4),
            "brier_score": round(MetricsEngine.brier_score(probs, outcomes), 4),
            "rps":         round(MetricsEngine.rps(probs, outcomes), 4),
            "log_loss":    round(MetricsEngine.log_loss(probs, outcomes), 4),
            "significance": MetricsEngine.significance_test(probs, outcomes),
        }
