"""
Professional Betting Markets Engine
=====================================
Full market pricing from expected goals (λ_h, λ_a).
No database dependency — feed it λ values from DC engine or xG model.

Markets covered:
  1X2, Asian Handicap, Over/Under (0.5–5.5), BTTS, Correct Score (top 12),
  Double Chance, Draw No Bet, Win to Nil, Team Goals, Half-Time/Full-Time

Also includes:
  - MarginRemover  (normalise / power / Shin margin stripping)
  - ValueDetector  (model edge vs bookmaker odds, Kelly sizing)
  - ArbitrageScanner (best price across multiple books)
  - CLVTracker      (closing line value tracker)
"""

import math
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import poisson
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — MARGIN REMOVER
# ══════════════════════════════════════════════════════════════════════════════

class MarginRemover:
    """Strip bookmaker vig from decimal odds using three methods."""

    @staticmethod
    def raw_implied(decimal_odds: float) -> float:
        return 1.0 / decimal_odds if decimal_odds > 0 else 0.0

    @staticmethod
    def overround(odds_list: list) -> float:
        return sum(1 / o for o in odds_list if o > 0) - 1.0

    @staticmethod
    def normalise(odds_list: list) -> list:
        raw   = [1/o for o in odds_list if o > 0]
        total = sum(raw) or 1
        return [r / total for r in raw]

    @staticmethod
    def power_method(odds_list: list, iterations: int = 50) -> list:
        raw = np.array([1/o for o in odds_list if o > 0])
        lo, hi = 0.5, 3.0
        for _ in range(iterations):
            k = (lo + hi) / 2
            s = np.sum(raw ** (1/k))
            if s > 1: lo = k
            else:     hi = k
        k = (lo + hi) / 2
        result = list(raw ** (1 / k))
        s = sum(result) or 1
        return [r / s for r in result]

    @staticmethod
    def shin_method(odds_list: list) -> list:
        raw = np.array([1/o for o in odds_list if o > 0])
        n   = len(raw)
        S   = raw.sum()

        def shin_z(z, raw, S, n):
            num = np.sqrt(z**2 + 4*(1-z)*raw**2/S) - z
            return sum(num / (2*(1-z))) - 1

        try:
            z = brentq(shin_z, 1e-6, 1 - 1e-6, args=(raw, S, n))
        except Exception:
            z = 0.02
        probs = (np.sqrt(z**2 + 4*(1-z)*raw**2/S) - z) / (2*(1-z))
        s = probs.sum() or 1
        return list(probs / s)


# ══════════════════════════════════════════════════════════════════════════════
# 2 — MARKET PRICER
# ══════════════════════════════════════════════════════════════════════════════

class MarketPricer:
    """
    Price ALL betting markets from expected goals λ_h and λ_a.
    Results are probabilities — convert to fair decimal odds via 1/p.
    """

    def __init__(self, lam_h: float, lam_a: float,
                 n_sim: int = 100_000, max_goals: int = 12):
        self.lam_h   = max(float(lam_h), 0.05)
        self.lam_a   = max(float(lam_a), 0.05)
        self.n_sim   = int(n_sim)
        self.max_g   = max_goals
        self._build_score_matrix()
        self._run_simulations()

    def _build_score_matrix(self):
        mat = np.zeros((self.max_g + 1, self.max_g + 1))
        for hg in range(self.max_g + 1):
            for ag in range(self.max_g + 1):
                mat[hg, ag] = poisson.pmf(hg, self.lam_h) * poisson.pmf(ag, self.lam_a)
        s = mat.sum()
        self.score_mat = mat / s if s > 0 else mat

    def _run_simulations(self):
        self.hg_sim = np.random.poisson(self.lam_h, self.n_sim)
        self.ag_sim = np.random.poisson(self.lam_a, self.n_sim)
        self.tg_sim = self.hg_sim + self.ag_sim

    # helpers
    def _fair(self, p: float) -> float:
        return round(1/p, 2) if p > 0 else 999.0

    def market_1x2(self) -> dict:
        hw = float(np.sum(np.tril(self.score_mat, -1)))
        d  = float(np.sum(np.diag(self.score_mat)))
        aw = float(np.sum(np.triu(self.score_mat, 1)))
        return {"home_win": hw, "draw": d, "away_win": aw}

    def market_asian_handicap(self,
                               handicaps: list = None) -> dict:
        if handicaps is None:
            handicaps = [-1.5, -1.0, -0.75, -0.5, -0.25,
                          0.0,  0.25,  0.5,  0.75,  1.0,  1.5]
        results = {}
        for h in handicaps:
            hg_adj = self.hg_sim - self.ag_sim + h
            if h % 0.5 == 0 and h % 1.0 != 0:
                home_cover = float((hg_adj > 0).mean())
                away_cover = float((hg_adj < 0).mean())
                push = 0.0
            elif h % 0.25 == 0 and h % 0.5 != 0:
                h_lo = h - 0.25; h_hi = h + 0.25
                hg_lo = self.hg_sim - self.ag_sim + h_lo
                hg_hi = self.hg_sim - self.ag_sim + h_hi
                home_cover = float(((hg_lo > 0).mean() + (hg_hi > 0).mean()) / 2)
                away_cover = float(((hg_lo < 0).mean() + (hg_hi < 0).mean()) / 2)
                push = 1 - home_cover - away_cover
            else:
                home_cover = float((hg_adj > 0).mean())
                away_cover = float((hg_adj < 0).mean())
                push = float((hg_adj == 0).mean())
            label = f"AH {h:+.2f}"
            results[label] = {
                "home": round(home_cover, 4),
                "push": round(push, 4),
                "away": round(away_cover, 4),
                "fair_home": self._fair(home_cover),
                "fair_away": self._fair(away_cover),
            }
        return results

    def market_over_under(self, lines: list = None) -> dict:
        if lines is None:
            lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        results = {}
        for line in lines:
            over  = float((self.tg_sim > line).mean())
            under = float((self.tg_sim < line).mean())
            push  = float((self.tg_sim == line).mean())
            results[f"O/U {line}"] = {
                "over": round(over, 4), "under": round(under, 4),
                "push": round(push, 4),
                "fair_over": self._fair(over), "fair_under": self._fair(under),
            }
        return results

    def market_btts(self) -> dict:
        yes = float(((self.hg_sim > 0) & (self.ag_sim > 0)).mean())
        return {"btts_yes": round(yes, 4), "btts_no": round(1 - yes, 4),
                "fair_yes": self._fair(yes), "fair_no": self._fair(1 - yes)}

    def market_correct_score(self, top_n: int = 12) -> list:
        scores = {}
        for hg in range(self.max_g + 1):
            for ag in range(self.max_g + 1):
                p = float(self.score_mat[hg, ag])
                scores[f"{hg}-{ag}"] = p
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        return [{"score": s, "probability": round(p, 5),
                 "fair_odds": self._fair(p)}
                for s, p in sorted_scores[:top_n]]

    def market_double_chance(self) -> dict:
        m = self.market_1x2()
        return {
            "1X": round(m["home_win"] + m["draw"], 4),
            "X2": round(m["draw"] + m["away_win"], 4),
            "12": round(m["home_win"] + m["away_win"], 4),
        }

    def market_draw_no_bet(self) -> dict:
        m  = self.market_1x2()
        nd = (m["home_win"] + m["away_win"]) or 1
        return {
            "home_dnb": round(m["home_win"] / nd, 4),
            "away_dnb": round(m["away_win"] / nd, 4),
        }

    def market_win_to_nil(self) -> dict:
        hwtn = float(((self.hg_sim > self.ag_sim) & (self.ag_sim == 0)).mean())
        awtn = float(((self.ag_sim > self.hg_sim) & (self.hg_sim == 0)).mean())
        return {
            "home_win_to_nil": round(hwtn, 4),
            "away_win_to_nil": round(awtn, 4),
            "fair_home": self._fair(hwtn),
            "fair_away": self._fair(awtn),
        }

    def market_team_goals(self) -> dict:
        return {
            "home_over_0.5":  round(float((self.hg_sim > 0).mean()), 4),
            "home_over_1.5":  round(float((self.hg_sim > 1).mean()), 4),
            "home_over_2.5":  round(float((self.hg_sim > 2).mean()), 4),
            "away_over_0.5":  round(float((self.ag_sim > 0).mean()), 4),
            "away_over_1.5":  round(float((self.ag_sim > 1).mean()), 4),
            "away_over_2.5":  round(float((self.ag_sim > 2).mean()), 4),
        }

    def full_sheet(self) -> dict:
        return {
            "1x2":             self.market_1x2(),
            "asian_handicap":  self.market_asian_handicap(),
            "over_under":      self.market_over_under(),
            "btts":            self.market_btts(),
            "correct_score":   self.market_correct_score(),
            "double_chance":   self.market_double_chance(),
            "draw_no_bet":     self.market_draw_no_bet(),
            "win_to_nil":      self.market_win_to_nil(),
            "team_goals":      self.market_team_goals(),
            "expected_goals":  {
                "home": round(self.lam_h, 3),
                "away": round(self.lam_a, 3),
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3 — VALUE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class ValueDetector:
    """Compare model probs vs bookmaker odds to find value bets."""

    def __init__(self, min_edge: float = 0.03, min_odds: float = 1.20,
                 max_odds: float = 15.0):
        self.min_edge = min_edge
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.remover  = MarginRemover()

    def edge(self, model_prob: float, market_odds: float) -> float:
        return model_prob - (1.0 / market_odds)

    def ev(self, model_prob: float, market_odds: float) -> float:
        return model_prob * (market_odds - 1) - (1 - model_prob)

    def kelly(self, model_prob: float, market_odds: float,
              fraction: float = 0.25) -> float:
        b = market_odds - 1
        q = 1 - model_prob
        full_kelly = max(0.0, (b * model_prob - q) / b)
        return round(full_kelly * fraction, 5)

    def _grade(self, edge: float, ev: float) -> str:
        if edge > 0.10 and ev > 0.15: return "A+"
        if edge > 0.07 and ev > 0.10: return "A"
        if edge > 0.05: return "B+"
        if edge > 0.03: return "B"
        return "C"

    def scan(self, model_probs: dict, market_odds: dict,
             market_name: str = "1x2") -> list:
        """
        model_probs: {outcome: probability}
        market_odds: {outcome: decimal_odds}
        Both dicts must use the same keys.
        """
        odds_vals = [v for v in market_odds.values()
                     if isinstance(v, (int, float)) and v > 0]
        if not odds_vals:
            return []
        fair_probs = self.remover.power_method(odds_vals)
        keys = [k for k, v in market_odds.items()
                if isinstance(v, (int, float)) and v > 0]
        fair_map = dict(zip(keys, fair_probs))

        bets = []
        for outcome, mkt_odds in market_odds.items():
            if not isinstance(mkt_odds, (int, float)) or mkt_odds <= 0:
                continue
            if mkt_odds < self.min_odds or mkt_odds > self.max_odds:
                continue
            mdl_prob = model_probs.get(outcome)
            if mdl_prob is None:
                continue
            edge_val = self.edge(mdl_prob, mkt_odds)
            ev_val   = self.ev(mdl_prob, mkt_odds)
            if edge_val >= self.min_edge:
                bets.append({
                    "market":        market_name,
                    "outcome":       outcome,
                    "model_prob":    round(mdl_prob, 4),
                    "fair_prob":     round(fair_map.get(outcome, 1/mkt_odds), 4),
                    "market_odds":   mkt_odds,
                    "fair_odds":     round(1 / mdl_prob, 2) if mdl_prob > 0 else 999,
                    "edge_pct":      round(edge_val * 100, 2),
                    "ev_per_unit":   round(ev_val, 4),
                    "kelly_25pct":   self.kelly(mdl_prob, mkt_odds, 0.25),
                    "kelly_full":    self.kelly(mdl_prob, mkt_odds, 1.0),
                    "grade":         self._grade(edge_val, ev_val),
                })
        return sorted(bets, key=lambda x: -x["edge_pct"])


# ══════════════════════════════════════════════════════════════════════════════
# 4 — ARBITRAGE SCANNER
# ══════════════════════════════════════════════════════════════════════════════

class ArbitrageScanner:
    """Find risk-free arb opportunities across multiple bookmakers."""

    @staticmethod
    def find_arb(books_odds: dict) -> dict:
        """
        books_odds: {outcome: {bookmaker: decimal_odds}}
        Returns arb info (with stakes for £100 target) if exists.
        """
        best = {}
        for outcome, book_map in books_odds.items():
            best_odds = max(book_map.values())
            best_book = max(book_map, key=book_map.get)
            best[outcome] = {"odds": best_odds, "book": best_book}

        implied_sum = sum(1 / v["odds"] for v in best.values() if v["odds"] > 0)
        margin = implied_sum - 1.0

        if margin < 0:
            total_return = 100.0
            stakes = {o: round(total_return / v["odds"], 2)
                      for o, v in best.items()}
            total_stake = sum(stakes.values())
            return {
                "arb_exists":   True,
                "margin_pct":   round(margin * 100, 3),
                "profit_pct":   round(-margin * 100, 3),
                "best_odds":    best,
                "stakes_for_100_return": stakes,
                "total_stake":  round(total_stake, 2),
                "guaranteed_profit": round(100.0 - total_stake, 2),
            }
        return {
            "arb_exists":     False,
            "margin_pct":     round(margin * 100, 3),
            "best_available": best,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5 — CLOSING LINE VALUE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class CLVTracker:
    """
    Track closing line value — the #1 indicator of long-term profitability.
    Positive average CLV = genuine edge. Recommended threshold: > 0%.
    """

    def __init__(self):
        self.records: list = []

    def log(self, outcome: str, your_odds: float, closing_odds: float,
            stake: float, won: bool):
        clv = (closing_odds / your_odds - 1) * 100 if your_odds > 0 else 0
        self.records.append({
            "outcome":      outcome,
            "your_odds":    your_odds,
            "closing_odds": closing_odds,
            "clv_pct":      round(clv, 3),
            "stake":        stake,
            "profit":       round(stake * (your_odds - 1) if won else -stake, 2),
            "won":          won,
        })

    def summary(self) -> dict:
        if not self.records:
            return {}
        clvs    = [r["clv_pct"] for r in self.records]
        profits = [r["profit"]  for r in self.records]
        stakes  = [r["stake"]   for r in self.records]
        avg_clv = float(np.mean(clvs))
        return {
            "bets":              len(self.records),
            "avg_clv_pct":       round(avg_clv, 3),
            "clv_positive_rate": round(sum(c > 0 for c in clvs) / len(clvs) * 100, 2),
            "total_profit":      round(sum(profits), 2),
            "roi_pct":           round(sum(profits) / max(sum(stakes), 0.01) * 100, 2),
            "positive_edge":     avg_clv > 0,
        }
