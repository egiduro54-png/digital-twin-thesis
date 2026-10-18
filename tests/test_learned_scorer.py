"""
Tests for learned_scorer.py (walk-forward learned risk scorers)
"""

import numpy as np
import pandas as pd
import pytest

from src.learned_scorer import (
    DT_FEATURES, STATIC_FEATURES, walk_forward_scores, permutation_importance,
)

FEATURES = DT_FEATURES + STATIC_FEATURES


def _rows(n_windows=6, n=80, seed=0, signal="ri_downside"):
    """Synthetic validation rows where only `signal` predicts the drawdown."""
    rng = np.random.default_rng(seed)
    recs = []
    for w in range(n_windows):
        X = rng.normal(size=(n, len(FEATURES)))
        level = rng.normal(-15, 5)  # market level differs by window
        dd = level - 5 * X[:, FEATURES.index(signal)] + rng.normal(0, 2, n)
        for i in range(n):
            recs.append({"window": w, **dict(zip(FEATURES, X[i])), "actual_drawdown": dd[i],
                         "baseline": 0.0})
    return pd.DataFrame(recs)


class TestWalkForward:
    def test_first_windows_unscored(self):
        rows = _rows()
        s = walk_forward_scores(rows, FEATURES, "ridge", min_train=3)
        assert s[rows["window"] < 3].isna().all()
        assert s[rows["window"] >= 3].notna().all()

    def test_learns_the_signal(self):
        rows = _rows()
        s = walk_forward_scores(rows, FEATURES, "ridge", min_train=3)
        # Scores are cross-sectional, so judge them within each window
        rhos = [s[g.index].corr(-g["actual_drawdown"], method="spearman")
                for w, g in rows.groupby("window") if w >= 3]
        assert min(rhos) > 0.8

    def test_no_leakage_from_future_windows(self):
        # Changing a later window's outcome must not change an earlier prediction
        rows = _rows()
        s1 = walk_forward_scores(rows, FEATURES, "ridge", min_train=3)
        altered = rows.copy()
        altered.loc[altered["window"] == 5, "actual_drawdown"] *= -1
        s2 = walk_forward_scores(altered, FEATURES, "ridge", min_train=3)
        early = rows["window"].isin([3, 4])
        assert np.allclose(s1[early], s2[early])


def test_permutation_importance_ranks_signal_first():
    rows = _rows(n_windows=5, n=60)
    imp = permutation_importance(rows, FEATURES, "ridge", min_train=3, n_repeats=2)
    assert imp.iloc[0]["feature"] == "ri_downside"
    assert imp.iloc[0]["importance"] > 0.3
