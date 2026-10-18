"""
Walk-forward learned risk scorers on the rolling-window validation rows.

    python -m scripts.run_learned_scorer --rows results/robust_validation_final/rows.csv

Outputs next to the rows file:
    learned_summary.csv      learned vs fixed-weight scorers, out-of-sample windows
    learned_importance.csv   permutation importance per feature set (Ridge)
    learned_coefficients.csv Ridge coefficients (standardised), for reading
    learned_explainer.json   Ridge on the DT features + reference window, for the app
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.learned_scorer import (
    FEATURE_SETS, evaluate_learned, fit_explainer, permutation_importance, ridge_coefficients,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/robust_validation_final/rows.csv")
    ap.add_argument("--min-train", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    rows_path = Path(args.rows)
    out = rows_path.parent
    rows = pd.read_csv(rows_path)

    summary, _ = evaluate_learned(rows, min_train=args.min_train, n_boot=args.bootstrap)
    summary.to_csv(out / "learned_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.to_string(index=False))

    imp, coef = [], []
    for name, feats in FEATURE_SETS.items():
        i = permutation_importance(rows, feats, "ridge", args.min_train, args.repeats)
        i.insert(0, "feature_set", name)
        imp.append(i)
        c = ridge_coefficients(rows, feats)
        c.insert(0, "feature_set", name)
        coef.append(c)
    imp, coef = pd.concat(imp), pd.concat(coef)
    imp.to_csv(out / "learned_importance.csv", index=False)
    coef.to_csv(out / "learned_coefficients.csv", index=False)
    print("\nPermutation importance (drop in mean Spearman ρ):")
    print(imp.to_string(index=False))
    print("\nRidge coefficients (all windows, standardised):")
    print(coef.to_string(index=False))

    import json
    (out / "learned_explainer.json").write_text(json.dumps(fit_explainer(rows), indent=1),
                                                encoding="utf-8")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {out}/")


if __name__ == "__main__":
    main()
