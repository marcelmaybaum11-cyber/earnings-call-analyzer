"""Correlate FinBERT sentiment with post-call returns and train a classifier.

Analysis:
  1. Spearman correlation between net sentiment (positive - negative) and
     the 1-day / 5-day post-call returns.
  2. Logistic regression predicting return DIRECTION (up vs down), with
     5-fold stratified cross-validation.

Predicting direction (classification) instead of magnitude (regression) is
deliberate: with ~190 samples and noisy markets, direction is the realistic
goal. Accuracy is always compared to the majority-class baseline, because
stocks drift upward and "always predict up" is deceptively strong.

FinBERT's three probabilities sum to 1, so one is redundant; we use
positive and negative as the two features.

Run from the project root:
    .venv\\Scripts\\python.exe src\\model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET_CSV = Path("data/processed/dataset.csv")
MODELS_DIR = Path("models")

FEATURES = ["positive", "negative"]
HORIZONS = ["return_1d", "return_5d"]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_CSV)
    df["net_sentiment"] = df["positive"] - df["negative"]
    return df


def train_for_horizon(df: pd.DataFrame, horizon: str) -> dict:
    """Train + cross-validate one horizon. Returns metrics and the model."""
    data = df.dropna(subset=[horizon])
    X = data[FEATURES].to_numpy()
    y = (data[horizon] > 0).astype(int).to_numpy()  # 1 = stock went up

    rho, pvalue = spearmanr(data["net_sentiment"], data[horizon])

    # Scaling first makes the regression coefficients comparable in size.
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    pipeline.fit(X, y)  # final model on all data

    return {
        "horizon": horizon,
        "n": len(data),
        "spearman_rho": rho,
        "spearman_p": pvalue,
        "baseline_accuracy": max(y.mean(), 1 - y.mean()),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "model": pipeline,
    }


def main() -> None:
    df = load_dataset()
    print(f"Dataset: {len(df)} calls, {df['ticker'].nunique()} tickers, "
          f"{df['call_date'].min()} to {df['call_date'].max()}\n")

    MODELS_DIR.mkdir(exist_ok=True)
    for horizon in HORIZONS:
        r = train_for_horizon(df, horizon)
        joblib.dump(r["model"], MODELS_DIR / f"logreg_{horizon}.joblib")

        print(f"--- {horizon} (n={r['n']}) ---")
        print(f"Spearman correlation (net sentiment vs return): "
              f"rho={r['spearman_rho']:+.3f}  p={r['spearman_p']:.3f}")
        print(f"Majority-class baseline accuracy: {r['baseline_accuracy']:.1%}")
        print(f"Logistic regression CV accuracy:  "
              f"{r['cv_accuracy_mean']:.1%} +/- {r['cv_accuracy_std']:.1%}")
        print(f"Model saved to {MODELS_DIR / f'logreg_{horizon}.joblib'}\n")


if __name__ == "__main__":
    main()
