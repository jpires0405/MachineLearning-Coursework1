"""visualize.py – Generate EDA and model-evaluation figures for the report.

Run from the repository root:
    python MachineLearning-Coursework1/src/visualize.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Allow sibling imports (features.py lives in the same src/ directory)
sys.path.insert(0, os.path.dirname(__file__))
from features import get_preprocessor, CATEGORICAL_FEATURES, TARGET  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "CW1_train.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_correlation_heatmap(df: pd.DataFrame, n_top: int = 10) -> None:
    """Save a heatmap of the top *n_top* numeric features by |correlation| with outcome."""
    numeric_df = df.drop(columns=CATEGORICAL_FEATURES)
    corr_with_target = numeric_df.corr()[TARGET].drop(TARGET).abs()
    top_features = corr_with_target.nlargest(n_top).index.tolist()

    subset = numeric_df[[TARGET] + top_features]
    corr_matrix = subset.corr()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Correlation Matrix: Top 10 Features vs. Outcome", fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pred_vs_actual(df: pd.DataFrame) -> None:
    """Train the tuned HistGBR pipeline on 80% of data and save a predicted-vs-actual plot."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=123)

    pipeline = Pipeline([
        ("preprocessor", get_preprocessor(X_tr.columns.tolist())),
        (
            "model",
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_depth=3,
                l2_regularization=0.1,
                random_state=123,
            ),
        ),
    ])
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)

    lims = [
        min(float(y_te.min()), float(y_pred.min())),
        max(float(y_te.max()), float(y_pred.max())),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_te, y_pred, alpha=0.3, s=8, color="steelblue", rasterized=True)
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual", fontsize=10)
    ax.set_ylabel("Predicted", fontsize=10)
    ax.set_title("Predicted vs. Actual (20% hold-out)", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "pred_vs_actual.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows from {DATA_PATH}")
    plot_correlation_heatmap(df)
    plot_pred_vs_actual(df)
    print("All figures generated.")


if __name__ == "__main__":
    main()
