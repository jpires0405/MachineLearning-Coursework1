import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold

from features import get_preprocessor

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUBMISSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")
SUBMISSION_FILE = "CW1_submission_23115639.csv"

PARAM_DISTRIBUTIONS = {
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__max_iter": [100, 200, 300, 500],
    "model__max_depth": [3, 5, 10, None],
    "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
}


def main():
    # --- Load data ---
    train_df = pd.read_csv(os.path.join(DATA_DIR, "CW1_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "CW1_test.csv"))

    assert "outcome" in train_df.columns, "Train set must contain 'outcome'"
    assert "outcome" not in test_df.columns, "Test set must NOT contain 'outcome'"

    # --- Prepare features / target ---
    X_train = train_df.drop(columns=["outcome"])
    y_train = train_df["outcome"]

    # --- Build pipeline ---
    pipeline = Pipeline([
        ("preprocessor", get_preprocessor(X_train.columns.tolist())),
        ("model", HistGradientBoostingRegressor(random_state=123)),
    ])

    # --- Hyperparameter tuning ---
    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=15,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        random_state=123,
    )
    search.fit(X_train, y_train)

    print(f"Best CV Mean R²: {search.best_score_:.6f}")
    print(f"Best params: {search.best_params_}")

    # --- Predict with best estimator ---
    predictions = search.best_estimator_.predict(test_df)

    assert len(predictions) == 1000, (
        f"Expected 1000 predictions, got {len(predictions)}"
    )

    # --- Save submission ---
    submission_df = pd.DataFrame({"yhat": predictions})
    assert "yhat" in submission_df.columns

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    submission_df.to_csv(
        os.path.join(SUBMISSIONS_DIR, SUBMISSION_FILE), index=False
    )
    print(f"Submission saved to submissions/{SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
