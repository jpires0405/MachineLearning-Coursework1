import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from features import get_preprocessor
from evaluate import evaluate_model

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUBMISSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")
SUBMISSION_FILE = "CW1_submission_23115639.csv"


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
    preprocessor = get_preprocessor(X_train.columns.tolist())
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(random_state=123)),
    ])

    # --- Cross-validation ---
    mean_r2, std_r2 = evaluate_model(pipeline, X_train, y_train)
    print(f"CV Mean R²: {mean_r2:.6f}")
    print(f"CV Std Dev: {std_r2:.6f}")

    # --- Refit on full training set and predict ---
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(test_df)

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
