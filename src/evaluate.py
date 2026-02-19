from sklearn.model_selection import cross_val_score, KFold


def evaluate_model(pipeline, X, y):
    """Run 5-Fold CV and return (mean_r2, std_r2)."""
    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    return scores.mean(), scores.std()
