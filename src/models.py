from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def get_models():
    """Return a dictionary of instantiated regression models."""
    return {
        "Ridge": Ridge(random_state=123),
        "RandomForest": RandomForestRegressor(random_state=123, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=123),
    }
