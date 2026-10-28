"""Tune a random forest using ordered time-series cross-validation."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

DATA = Path('data/processed')
OUT = Path('models/models')


def main():
    # The forest is invariant to feature scaling. Model selection uses unscaled
    # features to avoid leaking global training-set scaling stats across CV folds.
    x_train = pd.read_csv(DATA/'X_train.csv')
    y_train = pd.read_csv(DATA/'y_train.csv').iloc[:, 0]
    search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        {'n_estimators': [100, 200], 'max_depth': [None, 12], 'min_samples_leaf': [1, 3]},
        scoring='neg_mean_squared_error',
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
        refit=True,
        error_score='raise',
    )
    search.fit(x_train, y_train)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({'best_params': search.best_params_, 'cv_rmse': float((-search.best_score_) ** 0.5)}, OUT/'best_params.pkl')
    print(f'GRID SEARCH OK: {search.best_params_}; CV RMSE={(-search.best_score_) ** 0.5:.5f}')


if __name__ == '__main__':
    main()
