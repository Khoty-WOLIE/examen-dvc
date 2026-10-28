"""Train the selected model on the full training period."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA = Path('data/processed')
OUT = Path('models/models')


def main():
    x_train = pd.read_csv(DATA/'X_train.csv')
    y_train = pd.read_csv(DATA/'y_train.csv').iloc[:, 0]
    tuning = joblib.load(OUT/'best_params.pkl')
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **tuning['best_params'])
    model.fit(x_train, y_train)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT/'trained_model.joblib')
    print(f'TRAIN OK: fitted with {len(x_train)} observations')


if __name__ == '__main__':
    main()
