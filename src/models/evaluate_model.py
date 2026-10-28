"""Evaluate once on held-out future observations; store predictions and scores."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path('data/processed')
MODEL = Path('models/models/trained_model.joblib')
METRICS = Path('metrics')


def main():
    x_test = pd.read_csv(DATA/'X_test.csv')
    y_test = pd.read_csv(DATA/'y_test.csv').iloc[:, 0]
    model = joblib.load(MODEL)
    pred = model.predict(x_test)
    scores = {
        'mae': float(mean_absolute_error(y_test, pred)),
        'rmse': float(mean_squared_error(y_test, pred) ** 0.5),
        'r2': float(r2_score(y_test, pred)),
        'n_test': int(len(y_test)),
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'y_true': y_test.to_numpy(), 'y_pred': pred}).to_csv(DATA/'predictions.csv', index=False)
    (METRICS/'scores.json').write_text(json.dumps(scores, indent=2), encoding='utf-8')
    assert len(pred) == len(y_test)
    print('EVALUATE OK:', json.dumps(scores, sort_keys=True))


if __name__ == '__main__':
    main()
