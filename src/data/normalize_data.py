"""Fit the scaler on training rows only, never on the test set."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA = Path('data/processed')
MODEL_DATA = Path('models/data')


def main():
    x_train = pd.read_csv(DATA/'X_train.csv')
    x_test = pd.read_csv(DATA/'X_test.csv')
    if list(x_train.columns) != list(x_test.columns):
        raise ValueError('Train/test columns are different')
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)
    DATA.mkdir(parents=True, exist_ok=True)
    MODEL_DATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_scaled, columns=x_train.columns).to_csv(DATA/'X_train_scaled.csv', index=False)
    pd.DataFrame(test_scaled, columns=x_train.columns).to_csv(DATA/'X_test_scaled.csv', index=False)
    joblib.dump(scaler, MODEL_DATA/'scaler.joblib')
    print(f'NORMALIZE OK: {len(x_train)} train, {len(x_test)} test; scaler fit on train only')


if __name__ == '__main__':
    main()
