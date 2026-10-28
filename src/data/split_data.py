"""Split chronologically, keeping the final observations as a holdout test set."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE = Path('data/raw/raw.csv')
DEST = Path('data/processed')
TARGET = 'silica_concentrate'


def main():
    df = pd.read_csv(SOURCE)
    if TARGET not in df or 'date' not in df:
        raise ValueError('Missing date or target column')
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    df = df.sort_values('date', kind='stable').reset_index(drop=True)
    if df.isna().any().any() or df.duplicated().any():
        raise ValueError('Source data contain missing values or duplicate rows')
    features = df.drop(columns=[TARGET, 'date'])
    if not all(pd.api.types.is_numeric_dtype(dt) for dt in features.dtypes):
        raise TypeError('All model features must be numeric')
    split = int(len(df) * 0.8)
    if split < 10 or len(df)-split < 2:
        raise ValueError('Not enough observations to split')
    DEST.mkdir(parents=True, exist_ok=True)
    outputs = {
        'X_train.csv': features.iloc[:split],
        'X_test.csv': features.iloc[split:],
        'y_train.csv': df[[TARGET]].iloc[:split],
        'y_test.csv': df[[TARGET]].iloc[split:],
    }
    for name, part in outputs.items():
        part.to_csv(DEST/name, index=False)
    print(f'SPLIT OK: {len(df)} rows, {split} train, {len(df)-split} test, {features.shape[1]} features')


if __name__ == '__main__':
    main()
