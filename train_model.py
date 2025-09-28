# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

DATA_FILE = "gym_data_synthetic.csv"
MODEL_FILE = "next_weight_model.joblib"

def prepare_features(df):
    df = df.copy()
    # Ensure chronological order
    df.sort_values(['user_id','date'], inplace=True)

    # Create lag features: previous weight, previous volume, session_index
    df['session_index'] = df.groupby(['user_id','exercise']).cumcount()
    df['prev_weight'] = df.groupby(['user_id','exercise'])['weight'].shift(1).fillna(df['weight'])
    df['prev_volume'] = df.groupby(['user_id','exercise'])['volume'].shift(1).fillna(df['volume'])
    df['prev_rpe'] = df.groupby(['user_id','exercise'])['rpe'].shift(1).fillna(df['rpe'])
    # Target: next session's weight
    df['next_weight'] = df.groupby(['user_id','exercise'])['weight'].shift(-1)
    df = df.dropna(subset=['next_weight'])

    # One-hot encode exercise
    exercises = pd.get_dummies(df['exercise'], prefix='ex')
    X = pd.concat([
        df[['user_id','session_index','sets','reps','weight','rpe','rest_sec','volume','prev_weight','prev_volume','prev_rpe']],
        exercises
    ], axis=1)
    y = df['next_weight']
    return X, y

def train_and_save(data_file=DATA_FILE, model_file=MODEL_FILE):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found. Run data_generator.py to create it.")
    df = pd.read_csv(data_file, parse_dates=['date'])
    X, y = prepare_features(df)

    # Simple split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Trained RandomForest. Test MAE: {mae:.2f} kg")

    joblib.dump({'model': model, 'feature_columns': X.columns.tolist()}, model_file)
    print(f"Saved model to {model_file}")
    return model, X.columns.tolist()

if __name__ == "__main__":
    m, cols = train_and_save()
