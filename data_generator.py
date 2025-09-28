# data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_gym_data(n_users=50, n_sessions_per_user=40, seed=42):
    np.random.seed(seed)
    rows = []
    exercises = ['bench_press', 'squat', 'deadlift', 'overhead_press', 'barbell_row']

    for user_id in range(1, n_users+1):
        start_date = datetime(2024,1,1) + timedelta(days=int(np.random.rand()*365))
        for s in range(n_sessions_per_user):
            date = start_date + timedelta(days=int(s* (np.random.choice([1,2,3]))))
            exercise = np.random.choice(exercises, p=[0.25,0.25,0.2,0.15,0.15])
            # previous progress trend
            base_weight = {
                'bench_press': 60,
                'squat': 80,
                'deadlift': 100,
                'overhead_press': 40,
                'barbell_row': 55
            }[exercise]
            # per-user strength variance
            user_strength = base_weight + np.random.normal(0, 10)
            # session variation: weight and reps
            weight = max(5, round(user_strength + np.random.normal(0, 6) + (s * 0.2), 1))
            reps = int(np.clip(np.round(np.random.normal(8, 2)), 1, 20))
            sets = int(np.clip(np.round(np.random.choice([3,4,5], p=[0.6,0.3,0.1])), 1, 6))
            rpe = round(np.clip(np.random.normal(7, 1.0), 5, 10), 1)  # rate of perceived exertion
            rest_sec = int(np.clip(np.random.normal(120,30), 30, 300))
            # compute volume
            volume = weight * reps * sets
            rows.append({
                'user_id': user_id,
                'date': date,
                'exercise': exercise,
                'sets': sets,
                'reps': reps,
                'weight': weight,
                'rpe': rpe,
                'rest_sec': rest_sec,
                'volume': volume
            })
    df = pd.DataFrame(rows)
    df.sort_values(['user_id','date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == "__main__":
    df = generate_synthetic_gym_data()
    print("Generated", len(df), "rows")
    df.to_csv("gym_data_synthetic.csv", index=False)
    print("Saved as gym_data_synthetic.csv")
