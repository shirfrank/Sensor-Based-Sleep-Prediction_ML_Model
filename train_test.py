import os
import pandas as pd
import numpy as np

def generate_multiple_splits_per_user(min_train_days=3, max_splits=5):
    print("=== Generating Multiple Train/Test Splits per User ===")

    all_dfs = []
    for name in ['A', 'B', 'C']:
        path = f'features_session_{name}.csv'
        print(f"📂 Loading {path}")
        df = pd.read_csv(path)
        if 'sleep_score' in df.columns:
            df = df.rename(columns={'sleep_score': 'sleep_quality'})
        df['session'] = name
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.dropna(subset=['sleep_quality'])
    users = full_df['uid'].unique()
    print(f"🔍 Found {len(users)} unique users")

    for uid in users:
        df_user = full_df[full_df['uid'] == uid].copy()
        df_user = df_user.sort_values('label_date')

        if len(df_user) < min_train_days + 2:
            print(f"⏩ Skipping UID {uid} — Not enough samples ({len(df_user)})")
            continue

        num_splits = min(max_splits, len(df_user) - min_train_days)
        print(f"🔧 UID {uid}: Creating {num_splits} splits")

        for i in range(num_splits):
            split_idx = min_train_days + i
            train_df = df_user.iloc[:split_idx]
            test_df = df_user.iloc[[split_idx]]

            train_df.to_csv(f"train_user_{uid}_split{i}.csv", index=False)
            test_df.to_csv(f"test_user_{uid}_split{i}.csv", index=False)

    print("🎯 Splits generation complete.")
