import os
import pandas as pd

def build_final_user_files(unified_folder='final_user_files', data_folder='.', output_folder='final_user_csvs'):
    os.makedirs(output_folder, exist_ok=True)

    print("📦 Building final unified train/test files per user...")

    for fname in os.listdir(unified_folder):
        if not fname.startswith('unified_features_uid_') or not fname.endswith('.csv'):
            continue

        uid = fname.split('_')[-1].replace('.csv', '')
        unified_path = os.path.join(unified_folder, fname)

        try:
            # Load selected features for this UID
            selected_features = pd.read_csv(unified_path, header=None)[0].tolist()
        except Exception as e:
            print(f"❌ Failed to read unified feature list for UID {uid}: {e}")
            continue

        # Locate all splits for this UID
        train_files = [
            f for f in os.listdir(data_folder)
            if f.startswith(f"selected_train_user_{uid}_split") and f.endswith(".csv")
        ]
        test_files = [
            f for f in os.listdir(data_folder)
            if f.startswith(f"selected_test_user_{uid}_split") and f.endswith(".csv")
        ]

        if not train_files or not test_files:
            print(f"⚠️  UID {uid} — Missing selected split files. Skipping.")
            continue

        train_dfs = []
        test_dfs = []

        for tfile in sorted(train_files):
            tpath = os.path.join(data_folder, tfile)
            df = pd.read_csv(tpath)
            columns = [c for c in selected_features if c in df.columns] + ['sleep_quality']
            train_dfs.append(df[columns])

        for tfile in sorted(test_files):
            tpath = os.path.join(data_folder, tfile)
            df = pd.read_csv(tpath)
            columns = [c for c in selected_features if c in df.columns] + ['sleep_quality']
            test_dfs.append(df[columns])

        try:
            train_final = pd.concat(train_dfs, ignore_index=True)
            test_final = pd.concat(test_dfs, ignore_index=True)

            train_final.to_csv(os.path.join(output_folder, f"train_user_{uid}.csv"), index=False)
            test_final.to_csv(os.path.join(output_folder, f"test_user_{uid}.csv"), index=False)

            print(f"✅ UID {uid}: Final train/test files created.")

        except Exception as e:
            print(f"❌ Failed to merge final files for UID {uid}: {e}")

    print("\n🎯 Final user files ready for modeling.")
