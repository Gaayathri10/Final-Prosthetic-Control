import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from preprocess import preprocess_emg
from windowing import create_windows


# config 

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

FS = 1259.26
WINDOW_SIZE = 200
STEP_SIZE = 40
NUM_CHANNELS = 8

# build functions 

def build_dataset():

    print("Current Working Directory:", os.getcwd())
    print("Looking inside:", RAW_DIR)

    if not os.path.exists(RAW_DIR):
        print(" RAW_DIR does not exist.")
        return

    print("Files found:", os.listdir(RAW_DIR))

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X_all = []
    y_all = []

    for filename in os.listdir(RAW_DIR):

        if not filename.endswith(".csv"):
            continue

        print(f"\nProcessing: {filename}")

        filepath = os.path.join(RAW_DIR, filename)
        chunks = []

        for chunk in pd.read_csv(
            filepath,
            skiprows=8,
            header=None,
            chunksize=200000,
            low_memory=False
        ):

            # Remove time column
            chunk = chunk.iloc[:, 1:1 + NUM_CHANNELS]

            # Convert safely
            chunk = chunk.apply(pd.to_numeric, errors='coerce')

            # Drop rows containing ANY NaN
            chunk = chunk.dropna()

            if len(chunk) > 0:
                chunks.append(chunk)

        if len(chunks) == 0:
            print("⚠ Skipping — no usable EMG data")
            continue

        df_clean = pd.concat(chunks, ignore_index=True)
        print("Clean shape:", df_clean.shape)

        if len(df_clean) < WINDOW_SIZE:
            print("⚠ Skipping — too short after cleaning")
            continue

        raw = df_clean.values.astype(np.float32)

        # Preprocess 
        print("Filtering...")
        filtered = preprocess_emg(raw, FS)

        # Remove any NaNs in the filtered 
        filtered = np.nan_to_num(filtered)

        label = filename.replace(".csv", "")

        X, y = create_windows(
            filtered,
            label,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE
        )

        print("Windows created:", len(X))

        if len(X) == 0:
            print("⚠ Skipping — no windows created")
            continue

        X_all.append(X)
        y_all.append(y)

    if len(X_all) == 0:
        print("No data processed. Nothing to save.")
        return

    # concatenate 
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)

    print("\nFinal dataset shape:", X_all.shape)

    # encode labels 
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    # Save the files in data/processed 
    np.save(os.path.join(PROCESSED_DIR, "X.npy"), X_all)
    np.save(os.path.join(PROCESSED_DIR, "y.npy"), y_encoded)

 


if __name__ == "__main__":
    build_dataset()