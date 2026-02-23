import os
import numpy as np
import pandas as pd

from preprocess import preprocess_emg

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

FS = 1259.26
WINDOW_SIZE = 200
STEP_SIZE = 80
NUM_EMG_CHANNELS = 8  # use first 8 EMG channels


def build_regression_dataset():

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X_all = []
    y_all = []

    for filename in os.listdir(RAW_DIR):

        if not filename.endswith(".csv"):
            continue

        print(f"\nProcessing: {filename}")

        filepath = os.path.join(RAW_DIR, filename)

        df = pd.read_csv(
            filepath,
            skiprows=8,
            header=None,
            low_memory=False
        )

        # Convert everything to numeric
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()

        if len(df) < WINDOW_SIZE:
            print("Skipping file (too short after cleaning)")
            continue

        emg_all = df.iloc[:, 1::2].values   # odd columns
        motion_all = df.iloc[:, 2::2].values  # even columns

        if emg_all.shape[1] < NUM_EMG_CHANNELS:
            print("Skipping file (not enough EMG channels)")
            continue

        # Select first 8 EMG channels
        emg = emg_all[:, :NUM_EMG_CHANNELS]

        # Select first motion channel as regression target
        motion = motion_all[:, 0]

        # Preprocess EMG
        emg = preprocess_emg(emg.astype(np.float32), FS)
        emg = np.nan_to_num(emg)

        # Windowing
        for start in range(0, len(emg) - WINDOW_SIZE, STEP_SIZE):

            end = start + WINDOW_SIZE

            window_emg = emg[start:end]

            # Regression target = last motion value in window
            target = motion[end - 1]

            X_all.append(window_emg)
            y_all.append(target)

    if len(X_all) == 0:
        print("\nNo usable data found.")
        return

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)

    print("\nFinal shapes:")
    print("X:", X_all.shape)
    print("y:", y_all.shape)

    np.save(os.path.join(PROCESSED_DIR, "X_reg.npy"), X_all)
    np.save(os.path.join(PROCESSED_DIR, "y_reg.npy"), y_all)

    print("\nRegression dataset saved to data/processed/")


