import os
import numpy as np
import pandas as pd
from filtering_RCNN import preprocess_rcnn

fs = 1359.26

raw_folder = "data/raw"
processed_folder = "data/processed"

os.makedirs(processed_folder, exist_ok=True)

for filename in os.listdir(raw_folder):

    if filename.endswith(".csv"):

        print("Processing:", filename)

        file_path = os.path.join(raw_folder, filename)
        save_path = os.path.join(
            processed_folder,
            filename.replace(".csv", ".npy")
        )

        first_pass = True
        total_rows = 0

        # First pass: determine total rows after cleaning
        for chunk in pd.read_csv(
            file_path,
            skiprows=8,
            header=None,
            sep=',',
            engine='python',
            chunksize=500000
        ):
            emg_df = chunk.iloc[:, 1:]
            emg_df = emg_df.apply(pd.to_numeric, errors='coerce')
            emg_df = emg_df.dropna(how='all')
            total_rows += emg_df.shape[0]

        if total_rows == 0:
            print("No valid data found.")
            continue

        print("Total rows after cleaning:", total_rows)

        # Create memory-mapped file
        memmap_array = np.lib.format.open_memmap(
            save_path,
            mode='w+',
            dtype=np.float32,
            shape=(total_rows, 107)
        )

        current_index = 0

        # Second pass: process + write directly to disk
        for chunk in pd.read_csv(
            file_path,
            skiprows=8,
            header=None,
            sep=',',
            engine='python',
            chunksize=500000
        ):

            emg_df = chunk.iloc[:, 1:]
            emg_df = emg_df.apply(pd.to_numeric, errors='coerce')
            emg_df = emg_df.dropna(how='all')

            if emg_df.shape[0] < 30:
                continue

            emg_df = emg_df.astype(np.float32)
            emg_processed = preprocess_rcnn(emg_df, fs)

            rows = emg_processed.shape[0]

            memmap_array[current_index:current_index+rows, :] = emg_processed
            current_index += rows

            del emg_df
            del emg_processed

        print("Saved:", save_path)
        print("--------------------")

print("All gestures successfully processed.")