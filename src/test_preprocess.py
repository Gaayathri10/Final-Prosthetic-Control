import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from preprocess import preprocess_emg

fs = 1359.26

chunks = []
chunk_size = 200000  

for chunk in pd.read_csv(
    "data/raw/hand_open.csv",
    skiprows=8,
    header=None,
    chunksize=chunk_size,
    low_memory=False
):
    chunk = chunk.iloc[:, 1:]  # remove time column
    chunk = chunk.apply(pd.to_numeric, errors='coerce')
    chunk = chunk.dropna()
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

raw = df.values.astype(np.float32)
filtered = preprocess_emg(raw, fs)

plt.figure(figsize=(12,6))
plt.plot(raw[:2000, 0], label="Raw", alpha=0.6)
plt.plot(filtered[:2000, 0], label="Filtered", alpha=0.8)
plt.legend()
plt.show()