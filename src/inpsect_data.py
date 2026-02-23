import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load a small data 
df = pd.read_csv(
    "data/raw/hand_open.csv",
    skiprows=8,
    header=None,
    nrows=5000,
    low_memory=False
)

print("Shape:", df.shape)

# Column 0 is time
time = df.iloc[:, 0].values

# Inspect first 20 columns
print("\nFirst 20 columns preview:")
print(df.iloc[:, :20].head())


odd_cols = df.iloc[:, 1::2]   # 1,3,5,7,...
even_cols = df.iloc[:, 2::2]  # 2,4,6,8,...

print("\nOdd columns shape:", odd_cols.shape)
print("Even columns shape:", even_cols.shape)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time[:2000], odd_cols.iloc[:2000, 0])
plt.title("Odd Column Example (likely EMG)")

plt.subplot(2, 1, 2)
plt.plot(time[:2000], even_cols.iloc[:2000, 0])
plt.title("Even Column Example (likely Motion)")

plt.tight_layout()
plt.show()


# print the stats 
print("\nOdd column mean/std:",
      odd_cols.iloc[:, 0].mean(),
      odd_cols.iloc[:, 0].std())

print("Even column mean/std:",
      even_cols.iloc[:, 0].mean(),
      even_cols.iloc[:, 0].std())