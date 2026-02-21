import pandas as pd
import numpy as np

file_path = r"C:\Users\gaaya\OneDrive\Desktop\Thesis\Final-Prosthetic-Control\data\raw\hand_open.csv"

df = pd.read_csv(
    file_path,
    skiprows=8,
    header=None,
    usecols=[0],   # load time column
    nrows=2000     # only first 2000 rows
)

time = df.iloc[:, 0].values 
dt = np.mean(np.diff(time))
fs = 1 / dt

print("Sampling rate:", fs, "Hz")

#sampling rate = 1259.26 Hz 