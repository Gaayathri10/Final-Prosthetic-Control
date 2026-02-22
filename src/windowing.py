import numpy as np


def create_windows(signal, label, window_size=200, step_size=80):
    
    # (samples, channels ) 
    # and it returns (no. of windows, window size, and channels ) in x, and in y (num_windows, ...)
 
    X = []
    y = []

    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]
        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)