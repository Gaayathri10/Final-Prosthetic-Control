"""
windowing.py
-------------
Imported by build_dataset.py (classification) and usable by build_regression_dataset.py.

create_windows(): slides a window over a filtered EMG array and attaches a string label.
Used for the classification pipeline (gesture classes → LDA/SVM baseline).
"""

import numpy as np


def create_windows(filtered_emg: np.ndarray,
                   label: str,
                   window_size: int = 200,
                   step_size: int   = 40) -> tuple:
    """
    Slide a window over filtered EMG data and produce labelled windows.

    Args:
        filtered_emg : [n_samples × n_channels]  pre-filtered EMG array
        label        : string label for all windows from this file (e.g. "hand_open")
        window_size  : number of samples per window  (default 200 ≈ 159 ms @ 1259 Hz)
        step_size    : hop between windows in samples (default 40 → 80% overlap)

    Returns:
        X : np.ndarray  [n_windows × window_size × n_channels]  float32
        y : np.ndarray  [n_windows]                              object (string labels)
    """
    n_samples, n_channels = filtered_emg.shape
    X, y = [], []

    start = 0
    while start + window_size <= n_samples:
        window = filtered_emg[start: start + window_size]   # [window_size, n_channels]
        X.append(window)
        y.append(label)
        start += step_size

    if not X:
        return np.empty((0, window_size, n_channels), dtype=np.float32), np.array([])

    return (
        np.array(X, dtype=np.float32),  # [n_windows, window_size, n_channels]
        np.array(y, dtype=object),       # [n_windows] string labels
    )
