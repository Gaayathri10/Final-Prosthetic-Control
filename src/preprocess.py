import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal, fs, low=20, high=450, order=4):
    
    #4th order zero phase butterworth bandpass filter (20–450 Hz)

    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq

    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal, fs, freq=60, Q=30):
    
    # 60 Hz notch filter to remove powerline interference

    nyq = fs / 2
    w0 = freq / nyq

    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, signal, axis=0)


def preprocess_emg(emg, fs):
    
    emg = emg.astype(np.float32)

    emg = notch_filter(emg, fs)
    emg = bandpass_filter(emg, fs)

    return emg
