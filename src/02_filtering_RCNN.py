import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

#RCNN preprocessing pipeline 
#Keep it as raw, only to remove noise and artifacts, but not to rectify, smooth or extract features 
# CNN layers are designed to learn features automatically 

# 4th order butterworth bandpass filter (20-450 Hz) which isolates the physiologically relevant EMG frequency band 
# butterworth is chosen b/c it has a maximally flat passband 
# zero-phase filtering (filtfilt) to avoid phase delay which is important to preserve temporal dynamics 

def bandpass(signal, fs, low=20, high=450, order=4):
    nyq = fs / 2  # Nyquist frequency

    low_norm = low / nyq
    high_norm = high / nyq

    b, a = butter(order, [low_norm, high_norm], btype='band')

    # axis=0 filtering happens along time dimension
    return filtfilt(b, a, signal, axis=0)


# 60 Hz notch filter to suppress powerline interference
# Q=30 keeps the notch narrow to avoid distorting nearby EMG frequencies
# fs directly in iirnotch (cleaner and less error-prone)

def notch(signal, fs, freq=60, Q=30):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal, axis=0)

# main RCNN preprocessing function:
# remove DC offset, baseline drift, 60 Hz notch, and return filtering EMG for windowing


def preprocess_rcnn(df, fs):
    
    # convert DataFrame to numpy array (samples x channels)
    emg = df.values.astype(np.float32)

    # removes DC offset/ channel to stabilize filtering
    emg = emg - np.mean(emg, axis=0)

    # remove motion artifacts + high-frequency noise
    emg = bandpass(emg, fs)

    # remove powerline interference
    emg = notch(emg, fs)

    return emg