import numpy as np


def mean_absolute_value(window):
    return np.mean(np.abs(window), axis=0)


def root_mean_square(window):
    return np.sqrt(np.mean(window**2, axis=0))


def waveform_length(window):
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def zero_crossings(window, threshold=1e-6):
    zc = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        crossings = np.where(
            (signal[:-1] * signal[1:] < 0) &
            (np.abs(signal[:-1] - signal[1:]) > threshold)
        )[0]
        zc.append(len(crossings))
    return np.array(zc)


def slope_sign_changes(window, threshold=1e-6):
    ssc = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        diff1 = signal[1:-1] - signal[:-2]
        diff2 = signal[1:-1] - signal[2:]
        changes = np.where(
            (diff1 * diff2 > 0) &
            (np.abs(diff1) > threshold) &
            (np.abs(diff2) > threshold)
        )[0]
        ssc.append(len(changes))
    return np.array(ssc)


def variance(window):
    return np.var(window, axis=0)


def integrated_emg(window):
    return np.sum(np.abs(window), axis=0)


def log_detector(window):
    return np.exp(np.mean(np.log(np.abs(window) + 1e-6), axis=0))


def extract_features(X_windows):
    feature_list = []

    for window in X_windows:

        mav = mean_absolute_value(window)
        rms = root_mean_square(window)
        wl = waveform_length(window)
        zc = zero_crossings(window)
        ssc = slope_sign_changes(window)
        var = variance(window)
        iemg = integrated_emg(window)
        logd = log_detector(window)

        features = np.concatenate([
            mav, rms, wl, zc,
            ssc, var, iemg, logd
        ])

        feature_list.append(features)

    return np.array(feature_list)