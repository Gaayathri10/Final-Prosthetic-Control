{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a39bf157",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from scipy.signal import butter, filtfilt, iirnotch\n",
    "\n",
    "#RCNN preprocessing pipeline \n",
    "#Keep it as raw, only to remove noise and artifacts, but not to rectify, smooth or extract features \n",
    "# CNN layers are designed to learn features automatically \n",
    "\n",
    "# 4th order butterworth bandpass filter (20-450 Hz) which isolates the physiologically relevant EMG frequency band \n",
    "# butterworth is chosen b/c it has a maximally flat passband \n",
    "# zero-phase filtering (filtfilt) to avoid phase delay which is important to preserve temporal dynamics \n",
    "\n",
    "def bandpass(signal, fs, low=20, high=450, order=4):\n",
    "    nyq = fs / 2  # Nyquist frequency\n",
    "\n",
    "    low_norm = low / nyq\n",
    "    high_norm = high / nyq\n",
    "\n",
    "    b, a = butter(order, [low_norm, high_norm], btype='band')\n",
    "\n",
    "    # axis=0 filtering happens along time dimension\n",
    "    return filtfilt(b, a, signal, axis=0)\n",
    "\n",
    "\n",
    "# 60 Hz notch filter to suppress powerline interference\n",
    "# Q=30 keeps the notch narrow to avoid distorting nearby EMG frequencies\n",
    "# fs directly in iirnotch (cleaner and less error-prone)\n",
    "\n",
    "def notch(signal, fs, freq=60, Q=30):\n",
    "    b, a = iirnotch(freq, Q, fs)\n",
    "    return filtfilt(b, a, signal, axis=0)\n",
    "\n",
    "# main RCNN preprocessing function:\n",
    "# remove DC offset, baseline drift, 60 Hz notch, and return filtering EMG for windowing\n",
    "\n",
    "\n",
    "def preprocess_rcnn(df, fs):\n",
    "    \n",
    "    # convert DataFrame to numpy array (samples x channels)\n",
    "    emg = df.values.astype(np.float32)\n",
    "\n",
    "    # removes DC offset/ channel to stabilize filtering\n",
    "    emg = emg - np.mean(emg, axis=0)\n",
    "\n",
    "    # remove motion artifacts + high-frequency noise\n",
    "    emg = bandpass(emg, fs)\n",
    "\n",
    "    # remove powerline interference\n",
    "    emg = notch(emg, fs)\n",
    "\n",
    "    return emg"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "emg_dl",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
