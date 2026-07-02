"""
main.py
--------
Real-time prosthetic control loop — wired to YOUR existing code:
  preprocess.py   →  filter_emg_offline / causal SOS filter
  rcnn.py         →  EMG_RCNN
  inference.py    →  predict_window()
  delsys_interface.py → EMG stream (mock or real)
  arduino_interface.py → serial to PCA9685

Run demo (no hardware):
    python main.py --mock

Run with hardware:
    python main.py --port COM3

Press Ctrl+C to stop.
"""

import argparse
import time
import numpy as np
import os
import joblib
import torch
from collections import deque

from rcnn import EMG_RCNN
from preprocess import preprocess_emg
from delsys_interface import get_emg_stream, MOCK as DELSYS_MOCK
# from arduino_interface import ArduinoInterface

# ── Config — matches your measured values ────────────────────────────────────
FS            = 1259.26          # your empirically measured sampling rate
WINDOW_SIZE   = 200              # samples per window
STEP_SAMPLES  = 40               # hop size (80% overlap)
NUM_CHANNELS  = 6                # Delsys Trigno Avanti, 6 sensors per thesis
OUTPUT_DIM    = 1                # single regression output (ACC-proxy joint angle)
HIDDEN_SIZE   = 128

RESULTS_DIR   = "results/rcnn"

# How often we sample from the EMG stream per loop iteration
# At 1259 Hz, 20 ms ≈ 25 samples
SAMPLES_PER_TICK = 25
TICK_S = SAMPLES_PER_TICK / FS  # ≈ 0.020 s

# EMA smoothing alpha for servo output
EMA_ALPHA = 0.3

# ── Joint angle → servo degree mapping (output_dim=1 → hand open/close) ──────
# Expand to 4 joints once you have multi-DOF ground truth data
SERVO_MIN_DEG = 0
SERVO_MAX_DEG = 180


# ── EMA helper ────────────────────────────────────────────────────────────────
class EMAFilter:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self._state = None

    def update(self, val: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = val.copy()
        else:
            self._state = self.alpha * val + (1 - self.alpha) * self._state
        return self._state.copy()


# ── RCNN inference (replicates inference.py but for real-time) ────────────────
class RealTimeRCNN:
    def __init__(self, n_channels=NUM_CHANNELS, device="cpu"):
        self.device = torch.device(device)
        self.model = EMG_RCNN(
            input_channels=n_channels,
            hidden_size=HIDDEN_SIZE,
            output_dim=OUTPUT_DIM
        ).to(self.device)
        self.model.eval()

        self.scaler = None
        self.y_mean, self.y_std = 0.0, 1.0

        weights_path = os.path.join(RESULTS_DIR, "best_rcnn.pth")
        scaler_path  = os.path.join(RESULTS_DIR, "rcnn_scaler.pkl")
        stats_path   = os.path.join(RESULTS_DIR, "y_stats.npy")

        if os.path.exists(weights_path):
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            print(f"[RCNN] Loaded weights from {weights_path}")
        else:
            print("[RCNN] WARNING: No weights found — running random-weight demo mode.")

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"[RCNN] Loaded scaler from {scaler_path}")
        else:
            print("[RCNN] WARNING: No scaler found — skipping normalisation.")

        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.y_mean, self.y_std = float(stats[0]), float(stats[1])

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> float:
        """
        window: [WINDOW_SIZE, NUM_CHANNELS]  raw (unfiltered) EMG
        returns: predicted joint angle in original units (denormalised)
        """
        # Filter
        filtered = preprocess_emg(window.astype(np.float32), fs=FS)
        filtered = np.nan_to_num(filtered)

        # Scale
        if self.scaler is not None:
            N, C = filtered.shape
            filtered = self.scaler.transform(filtered)

        x = torch.tensor(filtered[np.newaxis], dtype=torch.float32).to(self.device)
        pred_norm = self.model(x).cpu().numpy().squeeze()
        return float(pred_norm) * self.y_std + self.y_mean


# ── Normalise joint angle to [0,1] for servo ─────────────────────────────────
def angle_to_normalized(angle: float, lo: float, hi: float) -> float:
    return float(np.clip((angle - lo) / (hi - lo + 1e-8), 0.0, 1.0))


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(mock_serial=True, port="COM3"):
    print("=" * 55)
    print(" Continuous Prosthetic Control — Gaayathri Ganesh")
    print("=" * 55)
    print(f"  EMG source   : {'MOCK (synthetic)' if DELSYS_MOCK else 'Delsys Trigno Avanti'}")
    print(f"  Arduino      : {'MOCK' if mock_serial else port}")
    print(f"  FS           : {FS} Hz")
    print(f"  Window       : {WINDOW_SIZE} samples")
    print()

    # 1. EMG stream
    emg_stream = get_emg_stream()
    emg_stream.connect()
    emg_stream.start()

    # 2. RCNN
    model = RealTimeRCNN(n_channels=NUM_CHANNELS)

    # 3. EMA
    ema = EMAFilter(alpha=EMA_ALPHA)

    # 4. Arduino
    # arduino = ArduinoInterface(port=port, mock=mock_serial)
    # arduino.connect()

    # 5. Rolling raw-sample buffer for windowing
    raw_buf = deque(maxlen=WINDOW_SIZE)
    samples_since_last_window = 0

    print("Control loop running. Press Ctrl+C to stop.\n")

    n_preds = 0
    t_start = time.time()

    # Calibrate output range over first 3 seconds of data
    # (you can replace these with values from your training set)
    angle_lo, angle_hi = -1.0, 1.0   # update after first training run

    try:
        while True:
            # Read new raw samples
            raw = emg_stream.read_samples(SAMPLES_PER_TICK)
            for sample in raw:
                raw_buf.append(sample)
            samples_since_last_window += SAMPLES_PER_TICK

            # Every STEP_SAMPLES, run inference
            if samples_since_last_window >= STEP_SAMPLES and len(raw_buf) == WINDOW_SIZE:
                samples_since_last_window = 0
                window = np.array(list(raw_buf), dtype=np.float32)  # [200, 6]

                raw_angle = model.predict(window)
                smooth_angle = ema.update(np.array([raw_angle]))[0]

                # Normalise to [0,1] for servo
                norm = angle_to_normalized(smooth_angle, angle_lo, angle_hi)

                # Send to Arduino — extend to 4-joint array with neutral for unused joints
                # angles_4 = np.array([0.5, 0.5, 0.5, norm], dtype=np.float32)
                # arduino.send_joint_angles(angles_4)

                n_preds += 1
                if n_preds % 20 == 0:
                    elapsed = time.time() - t_start
                    print(f"[{elapsed:5.1f}s] {n_preds/elapsed:.1f} Hz  "
                          f"angle={smooth_angle:.4f}  servo={norm*100:.0f}%")

            time.sleep(TICK_S)

    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
    finally:
        emg_stream.stop()
        # arduino.disconnect()
        print("[Main] Clean shutdown.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock",  action="store_true", default=True,
                        help="Mock serial / no Arduino")
    parser.add_argument("--port",  default="COM3",
                        help="Arduino COM port")
    args = parser.parse_args()
    run(mock_serial=args.mock, port=args.port)
