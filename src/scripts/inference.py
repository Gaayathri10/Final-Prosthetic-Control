"""
inference.py
────────────
Run the trained RCNN model on new EMG data.

Two modes:
  1. Batch evaluation on saved test numpy arrays  (default)
  2. Single-window prediction                     (call predict_window())

Usage:
  python inference.py                  # evaluates on X_reg.npy test split
  python inference.py --window         # demo with a random window
"""

import os
import argparse
import numpy as np
import joblib
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from rcnn import EMG_RCNN
from preprocess import preprocess_emg

# ── Config (must match train_rcnn.py) ────────────────────────────────────────

PROCESSED_DIR = "data/processed"
RESULTS_DIR   = "results/rcnn"

NUM_CHANNELS = 8
HIDDEN_SIZE  = 128
OUTPUT_DIM   = 1
WINDOW_SIZE  = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the best saved RCNN weights."""
    weights_path = os.path.join(RESULTS_DIR, "best_rcnn.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"\nNo trained model found at '{weights_path}'.\n"
            "Run train_rcnn.py first:\n"
            "  python train_rcnn.py"
        )
    model = EMG_RCNN(
        input_channels=NUM_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        output_dim=OUTPUT_DIM
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {weights_path}")
    return model


def predict_window(emg_window: np.ndarray, model, scaler) -> float:
    """
    Predict joint angle for a single EMG window.

    Args:
        emg_window : raw (unfiltered) EMG array of shape (WINDOW_SIZE, NUM_CHANNELS)
        model      : loaded EMG_RCNN model
        scaler     : fitted StandardScaler from training

    Returns:
        Predicted joint angle (denormalised, original units)
    """
    assert emg_window.shape == (WINDOW_SIZE, NUM_CHANNELS), \
        f"Expected ({WINDOW_SIZE}, {NUM_CHANNELS}), got {emg_window.shape}"

    # Preprocess
    from preprocess import preprocess_emg
    filtered = preprocess_emg(emg_window.astype(np.float32), fs=1259.26)
    filtered = np.nan_to_num(filtered)

    # Scale
    scaled = scaler.transform(filtered)              # (200, 8)

    # To tensor → (1, 200, 8)
    x = torch.tensor(scaled[np.newaxis], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_norm = model(x).cpu().numpy().squeeze()

    # Denormalise using saved y stats
    y_mean, y_std = load_y_stats()
    return float(pred_norm * y_std + y_mean)


def load_y_stats():
    """Load the label mean/std saved during training."""
    stats_path = os.path.join(RESULTS_DIR, "y_stats.npy")
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        return float(stats[0]), float(stats[1])
    # Fallback: recompute from full dataset
    y = np.load(os.path.join(PROCESSED_DIR, "y_reg.npy"))
    return float(y.mean()), float(y.std() + 1e-8)


def evaluate_test_set(model, scaler):
    """Evaluate on the held-out test split and print metrics."""
    X = np.load(os.path.join(PROCESSED_DIR, "X_reg.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y_reg.npy"))

    y_mean, y_std = y.mean(), y.std() + 1e-8
    y_norm = (y - y_mean) / y_std

    # Reproduce same split as training
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y_norm, test_size=0.30, random_state=42)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.667, random_state=42)

    N, T, C = X_test.shape
    X_test_sc = scaler.transform(X_test.reshape(-1, C)).reshape(N, T, C)

    x_tensor = torch.tensor(X_test_sc, dtype=torch.float32).to(DEVICE)

    all_preds = []
    batch = 256
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch):
            pred = model(x_tensor[i:i+batch]).cpu().numpy().squeeze()
            all_preds.extend(pred if pred.ndim > 0 else [pred.item()])

    all_preds   = np.array(all_preds)   * y_std + y_mean
    all_targets = np.array(y_test)      * y_std + y_mean

    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae  = np.mean(np.abs(all_preds - all_targets))
    r, _ = pearsonr(all_targets, all_preds)
    nrmse = rmse / (all_targets.max() - all_targets.min() + 1e-8)

    print("\n══ Inference Results ══")
    print(f"  Test samples : {len(all_targets)}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  MAE          : {mae:.4f}")
    print(f"  Pearson R    : {r:.4f}")
    print(f"  NRMSE        : {nrmse*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", action="store_true",
                        help="Demo: predict on one random window")
    args = parser.parse_args()

    model  = load_model()
    scaler = joblib.load(os.path.join(RESULTS_DIR, "rcnn_scaler.pkl"))

    if args.window:
        # Generate a random window for demo purposes
        demo_window = np.random.randn(WINDOW_SIZE, NUM_CHANNELS).astype(np.float32) * 0.02
        pred = predict_window(demo_window, model, scaler)
        print(f"Demo prediction (random window): {pred:.4f}")
    else:
        evaluate_test_set(model, scaler)
