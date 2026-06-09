"""
train_rcnn.py
--------------
Trains the EMG_RCNN model on the regression dataset built by build_regression_dataset.py.

Saves to results/rcnn/:
  best_rcnn.pth      — best validation-loss weights
  rcnn_scaler.pkl    — fitted StandardScaler (needed by inference.py)
  y_stats.npy        — [mean, std] of y_reg.npy (for denormalisation)
  training_curve.png — loss curves

Usage:
  python build_regression_dataset.py   # first — generates data/processed/X_reg.npy
  python train_rcnn.py                 # then train

Hyperparameters (thesis Section 4.7.3):
  Optimizer  : Adam
  Batch size : 64
  Max epochs : 200
  Early stop : patience=15 on val loss
  LR schedule: CosineAnnealingLR after epoch 50
  Split      : 70% train / 20% val / 10% test (trial-level, no leakage)
"""

import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rcnn import EMG_RCNN

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
RESULTS_DIR   = "results/rcnn"

FS           = 1259.26      # your empirically measured sampling rate
NUM_CHANNELS = 6            # Delsys Trigno Avanti — 6 sensors per thesis Table 4.41
HIDDEN_SIZE  = 128
OUTPUT_DIM   = 1            # single regression target (ACC X as joint-angle proxy)
WINDOW_SIZE  = 200

BATCH_SIZE   = 64
MAX_EPOCHS   = 200
PATIENCE     = 15
LR           = 1e-3
LR_WARMUP_EPOCH = 50        # cosine annealing starts after this epoch
LAMBDA_SMOOTH   = 0.01      # smoothness regularisation weight

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Loss (MSE + smoothness from thesis 4.7.2) ─────────────────────────────────

def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  lambda_s: float = LAMBDA_SMOOTH) -> torch.Tensor:
    mse = nn.functional.mse_loss(pred.squeeze(-1), target)
    if pred.shape[0] > 1:
        smooth = torch.mean((pred[1:] - pred[:-1]) ** 2)
    else:
        smooth = torch.tensor(0.0, device=pred.device)
    return mse + lambda_s * smooth


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_path = os.path.join(PROCESSED_DIR, "X_reg.npy")
    y_path = os.path.join(PROCESSED_DIR, "y_reg.npy")

    if not os.path.exists(X_path):
        raise FileNotFoundError(
            f"No dataset at '{X_path}'.\nRun:  python build_regression_dataset.py"
        )

    X = np.load(X_path)    # [N, WINDOW_SIZE, N_CHANNELS]
    y = np.load(y_path)    # [N]

    print(f"Loaded  X: {X.shape}   y: {y.shape}")
    print(f"y range : {y.min():.4f} – {y.max():.4f}")
    print(f"Device  : {DEVICE}")

    # ── Normalise targets ──────────────────────────────────────────────────────
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-8)
    y_norm = (y - y_mean) / y_std
    np.save(os.path.join(RESULTS_DIR, "y_stats.npy"), np.array([y_mean, y_std]))

    # ── Train / val / test split (70 / 20 / 10) ───────────────────────────────
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y_norm, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.667, random_state=42
    )
    print(f"Split   : train={len(X_tr)}  val={len(X_val)}  test={len(X_test)}")

    # ── Scale features (StandardScaler on flattened, then reshape) ────────────
    N_tr, T, C = X_tr.shape
    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr.reshape(-1, C)).reshape(N_tr, T, C)
    X_val_sc = scaler.transform(X_val.reshape(-1, C)).reshape(len(X_val), T, C)
    joblib.dump(scaler, os.path.join(RESULTS_DIR, "rcnn_scaler.pkl"))

    # ── Dataloaders ───────────────────────────────────────────────────────────
    def make_loader(X_arr, y_arr, shuffle):
        ds = TensorDataset(
            torch.tensor(X_arr,  dtype=torch.float32),
            torch.tensor(y_arr,  dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(X_tr_sc,  y_tr,  shuffle=True)
    val_loader   = make_loader(X_val_sc, y_val, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EMG_RCNN(
        input_channels=C,            # use actual channel count from data
        hidden_size=HIDDEN_SIZE,
        output_dim=OUTPUT_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS - LR_WARMUP_EPOCH, eta_min=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = combined_loss(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += combined_loss(pred, yb).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # LR schedule kicks in after warmup
        if epoch > LR_WARMUP_EPOCH:
            scheduler.step()

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, "best_rcnn.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stop at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved to {RESULTS_DIR}/best_rcnn.pth")

    # ── Save loss curve ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="train")
        plt.plot(val_losses,   label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("RCNN training curve")
        plt.legend()
        plt.tight_layout()
        curve_path = os.path.join(RESULTS_DIR, "training_curve.png")
        plt.savefig(curve_path, dpi=100)
        plt.close()
        print(f"Loss curve saved to {curve_path}")
    except ImportError:
        print("matplotlib not installed — skipping curve plot")


if __name__ == "__main__":
    train()
