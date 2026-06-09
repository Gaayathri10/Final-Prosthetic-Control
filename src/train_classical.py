import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves figures without a display
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

from feature_extraction import extract_features

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
RESULTS_DIR   = "results/classical"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)
print("NaNs in X:", np.isnan(X).sum())

# Load label names from the saved encoder if available, otherwise fall back
# to integer strings — handles any number of classes automatically.
encoder_path = os.path.join(PROCESSED_DIR, "label_encoder.pkl")
if os.path.exists(encoder_path):
    le = joblib.load(encoder_path)
    label_names = list(le.classes_)
else:
    label_names = [str(i) for i in sorted(np.unique(y))]
print(f"Classes ({len(label_names)}): {label_names}")

# ── Feature extraction ────────────────────────────────────────────────────────

print("\nExtracting EMG features...")
X_feat = extract_features(X)
print("Feature matrix shape:", X_feat.shape)   # (n_windows, 64)

# ── Train / Val / Test split  70 / 10 / 20 ───────────────────────────────────

X_train, X_temp, y_train, y_temp = train_test_split(
    X_feat, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.667, stratify=y_temp, random_state=42
)
# 0.667 of 0.30 ≈ 0.20 of total  →  final split is 70 / 10 / 20

print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ── Normalise ─────────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.pkl"))

# ── Helper: save confusion matrix ─────────────────────────────────────────────

def save_confusion_matrix(cm, title, filename, cmap):
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap=cmap, xticks_rotation=45, ax=ax, colorbar=False)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ── LDA ───────────────────────────────────────────────────────────────────────

print("\n── Training LDA ──")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

y_pred_val_lda = lda.predict(X_val)
y_pred_lda     = lda.predict(X_test)

lda_val_acc  = accuracy_score(y_val,  y_pred_val_lda)
lda_test_acc = accuracy_score(y_test, y_pred_lda)

print(f"  Val  accuracy : {lda_val_acc:.4f}")
print(f"  Test accuracy : {lda_test_acc:.4f}")
print(classification_report(y_test, y_pred_lda,
                             target_names=label_names, zero_division=0))

cm_lda = confusion_matrix(y_test, y_pred_lda)
save_confusion_matrix(cm_lda, f"LDA Confusion Matrix  (Test Acc = {lda_test_acc:.2%})",
                      "lda_confusion_matrix.png", "Blues")

joblib.dump(lda, os.path.join(RESULTS_DIR, "lda_model.pkl"))

# ── SVM with GridSearchCV ─────────────────────────────────────────────────────

print("\n── Tuning SVM ──")
param_grid = {
    "C":      [0.1, 1, 10, 100],
    "gamma":  ["scale", 0.01, 0.001, 0.0001],
    "kernel": ["rbf"]
}

grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best SVM params:", grid.best_params_)

best_svm = grid.best_estimator_

y_pred_val_svm = best_svm.predict(X_val)
y_pred_svm     = best_svm.predict(X_test)

svm_val_acc  = accuracy_score(y_val,  y_pred_val_svm)
svm_test_acc = accuracy_score(y_test, y_pred_svm)

print(f"  Val  accuracy : {svm_val_acc:.4f}")
print(f"  Test accuracy : {svm_test_acc:.4f}")
print(classification_report(y_test, y_pred_svm,
                             target_names=label_names, zero_division=0))

cm_svm = confusion_matrix(y_test, y_pred_svm)
save_confusion_matrix(cm_svm, f"SVM Confusion Matrix  (Test Acc = {svm_test_acc:.2%})",
                      "svm_confusion_matrix.png", "Greens")

joblib.dump(best_svm, os.path.join(RESULTS_DIR, "svm_model.pkl"))

# ── Side-by-side accuracy bar chart ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
models  = ["LDA", "SVM"]
accs    = [lda_test_acc, svm_test_acc]
colors  = ["#4c8fcb", "#4cb87a"]
bars    = ax.bar(models, [a * 100 for a in accs], color=colors, width=0.4, edgecolor="black")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5, f"{acc:.2%}", ha="center", fontsize=11)
ax.set_ylim(0, 105)
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("LDA vs SVM — Classification Accuracy")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path = os.path.join(RESULTS_DIR, "lda_vs_svm_accuracy.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n══ Results Summary ══")
print(f"  LDA  test accuracy : {lda_test_acc:.4f} ({lda_test_acc:.2%})")
print(f"  SVM  test accuracy : {svm_test_acc:.4f} ({svm_test_acc:.2%})")
print(f"  Best SVM params    : {grid.best_params_}")
print(f"\nAll outputs saved to: {RESULTS_DIR}/")
