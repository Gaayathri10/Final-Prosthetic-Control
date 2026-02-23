import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from feature_extraction import extract_features

# loading the data 

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)

# remove the zeros 

print("NaNs in dataset:", np.isnan(X).sum())

# extracting the features 

print("\nExtracting EMG features...")
X = extract_features(X)

print("Feature shape:", X.shape)  # should be (num_windows, 32)


# spliting 70/10/20

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.33, stratify=y_temp, random_state=42
)

print("\nTrain size:", X_train.shape[0])
print("Val size:", X_val.shape[0])
print("Test size:", X_test.shape[0])


# normalizing 
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# lda 

print("\nTraining LDA...")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

y_pred_lda = lda.predict(X_test)

print("\nLDA Accuracy:", accuracy_score(y_test, y_pred_lda))
print(classification_report(y_test, y_pred_lda, zero_division=0))

# Confusion matrix for LDA
cm_lda = confusion_matrix(y_test, y_pred_lda)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_lda)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("LDA Confusion Matrix")
plt.show()

# svm 

from sklearn.model_selection import GridSearchCV

print("\nTuning SVM...")

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.001, 0.0001],
    "kernel": ["rbf"]
}

svm = SVC()

grid = GridSearchCV(
    svm,
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("Best SVM Params:", grid.best_params_)

best_svm = grid.best_estimator_

y_pred_svm = best_svm.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, zero_division=0))

# Confusion matrix SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp.plot(cmap="Greens", xticks_rotation=45)
plt.title("SVM Confusion Matrix")
plt.show()