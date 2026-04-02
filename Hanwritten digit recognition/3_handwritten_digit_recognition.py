"""
=========================================
 Project 3: Handwritten Digit Recognition
=========================================
Libraries : scikit-learn, numpy, matplotlib
Algorithm : SVM (RBF kernel)  +  MLP Neural Net  +  Random Forest
Dataset   : sklearn digits (8×8 px) — zero downloads required
            Optionally uses MNIST (28×28) via OpenML if internet available
=========================================
Install   : pip install scikit-learn numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)


# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("Loading sklearn Digits dataset (8×8 grayscale images, 10 classes)...")
digits = load_digits()

X = digits.data          # shape (1797, 64)
y = digits.target        # labels 0-9

print(f"Dataset shape : {X.shape}")
print(f"Classes       : {digits.target_names}")


# ── 2. Visualise Sample Images ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
fig.suptitle("Sample Digit Images (8×8 pixels)", fontsize=12)
for digit in range(10):
    idx = np.where(y == digit)[0][0]
    axes[0, digit].imshow(digits.images[idx], cmap="gray_r")
    axes[0, digit].set_title(str(digit))
    axes[0, digit].axis("off")
    idx2 = np.where(y == digit)[0][1]
    axes[1, digit].imshow(digits.images[idx2], cmap="gray_r")
    axes[1, digit].axis("off")
plt.tight_layout()
plt.savefig("digit_samples.png", dpi=150)
plt.show()
print("Digit samples saved as 'digit_samples.png'")


# ── 3. Train / Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")


# ── 4. Scaling ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# ── 5. Model Training & Evaluation ────────────────────────────────────────────
models = {
    "SVM (RBF)": SVC(kernel="rbf", C=10, gamma=0.01, random_state=42),
    "MLP Neural Net": MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=500,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "preds": y_pred, "acc": acc}
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, digits=3))


# ── 6. Best Model Confusion Matrix ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["acc"])
best_preds = results[best_name]["preds"]

print(f"\nBest model: {best_name}  (Accuracy = {results[best_name]['acc']*100:.2f}%)")

cm = confusion_matrix(y_test, best_preds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Handwritten Digit Recognition", fontsize=13, fontweight="bold")

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=digits.target_names)
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Confusion Matrix – {best_name}")

# Model accuracy bar chart
names = list(results.keys())
accs  = [results[n]["acc"] * 100 for n in names]
bars  = axes[1].bar(names, accs,
                    color=["steelblue", "coral", "mediumseagreen"])
axes[1].set_ylim(85, 101)
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Model Comparison")
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 f"{acc:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("digit_recognition_results.png", dpi=150)
plt.show()
print("Results plot saved as 'digit_recognition_results.png'")


# ── 7. Visualise Predictions vs Ground Truth ──────────────────────────────────
best_model = results[best_name]["model"]

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
fig.suptitle(f"Predictions vs Ground Truth – {best_name}", fontsize=11)

wrong_idx = np.where(best_preds != y_test)[0]
right_idx = np.where(best_preds == y_test)[0]

sample_idx = list(right_idx[:10])           # first row: correct
sample_idx2 = list(wrong_idx[:10]) if len(wrong_idx) >= 10 else list(wrong_idx)

for col, idx in enumerate(sample_idx):
    axes[0, col].imshow(X_test[idx].reshape(8, 8), cmap="gray_r")
    axes[0, col].set_title(f"P:{best_preds[idx]}\nA:{y_test[idx]}",
                            fontsize=7, color="green")
    axes[0, col].axis("off")

for col, idx in enumerate(sample_idx2):
    axes[1, col].imshow(X_test[idx].reshape(8, 8), cmap="gray_r")
    axes[1, col].set_title(f"P:{best_preds[idx]}\nA:{y_test[idx]}",
                            fontsize=7, color="red")
    axes[1, col].axis("off")

# hide unused axes in row 1
for col in range(len(sample_idx2), 10):
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Correct", fontsize=8, color="green")
axes[1, 0].set_ylabel("Wrong", fontsize=8, color="red")

plt.tight_layout()
plt.savefig("digit_predictions.png", dpi=150)
plt.show()
print("Predictions plot saved as 'digit_predictions.png'")


# ── 8. Predict a Single Image ─────────────────────────────────────────────────
sample_img = X_test[[0]]
sample_img_s = scaler.transform(sample_img)
pred_label = best_model.predict(sample_img_s)[0]
true_label = y_test[0]
print(f"\nSingle image → Predicted: {pred_label}  |  Actual: {true_label}")
