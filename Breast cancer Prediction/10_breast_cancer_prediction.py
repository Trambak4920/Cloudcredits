"""
=========================================
 Project 10: Breast Cancer Prediction
=========================================
Libraries : scikit-learn, pandas, numpy, matplotlib
Algorithm : Logistic Regression, SVM, Random Forest, Gradient Boosting
Dataset   : sklearn's built-in Breast Cancer Wisconsin dataset
            (569 samples, 30 features, binary: Malignant / Benign)
=========================================
Install   : pip install scikit-learn pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)


# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("Loading Breast Cancer Wisconsin dataset...")
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target        # 0 = Malignant, 1 = Benign

print(f"Dataset shape : {df.shape}")
print(f"Class labels  : {cancer.target_names}")   # ['malignant', 'benign']
print("\nClass distribution:")
print(df["target"].value_counts().rename(index={0: "Malignant", 1: "Benign"}))


# ── 2. EDA ─────────────────────────────────────────────────────────────────────
print("\n--- Basic Statistics ---")
print(df.describe().round(3))

print("\n--- Missing Values ---")
print(df.isnull().sum().sum(), "missing values")


# ── 3. Feature / Target Split ──────────────────────────────────────────────────
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")


# ── 4. Scaling ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# ── 5. Model Training ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression"  : LogisticRegression(max_iter=2000, random_state=42),
    "SVM (RBF)"            : SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest"        : RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred      = model.predict(X_test_s)
    y_prob      = model.predict_proba(X_test_s)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)
    cv_sc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")

    results[name] = {
        "model"  : model,
        "preds"  : y_pred,
        "probs"  : y_prob,
        "acc"    : acc,
        "auc"    : auc,
        "cv_mean": cv_sc.mean(),
        "cv_std" : cv_sc.std(),
    }
    print(f"\n--- {name} ---")
    print(f"  Test Accuracy  : {acc * 100:.2f}%")
    print(f"  ROC-AUC        : {auc:.4f}")
    print(f"  CV Accuracy    : {cv_sc.mean() * 100:.2f}% ± {cv_sc.std() * 100:.2f}%")
    print(classification_report(y_test, y_pred,
                                 target_names=["Malignant", "Benign"]))


# ── 6. Best Model ─────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]
print(f"\nBest model (by AUC) : {best_name}")
print(f"  Accuracy : {best['acc'] * 100:.2f}%")
print(f"  AUC      : {best['auc']:.4f}")


# ── 7. Visualisation ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Breast Cancer Prediction", fontsize=14, fontweight="bold")

# (a) Confusion Matrix
cm = confusion_matrix(y_test, best["preds"])
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Malignant", "Benign"]
).plot(ax=axes[0, 0], colorbar=False, cmap="Reds")
axes[0, 0].set_title(f"Confusion Matrix – {best_name}")

# (b) ROC Curves (all models)
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["probs"])
    axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("ROC Curves")
axes[0, 1].legend(fontsize=8)

# (c) Precision-Recall Curve (best model)
precision, recall, _ = precision_recall_curve(y_test, best["probs"])
ap = average_precision_score(y_test, best["probs"])
axes[1, 0].plot(recall, precision, color="darkorange", lw=2,
                label=f"AP = {ap:.3f}")
axes[1, 0].set_xlabel("Recall")
axes[1, 0].set_ylabel("Precision")
axes[1, 0].set_title(f"Precision-Recall Curve – {best_name}")
axes[1, 0].legend()

# (d) Feature Importance / Coefficients
best_model = best["model"]
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances.nlargest(15).sort_values().plot(
        kind="barh", ax=axes[1, 1], color="steelblue"
    )
    axes[1, 1].set_title("Top 15 Feature Importances")
elif hasattr(best_model, "coef_"):
    coefs = pd.Series(np.abs(best_model.coef_[0]), index=X.columns)
    coefs.nlargest(15).sort_values().plot(
        kind="barh", ax=axes[1, 1], color="mediumseagreen"
    )
    axes[1, 1].set_title("Top 15 Feature Coefficients (|value|)")
else:
    axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("breast_cancer_prediction.png", dpi=150)
plt.show()
print("\nPlot saved as 'breast_cancer_prediction.png'")


# ── 8. Predict on a New Patient Sample ────────────────────────────────────────
sample   = X_test.iloc[[0]]
sample_s = scaler.transform(sample)
pred     = best_model.predict(sample_s)[0]
prob     = best_model.predict_proba(sample_s)[0]
label    = cancer.target_names[pred]

print("\n--- Single Patient Prediction ---")
print(f"  Prediction  : {label.upper()}")
print(f"  Malignant probability : {prob[0] * 100:.1f}%")
print(f"  Benign    probability : {prob[1] * 100:.1f}%")
print(f"  Actual label          : {cancer.target_names[y_test.iloc[0]].upper()}")
