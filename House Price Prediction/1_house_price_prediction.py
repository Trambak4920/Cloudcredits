"""
=========================================
 Project 1: House Price Prediction
=========================================
Libraries : scikit-learn, pandas, numpy, matplotlib
Algorithm : Random Forest Regressor  +  Linear Regression (comparison)
Dataset   : sklearn's built-in California Housing dataset
=========================================
Install   : pip install scikit-learn pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nFeatures: {housing.feature_names}")
print(f"Target  : MedHouseVal (median house value in $100k)")


# ── 2. Exploratory Data Analysis ───────────────────────────────────────────────
print("\n--- Basic Stats ---")
print(df.describe())

# Correlation heatmap (text-based for simplicity)
print("\n--- Correlation with Target ---")
print(df.corr()["MedHouseVal"].sort_values(ascending=False))


# ── 3. Feature / Target Split ──────────────────────────────────────────────────
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")


# ── 4. Scaling ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ── 5. Model Training ──────────────────────────────────────────────────────────
models = {
    "Linear Regression"       : LinearRegression(),
    "Random Forest"           : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"       : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")


# ── 6. Best Model Deep-Dive (Random Forest) ────────────────────────────────────
best_model = models["Random Forest"]
y_pred_best = best_model.predict(X_test_scaled)

# Feature importance
importances = pd.Series(best_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\n--- Feature Importances (Random Forest) ---")
print(importances)


# ── 7. Visualisation ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("House Price Prediction", fontsize=14, fontweight="bold")

# (a) Actual vs Predicted
axes[0].scatter(y_test, y_pred_best, alpha=0.3, color="steelblue", s=10)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--", lw=2)
axes[0].set_xlabel("Actual Price ($100k)")
axes[0].set_ylabel("Predicted Price ($100k)")
axes[0].set_title("Actual vs Predicted (Random Forest)")

# (b) Residuals
residuals = y_test - y_pred_best
axes[1].hist(residuals, bins=50, color="coral", edgecolor="white")
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")

# (c) Feature Importance
importances.plot(kind="barh", ax=axes[2], color="mediumseagreen")
axes[2].set_title("Feature Importances")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("house_price_prediction.png", dpi=150)
plt.show()
print("\nPlot saved as 'house_price_prediction.png'")


# ── 8. Predict on a Single Sample ─────────────────────────────────────────────
sample = X_test.iloc[[0]]
sample_scaled = scaler.transform(sample)
prediction = best_model.predict(sample_scaled)[0]
print(f"\nSample Prediction  : ${prediction * 100_000:,.0f}")
print(f"Actual Value       : ${y_test.iloc[0] * 100_000:,.0f}")
