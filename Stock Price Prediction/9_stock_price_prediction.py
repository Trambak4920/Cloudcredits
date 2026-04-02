"""
=========================================
 Project 9: Stock Price Prediction
=========================================
Libraries : scikit-learn, pandas, numpy, matplotlib
Algorithm : Random Forest Regressor  +  SVR  +  Gradient Boosting
            Feature Engineering: lag features, rolling statistics
Dataset   : Synthetic OHLCV data (runs offline)
            ── OR ──  real data via yfinance (see Section 1b)
=========================================
Install   : pip install scikit-learn pandas numpy matplotlib
Optional  : pip install yfinance          (for real stock data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── 1a. Synthetic Stock Data (works 100% offline) ─────────────────────────────
np.random.seed(42)

n_days = 500
dates  = pd.date_range(start="2022-01-01", periods=n_days, freq="B")  # business days

# Simulate a realistic-looking stock price using random walk
price = 150.0
prices = []
for _ in range(n_days):
    daily_return = np.random.normal(0.0003, 0.012)
    price *= (1 + daily_return)
    prices.append(round(price, 2))

close  = np.array(prices)
open_  = close * (1 + np.random.normal(0, 0.004, n_days))
high   = np.maximum(close, open_) * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
low    = np.minimum(close, open_) * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
volume = np.random.randint(1_000_000, 10_000_000, n_days)

df = pd.DataFrame({
    "Date"  : dates,
    "Open"  : open_,
    "High"  : high,
    "Low"   : low,
    "Close" : close,
    "Volume": volume,
})
df.set_index("Date", inplace=True)

print("=== Stock Price Prediction ===")
print(f"Ticker : Synthetic (replace with yfinance for real data)")
print(df.tail())


# ── 1b. Real Data with yfinance (uncomment to use) ────────────────────────────
# import yfinance as yf
# df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
# df = df[["Open", "High", "Low", "Close", "Volume"]]
# print(df.tail())


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Lag features (previous N days' close)
    for lag in [1, 2, 3, 5, 10, 20]:
        d[f"Lag_{lag}"] = d["Close"].shift(lag)

    # Rolling statistics
    for window in [5, 10, 20, 50]:
        d[f"MA_{window}"]   = d["Close"].rolling(window).mean()
        d[f"Std_{window}"]  = d["Close"].rolling(window).std()

    # Daily return & volatility
    d["Daily_Return"]  = d["Close"].pct_change()
    d["Volatility_5"]  = d["Daily_Return"].rolling(5).std()
    d["Volatility_20"] = d["Daily_Return"].rolling(20).std()

    # High-Low spread, Open-Close spread
    d["HL_Spread"]  = d["High"] - d["Low"]
    d["OC_Spread"]  = d["Open"] - d["Close"]

    # Volume change
    d["Volume_Change"] = d["Volume"].pct_change()

    # Target: next day's close price
    d["Target"] = d["Close"].shift(-1)

    return d.dropna()

df_feat = engineer_features(df)
print(f"\nFeature matrix shape : {df_feat.shape}")
print(f"Features             : {[c for c in df_feat.columns if c != 'Target']}")


# ── 3. Train / Test Split (time-series aware — no shuffling) ──────────────────
feature_cols = [c for c in df_feat.columns if c not in ["Close", "Target"]]

X = df_feat[feature_cols].values
y = df_feat["Target"].values

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

test_dates = df_feat.index[split:]

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")


# ── 4. Scaling ─────────────────────────────────────────────────────────────────
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_test_s  = scaler_X.transform(X_test)

y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()


# ── 5. Model Training ──────────────────────────────────────────────────────────
models = {
    "Random Forest"    : RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "SVR (RBF)"        : SVR(kernel="rbf", C=10, epsilon=0.01),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_s, y_train_s)
    y_pred_s = model.predict(X_test_s)

    # Inverse-transform to actual price scale
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_actual = y_test

    mae  = mean_absolute_error(y_actual, y_pred)
    rmse = mean_squared_error(y_actual, y_pred) ** 0.5
    r2   = r2_score(y_actual, y_pred)

    results[name] = {"model": model, "preds": y_pred, "MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"  MAE  : ${mae:.2f}")
    print(f"  RMSE : ${rmse:.2f}")
    print(f"  R²   : {r2:.4f}")


# ── 6. Visualisation ───────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["R2"])
best_preds = results[best_name]["preds"]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Stock Price Prediction", fontsize=14, fontweight="bold")

# (a) Actual vs Predicted (time series)
axes[0, 0].plot(test_dates, y_test,   label="Actual",    color="steelblue", linewidth=1.5)
axes[0, 0].plot(test_dates, best_preds, label="Predicted", color="coral",  linewidth=1.5, linestyle="--")
axes[0, 0].set_title(f"Actual vs Predicted – {best_name}")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Price ($)")
axes[0, 0].legend()
axes[0, 0].tick_params(axis="x", rotation=30)

# (b) Scatter – Actual vs Predicted
axes[0, 1].scatter(y_test, best_preds, alpha=0.4, color="purple", s=10)
axes[0, 1].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], "r--", lw=2)
axes[0, 1].set_xlabel("Actual Price ($)")
axes[0, 1].set_ylabel("Predicted Price ($)")
axes[0, 1].set_title("Actual vs Predicted (Scatter)")

# (c) Residuals
residuals = y_test - best_preds
axes[1, 0].plot(test_dates, residuals, color="green", linewidth=1)
axes[1, 0].axhline(0, color="black", linestyle="--")
axes[1, 0].set_title("Residuals Over Time")
axes[1, 0].set_xlabel("Date")
axes[1, 0].set_ylabel("Residual ($)")
axes[1, 0].tick_params(axis="x", rotation=30)

# (d) Feature importance (for RF / GB)
best_model = results[best_name]["model"]
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances.nlargest(15).sort_values().plot(kind="barh", ax=axes[1, 1],
                                                 color="darkorange")
    axes[1, 1].set_title("Top 15 Feature Importances")
    axes[1, 1].set_xlabel("Importance")
else:
    axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("stock_price_prediction.png", dpi=150)
plt.show()
print("\nPlot saved as 'stock_price_prediction.png'")


# ── 7. Next-Day Forecast ──────────────────────────────────────────────────────
last_features = scaler_X.transform(X_test[[-1]])
next_price_s  = best_model.predict(last_features)
next_price    = scaler_y.inverse_transform(next_price_s.reshape(-1, 1))[0][0]
print(f"\nNext-day predicted close price : ${next_price:.2f}")
print(f"Last known close price         : ${y_test[-1]:.2f}")
