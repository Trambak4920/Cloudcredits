# Cloudcredits — Machine Learning & AI Internship Projects

This repository contains Machine Learning projects completed as part of the
**ML & AI Internship**.

---

## 📁 Project Index

| # | File | Task | Algorithm | Dataset |
|---|------|------|-----------|---------|
| 1 | `House Price Prediction`      | House Price Prediction        | Linear Regression          | California Housing (sklearn) |
| 3 | `Handwritten Digit Recognition`| Handwritten Digit Recognition | CNN (Deep Learning)        | MNIST                        |
| 6 | `Sentiment Analysis`          | Sentiment Analysis            | Naive Bayes + LSTM         | IMDb (Keras built-in)        |
| 9 | `Stock Price Prediction`      | Stock Price Prediction        | LSTM                       | Yahoo Finance (AAPL)         |
| 10| `Breast Cancer Prediction`   | Breast Cancer Prediction      | SVM + Random Forest        | Breast Cancer Wisconsin      |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/Trambak4920/Cloudcredits.git
cd Cloudcredits

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Projects

Each script is fully self-contained. Run any of them directly:

```bash
python house_price_prediction.py
python handwritten_digit_recognition.py
python sentiment_analysis.py
python stock_price_prediction.py
python breast_cancer_prediction.py
```

Each script will:
- Load/download its dataset automatically
- Print evaluation metrics to the console
- Save plots as `.png` files in the same directory

---

## 📊 Project Details

### Task 1 — House Price Prediction
- **Objective**: Predict median house values based on demographic and geographic features.
- **Algorithm**: Linear Regression
- **Evaluation**: MSE, RMSE, R² Score
- **Output plots**: `task1_eda.png`, `task1_predictions.png`

### Task 3 — Handwritten Digit Recognition
- **Objective**: Classify handwritten digits (0–9) from 28×28 grayscale images.
- **Algorithm**: Convolutional Neural Network (CNN) with BatchNorm & Dropout
- **Evaluation**: Accuracy, Confusion Matrix
- **Output plots**: `task3_eda.png`, `task3_training_curves.png`, `task3_confusion_matrix.png`, `task3_sample_predictions.png`

### Task 6 — Sentiment Analysis on Movie Reviews
- **Objective**: Classify IMDb reviews as positive or negative.
- **Algorithm**: Naive Bayes (TF-IDF features) + LSTM (word embeddings)
- **Evaluation**: Accuracy, F1 Score, Confusion Matrix
- **Output plots**: `task6_nb_confusion.png`, `task6_lstm_curves.png`, `task6_lstm_confusion.png`

### Task 9 — Stock Price Prediction
- **Objective**: Predict next-day closing price of AAPL using historical prices.
- **Algorithm**: LSTM (sequence model with 60-day look-back window)
- **Evaluation**: MAE, RMSE
- **Output plots**: `task9_raw_prices.png`, `task9_loss_curve.png`, `task9_predictions.png`
- **Note**: Requires `yfinance`. Falls back to synthetic data if unavailable.

### Task 10 — Breast Cancer Prediction
- **Objective**: Classify tumors as malignant or benign from medical features.
- **Algorithm**: SVM (RBF Kernel) + Random Forest
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Output plots**: `task10_eda.png`, `task10_correlation_heatmap.png`, `task10_confusion_matrices.png`, `task10_roc_curves.png`, `task10_feature_importance.png`
---

## 🏷️ Tags

`#Cloudcredits` `#internship` `#machinelearning` `#artificialintelligence`
