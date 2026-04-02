"""
=========================================
 Project 6: Sentiment Analysis
=========================================
Libraries : scikit-learn, pandas, numpy, matplotlib
Algorithm : TF-IDF  +  Logistic Regression / Naive Bayes / SVM
Dataset   : sklearn's built-in 20newsgroups  ──  OR ──
            your own CSV (see Section 1b)
=========================================
Install   : pip install scikit-learn pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline


# ── 1a. Built-in Demo Dataset (movie-style reviews synthesised) ───────────────
#        Switch to Section 1b if you have your own CSV.

POSITIVE_REVIEWS = [
    "This movie was absolutely fantastic and I loved every moment of it.",
    "An outstanding film with brilliant performances and an engaging plot.",
    "One of the best movies I have ever seen. Highly recommended!",
    "The acting was superb and the story was incredibly moving.",
    "A masterpiece of cinema. Beautiful, emotional, and thought-provoking.",
    "Loved the direction and the script. Will watch again for sure.",
    "The cinematography was stunning and the soundtrack was perfect.",
    "Brilliant storytelling with a surprising twist at the end.",
    "A feel-good movie that left me smiling for hours.",
    "Wonderful characters and a heartwarming story. 10 out of 10.",
    "The best film of the year, without a doubt.",
    "Amazing visual effects paired with a solid story.",
    "Great pacing and excellent performances throughout.",
    "I was on the edge of my seat the whole time. Thrilling!",
    "A delightful experience from start to finish.",
]

NEGATIVE_REVIEWS = [
    "This was a complete waste of time. Boring and predictable.",
    "Terrible acting and a nonsensical plot. Avoid at all costs.",
    "I fell asleep halfway through. Absolutely dreadful film.",
    "The worst movie I have seen in years. Deeply disappointing.",
    "Poor direction, weak script, and forgettable characters.",
    "Nothing made sense and the ending was a letdown.",
    "I could not connect with any of the characters. Flat and dull.",
    "An overlong, self-indulgent mess with no clear direction.",
    "The special effects were cheap and the story was hollow.",
    "Not worth watching. Save your money and time.",
    "Painful to sit through. Awful dialogue and zero chemistry.",
    "A boring slog with no interesting moments whatsoever.",
    "The plot had more holes than a colander.",
    "Completely unoriginal and a pale imitation of better films.",
    "Disappointing in every single way. Expected much more.",
]

texts  = POSITIVE_REVIEWS + NEGATIVE_REVIEWS
labels = [1] * len(POSITIVE_REVIEWS) + [0] * len(NEGATIVE_REVIEWS)   # 1=Positive 0=Negative

df = pd.DataFrame({"text": texts, "label": labels})
df["sentiment"] = df["label"].map({1: "Positive", 0: "Negative"})

print("=== Sentiment Analysis ===")
print(f"Total samples   : {len(df)}")
print(df["sentiment"].value_counts())


# ── 1b. Your Own CSV (uncomment and adjust) ───────────────────────────────────
# df = pd.read_csv("reviews.csv")          # columns: 'text', 'label'  (0/1)
# df["sentiment"] = df["label"].map({1: "Positive", 0: "Negative"})


# ── 2. Text Preprocessing ──────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)
print("\nSample cleaned texts:")
print(df[["text", "clean_text"]].head(3).to_string())


# ── 3. Train / Test Split ──────────────────────────────────────────────────────
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_train)} samples")
print(f"Test  : {len(X_test)} samples")


# ── 4. Pipelines (TF-IDF + Classifier) ────────────────────────────────────────
pipelines = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)),
        ("clf",   MultinomialNB()),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)),
        ("clf",   LinearSVC(max_iter=2000, random_state=42)),
    ]),
}

results = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))


# ── 5. Best Model ─────────────────────────────────────────────────────────────
best_name = max(results, key=results.get)
best_pipe  = pipelines[best_name]
print(f"\nBest model: {best_name} (Accuracy = {results[best_name]:.4f})")


# ── 6. Confusion Matrix Plot ──────────────────────────────────────────────────
y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Sentiment Analysis Results", fontsize=13, fontweight="bold")

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Negative", "Positive"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Confusion Matrix – {best_name}")

# Model Accuracy Bar Chart
names = list(results.keys())
accs  = [results[n] for n in names]
bars  = axes[1].bar(names, accs, color=["steelblue", "coral", "mediumseagreen"])
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Model Comparison")
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{acc:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("sentiment_analysis.png", dpi=150)
plt.show()
print("Plot saved as 'sentiment_analysis.png'")


# ── 7. Live Prediction ────────────────────────────────────────────────────────
def predict_sentiment(text: str) -> str:
    cleaned = clean_text(text)
    pred = best_pipe.predict([cleaned])[0]
    return "Positive 😊" if pred == 1 else "Negative 😞"

print("\n--- Live Predictions ---")
test_sentences = [
    "This product is absolutely amazing, I love it!",
    "Terrible experience, would not recommend to anyone.",
    "It was okay, nothing special but not bad either.",
]
for s in test_sentences:
    print(f"  '{s}'\n  → {predict_sentiment(s)}\n")
