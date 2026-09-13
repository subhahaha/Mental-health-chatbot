"""
Generates a per-class precision/recall/F1 comparison chart between the
Logistic Regression baseline and the fine-tuned DistilBERT model.

Retrains/evaluates Logistic Regression fresh (fast, no GPU needed).
DistilBERT's numbers are hardcoded from its actual test-set evaluation -
see results/distilbert_classification_report.txt for the source. We don't
reload the DistilBERT model here since that would need the full model
file and more setup just to re-print numbers we already have recorded.

Run from inside src/: python generate_comparison_chart.py
Expects train.csv and test.csv (from data_cleaning.py) in ../data/
Saves output to ../results/per_class_comparison.png
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_PATH = '../data/train.csv'
TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../results/per_class_comparison.png'

# --- Logistic Regression: train fresh and evaluate on the real test set ---
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
X_train = vectorizer.fit_transform(train_df['statement_heavy'])
X_test = vectorizer.transform(test_df['statement_heavy'])

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, train_df['status'])
y_pred = model.predict(X_test)

lr_report = classification_report(test_df['status'], y_pred, output_dict=True)
print(classification_report(test_df['status'], y_pred, digits=2))

# --- DistilBERT: hardcoded from its actual test-set run ---
# Source: results/distilbert_classification_report.txt
distilbert_report = {
    'Anxiety': {'precision': 0.85, 'recall': 0.89, 'f1-score': 0.87},
    'Bipolar': {'precision': 0.83, 'recall': 0.84, 'f1-score': 0.84},
    'Depression': {'precision': 0.83, 'recall': 0.70, 'f1-score': 0.76},
    'Normal': {'precision': 0.96, 'recall': 0.95, 'f1-score': 0.95},
    'Personality disorder': {'precision': 0.71, 'recall': 0.73, 'f1-score': 0.72},
    'Stress': {'precision': 0.71, 'recall': 0.79, 'f1-score': 0.75},
    'Suicidal': {'precision': 0.69, 'recall': 0.81, 'f1-score': 0.75},
}

labels = ['Normal', 'Anxiety', 'Bipolar', 'Depression', 'Stress', 'Suicidal', 'Personality disorder']
metrics = ['precision', 'recall', 'f1-score']

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for i, metric in enumerate(metrics):
    ax = axes[i]
    lr_vals = [lr_report[l][metric] for l in labels]
    db_vals = [distilbert_report[l][metric] for l in labels]

    x = np.arange(len(labels))
    width = 0.38

    ax.bar(x - width / 2, lr_vals, width, label='Logistic Regression', color='#B0B7B4')
    ax.bar(x + width / 2, db_vals, width, label='DistilBERT', color='#5B8A80')

    ax.set_title(metric.capitalize(), fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
    if i == 0:
        ax.set_ylabel('Score')
    ax.legend(fontsize=9)

plt.suptitle('Per-class Precision, Recall, and F1: Logistic Regression vs DistilBERT (test set)', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"\nSaved: {OUTPUT_PATH}")