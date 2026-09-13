"""
Generates validation and learning curves for the Logistic Regression
baseline - diagnostic plots that reveal overfitting/underfitting and
whether more training data would actually help.

Run from inside src/: python generate_validation_curves.py
Expects train.csv (from data_cleaning.py) in ../data/
Saves output to ../results/validation_learning_curves.png
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve, learning_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_PATH = '../data/train.csv'
OUTPUT_PATH = '../results/validation_learning_curves.png'

df = pd.read_csv(TRAIN_PATH)

# Subsampled for speed - this is a diagnostic tool, not the final model.
# Remove this line to run on the full training set (slower, more precise).
df = df.sample(n=8000, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3)
X = vectorizer.fit_transform(df['statement_heavy'])
y = df['status']

print("Running validation curve (varying C, the regularization strength)...")
C_range = [0.001, 0.01, 0.1, 1, 10, 100]
vc_train_scores, vc_val_scores = validation_curve(
    LogisticRegression(class_weight='balanced', max_iter=1000),
    X, y, param_name='C', param_range=C_range,
    cv=3, scoring='f1_macro', n_jobs=-1
)
print("Train means:", vc_train_scores.mean(axis=1).round(3))
print("Val means:  ", vc_val_scores.mean(axis=1).round(3))

print("\nRunning learning curve (varying training set size)...")
train_sizes, lc_train_scores, lc_val_scores = learning_curve(
    LogisticRegression(class_weight='balanced', max_iter=1000),
    X, y, train_sizes=np.linspace(0.1, 1.0, 6),
    cv=3, scoring='f1_macro', n_jobs=-1
)
print("Sizes:      ", train_sizes)
print("Train means:", lc_train_scores.mean(axis=1).round(3))
print("Val means:  ", lc_val_scores.mean(axis=1).round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(C_range, vc_train_scores.mean(axis=1), 'o-', label='Training score', color='#5B8A80')
ax.fill_between(C_range,
                 vc_train_scores.mean(axis=1) - vc_train_scores.std(axis=1),
                 vc_train_scores.mean(axis=1) + vc_train_scores.std(axis=1),
                 alpha=0.15, color='#5B8A80')
ax.plot(C_range, vc_val_scores.mean(axis=1), 'o-', label='Validation score', color='#D9A05B')
ax.fill_between(C_range,
                 vc_val_scores.mean(axis=1) - vc_val_scores.std(axis=1),
                 vc_val_scores.mean(axis=1) + vc_val_scores.std(axis=1),
                 alpha=0.15, color='#D9A05B')
ax.set_xscale('log')
ax.set_xlabel('C (regularization strength - higher = less regularization)')
ax.set_ylabel('Macro F1 score')
ax.set_title('Validation Curve - Logistic Regression')
ax.legend()
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

ax = axes[1]
ax.plot(train_sizes, lc_train_scores.mean(axis=1), 'o-', label='Training score', color='#5B8A80')
ax.fill_between(train_sizes,
                 lc_train_scores.mean(axis=1) - lc_train_scores.std(axis=1),
                 lc_train_scores.mean(axis=1) + lc_train_scores.std(axis=1),
                 alpha=0.15, color='#5B8A80')
ax.plot(train_sizes, lc_val_scores.mean(axis=1), 'o-', label='Validation score', color='#D9A05B')
ax.fill_between(train_sizes,
                 lc_val_scores.mean(axis=1) - lc_val_scores.std(axis=1),
                 lc_val_scores.mean(axis=1) + lc_val_scores.std(axis=1),
                 alpha=0.15, color='#D9A05B')
ax.set_xlabel('Training set size')
ax.set_ylabel('Macro F1 score')
ax.set_title('Learning Curve - Logistic Regression')
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"\nSaved: {OUTPUT_PATH}")